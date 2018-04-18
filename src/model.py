import os
import numpy as np
import utils.helenUtils as helenUtils
import utils.generalUtils as utils
import json
import time
import sys
import BatchGenerator
from keras import optimizers
from keras import losses
from keras.layers import Dense, Reshape, BatchNormalization, Flatten, Multiply, Activation
from keras.layers import Dropout, Conv2DTranspose, Lambda, Concatenate, Add, MaxPool2D
from keras.models import Model, Sequential
from keras.models import load_model
from keras.applications import mobilenet
from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from losses import pointMaskDistanceLossPresetDims, cascadedPointMaskSigmoidLoss
from keras.models import load_model
import utils.layerUtils as layerUtils
import tensorflow as tf

# mobilenet things
from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, \
    GlobalAveragePooling2D, Dense, BatchNormalization, Activation
from keras.models import Model
from keras.engine.topology import get_source_inputs
from depthwise_conv2d import DepthwiseConvolution2D


def getCustomObjects():
    return {
        'pointMaskSigmoidLoss': pointMaskSigmoidLoss,
        'relu6': mobilenet.relu6,
        'DepthwiseConvolution2d': DepthwiseConvolution2D,
        'DepthwiseConv2D': DepthwiseConvolution2D, #mobilenet.DepthwiseConv2D, # there seems to be a name conflict lol
        'CropAndResize' : layerUtils.CropAndResize,
        'Resize': layerUtils.Resize,
        'PerturbBboxes' : layerUtils.PerturbBboxes,
        'PointMaskSoftmaxLossLayer': layerUtils.PointMaskSoftmaxLossLayer,
        'StopGradientLayer': layerUtils.StopGradientLayer,
        'pointMaskDistanceLoss': pointMaskDistanceLoss,
        'pointMaskDistance': pointMaskDistance,
        'pointMaskDistanceLossPresetDims': pointMaskDistanceLossPresetDims,
        'MaskMean': layerUtils.MaskMean,
        'BoxesFromCenters': layerUtils.BoxesFromCenters,
        'SliceBboxes': layerUtils.SliceBboxes,
        'cascadedPointMaskSigmoidLoss': cascadedPointMaskSigmoidLoss,
    }


def loadPointMaskerRefined(path):
    model = load_model(path, custom_objects=getCustomObjects)
    return model

def getPointMaskerRefinedHead(backbone, out_side_len, in_channels):
    out_shape = (out_side_len, out_side_len, 1)

    method = tf.image.ResizeMethod.BILINEAR
    x = layerUtils.depthwiseConvBlock(backbone, in_channels, 64, dilation_rate=[2,2])
    x = layerUtils.depthwiseConvBlock(x, 64, 16)
    x = layerUtils.Resize(14, method)(x)
    x = layerUtils.depthwiseConvBlock(x, 16, 1, final_activation='linear')
    x = layerUtils.Resize(out_side_len, method)(x)
    return x


def getPointMasker(im_side_len, mask_side_len, compile=True):
    im_shape = (im_side_len, im_side_len, 3)

    # lip only for now
    l = mask_side_len
    num_coords = 13
    img_input = Input(shape=im_shape)
    label_masks = Input(shape=(l, l, num_coords))

    z = []

    # 224x224
    x = Convolution2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)

    # 112x112
    x = layerUtils.depthwiseConvBlock(x, 32, 64, down_sample=True)
    x = layerUtils.depthwiseConvBlock(x, 64, 64)
    backbone = layerUtils.depthwiseConvBlock(x, 64, 64)

    # 56x56
    x = layerUtils.depthwiseConvBlock(x, 64, 128, down_sample=True)
    
    # 28x28
    x = layerUtils.depthwiseConvBlock(x, 128, 256, down_sample=True)

    # 14x14
    # having a larger kernel size gives a larger receptive field, which helps prevent misclassification
    x = layerUtils.depthwiseConvBlock(x, 256, 256, dilation_rate=[2,2])
    x = layerUtils.depthwiseConvBlock(x, 256, 256, dilation_rate=[4,4])
    x = layerUtils.depthwiseConvBlock(x, 256, 256, dilation_rate=[8,8])

    method = tf.image.ResizeMethod.BILINEAR
    x = layerUtils.depthwiseConvBlock(x, 256, 128)
    x = layerUtils.Resize(28, method)(x)
    x = layerUtils.depthwiseConvBlock(x, 128, num_coords, final_activation='linear')
    pred = x
    
    model = Model(
        inputs=[img_input], 
        outputs=[pred]
    )
    
    #optimizer = optimizers.adam(lr=6E-2)
    if compile:
        optimizer = optimizers.SGD(lr=5E-5, momentum=0.9, nesterov=True)
        model.compile(loss=[ pointMaskSigmoidLoss ], optimizer=optimizer)
    return model, backbone

# formerly getPointMaskerConcatRefined
def getPointMaskerRefined(im_side_len, mask_side_len):

    # outerlip only for now
    l = mask_side_len
    num_coords = 13
    method = tf.image.ResizeMethod.BILINEAR
    base_model, backbone = getPointMasker(im_side_len, mask_side_len, compile=False)
    img_input = base_model.input
    base_preds = base_model.output
    base_preds_normalized = Activation('sigmoid')(base_preds)

    # get crop for each coordinate
    # note that boxes and mask means are all in normalized image coordinates, i.e [0, 1]
    mask_means = layerUtils.MaskMean()(base_preds_normalized)
    boxes = layerUtils.BoxesFromCenters(28.0 / im_side_len)(mask_means)

    # slice and join to a separate refine model for each coordinate
    refined_coords = []
    for i in range(num_coords):
        box = layerUtils.SliceBboxes(i)(boxes)

        # using mask-rcnn's roi align to focus on feature crops
        crop = layerUtils.CropAndResize(7)([backbone, box])
        refined_output = getPointMaskerRefinedHead(crop, 28, 64)
        refined_coords.append(refined_output)

        # using crops from input directly 
        # crop = layerUtils.CropAndResize(28)([base_model.input, box])
        # refine_model = self.getPointMaskerSmall(28, 28, 3, 1)
        # output = refine_model(crop)
        # refined_coords.append(output)

    refined_preds = Concatenate()(refined_coords)
    all_preds = Concatenate()([base_preds, refined_preds])
    model = Model(inputs=base_model.input, outputs=[base_preds, all_preds])
    optimizer = optimizers.SGD(lr=5E-5, momentum=0.9, nesterov=True)
    model.compile(
        loss=[ pointMaskDistanceLossPresetDims, cascadedPointMaskSigmoidLoss ], 
        optimizer=optimizer
    )
    return model