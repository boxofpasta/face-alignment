"""
Includes a lot of code that was used for quick experimenting. 
Junk builds quickly over months of experimentation.
Most of it is no longer really needed -- the parts that are have been extracted out of this file.
"""

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

class ModelFactory:

    def __init__(self, ibug_version=True):
        self.im_width = 224
        self.im_height = 224
        self.coords_sparsity = 1
        self.num_coords = helenUtils.getNumCoords(self.coords_sparsity, ibug_version=ibug_version)
        self.mask_side_len = 28
        self.epsilon = 1E-8
        self.ibug_version = ibug_version
        self.custom_objects = {
            'squaredDistanceLoss': self.squaredDistanceLoss, 
            'pointMaskSoftmaxLoss': self.pointMaskSoftmaxLoss,
            'identityLoss': self.identityLoss,
            'maskSigmoidLoss': self.maskSigmoidLoss,
            'MaskSigmoidLossLayer': layerUtils.MaskSigmoidLossLayer,
            'MaskSigmoidLossLayerNoCrop': layerUtils.MaskSigmoidLossLayerNoCrop,
            'SquaredDistanceLossLayer': layerUtils.SquaredDistanceLossLayer,
            'iouLoss': self.iouLoss,
            'scaledSquaredDistanceLoss': self.percentageBboxDistanceLoss,
            'percentageBboxDistanceLoss': self.percentageBboxDistanceLoss,
            'pointMaskSigmoidLoss': self.pointMaskSigmoidLoss,
            'relu6': mobilenet.relu6,
            'DepthwiseConvolution2d': DepthwiseConvolution2D,
            'DepthwiseConv2D': DepthwiseConvolution2D, #mobilenet.DepthwiseConv2D, # there seems to be a name conflict lol
            'CropAndResize' : layerUtils.CropAndResize,
            'Resize': layerUtils.Resize,
            'PerturbBboxes' : layerUtils.PerturbBboxes,
            'PointMaskSoftmaxLossLayer': layerUtils.PointMaskSoftmaxLossLayer,
            'StopGradientLayer': layerUtils.StopGradientLayer,
            'pointMaskDistanceLoss': self.pointMaskDistanceLoss,
            'pointMaskDistance': self.pointMaskDistance,
            'pointMaskDistanceLossPresetDims': self.pointMaskDistanceLossPresetDims,
            'TileMultiply': layerUtils.TileMultiply,
            'TileSubtract': layerUtils.TileSubtract,
            'MaskMean': layerUtils.MaskMean,
            'BoxesFromCenters': layerUtils.BoxesFromCenters,
            'SliceBboxes': layerUtils.SliceBboxes,
            'cascadedPointMaskSigmoidLoss': self.cascadedPointMaskSigmoidLoss,
        }

    def getSaved(self, path, frozen=False):
        print 'Loading model ...'
        if frozen:
            model = load_model(path, custom_objects=self.custom_objects, compile=False)
            for layer in model.layers:
                layer.trainable = False
            model.compile()
        else:
            model = load_model(path, custom_objects=self.custom_objects)
        print 'Loading complete '
        return model

    """ 
    ----------------------------------------------------------------
        Collection of factory methods to build keras models. 
    ----------------------------------------------------------------
    """

    def getFullyConnected(self, alpha=1.0):
        alpha_1 = alpha
        input_tensor = None
        shallow = False
        input_shape = (self.im_height, self.im_width, 3)

        # https://github.com/rcmalli/keras-mobilenet/blob/master/keras_mobilenet/mobilenet.py
        input_shape = _obtain_input_shape(input_shape,
                                        default_size=224,
                                        min_size=96,
                                        data_format=K.image_data_format(),
                                        require_flatten=True)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        x = Convolution2D(int(32 * alpha_1), (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layerUtils.depthwiseConvBlock(x, 32 * alpha_1, 64 * alpha_1)

        # 112x112
        x = layerUtils.depthwiseConvBlock(x, 64 * alpha_1, 128 * alpha_1, down_sample=True)
        x = layerUtils.depthwiseConvBlock(x, 128 * alpha_1, 128 * alpha_1)

        # 56x56
        x = layerUtils.depthwiseConvBlock(x, 128 * alpha_1, 256 * alpha_1, down_sample=True)
        x = layerUtils.depthwiseConvBlock(x, 256 * alpha_1, 256 * alpha_1)

        # 28x28
        x = layerUtils.depthwiseConvBlock(x, 256 * alpha_1, 512 * alpha_1, down_sample=True)

        if not shallow:
            for _ in range(5):
                x = layerUtils.depthwiseConvBlock(x, 512 * alpha_1, 512 * alpha_1)

        # End of backbone:
        # Output dims are 14 x 14 x (512 * alpha_1)
        x = layerUtils.depthwiseConvBlock(x, 512 * alpha_1, 1024 * alpha_1, down_sample=True)
        x = layerUtils.depthwiseConvBlock(x, 1024 * alpha_1, 1024 * alpha_1)
        
        #x = GlobalAveragePooling2D()(x)
        #x = Dense(units=24, activation='linear')(x)
        #x = Reshape((12, 2))(x)
        model = Model(inputs=[img_input], outputs=[x])
        optimizer = optimizers.adam(lr=3E-4)
        model.compile(loss=self.squaredDistanceLoss, optimizer=optimizer)
        return model

    def getBboxRegressor(self):
        in_shape = (self.im_width, self.im_height, 3)
        base_model = mobilenet.MobileNet(include_top=False, input_shape=in_shape, alpha=0.25)

        # without the final pooling layer
        #x = base_model.layers[-2].output
        x = base_model.output
        x = Flatten()(x)
        x = Dense(units=4, activation='linear')(x)
        model = Model(inputs=base_model.input, outputs=x)
        model.compile(loss=self.scaledSquaredDistanceBboxLoss, optimizer='adam')
        return model

    def getPointMaskerAttention(self):
        im_shape = (self.im_width, self.im_height, 3)
        num_coords = 13
        img_input = Input(shape=im_shape)
        label_masks = Input(shape=(self.mask_side_len, self.mask_side_len, num_coords))
        
        x = Convolution2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
        # 112x112

        x = MaxPool2D()(x)
        # 56x56

        x = layerUtils.depthwiseConvBlock(x, 32, 64)
        x = layerUtils.depthwiseConvBlock(x, 64, 64)

        x = MaxPool2D()(x)
        # 28x28

        x = layerUtils.depthwiseConvBlock(x, 64, 128)
        x = layerUtils.depthwiseConvBlock(x, 128, 128)

        z = layerUtils.depthwiseConvBlock(x, 128, 128)
        z = layerUtils.depthwiseConvBlock(z, 128, 4)
        z = Conv2DTranspose(
            4, kernel_size=(3, 3),
            strides=(2, 2),
            activation='linear',
            padding='same')(z)
        # 56x56

        x = layerUtils.depthwiseConvBlock(x, 128, 256, down_sample=True)
        # 14x14

        #x = layerUtils.depthwiseConvBlock(x, 256, 256)
        #x = layerUtils.depthwiseConvBlock(x, 256, 512, down_sample=True)

        x = layerUtils.depthwiseConvBlock(x, 256, 256, dilation_rate=[2,2])
        x = layerUtils.depthwiseConvBlock(x, 256, 256, dilation_rate=[4,4])
        x = layerUtils.depthwiseConvBlock(x, 256, 16, final_activation='leaky_relu')

        #x = layerUtils.depthwiseConvBlock(x, 256, 512, dilation_rate=[4,4])
        #x = layerUtils.depthwiseConvBlock(x, 512, 16, final_activation='tanh')
        
        method = tf.image.ResizeMethod.BILINEAR
        #x = layerUtils.Resize(28, method)(x)
        #x = layerUtils.depthwiseConvBlock(x, num_coords, num_coords, final_activation='tanh')
        #x = layerUtils.Resize(112, method)(x)
        x = Conv2DTranspose(
            16, kernel_size=(3, 3),
            strides=(4, 4),
            activation='linear',
            padding='same')(x)

        #x = layerUtils.TileMultiply(16 / 4)([z, x])
        x = layerUtils.TileSubtract(16 / 4)([z, x])
        #x = Multiply()([x, z])
        # 56x56

        #x = layerUtils.depthwiseConvBlock(x, 128, 128)
        x = layerUtils.depthwiseConvBlock(x, 16, 64)
        x = layerUtils.depthwiseConvBlock(x, 64, 16, final_activation='leaky_relu')
        x = layerUtils.depthwiseConvBlock(x, 16, num_coords, kernel_size=(7,7), final_activation='linear')
        pred = x
        
        model = Model(
            inputs=[img_input], 
            outputs=[pred]
        )

        optimizer = optimizers.SGD(lr=5E-5, momentum=0.9, nesterov=True)
        #optimizer = optimizers.adam(lr=6E-2)
        model.compile(loss=[self.pointMaskDistanceLoss], metrics=[self.pointMaskDistance], optimizer=optimizer)
        return model

    def getPointMaskerDilated(self):
        im_shape = (self.im_width, self.im_height, 3)

        # lip only for now
        l = self.mask_side_len
        num_coords = 12 
        img_input = Input(shape=im_shape)
        # 224x224 resolution

        label_masks = Input(shape=(l, l, num_coords))

        x = Convolution2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
        # 112x112 resolution
        
        x = layerUtils.depthwiseConvBlock(x, 32, 32)
        x = layerUtils.depthwiseConvBlock(x, 32, 64, down_sample=True)
        # 56x56 resolution
        # 11x11 receptive field wrt image
        
        """
        # using odd numbers helps with decreasing feature overlap, and increases information coverage.
        x = layerUtils.depthwiseConvBlock(x, 64, 64, dilation_rate=[2,2])
        x = layerUtils.depthwiseConvBlock(x, 64, 64, dilation_rate=[5,5])
        x = layerUtils.depthwiseConvBlock(x, 64, 64, dilation_rate=[9,9])
        x = layerUtils.depthwiseConvBlock(x, 64, 128, dilation_rate=[16,16])

        x = layerUtils.depthwiseConvBlock(x, 128, 128)
        #x = layerUtils.depthwiseConvBlock(x, 64, 64)
        # 139x139 receptive field wrt image (deprecated, needs manual update)

        x = layerUtils.depthwiseConvBlock(x, 128, num_coords, final_activation='linear')
        """
        x = layerUtils.depthwiseConvBlock(x, 64, num_coords, final_activation='linear')

        #loss = layerUtils.PointMaskSoftmaxLossLayer(l)([label_masks, x])
        #loss = layerUtils.MaskSigmoidLossLayerNoCrop(l)([label_masks, x])
        #x = Activation('sigmoid')(x)
        pred = x
        
        model = Model(
            inputs=[img_input], 
            outputs=[pred]
        )
        optimizer = optimizers.adam(lr=6E-2)
        model.compile(loss=[self.pointMaskSigmoidLoss], metrics=[self.pointMaskDistance], optimizer=optimizer)
        return model


    def getPointMaskerSmall(self, in_side_len, out_side_len, in_channels, out_channels):
        # This one does not compile on its own
        im_shape = (in_side_len, in_side_len, in_channels)
        out_shape = (out_side_len, out_side_len, out_channels)
        img_input = Input(im_shape)

        method = tf.image.ResizeMethod.BILINEAR
        x = Convolution2D(16, (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
        x = layerUtils.depthwiseConvBlock(x, 16, 64, down_sample=True)
        x = layerUtils.depthwiseConvBlock(x, 64, 64, dilation_rate=[2,2])
        x = layerUtils.depthwiseConvBlock(x, 64, 64, dilation_rate=[4,4])
        x = layerUtils.depthwiseConvBlock(x, 64, 16)
        x = layerUtils.Resize(14, method)(x)
        x = layerUtils.depthwiseConvBlock(x, 16, out_channels, final_activation='linear')
        x = layerUtils.Resize(out_side_len, method)(x)
        model = Model(inputs=img_input, outputs=x)
        return model

    def getPointMaskerCascadedHead(self, backbone, out_side_len, in_channels):
        out_shape = (out_side_len, out_side_len, 1)

        method = tf.image.ResizeMethod.BILINEAR
        x = layerUtils.depthwiseConvBlock(backbone, in_channels, 64, dilation_rate=[2,2])
        x = layerUtils.depthwiseConvBlock(x, 64, 16)
        x = layerUtils.Resize(14, method)(x)
        x = layerUtils.depthwiseConvBlock(x, 16, 1, final_activation='linear')
        x = layerUtils.Resize(out_side_len, method)(x)
        return x

    def getPointMaskerConcatCascaded(self):

        # outerlip only for now
        l = self.mask_side_len
        num_coords = 13
        method = tf.image.ResizeMethod.BILINEAR
        base_model, backbone = self.getPointMaskerConcat(compile=False)
        img_input = base_model.input
        base_preds = base_model.output
        base_preds_normalized = Activation('sigmoid')(base_preds)

        # get crop for each coordinate
        # note that boxes and mask means are all in normalized image coordinates, i.e [0, 1]
        mask_means = layerUtils.MaskMean()(base_preds_normalized)
        boxes = layerUtils.BoxesFromCenters(28.0 / self.im_height)(mask_means)

        # slice and join to a separate refine model for each coordinate
        refined_coords = []
        for i in range(num_coords):
            box = layerUtils.SliceBboxes(i)(boxes)

            # using mask-rcnn's roi pooling
            crop = layerUtils.CropAndResize(7)([backbone, box])
            refined_output = self.getPointMaskerCascadedHead(crop, 28, 64)
            refined_coords.append(refined_output)

            # using crops from input directly 
            # crop = layerUtils.CropAndResize(28)([base_model.input, box])
            # refine_model = self.getPointMaskerSmall(28, 28, 3, 1)
            # output = refine_model(crop)
            # refined_coords.append(output)

        refined_preds = Concatenate()(refined_coords)
        all_preds = Concatenate()([base_preds, refined_preds])
        model = Model(inputs=base_model.input, outputs=[base_preds, all_preds])
        
        #optimizer = optimizers.adam(lr=6E-2)

        optimizer = optimizers.SGD(lr=5E-5, momentum=0.9, nesterov=True)
        model.compile(
            loss=[ self.pointMaskDistanceLossPresetDims, self.cascadedPointMaskSigmoidLoss ], 
            # metrics=[ self.pointMaskDistance, self.zeroLoss ], 
            optimizer=optimizer
        )
        return model

    def cascadedPointMaskSigmoidLoss(self, y_true, y_pred):
        num_coords = 13
        method = tf.image.ResizeMethod.BILINEAR

        # split the preds up into their component parts (dammit keras!)
        base_preds = y_pred[:,:,:,:13]
        base_preds = tf.stop_gradient(base_preds)
        refined_preds = y_pred[:,:,:,13:]

        # get crops;
        # this is actually repetitive code from the model architecture, 
        # blame keras for inflexible loss function arguments
        base_preds_normalized = Activation('sigmoid')(base_preds)
        mask_means = layerUtils.MaskMean()(base_preds_normalized)
        true_means = layerUtils.MaskMean()(y_true)
        #boxes = layerUtils.BoxesFromCenters(28.0 / self.im_height)(true_means)
        #boxes = layerUtils.PerturbBboxes([0.8, 1.2], [-0.25, 0.25])(boxes)
        boxes = layerUtils.BoxesFromCenters(28.0 / self.im_height)(mask_means)

        # avoid penalizing refined mask when the initial estimate is not even close to truth
        sqrd_diffs = tf.squared_difference(mask_means, true_means)
        dists = tf.sqrt(tf.reduce_sum(sqrd_diffs, axis=-1))
        thresh = 0.30 * 28.0 / self.im_height
        loss_mask = tf.where(dists < thresh, tf.ones(tf.shape(dists)), tf.zeros(tf.shape(dists)))
        loss_mask = tf.expand_dims(loss_mask, 1)
        loss_mask = tf.expand_dims(loss_mask, 1)

        label_crops = []
        for i in range(num_coords):
            box = Lambda( lambda x: x[:,i,:] )(boxes)
            label_mask = Lambda( lambda x: x[:,:,:,i] )(y_true)
            label_mask = tf.expand_dims(label_mask, axis=-1)
            label_crop = layerUtils.CropAndResize(28)([label_mask, box])
            label_crops.append(label_crop)

        labels = Concatenate()(label_crops)
        labels *= loss_mask
        refined_preds = layerUtils.Resize(28, method)(refined_preds)
        refined_preds *= loss_mask

        return self.pointMaskDistanceLoss(labels, refined_preds)


    def getPointMaskerConcat(self, compile=True):
        im_shape = (self.im_width, self.im_height, 3)

        # lip only for now
        l = self.mask_side_len
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
        #z.append(layerUtils.depthwiseConvBlock(b, 128, 128))
        #z.append(layerUtils.depthwiseConvBlock(x, 128, 128))
        x = layerUtils.depthwiseConvBlock(x, 128, 256, down_sample=True)

        # 14x14
        # having a larger kernel size gives a larger receptive field, which helps prevent misclassification
        x = layerUtils.depthwiseConvBlock(x, 256, 256, dilation_rate=[2,2])
        x = layerUtils.depthwiseConvBlock(x, 256, 256, dilation_rate=[4,4])
        x = layerUtils.depthwiseConvBlock(x, 256, 256, dilation_rate=[8,8])
        
        #z.append(x)
    
        method = tf.image.ResizeMethod.BILINEAR
        x = layerUtils.depthwiseConvBlock(x, 256, 128)
        x = layerUtils.Resize(28, method)(x)
        
        #z[0] = layerUtils.depthwiseConvBlock(z[0], 128, 64)
        #x = Concatenate()([x, z[0]])
        
        #x = layerUtils.depthwiseConvBlock(x, 192, 128)
        x = layerUtils.depthwiseConvBlock(x, 128, num_coords, final_activation='linear')
        #x = layerUtils.depthwiseConvBlock(x, 192, num_coords, final_activation='linear')
        #x = layerUtils.depthwiseConvBlock(x, 32, num_coords, final_activation='linear')

        #loss = layerUtils.PointMaskSoftmaxLossLayer(l)([label_masks, x])
        #loss = layerUtils.MaskSigmoidLossLayerNoCrop(l)([label_masks, x])
        #x = Activation('sigmoid')(x)
        pred = x
        #loss = Lambda(lambda x: x, name='f0')(loss)
        
        model = Model(
            inputs=[img_input], 
            outputs=[pred]
        )
        
        #optimizer = optimizers.adam(lr=6E-2)
        if compile:
            optimizer = optimizers.SGD(lr=5E-5, momentum=0.9, nesterov=True)
            model.compile(loss=[ self.pointMaskSigmoidLoss ], metrics=[ self.pointMaskDistance ], optimizer=optimizer)
        return model, backbone

        

    def getPointMasker(self):
        im_shape = (self.im_width, self.im_height, 3)
        masks_shape = (self.mask_side_len, self.mask_side_len, self.num_coords)
        summed_masks_shape = (self.mask_side_len, self.mask_side_len, 1)
        img_input = Input(shape=im_shape)
        label_masks = Input(shape=masks_shape)
        label_summed_masks = Input(shape=summed_masks_shape)

        x = Convolution2D(32, (3, 3), strides=(1, 1), padding='same', use_bias=False)(img_input)

        num_features = [64, 128, 256, 512, 512]
        z_layers = [None] * 4
        x, z_layers[0] = layerUtils.rcfBlock(x, 32, num_features[0], 2, z_out_layers=2) 
        x, z_layers[1] = layerUtils.rcfBlock(x, num_features[0], num_features[1], 2, z_out_layers=4) 
        x, z_layers[2] = layerUtils.rcfBlock(x, num_features[1], num_features[2], 3, z_out_layers=1)
        x, z_layers[3] = layerUtils.rcfBlock(x, num_features[2], num_features[3], 3, z_out_layers=1)
        #x, z_layers[4] = layerUtils.rcfBlock(x, num_features[3], num_features[4], 3, z_out_layers=1)
        
        # want 112x112 feature maps
        z_layers[0] = layerUtils.depthwiseConvBlock(z_layers[0], 2, 4, down_sample=True)

        # upsample 
        z_layers[2] = Conv2DTranspose(
            1, kernel_size=(3, 3),
            strides=(2, 2),
            activation='linear',
            padding='same')(z_layers[2])
        z_layers[2] = Convolution2D(1, (1,1))(z_layers[2])

        z_layers[3] = Conv2DTranspose(
            1, kernel_size=(3, 3),
            strides=(4, 4),
            activation='linear',
            padding='same')(z_layers[3])
        z_layers[3] = Convolution2D(1, (1,1))(z_layers[3])

        """
        # long strides xD
        z_layers[4] = Conv2DTranspose(
            1, kernel_size=(3, 3),
            strides=(8, 8),
            activation='linear',
            padding='same')(z_layers[4])"""

        final = Concatenate()(z_layers)
        final = layerUtils.depthwiseConvBlock(final, 10, 32, down_sample=True)
        final = layerUtils.depthwiseConvBlock(final, 32, self.num_coords)

        # losses
        losses = 3 * [None]

        # final prediction is 56x56
        label_masks_downsampled = layerUtils.Resize(self.mask_side_len/2, tf.image.ResizeMethod.AREA)(label_masks)
        losses[0] = layerUtils.MaskSigmoidLossLayerNoCrop(self.mask_side_len)([label_summed_masks, z_layers[2]])
        losses[1] = layerUtils.MaskSigmoidLossLayerNoCrop(self.mask_side_len)([label_summed_masks, z_layers[3]])
        losses[2] = layerUtils.PointMaskSoftmaxLossLayer(self.mask_side_len/2)([label_masks_downsampled, final])

        # names
        losses[0] = Lambda(lambda x: x, name='z2')(losses[0])
        losses[1] = Lambda(lambda x: x, name='z3')(losses[1])
        losses[2] = Lambda(lambda x: x, name='final')(losses[2])

        model = Model(
            inputs=[img_input, label_masks, label_summed_masks], 
            outputs=[losses[0], losses[1], losses[2], z_layers[2], z_layers[3], final]
        )
        optimizer = optimizers.adam(lr=3E-3)
        model.compile(loss=[self.identityLoss, self.identityLoss, self.identityLoss, None, None, None], optimizer=optimizer)
        return model

    def getLipMasker(self, alpha_1=1, alpha_2=1):
        
        input_tensor = None
        shallow = False
        input_shape = (self.im_height, self.im_width, 3)

        # https://github.com/rcmalli/keras-mobilenet/blob/master/keras_mobilenet/mobilenet.py
        input_shape = _obtain_input_shape(input_shape,
                                        default_size=224,
                                        min_size=96,
                                        data_format=K.image_data_format(),
                                        require_flatten=True)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        # labels to be set as inputs as well
        bbox_gts = Input(shape=([4]))
        mask_gts = Input(shape=(self.im_height, self.im_width, 1))

        x = Convolution2D(int(32 * alpha_1), (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = layerUtils.depthwiseConvBlock(x, 32 * alpha_1, 64 * alpha_1)

        # 112x112
        cf1 = x 
        x = layerUtils.depthwiseConvBlock(x, 64 * alpha_1, 128 * alpha_1, down_sample=True)
        x = layerUtils.depthwiseConvBlock(x, 128 * alpha_1, 128 * alpha_1)

        # 56x56
        cf2 = x
        x = layerUtils.depthwiseConvBlock(x, 128 * alpha_1, 256 * alpha_1, down_sample=True)
        x = layerUtils.depthwiseConvBlock(x, 256 * alpha_1, 256 * alpha_1)

        # 28x28
        cf3 = x
        x = layerUtils.depthwiseConvBlock(x, 256 * alpha_1, 512 * alpha_1, down_sample=True)

        if not shallow:
            for _ in range(5):
                x = layerUtils.depthwiseConvBlock(x, 512 * alpha_1, 512 * alpha_1)

        cf4 = x

        # End of backbone:
        # Output dims are 14 x 14 x (512 * alpha_1)

        # Bbox regressor head: 
        b = layerUtils.depthwiseConvBlock(x, 512 * alpha_1, 1024 * alpha_1, down_sample=True)
        b = layerUtils.depthwiseConvBlock(b, 1024 * alpha_1, 1024 * alpha_1)
        b = GlobalAveragePooling2D()(b)
        b = Dense(4)(b)

        # use for the mask branch, to help mask predictions be more robust even for poor bounding boxes
        randomized_bboxes = layerUtils.PerturbBboxes([0.7, 1.4], [-0.2, 0.2])(b)

        cfs = [cf1, cf2, cf3, cf4]

        # don't want the randomized boxes for test-time inference
        randomized_masks = layerUtils.getMaskHead(randomized_bboxes, cfs, alpha_1)
        #masks = layerUtils.getMaskHead(b, cfs, alpha_1)

        #a = Lambda(lambda a: K.squeeze(a, axis=-1))(a)
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

        #masks = Activation('sigmoid', name='masks')(masks)
        bboxes = Lambda(lambda b: b, name='bboxes')(b)
        
        #mask_loss = layerUtils.MaskSigmoidLossLayer(self.mask_side_len, name='mask_obj')([mask_gts, a, bboxes])

        # try to generate ground truth masks (which were obtained from ground truth crops)
        mask_loss = layerUtils.MaskSigmoidLossLayer(self.mask_side_len, name='mask_obj')([mask_gts, randomized_masks, randomized_bboxes])
        #mask_gts_cropped = layerUtils.CropAndResize(self.mask_side_len)([mask_gts, bbox_gts])
        #mask_gts_cropped = Lambda(lambda a: K.squeeze(a, axis=-1))(mask_gts_cropped)
        bbox_loss = layerUtils.SquaredDistanceLossLayer(name='bbox_obj')([bbox_gts, bboxes])
        #total_loss = Lambda(lambda(l1, l2) : l1 + l2)([mask_loss, bbox_loss])

        #model = Model(inputs=[inputs, bbox_gts, mask_gts], outputs=[mask_loss, bbox_loss, bboxes, mask_gts_cropped])
        model = Model(inputs=[inputs, bbox_gts, mask_gts], outputs=[mask_loss, bbox_loss, randomized_bboxes, randomized_masks])
        #model =  Model(inputs=[inputs, bbox_gts, mask_gts], outputs=[mask_loss, bbox_loss])
        optimizer = optimizers.adam(lr=1E-4)
        #model.compile(loss=[self.identityLoss, self.identityLoss], optimizer=optimizer)
        model.compile(loss=[self.identityLoss, self.identityLoss, None, None], optimizer=optimizer)
        #model = Model(inputs=[inputs, bbox_gts, mask_gts], outputs=[mask_loss])
        #model.compile(loss=[self.identityLoss], optimizer='adam')
        #model.summary()
        return model

    def getLipMaskerZoomed(self, alpha=1):

        # for bbox regressor
        alpha_1 = alpha

        # for mask cnn
        alpha_2 = 1.0
        input_tensor = None
        shallow = False
        input_shape = (self.im_height, self.im_width, 3)

        # https://github.com/rcmalli/keras-mobilenet/blob/master/keras_mobilenet/mobilenet.py
        input_shape = _obtain_input_shape(input_shape,
                                        default_size=224,
                                        min_size=96,
                                        data_format=K.image_data_format(),
                                        require_flatten=True)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        # labels to be set as inputs as well
        mask_gts = Input(shape=(self.im_height, self.im_width, 1))

        """
        # Mask head: 
        # https://arxiv.org/pdf/1703.06870.pdf
        #a = layerUtils.CropAndResize(7)([x, b])
        #a = Convolution2D(int(512 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
        #a = layerUtils.depthwiseConvBlock(x, 512 * alpha, 512 * alpha)
        """

        # note to self: alternative to sharing features -- just use a new fully-convolutional architecture 
        a = layerUtils.Resize(112, tf.image.ResizeMethods.BILINEAR)(img_input)
        a = Convolution2D(int(32 * alpha_2), (3, 3), strides=(2, 2), padding='same', use_bias=False)(a)
        a = BatchNormalization()(a)
        a = Activation('relu')(a)
        a = layerUtils.depthwiseConvBlock(a, 32 * alpha_2, 64 * alpha_2)
        a = layerUtils.depthwiseConvBlock(a, 64 * alpha_2, 128 * alpha_2, down_sample=True)
        a = layerUtils.depthwiseConvBlock(a, 128 * alpha_2, 128 * alpha_2)
        a = layerUtils.depthwiseConvBlock(a, 128 * alpha_2, 256 * alpha_2, down_sample=True)
        a = layerUtils.depthwiseConvBlock(a, 256 * alpha_2, 256 * alpha_2)
        a = layerUtils.depthwiseConvBlock(a, 256 * alpha_2, 512 * alpha_2, down_sample=True)
        if not shallow:
            for _ in range(5):
                a = layerUtils.depthwiseConvBlock(a, 512 * alpha_2, 512 * alpha_2)

        # 7x7
        conv_transpose_depth = 128
        a = Conv2DTranspose(int(conv_transpose_depth * alpha_2), kernel_size=(3, 3),
                strides=(2, 2),
                activation='relu',
                padding='same',
                data_format='channels_last')(a)
        for i in range(3):
            a = layerUtils.depthwiseConvBlock(a, conv_transpose_depth * alpha_2, conv_transpose_depth * alpha_2)

        # 14x14
        a = Conv2DTranspose(int(conv_transpose_depth * alpha_2), kernel_size=(3, 3),
                strides=(2, 2),
                activation='relu',
                padding='same',
                data_format='channels_last')(a)
        for i in range(3):
            a = layerUtils.depthwiseConvBlock(a, conv_transpose_depth * alpha_2, conv_transpose_depth * alpha_2)

        # 28x28
        a = Conv2DTranspose(int(conv_transpose_depth * alpha_2), kernel_size=(3, 3),
                strides=(2, 2),
                activation='relu',
                padding='same',
                data_format='channels_last')(a)
        a = layerUtils.depthwiseConvBlock(a, conv_transpose_depth * alpha_2, 1)

        #a = Lambda(lambda a: K.squeeze(a, axis=-1))(a)
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

        # a is the unnormalized bboxes
        masks = Activation('sigmoid', name='masks')(a)
        #bboxes = Lambda(lambda b: b, name='bboxes')(b)
        
        #mask_loss = layerUtils.MaskSigmoidLossLayer(self.mask_side_len, name='mask_obj')([mask_gts, a, bboxes])

        # try to generate ground truth masks (which were obtained from ground truth crops)
        #mask_loss = layerUtils.MaskSigmoidLossLayer(self.mask_side_len, name='mask_obj')([mask_gts, a, bboxes])
        mask_loss = layerUtils.MaskSigmoidLossLayerNoCrop(self.mask_side_len, name='mask_obj')([mask_gts, a])
        #mask_gts_cropped = layerUtils.CropAndResize(self.mask_side_len)([mask_gts, bbox_gts])
        #mask_gts_cropped = Lambda(lambda a: K.squeeze(a, axis=-1))(mask_gts_cropped)
        #bbox_loss = layerUtils.SquaredDistanceLossLayer(name='bbox_obj')([bbox_gts, bboxes])
        #total_loss = Lambda(lambda(l1, l2) : l1 + l2)([mask_loss, bbox_loss])

        #model = Model(inputs=[inputs, bbox_gts, mask_gts], outputs=[mask_loss, bbox_loss, bboxes, mask_gts_cropped])
        model = Model(inputs=[inputs, mask_gts], outputs=[mask_loss, masks])
        optimizer = optimizers.adam(lr=4E-4)
        model.compile(loss=[self.identityLoss, None], optimizer=optimizer)
        #model = Model(inputs=[inputs, bbox_gts, mask_gts], outputs=[mask_loss])
        #model.compile(loss=[self.identityLoss], optimizer='adam')
        #model.summary()
        return model

    """ 
    ----------------------------------------------------------------
        Helpers to build custom tensorflow loss functions.
    ----------------------------------------------------------------
    """
    def pointMaskSoftmaxLoss(self, labels, preds):
        labels = tf.reshape(labels, (-1, self.mask_side_len * self.mask_side_len, self.num_coords))
        preds = tf.reshape(preds, (-1, self.mask_side_len * self.mask_side_len, self.num_coords))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=preds, dim=1)
        return tf.reduce_sum(cross_entropy)

    def pointMaskDistanceLossPresetDims(self, labels, preds):
        method = tf.image.ResizeMethod.BILINEAR
        labels = layerUtils.Resize(28, method)(labels)
        preds = layerUtils.Resize(28, method)(preds)
        return self.pointMaskDistanceLoss(labels, preds)

    def pointMaskSigmoidLoss(self, labels, preds):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
        return tf.reduce_sum(cross_entropy)

    def pointMaskDistanceLoss(self, labels, preds):
        """
        We want to penalize mask output errors that are spatially further away from ground truth.
        This is accomplished by multiplying the usual cross entropy loss with squared distance.
        """
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
        width = tf.shape(labels)[2]
        height = tf.shape(labels)[1]
        x_inds = tf.cast(tf.expand_dims(tf.range(0, width), 0), tf.float32)
        y_inds = tf.cast(tf.expand_dims(tf.range(0, height), 1), tf.float32)

        # move channels (corresponding to the coords dim) together with batch dim to avoid confusion
        labels = tf.transpose(labels, [0, 3, 1, 2])
        preds = tf.transpose(preds, [0, 3, 1, 2])

        # even the labels are not necessarily normalized
        labels /= utils.expandDimsRepeatedly(tf.reduce_sum(labels, axis=[2,3]) + self.epsilon, 2, False)
        preds /= utils.expandDimsRepeatedly(tf.reduce_sum(preds, axis=[2,3]) + self.epsilon, 2, False)

        # performs a weighted sum to get center coordinates
        x_label = tf.reduce_sum(x_inds * labels, axis=[2,3])
        y_label = tf.reduce_sum(y_inds * labels, axis=[2,3])

        # lots of massaging things into the right shape
        x_inds = tf.expand_dims(x_inds, axis=0)
        y_inds = tf.transpose(y_inds, [1, 0])
        y_inds = tf.expand_dims(y_inds, axis=0)
        x_label = tf.expand_dims(x_label, axis=-1)
        y_label = tf.expand_dims(y_label, axis=-1)
        
        # get distances of each point to ground truth
        x_dists = tf.square(x_inds - x_label)
        y_dists = tf.square(y_inds - y_label)
        x_dists = tf.expand_dims(x_dists, axis=-2)
        y_dists = tf.expand_dims(y_dists, axis=-1)
        squared_dists = x_dists + y_dists
        squared_dists = tf.transpose(squared_dists, [0, 2, 3, 1])

        # normalize, since these numbers are pretty large
        squared_dists /= tf.cast(width, tf.float32) ** 2

        # penalize only in spots that should be pretty close to 0
        #diff = tf.maximum(preds - labels, 0)
        """
        epsilon = 1E-5
        zeros = tf.zeros(tf.shape(labels))
        ones = tf.ones(tf.shape(labels))
        mask = tf.where(labels > epsilon, x = zeros, y = ones)
        mask = tf.transpose(mask, [0, 2, 3, 1])
        """

        return tf.reduce_sum((0.1 + squared_dists) * cross_entropy)
        #return tf.reduce_sum(mask * squared_dists * cross_entropy)
        #x_diff = (x_label - x_pred) / tf.cast(width, tf.float32)
        #y_diff = (y_label - y_pred) / tf.cast(height, tf.float32)
        #return tf.reduce_sum(tf.square(x_diff) + tf.square(y_diff))

    def pointMaskDistance(self, labels, preds):
        """
        Sum of euclidean distances squared between the centers of preds and labels.
        """
        #labels = tf.reshape(labels, (-1, self.mask_side_len, self.mask_side_len, 12))
        #preds = tf.reshape(preds, (-1, self.mask_side_len, self.mask_side_len, 12))
        preds = tf.sigmoid(preds)
        width = tf.shape(labels)[2]
        height = tf.shape(labels)[1]
        x_inds = tf.cast(tf.expand_dims(tf.range(0, width), 0), tf.float32) / tf.cast(width, tf.float32)
        y_inds = tf.cast(tf.expand_dims(tf.range(0, height), 1), tf.float32) / tf.cast(height, tf.float32)

        # move channels (corresponding to the coords dim) together with batch dim to avoid confusion
        labels = tf.transpose(labels, [0, 3, 1, 2])
        preds = tf.transpose(preds, [0, 3, 1, 2])

        # even the labels are not necessarily normalized
        labels /= utils.expandDimsRepeatedly(tf.reduce_sum(labels, axis=[2,3]) + self.epsilon, 2, False)
        preds /= utils.expandDimsRepeatedly(tf.reduce_sum(preds, axis=[2,3]) + self.epsilon, 2, False)

        # performs a weighted sum to get center coordinates
        x_label = tf.reduce_sum(x_inds * labels, axis=[2,3])
        y_label = tf.reduce_sum(y_inds * labels, axis=[2,3])
        x_pred = tf.reduce_sum(x_inds * preds, axis=[2,3])
        y_pred = tf.reduce_sum(y_inds * preds, axis=[2,3])

        x_dist = tf.squared_difference(x_label, x_pred)
        y_dist = tf.squared_difference(y_label, y_pred)
        return tf.reduce_sum(x_dist + y_dist)

    """def maskSigmoidLoss(self, bboxes):
        def maskSigmoidLossHelper(labels, preds):
            labels = tf.expand_dims(labels, axis=-1)
            cropped_labels = layerUtils.CropAndResize([self.mask_side_len, self.mask_side_len])([labels, bboxes])
            cropped_labels = tf.squeeze(cropped_labels)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=cropped_labels)
            return tf.reduce_sum(cross_entropy)
        return maskSigmoidLossHelper"""

    def identityLoss(self, labels, preds):
        return tf.reshape(tf.reduce_sum(preds), (1,))

    def zeroLoss(self, labels, preds):
        return tf.constant(0.0)

    def maskSigmoidLoss(self, labels, preds, bboxes):
        labels = tf.expand_dims(labels, axis=-1)
        cropped_labels = layerUtils.CropAndResize(self.mask_side_len)([labels, bboxes])
        cropped_labels = tf.squeeze(cropped_labels)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=cropped_labels)
        return tf.reduce_sum(cross_entropy)

    def squaredDistanceLoss(self, labels, preds):
        return tf.reduce_sum(tf.square(labels - preds))

    def iouLoss(self, labels, preds):
        """
        Training does not converge well for this loss. There might be an error somewhere.
        """
        iouWeight = 0.9999
        distanceLoss = self.squaredDistanceLoss(labels, preds)
        intersection_dx = tf.maximum(tf.minimum(preds[:,2], labels[:,2]) - tf.maximum(preds[:,0], labels[:,0]), 0)
        intersection_dy = tf.maximum(tf.minimum(preds[:,3], labels[:,3]) - tf.maximum(preds[:,1], labels[:,1]), 0)
        
        intersection_area = intersection_dx * intersection_dy
        preds_area = tf.maximum(preds[:,2] - preds[:,0], 0) * tf.maximum(preds[:,3] - preds[:,1], 0)
        true_area = tf.maximum(labels[:,2] - labels[:,0], 0) * tf.maximum(labels[:,3] - labels[:,1], 0)
        union = preds_area + true_area - intersection_area
        return (1.0 - iouWeight) * distanceLoss - iouWeight * tf.reduce_sum(intersection_area / (union + self.epsilon))
    
    def percentageDistanceLoss(self, labels, preds):
        """
        Parameters
        ----------
        labels: 
            Should have shape (batch_size, num_coords, 2)
        """
        preds = tf.reshape(preds, (-1, self.num_coords, 2))
        labels = tf.reshape(labels, (-1, self.num_coords, 2))
        x_coords = labels[:,:,0]
        y_coords = labels[:,:,1]
        widths = tf.reduce_max(x_coords, axis=1, keep_dims=True) - tf.reduce_min(x_coords, axis=1, keep_dims=True)
        heights = tf.reduce_max(y_coords, axis=1, keep_dims=True) - tf.reduce_min(y_coords, axis=1, keep_dims=True)
        divisor = tf.concat([widths, heights], axis=1)
        dist_squared = tf.square(labels - preds)
        scaled = dist_squared / tf.square(divisor)
        return tf.reduce_sum(scaled)

    def percentageDistanceLossNp(self, labels, preds):
        """
        Just a port of scaledSquaredDistanceLoss for numpy.
        """
        preds = np.reshape(preds, (-1, self.num_coords, 2))
        x_coords = labels[:,:,0]
        y_coords = labels[:,:,1]
        widths = np.max(x_coords, axis=1) - np.max(x_coords, axis=1)
        heights = np.max(y_coords, axis=1) - np.min(y_coords, axis=1)
        divisor = np.concatenate([widths, heights], axis=1)
        dist_squared = np.square(labels - preds)
        scaled = dist_squared / np.square(divisor)
        return np.sum(scaled)

    def scaledSquaredDistanceLossBbox(self, labels, preds):
        """
        Labels are bboxes in this case.
        y and x distances scaled by height / width of the ground truth box.
        """
        widths = tf.expand_dims(labels[:,2] - labels[:,0], axis=1)
        heights = tf.expand_dims(labels[:,3] - labels[:,1], axis=1)
        
        # [widths, heights, widths, heights]
        divisor = tf.concat([widths, heights], axis=1)
        divisor = tf.concat([divisor, divisor], axis=1)

        dist_squared = tf.square(labels - preds)
        scaled = dist_squared / divisor
        return tf.reduce_sum(scaled)

    def percentageBboxDistanceLoss(self, labels, preds):
        """
        y and x distances scaled by height / width of the ground truth box.
        """
        widths = tf.expand_dims(labels[:,2] - labels[:,0], axis=1)
        heights = tf.expand_dims(labels[:,3] - labels[:,1], axis=1)
        
        # [widths, heights, widths, heights]
        divisor = tf.concat([widths, heights], axis=1)
        divisor = tf.concat([divisor, divisor], axis=1)

        dist_squared = tf.square(labels - preds)
        scaled = dist_squared / tf.square(divisor)
        return tf.reduce_sum(scaled)

