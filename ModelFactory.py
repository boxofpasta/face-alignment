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

    def __init__(self):
        self.im_width = 224
        self.im_height = 224
        self.coords_sparsity = 1
        self.num_coords = helenUtils.getNumCoords(self.coords_sparsity)
        self.mask_side_len = 56
        self.epsilon = 1E-5
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
            'pointMaskDistance': self.pointMaskDistance
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
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=24, activation='linear')(x)
        x = Reshape((12, 2))(x)
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

    def getPointMaskerSmall(self):
        im_shape = (self.im_width, self.im_height, 3)

        # lip only for now
        l = self.mask_side_len
        num_coords = 12 
        img_input = Input(shape=im_shape)

        # different resolutions. 
        # we could also just downsample directly here but doing it externally gives better control.
        label_masks = [
            Input(shape=(l, l, num_coords)),
            Input(shape=(l/2, l/2, num_coords)),
            Input(shape=(l/4, l/4, num_coords)),
            #Input(shape=(l/8, l/8, num_coords))
        ]

        z = []

        # at different resolutions
        preds = []

        # 224x224
        x = Convolution2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
        #x = MaxPool2D()(x)

        # 112x112
        b = layerUtils.StopGradientLayer()(x)
        z.append(layerUtils.depthwiseConvBlock(b, 32, num_coords))
        x = layerUtils.depthwiseConvBlock(x, 32, 64, down_sample=True)
        #x = layerUtils.depthwiseConvBlock(x, 64, 64)
        #x = MaxPool2D()(x)

        # 56x56
        b = layerUtils.StopGradientLayer()(x)
        z.append(layerUtils.depthwiseConvBlock(b, 64, num_coords))
        x = layerUtils.depthwiseConvBlock(x, 64, 128, down_sample=True)
        #x = layerUtils.depthwiseConvBlock(x, 128, 128)
        #x = MaxPool2D()(x)
        
        # 28x28
        b = layerUtils.StopGradientLayer()(x)
        z.append(layerUtils.depthwiseConvBlock(b, 128, num_coords))
        x = layerUtils.depthwiseConvBlock(x, 128, 256, down_sample=True)
        #x = layerUtils.depthwiseConvBlock(x, 256, 256)
        #x = MaxPool2D()(x)

        # 14x14
        x = layerUtils.depthwiseConvBlock(x, 256, 64)
        x = layerUtils.depthwiseConvBlock(x, 64, num_coords)
        x = layerUtils.depthwiseConvBlock(x, num_coords, num_coords, kernel_size=(5,5), final_activation='linear')
        z.append(x)

        losses = []
        losses.append(layerUtils.MaskSigmoidLossLayerNoCrop(l/4)([label_masks[-1], x]))
        x = Activation('sigmoid')(x)
        preds.append(x)
        
        # up-sampling chain
        method = tf.image.ResizeMethod.BILINEAR
        x = layerUtils.Resize(28, method)(x)
        x = layerUtils.StopGradientLayer()(x)
        x = Multiply()([x, z[2]])
        x = DepthwiseConvolution2D(int(num_coords), (3,3), strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = DepthwiseConvolution2D(int(num_coords), (3,3), strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        losses.append(layerUtils.MaskSigmoidLossLayerNoCrop(l/2)([label_masks[-2], x]))
        x = Activation('sigmoid')(x)
        preds.append(x)

        x = layerUtils.Resize(56, method)(x)
        x = layerUtils.StopGradientLayer()(x)
        x = Multiply()([x, z[1]])
        x = DepthwiseConvolution2D(int(num_coords), (3,3), strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = DepthwiseConvolution2D(int(num_coords), (3,3), strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        losses.append(layerUtils.MaskSigmoidLossLayerNoCrop(l)([label_masks[-3], x]))
        x = Activation('sigmoid')(x)
        preds.append(x)

        """
        x = layerUtils.Resize(112, method)(x)
        x = layerUtils.StopGradientLayer()(x)
        x = Multiply()([x, z[0]])
        x = DepthwiseConvolution2D(int(num_coords), (3,3), strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = DepthwiseConvolution2D(int(num_coords), (3,3), strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        losses.append(layerUtils.MaskSigmoidLossLayerNoCrop(l)([label_masks[-4], x]))
        x = Activation('sigmoid')(x)
        preds.append(x)
        """

        # they both end up being from highest to lowest resolution, same order as z
        losses.reverse()
        preds.reverse()
        losses[0] = Lambda(lambda x: x, name='f0')(losses[0])
        losses[1] = Lambda(lambda x: x, name='f1')(losses[1])
        losses[2] = Lambda(lambda x: x, name='f2')(losses[2])
        #losses[3] = Lambda(lambda x: x, name='f3')(losses[3])
        
        model = Model(
            inputs=[img_input] + label_masks, 
            outputs=[losses[0], losses[1], losses[2], preds[1], preds[-1]]
        )
        optimizer = optimizers.adam(lr=1E-2)
        model.compile(loss=[self.identityLoss, self.identityLoss, self.identityLoss, None, None], optimizer=optimizer)
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
        
        x = layerUtils.depthwiseConvBlock(x, 64, 64, dilation_rate=[2,2])
        x = layerUtils.depthwiseConvBlock(x, 64, 64, dilation_rate=[4,4])
        x = layerUtils.depthwiseConvBlock(x, 64, 64, dilation_rate=[8,8])
        x = layerUtils.depthwiseConvBlock(x, 64, 128, dilation_rate=[16,16])
        
        x = layerUtils.depthwiseConvBlock(x, 128, 128)
        #x = layerUtils.depthwiseConvBlock(x, 64, 64)
        # 139x139 receptive field wrt image (deprecated, needs manual update)

        x = layerUtils.depthwiseConvBlock(x, 128, num_coords, final_activation='linear')

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

    def getPointMaskerVanilla(self):
        """ 
        Nothing fancy about this one. Just a few skip connections.
        """
        im_shape = (self.im_width, self.im_height, 3)

        # lip only for now
        l = self.mask_side_len
        num_coords = 12 
        img_input = Input(shape=im_shape)
        label_masks = Input(shape=(l, l, num_coords))

        z = []

        # 224x224
        x = Convolution2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)

        # 112x112
        x = layerUtils.depthwiseConvBlock(x, 32, 64, down_sample=True)
        x = layerUtils.depthwiseConvBlock(x, 64, 64)

        # 56x56
        b = layerUtils.StopGradientLayer()(x)
        z.append(layerUtils.depthwiseConvBlock(b, 64, 64))
        x = layerUtils.depthwiseConvBlock(x, 64, 128, down_sample=True)
        
        # 28x28
        b = layerUtils.StopGradientLayer()(x)
        z.append(layerUtils.depthwiseConvBlock(b, 128, 128))
        x = layerUtils.depthwiseConvBlock(x, 128, 128, down_sample=True)

        # 14x14
        # having a larger kernel size gives a larger receptive field, which helps prevent misclassification
        x = layerUtils.depthwiseConvBlock(x, 128, 128, kernel_size=(7,7))
        z.append(x)

        method = tf.image.ResizeMethod.BILINEAR
        x = layerUtils.Resize(28, method)(x)
        x = Add()([x, z[1]])
        x = layerUtils.depthwiseConvBlock(x, 128, 64)
        x = layerUtils.depthwiseConvBlock(x, 64, 64)

        x = layerUtils.Resize(56, method)(x)
        x = Add()([x, z[0]])
        x = layerUtils.depthwiseConvBlock(x, 64, 32)
        x = layerUtils.depthwiseConvBlock(x, 32, num_coords, final_activation='linear')

        #loss = layerUtils.PointMaskSoftmaxLossLayer(l)([label_masks, x])
        loss = layerUtils.MaskSigmoidLossLayerNoCrop(l)([label_masks, x])
        x = Activation('sigmoid')(x)
        pred = x
        loss = Lambda(lambda x: x, name='f0')(loss)
        
        model = Model(
            inputs=[img_input, label_masks], 
            outputs=[loss, pred]
        )
        optimizer = optimizers.adam(lr=5E-3)
        model.compile(loss=[self.identityLoss, None], optimizer=optimizer)
        return model

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

        # even the labels are not necessarily normalized
        labels /= utils.expandDimsRepeatedly(tf.reduce_sum(labels, axis=[2,3]), 2, False)

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

        return tf.reduce_sum(squared_dists * cross_entropy)
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
        x_inds = tf.cast(tf.expand_dims(tf.range(0, width), 0), tf.float32)
        y_inds = tf.cast(tf.expand_dims(tf.range(0, height), 1), tf.float32)

        # move channels (corresponding to the coords dim) together with batch dim to avoid confusion
        labels = tf.transpose(labels, [0, 3, 1, 2])
        preds = tf.transpose(preds, [0, 3, 1, 2])

        # even the labels are not necessarily normalized
        labels /= utils.expandDimsRepeatedly(tf.reduce_sum(labels, axis=[2,3]), 2, False)
        preds /= utils.expandDimsRepeatedly(tf.reduce_sum(preds, axis=[2,3]), 2, False)

        # performs a weighted sum to get center coordinates
        x_label = tf.reduce_sum(x_inds * labels, axis=[2,3])
        y_label = tf.reduce_sum(y_inds * labels, axis=[2,3])
        x_pred = tf.reduce_sum(x_inds * preds, axis=[2,3])
        y_pred = tf.reduce_sum(y_inds * preds, axis=[2,3])

        x_dist = tf.squared_difference(x_label, x_pred) / tf.cast(width * width, tf.float32)
        y_dist = tf.squared_difference(y_label, y_pred) / tf.cast(height * height, tf.float32)
        return tf.reduce_sum(x_dist + y_dist)
        
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

        # even the labels are not necessarily normalized
        labels /= utils.expandDimsRepeatedly(tf.reduce_sum(labels, axis=[2,3]), 2, False)

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

        return tf.reduce_sum(squared_dists * cross_entropy)
        #x_diff = (x_label - x_pred) / tf.cast(width, tf.float32)
        #y_diff = (y_label - y_pred) / tf.cast(height, tf.float32)
        #return tf.reduce_sum(tf.square(x_diff) + tf.square(y_diff))

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

