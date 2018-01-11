import os
import numpy as np
import utils.helenUtils as helenUtils
import utils.generalUtils as utils
import json
import time
import sys
import BatchGenerator
from keras.layers import Dense, Reshape, BatchNormalization, Flatten
from keras.layers import Dropout, Conv2DTranspose
from keras.models import Model, Sequential
from keras.models import load_model
from keras.applications import mobilenet
from keras.applications.imagenet_utils import _obtain_input_shape
import tensorflow as tf

# mobilenet things
import keras.backend as K
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

    def getSaved(self, path):
        print 'Loading model ...'
        model = load_model(path, custom_objects={
            'squaredDistanceLoss': self.squaredDistanceLoss,
            'pointMaskSoftmaxLoss': self.pointMaskSoftmaxLoss,
            'iouLoss': self.iouLoss,
            'scaledSquaredDistanceLoss': self.percentageBboxDistanceLoss,
            'percentageBboxDistanceLoss': self.percentageBboxDistanceLoss,
            'relu6': mobilenet.relu6,
            'DepthwiseConv2D': mobilenet.DepthwiseConv2D
        })
        print 'COMPLETE'
        return model

    """ 
    ----------------------------------------------------------------
        Collection of factory methods to build keras models. 
    ----------------------------------------------------------------
    """

    def getFullyConnected(self):

        """ Mobilenet with the last layer replaced by coordinate regression """
        in_shape = (self.im_width, self.im_height, 3)
        base_model = mobilenet.MobileNet(include_top=False, input_shape=in_shape, alpha=0.25)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(units=2 * self.num_coords, activation='linear')(x)
        x = Reshape((self.num_coords, 2))(x)
        model = Model(inputs=base_model.input, outputs=x)

        model.compile(loss=self.squaredDistanceLoss, optimizer='adam')
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
        model.compile(loss=self.scaledSquaredDistanceLoss, optimizer='adam')
        return model

    def getPointMasker(self):

        """ Mobilenet with some 'deconv' layers near the end """
        in_shape = (self.im_width, self.im_height, 3)
        base_model = mobilenet.MobileNet(include_top=False, input_shape=in_shape)
        x = base_model.output

        # 7x7 head resolution, need 2^3 to get 56x56 resolution
        x = Conv2DTranspose(256, kernel_size=(3, 3),
                strides=(2, 2),
                activation='relu',
                padding='same')(x)
        x = Conv2DTranspose(256, kernel_size=(3, 3),
                strides=(2, 2),
                activation='relu',
                padding='same')(x)
        x = Conv2DTranspose(self.num_coords, kernel_size=(3, 3),
                strides=(2, 2),
                activation='linear',
                padding='same')(x)
        model = Model(inputs=base_model.input, outputs=x)
        model.compile(loss=self.pointMaskSoftmaxLoss, optimizer='adam')
        return model

    def getLipMasker(self, alpha=1):
        input_tensor = None
        shallow = False
        input_shape = (self.im_width, self.im_height, 3)

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

        x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = DepthwiseConvolution2D(int(32 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = DepthwiseConvolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        if not shallow:
            for _ in range(5):
                x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)

        # End of backbone:
        # Output dims are 14 x 14 x (512 * alpha)

        # Mask head: 
        # mask code here ...
        a = x

        # Bbox regressor head: 
        b = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
        b = BatchNormalization()(b)
        b = Activation('relu')(b)
        b = Convolution2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(b)
        b = BatchNormalization()(b)
        b = Activation('relu')(b)

        b = DepthwiseConvolution2D(int(1024 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(b)
        b = BatchNormalization()(b)
        b = Activation('relu')(b)
        b = Convolution2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(b)
        b = BatchNormalization()(b)
        b = Activation('relu')(b)

        b = GlobalAveragePooling2D()(b)
        b = Dense(4)(b)

        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

        masks = a
        bbox_coords = b
        model = Model(inputs, outputs=[masks, bbox_coords])
        model.compile(loss=[self.maskerLoss, self.squaredDistanceLoss], optimizer='adam')
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

    def maskerLoss(self, labels, preds):
        #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
        #return tf.reduce_sum(cross_entropy)
        return 0

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

