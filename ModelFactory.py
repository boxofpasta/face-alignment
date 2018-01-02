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
import tensorflow as tf

class ModelFactory:

    def __init__(self):
        self.im_width = 224
        self.im_height = 224
        self.num_coords = 194
        self.mask_side_len = 56

    """ 
    ----------------------------------------------------------------
        Collection of factory methods to build keras models. 
    ----------------------------------------------------------------
    """

    def getFullyConnected(self):

        """ Mobilenet with the last layer replaced by coordinate regression """
        in_shape = (self.im_width, self.im_height, 3)
        base_model = mobilenet.MobileNet(include_top=False, input_shape=in_shape)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(units=2 * self.num_coords, activation='linear')(x)
        model = Model(inputs=base_model.input, outputs=x)

        model.compile(loss=self.squaredDistanceLoss, optimizer='adam')
        return model

    def getFullyConvolutional(self):

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
        model.compile(loss=self.heatmapSoftmaxLoss, optimizer='adam')
        return model

    """ 
    ----------------------------------------------------------------
        Helpers to build custom tensorflow loss functions.
    ----------------------------------------------------------------
    """

    def heatmapSoftmaxLoss(self, y_true, y_pred):
        y_true = tf.reshape(y_true, (-1, self.mask_side_len * self.mask_side_len, self.num_coords))
        y_pred = tf.reshape(y_pred, (-1, self.mask_side_len * self.mask_side_len, self.num_coords))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred, dim=1)
        return tf.reduce_sum(cross_entropy)

    def squaredDistanceLoss(self, y_true, y_pred):
        y_true = tf.reshape(y_true, (-1, self.num_coords, 2))
        y_pred = tf.reshape(y_pred, (-1, self.num_coords, 2))
        sqrd_diff = tf.reduce_sum(tf.square(y_true - y_pred), axis=2)
        return tf.sum(sqrd_diff)
