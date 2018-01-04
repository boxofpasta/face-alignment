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
        self.heatmap_side_len = 56
        self.epsilon = 1E-5

    def getSaved(self, path):
        print 'Loading model ...'
        model = load_model(path, custom_objects={
            'squaredDistanceLoss': self.squaredDistanceLoss,
            'heatmapSoftmaxLoss': self.heatmapSoftmaxLoss,
            'iouLoss': self.iouLoss,
            'scaledSquaredDistanceLoss': self.scaledSquaredDistanceLoss,
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
        base_model = mobilenet.MobileNet(include_top=False, input_shape=in_shape)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(units=2 * self.num_coords, activation='linear')(x)
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
    def heatmapSoftmaxLoss(self, label, pred):
        label = tf.reshape(label, (-1, self.heatmap_side_len * self.heatmap_side_len, self.num_coords))
        pred = tf.reshape(pred, (-1, self.heatmap_side_len * self.heatmap_side_len, self.num_coords))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=pred, dim=1)
        return tf.reduce_sum(cross_entropy)

    def squaredDistanceLoss(self, label, pred):
        return tf.reduce_sum(tf.square(label - pred))

    def iouLoss(self, label, pred):
        """
        Training does not converge well for this loss. There might be an error somewhere.
        """
        iouWeight = 0.9999
        distanceLoss = self.squaredDistanceLoss(label, pred)
        intersection_dx = tf.maximum(tf.minimum(pred[:,2], label[:,2]) - tf.maximum(pred[:,0], label[:,0]), 0)
        intersection_dy = tf.maximum(tf.minimum(pred[:,3], label[:,3]) - tf.maximum(pred[:,1], label[:,1]), 0)
        
        intersection_area = intersection_dx * intersection_dy
        pred_area = tf.maximum(pred[:,2] - pred[:,0], 0) * tf.maximum(pred[:,3] - pred[:,1], 0)
        true_area = tf.maximum(label[:,2] - label[:,0], 0) * tf.maximum(label[:,3] - label[:,1], 0)
        union = pred_area + true_area - intersection_area
        return (1.0 - iouWeight) * distanceLoss - iouWeight * tf.reduce_sum(intersection_area / (union + self.epsilon))
    
    def scaledSquaredDistanceLoss(self, label, pred):
        """
        y and x distances scaled by height / width of the ground truth box.
        """
        widths = tf.expand_dims(label[:,2] - label[:,0], axis=1)
        heights = tf.expand_dims(label[:,3] - label[:,1], axis=1)
        
        # [widths, heights, widths, heights]
        divisor = tf.concat([widths, heights], axis=1)
        divisor = tf.concat([divisor, divisor], axis=1)

        dist_squared = tf.square(label - pred)
        scaled = dist_squared / divisor
        return tf.reduce_sum(dist_squared)
