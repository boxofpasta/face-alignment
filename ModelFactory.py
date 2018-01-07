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

    def __init__(self, num_coords):
        self.im_width = 224
        self.im_height = 224
        self.num_coords = num_coords
        self.heatmap_side_len = 56
        self.epsilon = 1E-5

    def getSaved(self, path):
        print 'Loading model ...'
        model = load_model(path, custom_objects={
            'squaredDistanceLoss': self.squaredDistanceLoss,
            'heatmapSoftmaxLoss': self.heatmapSoftmaxLoss,
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
    def heatmapSoftmaxLoss(self, labels, preds):
        labels = tf.reshape(labels, (-1, self.heatmap_side_len * self.heatmap_side_len, self.num_coords))
        preds = tf.reshape(preds, (-1, self.heatmap_side_len * self.heatmap_side_len, self.num_coords))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labelss=labels, logits=preds, dim=1)
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
    
    def scaledSquaredDistanceLoss(self, labels, preds):
        """
        Parameters
        ----------
        labels: 
            Should have shape (batch_size, num_coords, 2)
        """
        preds = tf.reshape(preds, (-1, self.num_coords, 2))
        x_coords = labels[:,:,0]
        y_coords = labels[:,:,1]
        widths = tf.reduce_max(x_coords, axis=1) - tf.reduce_min(x_coords, axis=1)
        heights = tf.reduce_max(y_coords, axis=1) - tf.reduce_min(y_coords, axis=1)
        
        # [widths, heights, widths, heights]
        divisor = tf.concat([widths, heights], axis=1)
        divisor = tf.concat([divisor, divisor], axis=1)

        dist_squared = tf.square(labels - preds)
        scaled = dist_squared / divisor
        return tf.reduce_sum(dist_squared)

    def scaledSquaredDistanceLossNp(self, labels, preds):
        """
        Just a port of scaledSquaredDistanceLoss for numpy.
        """
        preds = np.reshape(preds, (-1, self.num_coords, 2))
        x_coords = labels[:,:,0]
        y_coords = labels[:,:,1]
        widths = np.max(x_coords, axis=1) - np.max(x_coords, axis=1)
        heights = np.max(y_coords, axis=1) - np.min(y_coords, axis=1)
        
        # [widths, heights, widths, heights]
        divisor = np.concatenate([widths, heights], axis=1)
        divisor = np.concatenate([divisor, divisor], axis=1)

        dist_squared = np.square(labels - preds)
        scaled = dist_squared / divisor
        return np.sum(dist_squared)

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
        return tf.reduce_sum(dist_squared)

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
        return tf.reduce_sum(dist_squared)

