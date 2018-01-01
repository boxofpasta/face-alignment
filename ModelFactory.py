import os
import numpy as np
import utils.helenUtils as helenUtils
import utils.generalUtils as utils
import json
import time
import sys
import BatchGenerator
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.applications import mobilenet
from keras import backend as K

class ModelFactory:

    def __init__(self):
        self.im_width = 224
        self.im_height = 224
        self.num_coords = 194

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

    """ 
    ----------------------------------------------------------------
        Helpers to build custom tensorflow loss functions.
    ----------------------------------------------------------------
    """

    def squaredDistanceLoss(self, y_true, y_pred):
        y_true = K.reshape(y_true, (-1, self.num_coords, 2))
        y_pred = K.reshape(y_pred, (-1, self.num_coords, 2))
        sqrd_diff = K.sum(K.square(y_true - y_pred), axis=2)
        return K.sum(sqrd_diff, axis=-1)
