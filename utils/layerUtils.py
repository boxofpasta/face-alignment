from keras.engine.topology import Layer
from depthwise_conv2d import DepthwiseConvolution2D
from keras import backend as K
from keras.layers import Input, Convolution2D, \
    GlobalAveragePooling2D, Dense, BatchNormalization, Activation
import numpy as np
import tensorflow as tf

class CropAndResize(Layer):

    def __init__(self, out_im_res, **kwargs):
        self.out_im_res = out_im_res
        self.output_dim = [out_im_res, out_im_res, 1]
        super(CropAndResize, self).__init__(**kwargs)

    def call(self, inputs):
        x, boxes = inputs
        batch_dim = tf.shape(boxes)[0]
        #indices = tf.linspace(0.0, tf.cast(batch_dim - 1, tf.float32), batch_dim)
        indices = tf.linspace(0.0, tf.cast(49, tf.float32), 50)
        indices = tf.cast(indices, tf.int32)
        crops = tf.image.crop_and_resize(x, boxes, indices, tf.constant(self.out_im_res))
        return crops

    def compute_output_shape(self, input_shape):
        return [input_shape[0]] + self.output_dim


def depthwiseConvBlock(x, features_in, features_out, down_sample=False):
    strides = (2, 2) if down_sample else (1, 1)
    x = DepthwiseConvolution2D(int(features_in), (3, 3), strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(features_out), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x