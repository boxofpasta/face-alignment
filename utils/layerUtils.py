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
        self.output_dim = [out_im_res, out_im_res]
        super(CropAndResize, self).__init__(**kwargs)

    def call(self, inputs):
        x, boxes = inputs
        batch_dim = tf.shape(boxes)[0]
        indices = tf.range(0, tf.cast(batch_dim, tf.float32), 1)
        indices = tf.cast(indices, tf.int32)
        crops = tf.image.crop_and_resize(x, boxes, indices, tf.constant(self.output_dim))
        return crops

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0][0]] + self.output_dim + [input_shape[0][3]])

    def get_config(self):
        config = {'out_im_res': self.out_im_res}
        base_config = super(CropAndResize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaskSigmoidLossLayer(Layer):

    def __init__(self,  mask_side_len, **kwargs):
        super(MaskSigmoidLossLayer, self).__init__(**kwargs)
        self.mask_side_len = mask_side_len

    def call(self, inputs):
        labels, preds, bboxes = inputs
        #labels = tf.expand_dims(labels, axis=-1)
        cropped_labels = CropAndResize(self.mask_side_len)([labels, bboxes])

        # should just be single channel images
        cropped_labels = tf.squeeze(cropped_labels, axis=-1)
        preds = tf.squeeze(preds, axis=-1)
        
        """batch_dim = tf.shape(bboxes)[0]
        indices = tf.range(0, tf.cast(batch_dim, tf.float32), 1)
        indices = tf.cast(indices, tf.int32)
        mask_dims = [self.mask_side_len, self.mask_side_len]
        cropped_labels = tf.image.crop_and_resize(labels, bboxes, indices, tf.constant(mask_dims))
        cropped_labels = tf.squeeze(cropped_labels)"""
        
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=cropped_labels)
        
        """labels = tf.expand_dims(labels, axis=-1)
        labels = tf.image.resize_images(labels, (self.mask_side_len, self.mask_side_len))
        labels = tf.squeeze(labels)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)"""

        #loss = tf.reshape(tf.reduce_sum(cross_entropy), (1,))
        return tf.expand_dims(tf.reduce_sum(cross_entropy,axis=[1,2]), axis=1)
        #return tf.reduce_sum(cross_entropy)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)
        #return (1,)

    def get_config(self):
        config = {'mask_side_len': self.mask_side_len}
        base_config = super(MaskSigmoidLossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SquaredDistanceLossLayer(Layer):

    def __init__(self,  **kwargs):
        super(SquaredDistanceLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        labels, preds = inputs
        batch_dim = tf.shape(labels)[0]
        labels = tf.reshape(labels, (batch_dim,-1))
        preds = tf.reshape(preds, (batch_dim,-1))
        return 0.0 * tf.expand_dims(tf.reduce_sum(tf.square(labels - preds), axis=1), axis=1)
        #return tf.reduce_sum(tf.square(labels - preds))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)


def depthwiseConvBlock(x, features_in, features_out, down_sample=False):
    strides = (2, 2) if down_sample else (1, 1)
    x = DepthwiseConvolution2D(int(features_in), (3, 3), strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(features_out), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x