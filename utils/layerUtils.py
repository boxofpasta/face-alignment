from keras.engine.topology import Layer
from depthwise_conv2d import DepthwiseConvolution2D
from keras import backend as K
from keras.layers import Input, Convolution2D, \
    GlobalAveragePooling2D, Dense, BatchNormalization, Activation
import numpy as np
import tensorflow as tf
import generalUtils as utils

class PerturbBboxes(Layer):
    def __init__(self, scale_extents, offset_extents):
        """
        Parameters
        ----------
        scale_extents: 
            [min_perc, max_perc]. e.g [0.2, 1.0] means at least scale to 20%, at most keep the same size.
            Applies to both width and height.
        offset_extents:
            [min_extent, max_extent]. Similar to scale_extents.
        """
        self.scale_extents = scale_extents
        self.offset_extents = offset_extents
        super(PerturbBboxes, self).__init__(**kwargs)

    def call(self, bboxes):
        """ 
        each bbox will come as [y1, x1, y2, x2].
        we want : [y1 + yo - h/2, x1 + xo - w/2, y2 + yo + h/2, x2 + xo + w/2].
        where yo, xo, w, h are randomly sampled according to scale and offset extents for each bbox in the batch.
        """

        batch_len = bboxes.shape()[0]
        scales = tf.random_uniform((batch_len,1), min_val=self.scale_extents[0], max_val=self.scale_extents[1])
        scales_offset_x = tf.random_uniform((batch_len,1), min_val=self.offset_extents[0], max_val=self.offset_extents[1])
        scales_offset_y = tf.random_uniform((batch_len,1), min_val=self.offset_extents[0], max_val=self.offset_extents[1])
        orig_heights = bboxes[:,2] - bboxes[:,0]
        orig_widths = bboxes[:,3] - bboxes[:,1]

        # get the randomized parameters
        rand_widths = scales * orig_heights
        rand_heights = scales * orig_heights
        offset_x = scales_offset_x * orig_widths
        offset_y = scales_offset_y * orig_heights

        # move things into the right places
        offsets = tf.stack([
            offset_y - rand_heights/2.0, 
            offset_x - rand_widths/2.0, 
            offset_y + rand_heights/2.0, 
            offset_x + rand_widths/2.0
        ], axis=1)

        utils.printTensorShape(offsets)
        utils.printTensorShape(bboxes)
        utils.printTensorShape(bboxes * offsets)
        return bboxes * offsets

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'scale_extents': self.scale_extents, 'offset_extents': self.offset_extents}
        base_config = super(PerturbBboxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
class CropAndResize(Layer):

    def __init__(self, out_im_res, **kwargs):
        self.out_im_res = out_im_res
        self.output_dim = [out_im_res, out_im_res]
        super(CropAndResize, self).__init__(**kwargs)

    def call(self, inputs):
        x, boxes = inputs

        # not sure if having a gradient flow through boxes makes sense
        boxes = tf.stop_gradient(boxes)
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

class Resize(Layer):

    def __init__(self, out_im_res, method, **kwargs):
        self.out_im_res = out_im_res
        self.output_dim = [out_im_res, out_im_res]
        self.method = method
        super(Resize, self).__init__(**kwargs)

    def call(self, x):
        batch_dim = tf.shape(x)[0]
        #x = tf.image.resize_area(x, self.output_dim)
        #x = tf.image.resize_images(x, self.output_dim, method=self.method)
        x = tf.image.resize_images(x, self.output_dim)
        return x

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0]] + self.output_dim + [input_shape[3]])

    def get_config(self):
        config = {'out_im_res': self.out_im_res, 'method': self.method}
        base_config = super(Resize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MaskSigmoidLossLayer(Layer):

    def __init__(self, mask_side_len, **kwargs):
        self.mask_side_len = mask_side_len
        super(MaskSigmoidLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        labels, preds, bboxes = inputs
        #labels = tf.expand_dims(labels, axis=-1)
        cropped_labels = CropAndResize(self.mask_side_len)([labels, bboxes])

        # should just be single channel images
        cropped_labels = tf.squeeze(cropped_labels, axis=-1)
        preds = tf.squeeze(preds, axis=-1)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=cropped_labels)
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

class MaskSigmoidLossLayerNoCrop(Layer):

    def __init__(self, mask_side_len, **kwargs):
        self.mask_side_len = mask_side_len
        super(MaskSigmoidLossLayerNoCrop, self).__init__(**kwargs)

    def call(self, inputs):
        labels, preds = inputs
        #labels = tf.expand_dims(labels, axis=-1)
        labels = Resize(self.mask_side_len, tf.image.ResizeMethod.BILINEAR)(labels)
        #labels = Resize(self.mask_side_len)(labels)

        # should just be single channel images
        labels = tf.squeeze(labels, axis=-1)
        preds = tf.squeeze(preds, axis=-1)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
        #loss = tf.reshape(tf.reduce_sum(cross_entropy), (1,))
        return tf.expand_dims(tf.reduce_sum(cross_entropy,axis=[1,2]), axis=1)
        #return tf.reduce_sum(cross_entropy)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)

    def get_config(self):
        config = {'mask_side_len': self.mask_side_len}
        base_config = super(MaskSigmoidLossLayerNoCrop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SquaredDistanceLossLayer(Layer):

    def __init__(self,  **kwargs):
        super(SquaredDistanceLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        labels, preds = inputs
        batch_dim = tf.shape(labels)[0]
        labels = tf.reshape(labels, (batch_dim,-1))
        preds = tf.reshape(preds, (batch_dim,-1))
        return tf.expand_dims(tf.reduce_sum(tf.square(labels - preds), axis=1), axis=1)
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

def resizeConvBlock(x, out_res, features_in, features_out):
    x = Resize(out_res, tf.image.ResizeMethod.NEAREST_NEIGHBOR)(x)
    x = depthwiseConvBlock(x, features_in, features_out)
    return x