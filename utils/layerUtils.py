from keras.engine.topology import Layer
from depthwise_conv2d import DepthwiseConvolution2D
from keras import backend as K
from keras.layers import Input, Convolution2D, Conv2DTranspose, Lambda, \
    GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Concatenate, Add, LeakyReLU
import numpy as np
import tensorflow as tf
import generalUtils as utils

class PerturbBboxes(Layer):
    def __init__(self, scale_extents, offset_extents, **kwargs):
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

        batch_len = tf.shape(bboxes)[0]
        scales = tf.random_uniform([batch_len], minval=self.scale_extents[0], maxval=self.scale_extents[1])
        scales_offset_x = tf.random_uniform([batch_len], minval=self.offset_extents[0], maxval=self.offset_extents[1])
        scales_offset_y = tf.random_uniform([batch_len], minval=self.offset_extents[0], maxval=self.offset_extents[1])
        orig_heights = bboxes[:,2] - bboxes[:,0]
        orig_widths = bboxes[:,3] - bboxes[:,1]

        # get the randomized parameters
        add_widths = (scales - 1.0) * orig_heights 
        add_heights = (scales - 1.0) * orig_widths 
        offset_x = scales_offset_x * orig_widths
        offset_y = scales_offset_y * orig_heights

        # move things into the right places
        offsets = tf.stack([
            offset_y - add_heights/2.0, 
            offset_x - add_widths/2.0, 
            offset_y + add_heights/2.0, 
            offset_x + add_widths/2.0
        ], axis=1)
        offsets = tf.stop_gradient(offsets)
        return bboxes + offsets

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'scale_extents': self.scale_extents, 'offset_extents': self.offset_extents}
        base_config = super(PerturbBboxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TileMultiply(Layer):

    def __init__(self, multiple, **kwargs):
        self.multiple = multiple
        super(TileMultiply, self).__init__(**kwargs)

    def call(self, inputs):
        x, z = inputs
        x_tiled = tf.tile(x, [1, 1, 1, self.multiple])
        return x_tiled * z

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = {'multiple': self.multiple}
        base_config = super(TileMultiply, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class TileSubtract(Layer):

    def __init__(self, multiple, **kwargs):
        self.multiple = multiple
        super(TileSubtract, self).__init__(**kwargs)

    def call(self, inputs):
        x, z = inputs
        x_tiled = tf.tile(x, [1, 1, 1, self.multiple])
        return x_tiled - z

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = {'multiple': self.multiple}
        base_config = super(TileSubtract, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MaskMean(Layer):

    def __init__(self, **kwargs):
        super(MaskMean, self).__init__(**kwargs)

    def call(self, masks):
        width = tf.shape(masks)[2]
        height = tf.shape(masks)[1]
        x_inds = tf.cast(tf.expand_dims(tf.range(0, width), 0), tf.float32) / tf.cast(width, tf.float32)
        y_inds = tf.cast(tf.expand_dims(tf.range(0, height), 1), tf.float32) / tf.cast(height, tf.float32)

        # move channels (corresponding to the coords dim) together with batch dim to avoid confusion
        epsilon = 1E-6
        masks = tf.transpose(masks, [0, 3, 1, 2])
        masks /= utils.expandDimsRepeatedly(tf.reduce_sum(masks, axis=[2,3]) + epsilon, 2, False)

        # performs a weighted sum to get center coordinates
        x_cen = tf.reduce_sum(x_inds * masks, axis=[2,3])
        y_cen = tf.reduce_sum(y_inds * masks, axis=[2,3])
        x_cen = tf.expand_dims(x_cen, axis=-1)
        y_cen = tf.expand_dims(y_cen, axis=-1)

        # [batch, coords, 2]
        centers = tf.concat([y_cen, x_cen], axis=-1)
        return centers

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0:2] + (2,))

    def get_config(self):
        config = {}
        base_config = super(MaskMean, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BoxesFromCenters(Layer):

    def __init__(self, side_len, **kwargs):
        self.side_len = side_len
        super(BoxesFromCenters, self).__init__(**kwargs)

    def call(self, centers):

        # centers should follow [y, x] and all fall between [0, 1] (must be normalized coordinates)
        # [batch, num_coords, 2]
        offset = self.side_len / 2.0
        offsets = tf.constant([-offset, -offset, offset, offset])
        boxes = tf.concat([centers, centers], axis=-1)

        boxes += offsets
        return boxes

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0:2] + (4,))

    def get_config(self):
        config = {'side_len': self.side_len}
        base_config = super(BoxesFromCenters, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class CropAndResize(Layer):

    def __init__(self, out_im_res, **kwargs):
        self.out_im_res = out_im_res
        self.output_dim = [out_im_res, out_im_res]
        super(CropAndResize, self).__init__(**kwargs)

    def call(self, inputs):
        # bboxes should follow [y1, x1, y2, x2]
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


 # Why do Lambda layers have to not work in keras ... :'(
class SliceBboxes(Layer):

    def __init__(self, index, **kwargs):
        self.index = index
        super(SliceBboxes, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs[:,self.index,:]

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0]] + [input_shape[2]])

    def get_config(self):
        config = {'index': self.index}
        base_config = super(SliceBboxes, self).get_config()
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
        x = tf.image.resize_images(x, self.output_dim, method=self.method)
        return x

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0]] + self.output_dim + [input_shape[3]])

    def get_config(self):
        config = {'out_im_res': self.out_im_res, 'method': self.method}
        base_config = super(Resize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PointMaskSoftmaxLossLayer(Layer):
    def __init__(self, mask_side_len, **kwargs):
        self.mask_side_len = mask_side_len
        super(PointMaskSoftmaxLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        labels, preds = inputs

        # flatten image dimensions to get distribution for softmax
        labels = tf.transpose(labels, [0, 3, 1, 2])
        preds = tf.transpose(preds, [0, 3, 1, 2])
        batch_dim = tf.shape(labels)[0]
        coords_dim = tf.shape(labels)[1]
        labels = tf.reshape(labels, [batch_dim * coords_dim, -1])
        preds = tf.reshape(preds, [batch_dim * coords_dim, -1])
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
        entropy = tf.reshape(entropy, [batch_dim, coords_dim])
        entropy = tf.reduce_sum(entropy, axis=[1])
        return tf.expand_dims(entropy, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)

    def get_config(self):
        config = {'mask_side_len': self.mask_side_len}
        base_config = super(PointMaskSoftmaxLossLayer, self).get_config()
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
        #labels = tf.squeeze(labels, axis=-1)
        #preds = tf.squeeze(preds, axis=-1)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
        #loss = tf.reshape(tf.reduce_sum(cross_entropy), (1,))
        return tf.expand_dims(tf.reduce_sum(cross_entropy,axis=[1,2,3]), axis=1)
        #return tf.reduce_sum(cross_entropy)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],1)

    def get_config(self):
        config = {'mask_side_len': self.mask_side_len}
        base_config = super(MaskSigmoidLossLayerNoCrop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))  

class StopGradientLayer(Layer):
    def __init__(self,  **kwargs):
        super(StopGradientLayer, self).__init__(**kwargs)

    def call(self, x):
        return tf.stop_gradient(x)

    def compute_output_shape(self, input_shape):
        return input_shape

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


def depthwiseConvBlock(x, features_in, features_out, down_sample=False, 
                        kernel_size=(3,3), final_activation='relu', dilation_rate=[1,1]):
    strides = (2, 2) if down_sample else (1, 1)
    x = DepthwiseConvolution2D(int(features_in), kernel_size, strides=strides, padding='same', use_bias=False, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(features_out), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    if final_activation == 'leaky_relu':
        x = LeakyReLU(alpha=0.3)(x)
    else:
        x = Activation(final_activation)(x)
    return x

def resizeConvBlock(x, out_res, features_in, features_out):
    x = Resize(out_res, tf.image.ResizeMethod.NEAREST_NEIGHBOR)(x)
    x = depthwiseConvBlock(x, features_in, features_out)
    return x

def rcfBlock(x, features_in, features_out, num_layers, z_out_layers=1):
    """
    https://arxiv.org/pdf/1612.02103.pdf
    Parameters
    ----------
    layers: 
        The number of layers to use in this block. Should be 2 or 3.
    """

    outputs = []
    for i in range(num_layers-1):
        x = depthwiseConvBlock(x, features_in, features_in)
        outputs.append(x)
    
    x = depthwiseConvBlock(x, features_in, features_out, down_sample=True)
    if len(outputs) > 1:
        z = Add()(outputs)
    else:
        z = outputs[0]
    z = depthwiseConvBlock(z, features_in, z_out_layers)
    return [x, z]


def getMaskHead(bboxes, cfs, alpha, use_resize_conv=False):
    """
    A relatively large block -- the entire mask head essentially. 
    cfs should contain 4 feature maps, in order from highest to lowest res (or shallow to deep).
    """
    
    # Mask head: 
    # https://arxiv.org/pdf/1703.06870.pdf
    #cf1_cropped = CropAndResize(56)([cfs[0], bboxes]) # 112x112, d = 64
    cf2_cropped = CropAndResize(28)([cfs[1], bboxes]) # 56x56, d = 128
    cf3_cropped = CropAndResize(14)([cfs[2], bboxes]) # 28x28, d = 256
    cf4_cropped = CropAndResize(7)([cfs[3], bboxes]) # 14x14, d = 512

    a = cf4_cropped
    #a = Concatenate(axis=-1)([cf1_cropped, cf2_cropped, cf3_cropped, cf4_cropped]) # d = 448 + 512 = 960
    #a = Convolution2D(int(512 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)

    #a = Concatenate()([a, cf3_cropped])
    a = depthwiseConvBlock(a, 512 * alpha, 512 * alpha)
    a = depthwiseConvBlock(a, 512 * alpha, 512 * alpha)
    a = depthwiseConvBlock(a, 512 * alpha, 256 * alpha)

    # 7x7
    conv_transpose_depth = 128
    
    if not use_resize_conv:
        a = Conv2DTranspose(int(conv_transpose_depth * alpha), kernel_size=(3, 3),
                strides=(2, 2),
                activation='relu',
                padding='same',
                data_format='channels_last')(a)
    else:
        a = resizeConvBlock(a, 14, 512 * alpha, conv_transpose_depth * alpha)

    # 14x14
    # fuse cf3_cropped
    cf3_cropped = depthwiseConvBlock(cf3_cropped, 256 * alpha, conv_transpose_depth * alpha, kernel_size=(1,1))
    a = Add()([a, cf3_cropped])
    for i in range(3):
        a = depthwiseConvBlock(a, conv_transpose_depth * alpha, conv_transpose_depth * alpha)

    if not use_resize_conv:
        a = Conv2DTranspose(int(conv_transpose_depth * alpha), kernel_size=(3, 3),
            strides=(2, 2),
            activation='relu',
            padding='same',
            data_format='channels_last')(a)
    else:
        a = resizeConvBlock(a, 28, conv_transpose_depth * alpha, conv_transpose_depth * alpha)
    
    # 28x28
    # fuse cf2_cropped
    cf2_cropped = depthwiseConvBlock(cf2_cropped, 128 * alpha, conv_transpose_depth * alpha, kernel_size=(1,1))
    a = Add()([a, cf2_cropped])

    for i in range(3):
        a = depthwiseConvBlock(a, conv_transpose_depth * alpha, conv_transpose_depth * alpha)

    if not use_resize_conv:
        a = Conv2DTranspose(int(conv_transpose_depth * alpha), kernel_size=(3, 3),
            strides=(2, 2),
            activation='relu',
            padding='same',
            data_format='channels_last')(a)
    else:
        a = resizeConvBlock(a, 56, conv_transpose_depth * alpha, conv_transpose_depth * alpha)
    
    # 56x56
    # fuse cf1_cropped (?)
    a = depthwiseConvBlock(a, conv_transpose_depth * alpha, conv_transpose_depth * alpha)
    a = depthwiseConvBlock(a, conv_transpose_depth * alpha, 1)
    return a