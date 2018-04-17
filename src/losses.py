import os
import numpy as np
import utils.helenUtils as helenUtils
import utils.generalUtils as utils
import json
import time
import sys
from keras import losses
import utils.layerUtils as layerUtils
import tensorflow as tf


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
    preds = tf.transpose(preds, [0, 3, 1, 2])

    # even the labels are not necessarily normalized
    labels /= utils.expandDimsRepeatedly(tf.reduce_sum(labels, axis=[2,3]) + self.epsilon, 2, False)
    preds /= utils.expandDimsRepeatedly(tf.reduce_sum(preds, axis=[2,3]) + self.epsilon, 2, False)

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

    # normalize, since these numbers are pretty large
    squared_dists /= tf.cast(width, tf.float32) ** 2

    return tf.reduce_sum((0.1 + squared_dists) * cross_entropy)


def pointMaskDistanceLossPresetDims(self, labels, preds):
    method = tf.image.ResizeMethod.BILINEAR
    labels = layerUtils.Resize(28, method)(labels)
    preds = layerUtils.Resize(28, method)(preds)
    return pointMaskDistanceLoss(labels, preds)


def cascadedPointMaskSigmoidLoss(self, y_true, y_pred):
    num_coords = 13
    method = tf.image.ResizeMethod.BILINEAR

    # split the preds up into their component parts (dammit keras!)
    base_preds = y_pred[:,:,:,:13]
    base_preds = tf.stop_gradient(base_preds)
    refined_preds = y_pred[:,:,:,13:]

    # get crops;
    # this is actually repetitive code from the model architecture, 
    # blame keras for inflexible loss function arguments
    base_preds_normalized = Activation('sigmoid')(base_preds)
    mask_means = layerUtils.MaskMean()(base_preds_normalized)
    true_means = layerUtils.MaskMean()(y_true)
    #boxes = layerUtils.BoxesFromCenters(28.0 / self.im_height)(true_means)
    #boxes = layerUtils.PerturbBboxes([0.8, 1.2], [-0.25, 0.25])(boxes)
    boxes = layerUtils.BoxesFromCenters(28.0 / self.im_height)(mask_means)

    # avoid penalizing refined mask when the initial estimate is not even close to truth
    sqrd_diffs = tf.squared_difference(mask_means, true_means)
    dists = tf.sqrt(tf.reduce_sum(sqrd_diffs, axis=-1))
    thresh = 0.30 * 28.0 / self.im_height
    loss_mask = tf.where(dists < thresh, tf.ones(tf.shape(dists)), tf.zeros(tf.shape(dists)))
    loss_mask = tf.expand_dims(loss_mask, 1)
    loss_mask = tf.expand_dims(loss_mask, 1)

    label_crops = []
    for i in range(num_coords):
        box = Lambda( lambda x: x[:,i,:] )(boxes)
        label_mask = Lambda( lambda x: x[:,:,:,i] )(y_true)
        label_mask = tf.expand_dims(label_mask, axis=-1)
        label_crop = layerUtils.CropAndResize(28)([label_mask, box])
        label_crops.append(label_crop)

    labels = Concatenate()(label_crops)
    labels *= loss_mask
    refined_preds = layerUtils.Resize(28, method)(refined_preds)
    refined_preds *= loss_mask

    return self.pointMaskDistanceLoss(labels, refined_preds)


def pointMaskDistance(self, labels, preds):
    """
    Sum of euclidean distances squared between the centers of preds and labels. 
    """
    preds = tf.sigmoid(preds)
    width = tf.shape(labels)[2]
    height = tf.shape(labels)[1]
    x_inds = tf.cast(tf.expand_dims(tf.range(0, width), 0), tf.float32) / tf.cast(width, tf.float32)
    y_inds = tf.cast(tf.expand_dims(tf.range(0, height), 1), tf.float32) / tf.cast(height, tf.float32)

    # move channels (corresponding to the coords dim) together with batch dim to avoid confusion
    labels = tf.transpose(labels, [0, 3, 1, 2])
    preds = tf.transpose(preds, [0, 3, 1, 2])

    # even the labels are not necessarily normalized
    labels /= utils.expandDimsRepeatedly(tf.reduce_sum(labels, axis=[2,3]) + self.epsilon, 2, False)
    preds /= utils.expandDimsRepeatedly(tf.reduce_sum(preds, axis=[2,3]) + self.epsilon, 2, False)

    # performs a weighted sum to get center coordinates
    x_label = tf.reduce_sum(x_inds * labels, axis=[2,3])
    y_label = tf.reduce_sum(y_inds * labels, axis=[2,3])
    x_pred = tf.reduce_sum(x_inds * preds, axis=[2,3])
    y_pred = tf.reduce_sum(y_inds * preds, axis=[2,3])

    x_dist = tf.squared_difference(x_label, x_pred)
    y_dist = tf.squared_difference(y_label, y_pred)
    return tf.reduce_sum(x_dist + y_dist)