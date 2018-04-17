

from scipy.stats import norm
from random import shuffle
import os
import numpy as np
import matplotlib.pyplot as plt
import utils.helenUtils as helenUtils
import utils.generalUtils as utils
import json 
import time
import cv2
import scipy
#from PIL import ImageEnhance

"""
This file contains BatchGenerator (which returns unaltered helen labels as float array),
along with any subclasses that need to do more advanced processing of labels.
"""

class BatchGenerator:
    """
    This class assumes a specific directory and file structure for your data:
        Each image needs to be saved as a .npy file in path path/ims.
        Each coords needs to be saved as a .npy file in path path/coords.
        Coords/masks and images that correspond to each other must have the same name (excluding file extension).
    Subclasses should override the getTrainingPair function.
    
    Parameters
    ----------
    augment_on_generate: 
        Applies only to the generate function. If true, augments data.

    """
    def __init__(self, names, path, augment_on_generate=True, coords_sparsity=1, ibug_version=True):
        self.ibug_version = ibug_version
        self.targ_im_len = 224
        self.batch_size = 32
        self.all_names = []
        self.augment_on_generate = augment_on_generate

        # essentially taking every self.sparsity points in the original coords
        self.coords_sparsity = int(coords_sparsity)
        self.num_coords = helenUtils.getNumCoords(self.coords_sparsity)
        
        # filled only if read_all == True
        self.all_ims = None
        self.all_labels = None

        self.ims_path = path + '/ims'
        self.coords_path = path + '/coords'
        self.im_extension = '.npy'
        self.label_extension = '.npy'
        
        self.all_names = names

        self.steps_per_epoch = len(self.all_names) / self.batch_size

    def getTrainingPairFromName(self, sample_name, augment=False):
        """
        Parameters
        ----------
        sample_name: 
            No extension or folder name included, just the name.

        Returns
        -------
        (inputs, labels) pair, just as the model would receive during training.
        """
        im = np.load(self.ims_path + '/' + sample_name + '.npy')
        coords = np.load(self.coords_path + '/' + sample_name + '.npy')
        coords = self.preprocessCoords(coords)
        inputs, labels = self.getTrainingPair(im, coords, augment=augment)
        return inputs, labels    

    def getAllData(self, augment=False):
        return self.getBatchFromNames(self.all_names, augment=augment)
        #all_ims, all_coords, all_masks = helenUtils.getAllData(path)
        #self.all_ims = all_ims
        #self.all_labels = [ self.getLabels(self.preprocessCoords(coords), all_ims[0]) for coords in all_coords]
        #self.all_labels = utils.transposeList(self.all_labels)
        #return self.all_ims, self.all_labels

    def numTotalSamples(self):
        return len(self.all_names)

    def preprocessCoords(self, coords):
        return coords[0::self.coords_sparsity]

    def getTrainingPair(self, im, coords, augment=False):
        if augment:
            raise ValueError('This class does not support training time augmentation')
        
        bbox = utils.getBbox(coords)
        rect = utils.getSquareFromRect(bbox)
        
        # make sure that rect does not go beyond image borders
        rect = utils.getClippedBbox(im, rect)
        #utils.visualizeCoords(im, coords)

        # crop
        coords[:,0] -= rect[0]
        coords[:,1] -= rect[1]
        im = utils.getCropped(im, rect)
        #utils.visualizeCoords(im, coords)
        coords = helenUtils.normalizeCoords(coords, rect[3] - rect[1], rect[2] - rect[0])

        im = cv2.resize(im, (self.targ_im_len, self.targ_im_len))
        #utils.visualizeCoords(im, 224 * coords)
        return [im], [coords]

    def getLabels(self, im, coords, flip_x=False):
        return [coords]

    """ 
    def visualizeBatch(self):
        for im_batch, labels_batch in self.generate():
            for i in range(len(im_batch)):
                utils.visualizeCoords(im_batch[i], len(im_batch[i]) * labels_batch[i])
    """
    
    def getBatchFromNames(self, sample_names, augment=False):
        X, Y = [], []
        for name in sample_names:
            x, y = self.getTrainingPairFromName(name, augment=augment)
            X.append(x)
            Y.append(y)
        return self.getBatchFromSamples(X, Y)
    
    def getBatchFromSamples(self, X, Y):
        if isinstance(Y[0], tuple):
            raise ValueError('Please use a list for multiple labels')
        if isinstance(Y[0], list):
            Y = utils.transposeList(Y)
            Y = [np.array(output_batch_type) for output_batch_type in Y]
        else:
            Y = np.array(Y)

        X = utils.transposeList(X)
        X = [np.array(in_array) for in_array in X]
        return X, Y

    def generate(self):
     
        num_batches = len(self.all_names) / self.batch_size
        while(True):

            # epoch complete
            rand_idx = np.arange(0, len(self.all_names))
            np.random.shuffle(rand_idx)

            for i in range(0, num_batches):
                past = time.clock()

                # get range of current batch
                start = (i * self.batch_size) % len(self.all_names)
                end = min(start + self.batch_size, len(self.all_names))
                wrap = max(start + self.batch_size - len(self.all_names), 0)
                indices = np.concatenate((rand_idx[start : end], (rand_idx[0 : wrap])), axis=0)
                
                # generate batch
                #if self.all_ims != None and len(self.all_ims) == len(self.all_names):
                #    X = [self.all_ims[k] for k in indices]
                #    Y = [self.all_labels[k] for k in indices]
                #else:
                cur_names = [self.all_names[k] for k in indices]
                X = []
                Y = []
                for name in cur_names:

                    """im = np.load(self.ims_path + '/' + name + self.im_extension)
                    coords = np.load(self.coords_path + '/' + name + self.label_extension)
                    coords = self.preprocessCoords(coords)
                    labels = self.getLabels(coords, im, flip_x)
                    inputs = self.getInputs(coords, im, flip_x)
                    """
                    inputs, labels = self.getTrainingPairFromName(name, augment=self.augment_on_generate)
                    Y.append(labels)
                    X.append(inputs)

                    #X.append([im])
                    #X.append([im] + label)
                
                X, Y = self.getBatchFromSamples(X, Y)
                # X also has the labels appended to itself, in case the model needs access to them internally
                # as part of/before computing the loss (https://github.com/keras-team/keras/issues/4781). 
                # The second part of the tuple is just a bunch of dummy arrays right now.
                #zeros = np.zeros((self.batch_size))
                yield (X, Y)
                #yield (X, [zeros, zeros])
                #yield (X, Y[1])
                #yield (X, Y)

class PointsBatchGenerator(BatchGenerator):
    def __init__(self, names, path):
        BatchGenerator.__init__(self, names, path)
    
    def getInputs(self, coords, im, flip_x=False):
        return [im]

    def getLabels(self, coords, im, flip_x=False):
        return [coords]


class MaskAndBboxBatchGenerator(BatchGenerator):

    def __init__(self, names, path, mask_side_len, **kwargs):
        BatchGenerator.__init__(self, names, path, **kwargs)
        self.mask_side_len = mask_side_len
        self.ibug_version = ibug_version

        # pdfs cache for speeding up coord mask expansions (makes a big difference in training times)
        #self.pdfs = utils.getGaussians(10000, self.mask_side_len)

    def _getLabels(self, coords, im):
        """coords = np.reshape(coords, (self.num_coords, 2))
        heatmap = utils.coordsToHeatmapsFast(coords, self.pdfs)
        heatmap = np.moveaxis(heatmap, 0, -1)
        return heatmap"""

        # need lip_coords in pixel-coordinate units for generating masks
        lip_coords = (len(im) * np.array(helenUtils.getLipCoords(coords, ibug_version=self.ibug_version))).astype(int)
        mask = utils.getMask([lip_coords], (len(im), len(im[0])), (len(im), len(im[0])))
        mask = np.expand_dims(mask, axis=-1)
        
        # bbox coords
        lip_coords_normalized = helenUtils.getLipCoords(coords)
        bbox = utils.getBbox(lip_coords_normalized)
        bbox = utils.getExpandedBbox(bbox, 0.5, 0.5)
        return [bbox, mask]

    def getInputs(self, coords, im):
        inputs = [im]
        labels = self._getLabels(coords, im)
        for label in labels:
            inputs += [label]
        return inputs

    def getLabels(self, coords, im):
        """ These are mock labels to satisfy some of Keras' checks """
        return [0, 0]



class MaskBatchGenerator(BatchGenerator):

    def __init__(self, names, path, mask_side_len, **kwargs):
        BatchGenerator.__init__(self, names, path, **kwargs)
        self.mask_side_len = mask_side_len

    def _getLabels(self, coords, im):

        # need lip_coords in pixel-coordinate units for generating masks
        lip_coords = (len(im) * np.array(helenUtils.getLipCoords(coords))).astype(int)
        mask = utils.getMask([lip_coords], (len(im), len(im[0])), (len(im), len(im[0])))
        mask = np.expand_dims(mask, axis=-1)
        return mask

    def getInputs(self, coords, im):
        inputs = im
        labels = self._getLabels(coords, im)
        return [inputs, labels]

    def getLabels(self, coords, im):
        """ These are mock labels to satisfy some of Keras' checks """
        return [0]


class PointMaskStrictBatchGenerator(BatchGenerator):

    def __init__(self, names, path, mask_side_len, **kwargs):
        BatchGenerator.__init__(self, names, path, **kwargs)
        self.mask_side_len = mask_side_len

        # 0 for most precise
        self.pdfs = 3 * [None]
        self.pdfs[0] = utils.getGaussians(10000, self.mask_side_len, stddev=0.02)
        self.pdfs[1] = utils.getGaussians(10000, self.mask_side_len/2, stddev=0.08)
        self.pdfs[2] = utils.getGaussians(10000, self.mask_side_len/4, stddev=0.2)
        #self.pdfs[3] = utils.getGaussians(10000, self.mask_side_len/8, stddev=0.06)    

    def _getLabels(self, coords, im):
        coords = np.reshape(coords, (self.num_coords, 2))
        lip_coords = helenUtils.getLipCoords(coords)

        labels = []
        for i in range(len(self.pdfs)):
            masks = utils.coordsToHeatmapsFast(lip_coords, self.pdfs[i])
            masks = np.moveaxis(masks, 0, -1)
            masks /= np.max(masks, axis=(0,1))

            l = self.mask_side_len / 2**i
            masks = cv2.resize(masks, (l, l), interpolation=cv2.INTER_LINEAR)
            labels.append(masks)

        """
        masks_0 = utils.coordsToHeatmapsFast(coords, self.pdfs)
        heatmap = np.moveaxis(heatmap, 0, -1)
        heatmap /= np.sum(heatmap, axis=(0,1))
        summed = np.sum(heatmap, axis=-1)
        summed /= (np.max(summed) / 4.0)
        summed = np.minimum(summed, 1.0)
        summed = np.expand_dims(summed, axis=-1)
        return [heatmap, summed]
        """
        return labels

    def getInputs(self, coords, im):
        labels = self._getLabels(coords, im)
        return [im] + labels

    def getLabels(self, coords, im):
        """ Dummy labels. Basically for however many non-None loss entries we have in the model."""
        return [0, 0, 0]


class PointMaskBatchGenerator(BatchGenerator):

    def __init__(self, names, path, mask_side_len, **kwargs):
        BatchGenerator.__init__(self, names, path, **kwargs)
        self.mask_side_len = mask_side_len
        self.pdfs = utils.getGaussians(10000, self.mask_side_len, stddev=0.03)

    def getTrainingPair(self, im, coords, augment=False):

        if augment:
            # random rotation
            width = len(im[0])
            height = len(im)
            center = [(height-1)/2.0, (width-1)/2.0]
            max_rot_deg = 10
            rot_deg = max_rot_deg * np.random.rand(1)
            rot_rad = np.deg2rad(rot_deg)
            im = scipy.misc.imrotate(im, rot_deg)
            coords = utils.getRotatedPoints(coords, center, rot_rad)

        bbox = utils.getBbox(coords)
        square = utils.getSquareFromRect(bbox)

        if augment:
            # random scale and shift
            rect = utils.getRandomlyExpandedBbox(square, -0.2, 0.3)
            max_shift = 0.1 * (square[2] - square[0])
            shifts = max_shift * np.random.rand(2)
            rect = np.array(utils.getShiftedBbox(rect, shifts)).astype(int)
        else:
            rect = utils.getExpandedBbox(square, 0.0, 0.0)

        # make sure that rect does not go beyond image borders
        rect = utils.getClippedBbox(im, rect)

        # crop
        coords[:,0] -= rect[0]
        coords[:,1] -= rect[1]
        im = utils.getCropped(im, rect)

        """
        # brightness and saturation adjustments
        if augment:
            rand_v_delta = (np.random.rand() - 0.5) * 0.2 * 255
            rand_s_delta = (np.random.rand() - 0.7) * 0.6 * 255
            im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            im_hsv = im_hsv.astype(np.float32)
            im_hsv[:,:,1] += rand_s_delta
            im_hsv[:,:,2] += rand_v_delta
            im_hsv = np.clip(im_hsv, 0, 255)
            im_hsv = im_hsv.astype(np.uint8)
            im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        """

        # just the lip coords for now
        # flip
        flipped = bool(np.random.randint(0, 2)) if augment else False
        lip_coords = helenUtils.getLipCoords(coords, len(im[0]), flip_x=flipped, ibug_version=self.ibug_version)
        im = np.fliplr(im) if flipped else im

        # normalize the coords and image
        im = cv2.resize(im, (self.targ_im_len, self.targ_im_len))
        crop_width = rect[3] - rect[1]
        crop_height = rect[2] - rect[0]
        normalized_lip_coords = helenUtils.normalizeCoords(lip_coords, crop_width, crop_height)
    
        # mask from coords
        masks = utils.coordsToHeatmapsFast(normalized_lip_coords, self.pdfs)
        masks = np.moveaxis(masks, 0, -1)
        masks /= np.max(masks, axis=(0,1))
        l = self.mask_side_len
        masks = cv2.resize(masks, (l, l), interpolation=cv2.INTER_AREA)
        return [im], [masks]

class PointMaskCascadedBatchGenerator(BatchGenerator):

    def __init__(self, names, path, mask_side_len, hd_mask_side_len, **kwargs):
        BatchGenerator.__init__(self, names, path, **kwargs)
        self.mask_side_len = mask_side_len
        self.hd_mask_side_len = hd_mask_side_len
        self.pdfs = utils.getGaussians(10000, self.mask_side_len, stddev=0.03)

        # hd for high definition
        self.hd_pdfs = utils.getGaussians(10000, self.hd_mask_side_len, stddev=0.01)

    def getTrainingPair(self, im, coords, augment=False):

        if augment:
            # random rotation
            width = len(im[0])
            height = len(im)
            center = [(height-1)/2.0, (width-1)/2.0]
            max_rot_deg = 10
            rot_deg = max_rot_deg * np.random.rand(1)
            rot_rad = np.deg2rad(rot_deg)
            im = scipy.misc.imrotate(im, rot_deg)
            coords = utils.getRotatedPoints(coords, center, rot_rad)

        bbox = utils.getBbox(coords)
        square = utils.getSquareFromRect(bbox)

        if augment:
            # random scale and shift
            rect = utils.getRandomlyExpandedBbox(square, -0.2, 0.3)
            max_shift = 0.1 * (square[2] - square[0])
            shifts = max_shift * np.random.rand(2)
            rect = np.array(utils.getShiftedBbox(rect, shifts)).astype(int)
        else:
            rect = utils.getExpandedBbox(square, 0.0, 0.0)

        # make sure that rect does not go beyond image borders
        rect = utils.getClippedBbox(im, rect)

        # crop
        coords[:,0] -= rect[0]
        coords[:,1] -= rect[1]
        im = utils.getCropped(im, rect)

        """
        # brightness and saturation adjustments
        if augment:
            rand_v_delta = (np.random.rand() - 0.5) * 0.2 * 255
            rand_s_delta = (np.random.rand() - 0.7) * 0.6 * 255
            im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            im_hsv = im_hsv.astype(np.float32)
            im_hsv[:,:,1] += rand_s_delta
            im_hsv[:,:,2] += rand_v_delta
            im_hsv = np.clip(im_hsv, 0, 255)
            im_hsv = im_hsv.astype(np.uint8)
            im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        """

        # just the lip coords for now
        # flip
        flipped = bool(np.random.randint(0, 2)) if augment else False
        lip_coords = helenUtils.getLipCoords(coords, len(im[0]), flip_x=flipped, ibug_version=self.ibug_version)
        im = np.fliplr(im) if flipped else im

        # normalize the coords and image
        im = cv2.resize(im, (self.targ_im_len, self.targ_im_len))
        crop_width = rect[3] - rect[1]
        crop_height = rect[2] - rect[0]
        normalized_lip_coords = helenUtils.normalizeCoords(lip_coords, crop_width, crop_height)
    
        # mask from coords
        masks = utils.coordsToHeatmapsFast(normalized_lip_coords, self.pdfs)
        masks = np.moveaxis(masks, 0, -1)
        masks /= np.max(masks, axis=(0,1))
        hd_masks = utils.coordsToHeatmapsFast(normalized_lip_coords, self.hd_pdfs)
        hd_masks = np.moveaxis(hd_masks, 0, -1)
        hd_masks /= np.max(hd_masks, axis=(0,1))
        l = self.mask_side_len
        hd_l = self.hd_mask_side_len

        # try using hd masks for everything!
        masks = cv2.resize(masks, (hd_l, hd_l), interpolation=cv2.INTER_LINEAR)
        hd_masks = cv2.resize(hd_masks, (hd_l, hd_l), interpolation=cv2.INTER_LINEAR)
        return [im], [masks, hd_masks]


class LineMaskBatchGenerator(BatchGenerator):

    def __init__(self, path, mask_side_len, **kwargs):
        BatchGenerator.__init__(self, path, **kwargs)
        self.mask_side_len = mask_side_len

    def getLabels(self, coords, im):
        return [0]
    
    def getInputs(self, coords, im):
        return [im] + self._getLabels(coords, im)

    def _getLabels(self, coords, im):
        lip_coords = helenUtils.getLipCoords(coords)
        line_mask = helenUtils.getLipLineMask(lip_coords, np.shape(im), (self.mask_side_len, self.mask_side_len))
        return [line_mask]
