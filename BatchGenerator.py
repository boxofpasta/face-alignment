from scipy.stats import norm
import os
import numpy as np
import matplotlib.pyplot as plt
import utils.helenUtils as helenUtils
import utils.generalUtils as utils
import json 
import time
import cv2

"""
This file contains BatchGenerator (which returns unaltered helen labels as float array),
along with any subclasses that need to do more advanced processing of labels.
"""

class BatchGenerator:
    """
    This class assumes a specific directory and file structure for your data:
        Each image needs to be saved as a .npy file in path path/ims.
        Each coords needs to be saved as a .npy file in path path/coords.
        Each mask needs to be saved as a .npy file in path path/masks.
        Coords/masks and images that correspond to each other must have the same name (excluding file extension).
        The names of all samples (training pairs) must be in path/names.json.
    """
    def __init__(self, path, coords_sparsity=1):
        self.batch_size = 32
        self.all_names = []

        # essentially taking every self.sparsity points in the original coords
        self.coords_sparsity = int(coords_sparsity)
        self.num_coords = helenUtils.getNumCoords(self.coords_sparsity)
        
        # filled only if read_all == True
        self.all_ims = None
        self.all_labels = None

        self.name_path = path + '/names.json'
        self.ims_path = path + '/ims'
        self.coords_path = path + '/coords'
        self.im_extension = '.npy'
        self.label_extension = '.npy'
        
        with open(self.name_path) as fp:
            self.all_names = json.load(fp)

        self.steps_per_epoch = len(self.all_names) / self.batch_size

    def getPair(self, sample_name):
        """
        Parameters
        ----------
        sample_name: 
            No extension or folder name included, just the name.

        Returns
        -------
        (inputs, outputs) pair, just as the model would receive during training.
        """
        im = np.load(self.ims_path + '/' + sample_name + '.npy')
        coords = np.load(self.coords_path + '/' + sample_name + '.npy')
        coords = self.preprocessCoords(coords)
        outputs = self.getOutputs(coords, im)  
        inputs = self.getInputs(coords, im)
        return inputs, outputs    

    def getAllPairs(self):
        print 'Not implemented yet'
        #all_ims, all_coords, all_masks = helenUtils.getAllData(path)
        #self.all_ims = all_ims
        #self.all_labels = [ self.getOutputs(self.preprocessCoords(coords), all_ims[0]) for coords in all_coords]
        #self.all_labels = utils.transposeList(self.all_labels)
        #return self.all_ims, self.all_labels

    def numTotalSamples(self):
        return len(self.all_names)

    def preprocessCoords(self, coords):
        return coords[0::self.coords_sparsity]

    def getOutputs(self, coords, im):
        return [coords]

    """ 
    def visualizeBatch(self):
        for im_batch, labels_batch in self.generate():
            for i in range(len(im_batch)):
                utils.visualizeCoords(im_batch[i], len(im_batch[i]) * labels_batch[i])
    """
    
    def getBatchFromNames(self, sample_names):
        X, Y = [], []
        for name in sample_names:
            x, y = self.getPair(name)
            X.append(x)
            Y.append(y)
        return self.getBatchFromSamples(X, Y)
    
    def getBatchFromSamples(self, X, Y):
        if isinstance(Y[0], tuple):
            raise ValueError('Please use a list for multiple outputs')
        if isinstance(Y[0], list):
            Y = utils.transposeList(Y)
            Y = [np.array(output_batch_type) for output_batch_type in Y]
        else:
            Y = np.array(Y)

        X = utils.transposeList(X)
        X = [np.array(in_array) for in_array in X]
        return X, Y

    def generate(self):
        while(True):

            # epoch complete
            num_batches = len(self.all_names) / self.batch_size
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
                    im = np.load(self.ims_path + '/' + name + self.im_extension)
                    coords = np.load(self.coords_path + '/' + name + self.label_extension)
                    coords = self.preprocessCoords(coords)
                    label = self.getOutputs(coords, im)
                    inputs = self.getInputs(coords, im)
                    Y.append(label)
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
    def __init__(self, path):
        BatchGenerator.__init__(self, path)
    
    def getInputs(self, coords, im):
        return [im]

    def getOutputs(self, coords, im):
        return [helenUtils.getLipCoords(coords)]


class MaskAndBboxBatchGenerator(BatchGenerator):

    def __init__(self, path, mask_side_len):
        BatchGenerator.__init__(self, path)
        self.mask_side_len = mask_side_len

        # pdfs cache for speeding up coord mask expansions (makes a big difference in training times)
        #self.pdfs = utils.getGaussians(10000, self.mask_side_len)

    def getLabels(self, coords, im):
        """coords = np.reshape(coords, (self.num_coords, 2))
        heatmap = utils.coordsToHeatmapsFast(coords, self.pdfs)
        heatmap = np.moveaxis(heatmap, 0, -1)
        return heatmap"""

        # need lip_coords in pixel-coordinate units for generating masks
        lip_coords = (len(im) * np.array(helenUtils.getLipCoords(coords))).astype(int)
        mask = utils.getMask([lip_coords], (len(im), len(im[0])), (len(im), len(im[0])))
        mask = np.expand_dims(mask, axis=-1)
        
        # bbox coords
        lip_coords_normalized = helenUtils.getLipCoords(coords)
        bbox = utils.getBbox(lip_coords_normalized)
        bbox = utils.getExpandedBbox(bbox, 0.5, 0.5)
        return [bbox, mask]

    def getInputs(self, coords, im):
        inputs = [im]
        labels = self.getLabels(coords, im)
        for label in labels:
            inputs += [label]
        return inputs

    def getOutputs(self, coords, im):
        """ These are mock outputs to satisfy some of Keras' checks """
        return [0, 0]


class MaskBatchGenerator(BatchGenerator):

    def __init__(self, path, mask_side_len):
        BatchGenerator.__init__(self, path)
        self.mask_side_len = mask_side_len

    def getLabels(self, coords, im):

        # need lip_coords in pixel-coordinate units for generating masks
        lip_coords = (len(im) * np.array(helenUtils.getLipCoords(coords))).astype(int)
        mask = utils.getMask([lip_coords], (len(im), len(im[0])), (len(im), len(im[0])))
        mask = np.expand_dims(mask, axis=-1)
        return mask

    def getInputs(self, coords, im):
        inputs = im
        labels = self.getLabels(coords, im)
        return [inputs, labels]

    def getOutputs(self, coords, im):
        """ These are mock outputs to satisfy some of Keras' checks """
        return [0]


class PointMaskBatchGenerator(BatchGenerator):

    def __init__(self, path, mask_side_len):
        BatchGenerator.__init__(self, path)
        self.mask_side_len = mask_side_len

        # 0 for most precise
        self.pdfs = 4 * [None]
        self.pdfs[0] = utils.getGaussians(10000, self.mask_side_len, stddev=0.008)
        self.pdfs[1] = utils.getGaussians(10000, self.mask_side_len/2, stddev=0.015)
        self.pdfs[2] = utils.getGaussians(10000, self.mask_side_len/4, stddev=0.03)
        self.pdfs[3] = utils.getGaussians(10000, self.mask_side_len/8, stddev=0.05)
    
    def getLabels(self, coords, im):
        coords = np.reshape(coords, (self.num_coords, 2))
        lip_coords = helenUtils.getLipCoords(coords)

        labels = []
        for i in range(len(self.pdfs)):
            masks = utils.coordsToHeatmapsFast(lip_coords, self.pdfs[i])
            masks = np.moveaxis(masks, 0, -1)
            masks /= (0.02 * np.max(masks, axis=(0,1)))
            masks = np.minimum(masks, 1.0)
            l = self.mask_side_len / 2**i
            masks = cv2.resize(masks, (l, l), interpolation=cv2.INTER_AREA)
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
        labels = self.getLabels(coords, im)
        return [im] + labels

    def getOutputs(self, coords, im):
        """ Dummy outputs. Basically for however many non-None loss entries we have in the model."""
        return [0, 0, 0, 0]


class LineMaskBatchGenerator(BatchGenerator):

    def __init__(self, path, mask_side_len):
        self.mask_side_len = mask_side_len
        BatchGenerator.__init__(self, path)

    def getOutputs(self, coords, im):
        return [0]
    
    def getInputs(self, coords, im):
        return [im] + self.getLabels(coords, im)

    def getLabels(self, coords, im):
        lip_coords = helenUtils.getLipCoords(coords)
        line_mask = helenUtils.getLipLineMask(lip_coords, np.shape(im), (self.mask_side_len, self.mask_side_len))
        return [line_mask]
