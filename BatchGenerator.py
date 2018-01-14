from scipy.stats import norm
import os
import numpy as np
import matplotlib.pyplot as plt
import utils.helenUtils as helenUtils
import utils.generalUtils as utils
import json 

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
        self.batch_size = 50
        self.names = []

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
            self.names = json.load(fp)

        self.steps_per_epoch = len(self.names) / self.batch_size

    def getPair(self, sample_name):
        im = np.load(self.ims_path + '/' + sample_name + '.npy')
        coords = np.load(self.coords_path + '/' + sample_name + '.npy')
        label = self.getLabel(self.preprocessCoords(coords), im)  
        return im, label    

    def getAllPairs(self):
        all_ims, all_coords, all_masks = helenUtils.getAllData(path)
        self.all_ims = all_ims
        self.all_labels = [ self.getLabel(self.preprocessCoords(coords), all_ims[0]) for coords in all_coords]
        self.all_labels = utils.transposeList(self.all_labels)
        return self.all_ims, self.all_labels

    def numTotalSamples(self):
        return len(self.names)

    def preprocessCoords(self, coords):
        return coords[0::self.coords_sparsity]

    def getLabel(self, coords, im):
        return coords

    def visualizeBatch(self):
        for im_batch, labels_batch in self.generate():
            for i in range(len(im_batch)):
                utils.visualizeCoords(im_batch[i], len(im_batch[i]) * labels_batch[i])

    def generate(self):
        while(True):

            # epoch complete
            num_batches = len(self.names) / self.batch_size
            rand_idx = np.arange(0, len(self.names))
            np.random.shuffle(rand_idx)

            for i in range(0, num_batches):

                # get range of current batch
                start = (i * self.batch_size) % len(self.names)
                end = min(start + self.batch_size, len(self.names))
                wrap = max(start + self.batch_size - len(self.names), 0)
                indices = np.concatenate((rand_idx[start : end], (rand_idx[0 : wrap])), axis=0)
                
                # generate batch
                #if self.all_ims != None and len(self.all_ims) == len(self.names):
                #    X = [self.all_ims[k] for k in indices]
                #    Y = [self.all_labels[k] for k in indices]
                #else:
                cur_names = [self.names[k] for k in indices]
                X = []
                Y = []
                for name in cur_names:
                    im = np.load(self.ims_path + '/' + name + self.im_extension)
                    coords = np.load(self.coords_path + '/' + name + self.label_extension)
                    coords = self.preprocessCoords(coords)
                    label = self.getLabel(coords, im)
                    Y.append(label)
                    X.append([im] + label)
                
                # in the case of multiple outputs
                if isinstance(Y[0], tuple):
                    raise ValueError('Please use a list for multiple outputs')
                if isinstance(Y[0], list):
                    Y = utils.transposeList(Y)
                    Y = [np.array(output_batch_type) for output_batch_type in Y]
                else:
                    Y = np.array(Y)

                X = utils.transposeList(X)
                X = [np.array(in_array) for in_array in X]

                # X also has the labels appended to itself, in case the model needs access to them internally
                # as part of/before computing the loss (https://github.com/keras-team/keras/issues/4781). 
                yield (X, [Y[0], Y[0]])
                #yield (X, Y[0])


class MaskBatchGenerator(BatchGenerator):

    def __init__(self, path, mask_side_len):
        BatchGenerator.__init__(self, path)
        self.mask_side_len = mask_side_len

        # pdfs cache for speeding up coord mask expansions (makes a big difference in training times)
        #self.pdfs = utils.getGaussians(10000, self.mask_side_len)

    def getLabel(self, coords, im):
        """coords = np.reshape(coords, (self.num_coords, 2))
        heatmap = utils.coordsToHeatmapsFast(coords, self.pdfs)
        heatmap = np.moveaxis(heatmap, 0, -1)
        return heatmap"""

        # need lip_coords in pixel-coordinate units for generating masks
        lip_coords = (len(im) * np.array(helenUtils.getLipCoords(coords))).astype(int)
        lip_coords = [tuple(lip_coord) for lip_coord in lip_coords]
        mask = utils.getMask([lip_coords], (len(im), len(im[0])), (len(im), len(im[0])))
        
        # bbox coords
        lip_coords_normalized = helenUtils.getLipCoords(coords)
        bbox = utils.getBbox(lip_coords_normalized)
        bbox = utils.getExpandedBbox(bbox, 0.5, 0.5)
        return [bbox, mask]