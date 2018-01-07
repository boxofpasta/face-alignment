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
        Each label or mask needs to be saved as a .npy file in path path/coords.
        Labels and images that correspond to each other must have the same name (excluding file extension).
        The names of all samples (training pairs) must be in path/names.json.
    """
    def __init__(self, path, sparse_coords=True, val_path=None, read_all=False):
        """
        Parameters
        ----------
        read_all: 
            If True, will read all the .npy files into an array at once. Better for small datasets.
        """
        self.batch_size = 50
        self.names = []
        self.sparse_coords = sparse_coords

        # essentially taking every self.sparsity points in the original coords
        self.sparsity = 4.0 
        self.num_coords = np.ceil(194 / sparsity) if sparse_coords else 194
        
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
        if read_all == True:
            all_ims, all_coords = helenUtils.getAllData(path)
            self.all_ims = all_ims
            self.all_labels = [self.getLabel(self.preprocessCoords(coords)) for coords in all_coords]

    def getPair(self, sample_name):
        im = np.load(self.ims_path + '/' + sample_name + '.npy')
        label = self.getLabel(np.load(self.coords_path + '/' + sample_name + '.npy'))  
        return im, label    

    def getAllPairs(self):
        if self.all_ims == None or self.all_labels == None:
            all_ims, all_coords = helenUtils.getAllData(self.name_path)
            self.all_ims = all_ims
            self.all_labels = [self.getLabel(coords) for coords in all_coords]
        return self.all_ims, self.all_labels

    def numTotalSamples(self):
        return len(self.names)

    def preprocessCoords(coords):
        coords = np.reshape(coords, (-1, 2))
        if self.sparse_coords:
            return coords[0::self.sparsity]
        return coords

    def getLabel(self, coords):
        return coords

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
                if self.all_ims != None and len(self.all_ims) == len(self.names):
                    X = [self.all_ims[k] for k in indices]
                    Y = [self.all_labels[k] for k in indices]
                else:
                    cur_names = [self.names[k] for k in indices]
                    X = []
                    Y = []
                    for name in cur_names:
                        X.append(np.load(self.ims_path + '/' + name + self.im_extension))
                        coords = np.load(self.coords_path + '/' + name + self.label_extension)
                        coords = self.preprocessCoords(coords)
                        Y.append(self.getLabel(coords))
                yield np.array(X), np.array(Y)


class HeatmapBatchGenerator(BatchGenerator):

    def __init__(self, path, heatmap_sidelen, val_path=None, read_all=False):
        BatchGenerator.__init__(self, path, val_path, read_all)

        # pdfs cache for speeding up heatmap expansions (makes a big difference in training times)
        self.heatmap_sidelen = heatmap_sidelen
        self.pdfs = utils.getGaussians(10000, self.heatmap_sidelen)

    def getLabel(self, coords):
        """coords = np.reshape(coords, (self.num_coords, 2))
        heatmap = utils.coordsToHeatmapsFast(coords, self.pdfs)
        heatmap = np.moveaxis(heatmap, 0, -1)
        return heatmap"""
        # eyepoints start [134, 153]

        # WARN: there must be 194 coords for this to work
        coords = np.reshape(coords, (-1, 2))
        eyecoords = coords[134:154]
        return utils.getBbox(eyecoords)