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
        self.mask_path = path + '/masks'
        self.im_extension = '.npy'
        self.label_extension = '.npy'
        
        with open(self.name_path) as fp:
            self.names = json.load(fp)

        self.steps_per_epoch = len(self.names) / self.batch_size
        """
        if read_all == True:
            all_ims, all_coords, all_masks = helenUtils.getAllData(path)
            self.all_ims = all_ims
            for i in range(len(all_coords)):
                label = self.getLabel(self.preprocessCoords(all_coords[i]), all_masks[i])
                self.all_labels.append(label)
        """

    def getPair(self, sample_name):
        im = np.load(self.ims_path + '/' + sample_name + '.npy')
        coords = np.load(self.coords_path + '/' + sample_name + '.npy')
        mask = np.load(self.mask_path + '/' + sample_name + '.npy')
        label = self.getLabel(np.load(self.coords_path + '/' + sample_name + '.npy'))  
        return im, label    

    def getAllPairs(self):
        all_ims, all_coords, all_masks = helenUtils.getAllData(path)
        self.all_ims = all_ims
        for i in range(len(all_coords)):
            label = self.getLabel(self.preprocessCoords(all_coords[i]), all_masks[i])
            self.all_labels.append(label)
        

    def numTotalSamples(self):
        return len(self.names)

    def preprocessCoords(self, coords):
        return coords[0::self.coords_sparsity]

    def getLabel(self, coords):
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
                        mask = np.load(self.masks_path + '/' + name + self.mask_extension)
                        coords = self.preprocessCoords(coords)
                        Y.append(self.getLabel(coords, mask))
                    Y = utils.transposeList(Y)
                yield np.array(X), np.array(Y)


class MaskBatchGenerator(BatchGenerator):

    def __init__(self, path, mask_sidelen, val_path=None, read_all=False):
        BatchGenerator.__init__(self, path, val_path=val_path, read_all=read_all)

        # pdfs cache for speeding up heatmap expansions (makes a big difference in training times)
        self.mask_sidelen = mask_sidelen
        self.pdfs = utils.getGaussians(10000, self.mask_sidelen)

    def getLabel(self, coords, mask):
        """coords = np.reshape(coords, (self.num_coords, 2))
        heatmap = utils.coordsToHeatmapsFast(coords, self.pdfs)
        heatmap = np.moveaxis(heatmap, 0, -1)
        return heatmap"""
        lip_coords = helenUtils.getLipCoords(coords)
        return utils.getBbox(lip_coords)