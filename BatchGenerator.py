from scipy.stats import norm
import os
import numpy as np
import matplotlib.pyplot as plt
import utils.helenUtils as helenUtils
import utils.generalUtils as utils
import json


class BatchGenerator:
    """
    This class assumes a specific directory and file structure for your data:
        Each image needs to be saved as a .npy file in path train_path/ims.
        Each label or mask needs to be saved as a .npy file in path train_path/labels.
        Labels and images that correspond to each other must have the same name (excluding file extension).
        The names of all samples (training pairs) must be in train_path/names.json.
    """

    def __init__(self, train_path, mask_side_len, val_path=None, read_all=False):
        """
        Parameters
        ----------
        read_all: 
            If True, will read all the .npy files into an array at once. Better for small datasets.
        """
        self.num_coords = 194
        self.batch_size = 50
        self.mask_side_len = mask_side_len
        self.names = []
        
        # filled only if read_all == True
        self.all_ims = None
        self.all_labels = None

        self.name_path = train_path + '/names.json'
        self.train_ims_path = train_path + '/ims'
        self.train_labels_path = train_path + '/labels'
        self.im_extension = '.npy'
        self.label_extension = '.npy'

        # pdfs cache for speeding up heatmap expansions
        self.pdfs = utils.get_gaussians(10000, self.mask_side_len)
        
        with open(self.name_path) as fp:
            self.names = json.load(fp)

        self.steps_per_epoch = len(self.names) / self.batch_size
        if read_all == True:
            self.all_ims, self.all_labels = helenUtils.get_all_data(train_path)
            raise NameError('hey programmer, labels need to be converted to heatmaps')
            """
            self.all_ims = []
            self.all_labels = []
            for name in self.names:
                im_path = self.train_ims_path + '/' + name + self.im_extension
                label_path = self.train_labels_path + '/' + name + self.label_extension
                self.all_ims.append(np.load(im_path))
                self.all_labels.append(np.load(label_path).flatten())
            """

    def num_total_samples(self):
        return len(self.names)

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
                        X.append(np.load(self.train_ims_path + '/' + name + self.im_extension))
                        coords = np.load(self.train_labels_path + '/' + name + self.label_extension)
                        coords = np.reshape(coords, (self.num_coords, 2))
                        heatmap = utils.coords_to_heatmaps_fast(coords, self.pdfs)
                        heatmap = np.moveaxis(heatmap, 0, -1)
                        Y.append(heatmap)
                yield np.array(X), np.array(Y)

