import os
import sys
import scipy.misc
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import utils.helenUtils as helenUtils
import utils.utils as utils
from matplotlib.patches import Circle
import json


""" Run to update the .npy files (resized images and labels) from the downloaded dataset. """

# for serializing a smaller dataset
use_small = True

# takes only a small sample for testing purposes
use_samples = False

# other common params
targ_im_len = 224
npy_path = 'data/train'

if use_small:
    train_paths = ['downloads/helen_1']
else:
    train_paths = ['downloads/helen_1', 'downloads/helen_2', 'downloads/helen_3', 'downloads/helen_4', 'downloads/helen_5']

if use_samples:
    sample_names = ['11564757_2', '1240746154_1', '1165647416_1', '1691766_1']
else:
    sample_names = None

for path in train_paths:
    train_props = helenUtils.DatasetProps('.jpg', '.txt', path, 'downloads/annotation')
    ims, labels = helenUtils.process_data(train_props, targ_im_len, sample_names=sample_names)
    helenUtils.serialize_data(ims, labels, npy_path)

# visualize the serialized samples
if use_samples:
    for name in sample_names:
        im = np.load(npy_path + '/ims/' + name + '.npy')
        if targ_im_len == -1:
            factor = 1
        else:
            factor = targ_im_len
        label = np.load(npy_path + '/labels/' + name + '.npy')
        for coord in label:
            coord[0] -= 1
            coord[1] -= 1
        label *= factor
        utils.visualize_labels(im, label)

#helenUtils.save_data(train_props, 'data/train', 224, append_to_names=False)
