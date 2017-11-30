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

#train_paths = ['downloads/helen_1', 'downloads/helen_2', 'downloads/helen_3', 'downloads/helen_4', 'downloads/helen_5']
targ_im_len = -1
npy_path = 'data/train'
train_paths = ['downloads/helen_1']
train_props = helenUtils.DatasetProps('.jpg', '.txt', train_paths[0], 'downloads/annotation')
sample_names = ['11564757_2', '1240746154_1', '1165647416_1', '1691766_1']
ims, labels = helenUtils.process_data(train_props, targ_im_len, sample_names=sample_names)
helenUtils.serialize_data(ims, labels, npy_path)

"""
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
    utils.visualize_labels(im, label)"""

#helenUtils.save_data(train_props, 'data/train', 224, append_to_names=False)
