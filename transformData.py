import os
import sys
import scipy.misc
import numpy as np
#import matplotlib
#matplotlib.use('Qt5Agg')
#import matplotlib.pyplot as plt
import utils.helenUtils as helenUtils
import utils.generalUtils as utils
#from matplotlib.patches import Circle
import json


""" Run to update the .npy files (resized images and labels) from the downloaded dataset. """

transform_train = False
transform_test = True

# takes only a small sample for testing purposes
use_samples = False
targ_im_len = 224

if transform_test:
    path = 'downloads/helen_test'
    npy_test_path = 'data/test'
    print "\nProcessing images in " + path + " and saving to " + npy_test_path + "... \n"
    test_props = helenUtils.DatasetProps('.jpg', '.txt', path, 'downloads/annotation')
    ims, labels = helenUtils.processData(test_props, targ_im_len, sample_names=None)
    helenUtils.serializeData(ims, labels, npy_test_path)

if transform_train:

    # for serializing a smaller dataset
    use_small = False

    if use_small:
        npy_path = 'data/train_small'
    else:
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
        print "\nProcessing images in " + path + " and saving to " + npy_path + "... \n"
        train_props = helenUtils.DatasetProps('.jpg', '.txt', path, 'downloads/annotation')
        ims, labels = helenUtils.processData(train_props, targ_im_len, sample_names=sample_names)
        helenUtils.serializeData(ims, labels, npy_path)

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
        utils.visualizeLabels(im, label)

#helenUtils.save_data(train_props, 'data/train', 224, append_to_names=False)
