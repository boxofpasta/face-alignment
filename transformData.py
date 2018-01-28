import os
import sys
import scipy.misc
import numpy as np
import matplotlib
import cv2
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import utils.helenUtils as helenUtils
import utils.generalUtils as utils
from matplotlib.patches import Circle
from matplotlib.patches import Polygon
import json

""" Run to update the .npy files (resized images and labels) from the downloaded dataset. """

transform_train = True
transform_test = False
ibug_version = True

# applies only if transform_train == True and ibug_version == False. For serializing just the helen_1 folder
use_small = True

# takes only a small sample for testing purposes
use_samples = False
targ_im_len = 224

im_extension = '.jpg'
coords_extension = '.pts' if ibug_version else '.txt'

if transform_test:
    if ibug_version:
        im_path = 'downloads/helen_ibug/testset'
        coords_path = im_path
        npy_test_path = 'data/test_ibug'
    else:
        im_path = 'downloads/helen_test'
        coords_path = 'downloads/annotation'
        npy_test_path = 'data/test'
    print "\nProcessing images in " + im_path + " and saving to " + npy_test_path + "..."
    test_props = helenUtils.DatasetProps(im_extension, coords_extension, im_path, coords_path)
    ims, coords = helenUtils.processData(test_props, targ_im_len, sample_names=None, ibug_version=ibug_version)
    helenUtils.serializeData(ims, coords, npy_test_path, ibug_version=ibug_version)

if transform_train:
    if ibug_version:
        im_paths = ['downloads/helen_ibug/trainset']
        coords_path = im_paths[0]
        npy_path = 'data/train_ibug_lip_zoomed'
    else:
        coords_path = 'downloads/annotation'
        if use_small:
            im_paths = ['downloads/helen_1']
        else:
            im_paths = ['downloads/helen_1', 'downloads/helen_2', 'downloads/helen_3', 'downloads/helen_4', 'downloads/helen_5']
        if use_small:
            npy_path = 'data/train_small'
        else:
            npy_path = 'data/train'

    if use_samples:
        sample_names = ['100466187_1', '11564757_2', '1240746154_1', '1165647416_1', '1691766_1']
    else:
        sample_names = None

    for im_path in im_paths:
        print "\nProcessing images in " + im_path + " and saving to " + npy_path + "..."

        # we'll operate under the assumption that you won't have too many samples (you shouldn't anyways)
        if use_samples:
            print "\nReading images ..."
            ims = helenUtils.readImagesHelen(im_path, im_extension, sample_names=sample_names)
            coords = helenUtils.readCoordsHelen(coords_path, coords_extension, sample_names=sample_names, ibug_version=ibug_version)
            helenUtils.processData(ims, coords, targ_im_len)
            helenUtils.serializeData(ims, coords, npy_path, ibug_version=ibug_version)
        else:
            im_reader = helenUtils.ImagesReader(im_path, im_extension, 2000)

            # operating under the assumption that coords/annotations take up a negligible 
            # amount of memory relative to the images.
            while not im_reader.complete:
                print "\nReading images..."
                ims_list, names_list = im_reader.read()
                ims = utils.getDictFromLists(names_list, ims_list)
                coords = helenUtils.readCoordsHelen(coords_path, coords_extension, sample_names=names_list, ibug_version=ibug_version)
                helenUtils.processData(ims, coords, targ_im_len)
                helenUtils.serializeData(ims, coords, npy_path, ibug_version=ibug_version)

                # test
                #helenUtils.trySerializedFolder(npy_path, targ_im_len)

# visualize the serialized samples
if use_samples:
    for name in sample_names:
        helenUtils.trySerializedSample(npy_path, name, targ_im_len)

#helenUtils.save_data(train_props, 'data/train', 224, append_to_names=False)
