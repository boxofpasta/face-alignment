# takes only a small sample for testing purposes
use_samples = False

import os
import sys
import scipy.misc
import numpy as np
import matplotlib
import cv2

cur_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_folder + '/../')
if use_samples:
    matplotlib.use('Qt5Agg')

# sys.path.append(os.getcwd() + '/../bin')
import matplotlib.pyplot as plt
from utils import helenUtils
from utils import generalUtils
from matplotlib.patches import Circle
from matplotlib.patches import Polygon
import json

""" Run to update the .npy files (resized images and labels) from the downloaded dataset. """

transform_train = False
transform_test = True # ignored if use_samples == True
ibug_version = True

# applies only if transform_train == True and ibug_version == False. For serializing just the helen_1 folder
use_small = True

targ_im_width = 224
im_extension = '.jpg'
coords_extension = '.pts' if ibug_version else '.txt'
downloads_path = cur_folder + '/../downloads'
data_path = cur_folder + '/../data'

if transform_test and not use_samples:
    if ibug_version:
        im_path = downloads_path + '/helen_ibug/testset'
        coords_path = im_path
        npy_path = data_path + '/test_ibug'
    else:
        im_path = downloads_path + '/helen_test'
        coords_path = downloads_path + '/annotation'
        npy_path = data_path + '/test'
    print "\nProcessing images in " + im_path + " and saving to " + npy_path + "..."
    helenUtils.reserializeFolderAsNpy(im_path, coords_path, im_extension, coords_extension, npy_path, targ_im_width, ibug_version=ibug_version)
    
    #coords = helenUtils.readCoordsHelen(coords_path, coords_extension, sample_names=[]], ibug_version=ibug_version)
    #helenUtils.serializeData(ims, coords, npy_path, ibug_version=ibug_version)
    
    """
    test_props = helenUtils.DatasetProps(im_extension, coords_extension, im_path, coords_path)
    ims, coords = helenUtils.processData(test_props, targ_im_len, sample_names=None, ibug_version=ibug_version)
    helenUtils.serializeData(ims, coords, npy_test_path, ibug_version=ibug_version)
    """

if transform_train:
    if ibug_version:
        im_paths = [downloads_path + '/helen_ibug/trainset']
        coords_path = im_paths[0]
        npy_path = data_path + '/train_ibug'
    else:
        coords_path = downloads_path + '/annotation'
        if use_small:
            im_paths = [downloads_path + '/helen_1']
        else:
            im_paths = [downloads_path + '/helen_1', downloads_path + '/helen_2', downloads_path + '/helen_3', downloads_path + '/helen_4', downloads_path + '/helen_5']
        if use_small:
            npy_path = data_path + '/train_small'
        else:
            npy_path = data_path + '/train'

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
            ims, coords = helenUtils.processData(ims, coords, targ_im_width)
            helenUtils.serializeData(ims, coords, npy_path, ibug_version=ibug_version)
        else:
            helenUtils.reserializeFolderAsNpy(im_path, coords_path, im_extension, coords_extension, npy_path, targ_im_width, ibug_version=ibug_version)
            
            """
            im_reader = helenUtils.ImagesReader(im_path, im_extension, 2000)

            # operating under the assumption that coords/annotations take up a negligible 
            # amount of memory relative to the images.
            while not im_reader.complete:
                print "\nReading images..."
                ims_list, names_list = im_reader.read()
                ims = utils.getDictFromLists(names_list, ims_list)
                coords = helenUtils.readCoordsHelen(coords_path, coords_extension, sample_names=names_list, ibug_version=ibug_version)
                helenUtils.serializeData(ims, coords, npy_path, ibug_version=ibug_version)
            """

            # test
            #helenUtils.trySerializedFolder(npy_path, targ_im_len)

# visualize the serialized samples
if use_samples:
    for name in sample_names:
        helenUtils.trySerializedSample(npy_path, name)

#helenUtils.save_data(train_props, data_path + '/train', 224, append_to_names=False)
