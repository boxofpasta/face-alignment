""" Run to update the .npy files (resized images and labels) from the downloaded dataset. """

#https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py
from os import path, sys
from sys import argv
import os
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import scipy.misc
import numpy as np
import matplotlib
import cv2
import matplotlib.pyplot as plt
from src.utils import helenUtils
from src.utils import generalUtils
from matplotlib.patches import Circle
from matplotlib.patches import Polygon
import json
import optparse

parser = optparse.OptionParser()
parser.add_option('-i', '--image-path',
    action="store", dest="image_path",
    help="query string", default="")
parser.add_option('-c', '--coords-path',
    action="store", dest="coords_path",
    help="query string", default="")
options, args = parser.parse_args()

# takes only a small sample for testing purposes
use_samples = False
transform_train = True
ibug_version = False

targ_im_width = 320
im_extension = '.jpg'
coords_extension = '.pts'

cur_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_folder + '/../')
data_path = cur_folder + '/../data'
im_path = options.image_path
coords_path = options.coords_path
npy_path = data_path + '/train'

potential_sample_names = ['100466187_1', '11564757_2', '1240746154_1', '1165647416_1', '1691766_1']
sample_names = potential_sample_names if use_samples else None

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
