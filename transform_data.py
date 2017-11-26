from __future__ import print_function
import os
import sys
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import utils.helenUtils as helenUtils
import utils.utils as utils
from matplotlib.patches import Circle
import json


""" Run to update the .npy files (resized images and labels) from the downloaded dataset. """

# train_paths = ['downloads/helen_1', 'downloads/helen_2', 'downloads/helen_3', 'downloads/helen_4']
train_paths = ['downloads/helen_1']
train_props = helenUtils.DatasetProps('.jpg', '.txt', train_paths, 'downloads/annotation')
helenUtils.save_data(train_props, 'data/train', 224)