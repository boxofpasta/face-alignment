# https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py
from os import path, sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
from keras.utils import plot_model
from src.tests import tryPointMaskerRefinedOnSamples
from src.model import loadPointMaskerRefined
import optparse

parser = optparse.OptionParser()
parser.add_option('-i', '--image-path',
    action="store", dest="image_path",
    help="path to folder of images that we will process", default="")
parser.add_option('-m', '--model-path')
    action="store", dest="model_path",
    help="path to model that we will load", default="")
parser.add_option('-o', '--output-path')
    action="store", dest="output_path",
    help="path to output folder", default="")

options, args = parser.parse_args()
model = loadPointMaskerRefined(args.model_path)
model.summary()

tryPointMaskerRefinedOnSamples(model, args.image_path, args.output_path)

# comparing to mftracker
# compare_coords = {}
# compare_folder = 'compare/mftracker-outputs-hard'
# for fname in os.listdir(compare_folder):
#     if fname.endswith('.txt'):
#         key = fname[:-4]
#         compare_coords[key] = utils.readCoords(compare_folder + '/' + fname)

# modelTests.tryPointMaskerCascadedOnSamples(model, 'downloads/hard-samples-2', 'outputs-2', compare_coords)