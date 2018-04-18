import utils.helenUtils as helenUtils
import utils.generalUtils as utils
import scipy.misc
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import sys


def tryPointMaskerRefinedOnSamples(model, folder, output_folder, compare_coords=None):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for fname in os.listdir(folder):
        if fname.endswith(('png', 'jpg')):
            im = cv2.imread(folder + '/' + fname)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            print folder + '/' + fname
            im = cv2.resize(im, (224, 224))
            width = len(im[0])
            height = len(im)
            t1 = time.time()
            
            base_masks, residual_masks = getNormalizedCascadedMasksFromImage(model, im)
            print time.time() - t1
            base_coords = utils.getCoordsFromPointMasks(base_masks, width, height, 'mean')
            residual_coords = utils.getCoordsFromPointMasks(residual_masks, 28, 28, 'mean')
            #max_coords = utils.getCoordsFromPointMasks(base_masks, width, height, 'max')
            coords = np.add(base_coords, residual_coords) - 28 / 2.0

            # scale back 
            full_im = cv2.imread(folder + '/' + fname)
            full_im = cv2.cvtColor(full_im, cv2.COLOR_BGR2RGB)
            scale_width = len(full_im[0]) / float(len(im[0]))
            scale_height = len(full_im) / float(len(im))
            print scale_width

            coords[:,0] *= scale_height
            coords[:,1] *= scale_width
            #for i in range(len(residual_masks)):
            #    plt.imshow(residual_masks[i])
            #    plt.show()

            #utils.visualizeCoords(im, base_coords)
            output_path = output_folder + '/' + fname

            key = fname[:-4]
            if key in compare_coords:
                utils.visualizeCoords(full_im, np.concatenate([compare_coords[key], coords], axis=0), np.arange(0, len(compare_coords[key])), output_name=output_path)
            else:
                utils.visualizeCoords(full_im, coords, output_name=output_path)