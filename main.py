import os
import time
import scipy.misc
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import utils.helenUtils as helenUtils
import utils.generalUtils as utils
from matplotlib.patches import Circle
from skimage.transform import resize
import cv2
import json
import time
import sys
import ModelFactory
import BatchGenerator
from keras.models import load_model
from keras.applications import mobilenet


def trySavedFullyConnected(path):
    model = get_saved_model(path)
    sample_paths = []
    for fname in os.listdir('downloads/samples'):
        if fname.endswith(('.jpg', '.png')):
            sample_paths.append('downloads/samples/' + fname)

    for path in sample_paths:
        im = scipy.misc.imread(path)
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)

        # alpha channel needs to be cutoff
        if im.shape[2] > 3:
           im = im[:,:,:3]

        label = model.predict(np.array([im]), batch_size=1)
        label = np.reshape(label, (194, 2))
        label *= 224
        utils.visualizeLabels(im, label)

def visualizeHeatmaps():
    fnames = ['13602254_1.npy', '2908549_1.npy', '100032540_1.npy', '1691766_1.npy', '11564757_2.npy', '110886318_1.npy']
    pdfs = utils.getGaussians(10000, 56)

    for fname in fnames:
        coords = np.load('data/train/labels/' + fname)
        coords = np.reshape(coords, (194, 2))
        heatmap = utils.coordsToHeatmapsFast(coords, pdfs)
        heatmap = np.moveaxis(heatmap, 0, -1)
        summed = np.sum(heatmap, axis=-1)
        plt.imshow(summed)
        plt.show()

def visualizeSamples(model=None):
    fnames = ['13602254_1.npy', '2908549_1.npy', '100032540_1.npy', '1691766_1.npy', '11564757_2.npy', '110886318_1.npy']
    pdfs = utils.getGaussians(10000, 56)
    for fname in fnames:
        im = np.load('data/train/ims/' + fname)

        if model == None:
            label = np.load('data/train/labels/' + fname)
        else:
            label = model.predict(np.array([im]), batch_size=1)

        label = np.reshape(label, (194, 2))
        label *= 224
        utils.visualizeLabels(im, label)


def getAvgTestError(model, test_path):
    all_ims, all_labels = helenUtils.getAllData(test_path)
    preds = model.predict(np.array(all_ims), batch_size=len(all_ims))
    error = 0
    for i in range(len(preds)):
        coords_shape = (194, 2)
        label = np.reshape(all_labels[i], coords_shape)
        pred = np.reshape(preds[i], coords_shape)

        # compute euclidean distance squared sum for all points
        error += np.sum(np.square(pred - label))

    return error / len(preds)

if __name__ == '__main__':
    #model = get_saved_model('models/fully_connected_v2.h5')
    #print get_avg_test_error(model, 'data/test')

    visualize_samples()
    #try_saved_model('models/fully_connected_v1.h5')
    #model = get_saved_model('models/tmp/fully_conv.h5')
    """factory = ModelFactory.ModelFactory()
    model = factory.getFullyConvolutional()
    batch_generator = BatchGenerator.BatchGenerator('data/train', factory.mask_side_len)
    model.fit_generator(generator=batch_generator.generate(),
                        steps_per_epoch=batch_generator.steps_per_epoch,
                        epochs=10)

    model.save('models/tmp/fully_conv.h5')"""
    #visualize_samples()