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

def visualizeHeatmaps(sample_names=[]):
    pdfs = utils.getGaussians(10000, 56)

    for sample_name in sample_names:
        coords = np.load('data/train/labels/' + sample_name + '.npy')
        coords = np.reshape(coords, (194, 2))
        heatmap = utils.coordsToHeatmapsFast(coords, pdfs)
        heatmap = np.moveaxis(heatmap, 0, -1)
        summed = np.sum(heatmap, axis=-1)
        plt.imshow(summed)
        plt.show()

def visualizeSamples(sample_names, model=None, special_indices=[]):
    for sample_name in sample_names:
        im = np.load('data/train/ims/' + sample_name + '.npy')

        if model == None:
            label = np.load('data/train/labels/' + sample_name + '.npy')
        else:
            label = model.predict(np.array([im]), batch_size=1)

        label = np.reshape(label, (194, 2))
        label *= 224
        utils.visualizeLabels(im, label, special_indices)

def queryCoordPositions():
    samples = ['13602254_1']
    while True:
        val = int(raw_input('enter indices to draw red up to: '))
        indices = [i for i in range(val+1)]
        visualizeSamples(samples, special_indices=indices)

def getAvgTestError(model, test_path):
    all_ims, all_labels = helenUtils.getAllData(test_path)
    preds = model.predict(np.array(all_ims), batch_size=len(all_ims))
    error = 0
    for i in range(len(preds)):
        # compute euclidean distance squared sum for all points
        error += np.sum(np.square(pred - label))

    return error / len(preds)

if __name__ == '__main__':
    samples = ['13602254_1', '2908549_1', '100032540_1', '1691766_1', '11564757_2', '110886318_1']
    
    #visualizeSamples(samples)
    #model = get_saved_model('models/fully_connected_v2.h5')
    #print get_avg_test_error(model, 'data/test')
    #try_saved_model('models/fully_connected_v1.h5')
    #model = get_saved_model('models/tmp/fully_conv.h5')
    
    factory = ModelFactory.ModelFactory()
    #model = factory.getBboxRegressor()
    model = factory.getSaved('models/bbox_lite_loss_scaled.h5')
    batch_generator = BatchGenerator.HeatmapBatchGenerator('data/train', factory.heatmap_side_len)

    for sample in samples:
        im, label = batch_generator.getPair(sample)
        pred = np.squeeze(model.predict(np.array([im]), batch_size=1))
        utils.visualizeBboxes(im, [224 * pred, 224 * label])
        #utils.visualizeBboxes(im, [224 * label])

    """model.fit_generator(generator=batch_generator.generate(),
                        steps_per_epoch=batch_generator.steps_per_epoch,
                        epochs=240)

    model.save('models/tmp/bbox_lite_iou.h5')"""
    #visualize_samples()