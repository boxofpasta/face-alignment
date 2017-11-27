import os
import scipy.misc
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import utils.helenUtils as helenUtils
import utils.utils as utils
from matplotlib.patches import Circle
from skimage.transform import resize
import json
import time
import sys
import ModelFactory
import BatchGenerator
from keras.models import load_model
from keras.applications import mobilenet


def try_saved_model(path):
    print 'Loading model ...'
    model = load_model(path, custom_objects={
        'relu6': mobilenet.relu6,
        'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
    print 'COMPLETE'

    sample_paths = []
    for fname in os.listdir('downloads/samples'):
        if fname.endswith(('.jpg', '.png')):
            sample_paths.append('downloads/samples/' + fname)

    for path in sample_paths:
        im = scipy.misc.imread(path)
        im = resize(im, (224, 224))
        im = im[:,:,:3]
        label = model.predict(np.array([im]), batch_size=1)
        label = np.reshape(label, (194, 2))
        label *= 224
        utils.visualize_labels(im, label)


def visualize_samples(model=None):
    fnames = ['100032540_1.npy', '1691766_1.npy', '11564757_2.npy', '110886318_1.npy']
    for fname in fnames:
        im = np.load('data/train/ims/' + fname)

        if model == None:
            label = np.load('data/train/labels/' + fname)
        else:
            label = model.predict(np.array([im]), batch_size=1)

        label = np.reshape(label, (194, 2))
        label *= 224
        utils.visualize_labels(im, label)


if __name__ == '__main__':
    """batch_generator = BatchGenerator.BatchGenerator('data/train')
    factory = ModelFactory.ModelFactory()
    model = factory.getFullyConnected()
    model.fit_generator(generator=batch_generator.generate(),
                        steps_per_epoch=batch_generator.steps_per_epoch,
                        epochs=40)

    model.save('models/saved/fully_connected.h5')"""
    visualize_samples()