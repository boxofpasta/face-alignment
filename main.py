import os
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import utils.helenUtils as helenUtils
import utils.utils as utils
from matplotlib.patches import Circle
import json
import BatchGenerator
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.models import Sequential
from keras.models import Model
from keras.applications import mobilenet

if __name__ == '__main__':

    # getting serialized data
    batch_generator = BatchGenerator.BatchGenerator('data/train')

    base_model = mobilenet.MobileNet(include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(units=1024, activation='relu')(x)
    x = Dense(units=2*194, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)

    """model = Sequential()
    model.add(Reshape((224 * 224 * 3,), input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=2*194, activation='sigmoid'))"""

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit_generator(generator=batch_generator.generate(),
                        steps_per_epoch=batch_generator.steps_per_epoch,
                        epochs=40)

    model.save('models/saved/fully_connected.h5')

    """fname = '100032540_1.npy'
    im = np.load('data/train/ims/' + fname)
    label = np.load('data/train/labels/' + fname)
    label = np.reshape(label, (194, 2))
    label *= 224
    utils.visualize_labels(im, label)"""

    """
    for name in ims:
        label = labels[name]
        im = ims[name]
        paired.append([im, label])
        visualize_labels(im, label)
    """