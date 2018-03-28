import sys
train = bool(int(sys.argv[1]))

import datetime
import os
import time
import scipy.misc
import numpy as np
import copy
np.random.seed(0)

import matplotlib
import keras
if not train:
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import utils.helenUtils as helenUtils
import utils.generalUtils as utils
import modelTests
from matplotlib.patches import Circle
from skimage.transform import resize
import cv2
import json
import time
import ModelFactory
import BatchGenerator
import keras.backend as K
from keras.utils import plot_model
from keras.models import load_model
from keras.applications import mobilenet
from tensorflow.python import debug as tf_debug


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

def visualizeSamples(folder, sample_names=[], model=None, special_indices=[]):
    if len(sample_names) == 0:
        with open(folder + '/names.json') as fp:
            sample_names = set(json.load(fp))

    for sample_name in sample_names:
        im = np.load(folder + '/ims/' + sample_name + '.npy')

        if model == None:
            coords = np.load(folder + '/coords/' + sample_name + '.npy')
        else:
            coords = model.predict(np.array([im]), batch_size=1)

        coords = np.reshape(coords, (-1, 2))
        coords *= len(im)-1
        utils.visualizeCoords(im, coords, special_indices)

def queryCoordPositions():
    samples = ['13602254_1']
    while True:
        val = int(raw_input('enter indices to draw red up to: '))
        indices = [i for i in range(val+1)]
        visualizeSamples(samples, special_indices=indices)

def getAvgTestError(model, batch_generator):
    """
    For all experiments we're using squared euclidean distance as the error metric.
    """
    all_ims = np.array(batch_generator.all_ims)
    all_labels = np.array(batch_generator.all_labels)
    # for some reason this doesn't work as expected (should be the same as evaluating loss through numpy?)
    #loss = model.evaluate(x=all_ims, y=all_labels)
    all_preds = model.predict(np.array(all_ims), batch_size=len(all_ims))
    error = np.sum(np.square(all_preds - all_labels))
    error /= (len(all_preds) * batch_generator.num_coords)
    return error

def visualizePointOrder(coords):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    min_y = np.min(coords[:,0])
    min_x = np.min(coords[:,1])
    max_y = np.max(coords[:,0])
    max_x = np.max(coords[:,1])
    for i in range(len(coords)):
        normalized_y = 1.0 - coords[i][0] / float(max_y)
        normalized_x = coords[i][1] / float(max_x)
        ax.text(normalized_x, normalized_y, str(i))
        #if i >= len(coords) - 2:
        #    print '(' + str(coords[i][0]) + ', ' + str(coords[i][1]) + ')'
    #print '(' + str(min_y) + ', ' + str(min_x) + ')'
    #print '(' + str(max_y) + ', ' + str(max_x) + ')'
    plt.show()


if __name__ == '__main__':
    ibug_version = not train
    notify_training_complete = True
    factory = ModelFactory.ModelFactory()
    
    samples = ['100466187_1', '13602254_1', '2908549_1', '100032540_1', '1691766_1', '11564757_2', '110886318_1']
    #samples = ['100466187_1', '11564757_2', '1240746154_1', '1165647416_1', '1691766_1']

    #visualizePointOrder(np.load('downloads/3214115970_1.npy'))
    #batch_generator = BatchGenerator.PointMaskBatchGenerator(samples, 'data/train_ibug', factory.mask_side_len)
    
    """
    for sample in samples:
        test_im = cv2.imread('downloads/helen_ibug/trainset/' + sample + '.jpg')
        hsv_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2HSV)
        hsv_im = hsv_im.astype(np.float32)
        hsv_im[:,:,1] *= 0.2
        hsv_im = hsv_im.astype(np.uint8)
        test_im = cv2.cvtColor(hsv_im, cv2.COLOR_HSV2RGB)
        plt.imshow(test_im)
        plt.show()
    """

    #batch_generator = BatchGenerator.PointMaskBatchGenerator(samples, 'data/train_ibug', factory.mask_side_len, 
    #                                                               augment_on_generate=True, ibug_version=ibug_version)
    
    """
    batch_generator = BatchGenerator.PointMaskCascadedBatchGenerator(samples, 'data/train_ibug', factory.mask_side_len, 224,
                                                                   augment_on_generate=True, ibug_version=ibug_version)
    X, Y = batch_generator.getBatchFromNames(samples, augment=True)
    ims, masks, hd_masks = X[0], Y[0], Y[1]
    for i in range(len(ims)):
        #plt.imshow(masks[i][:,:,0])
        #plt.show()
        #plt.imshow(hd_masks[i][:,:,0])
        #plt.show()

        coord_masks = np.moveaxis(masks[i], -1, 0)
        coord_hd_masks = np.moveaxis(hd_masks[i], -1, 0)
        coarse_coords = utils.getCoordsFromPointMasks(coord_masks, 224, 224)
        fine_coords = utils.getCoordsFromPointMasks(coord_hd_masks, 224, 224)
        utils.visualizeCoords(ims[i], coarse_coords + fine_coords, np.arange(0, len(coarse_coords)))
        #utils.visualizeCoordMasks(ims[i], masks[i])
    """

    now = datetime.datetime.now()
    #time_str = '2018-03-01:04:33' 
    time_str = now.strftime("%m-%d:%H:%M")
    #model_name = 'point_masker_attention'
    model_name = 'point_masker_cascaded'
    model_folder = 'models/' + model_name + '/' + time_str
    model_path = model_folder + '/model.h5'
    #model_path = '/home/tian/fun/face-alignment/models/point_masker_cascaded/03-21:22:46/model.h5'
    if not train:
        model_path = '/Users/tianxingli/Desktop/machine-learning/face-alignment/models/point_masker_cascaded/03-26:23:30/model.h5'
    #model_path = '/Users/tianxingli/Desktop/machine-learning/face-alignment/models/point_masker_attention/03-14:23:06/model.h5'
    #model_path = '/Users/tianxingli/Desktop/machine-learning/face-alignment/models/point_masker_concat/03-17:00:38/model.h5'
    #model_path = '/Users/tianxingli/Desktop/machine-learning/face-alignment/models/point_masker_attention/2018-03-06:22:31/model.h5'
    #model_path = '/Users/tianxingli/Desktop/machine-learning/face-alignment/models/point_masker_concat/model.h5'
    #model_path = "/Users/tianxingli/Desktop/machine-learning/face-alignment/models/point_masker_concat/2018-03-06:20:17/model.h5"
    #model = factory.getPointMaskerSmall()
    #model = factory.getPointMaskerConcat()
    if train:
        model = factory.getPointMaskerConcatCascaded()
    else:
        model = factory.getSaved(model_path)
    #model = factory.getPointMaskerDilated()
    #model = factory.getPointMaskerAttention()
    #model = factory.getPointMaskerDilated()
    #path = 'models/point_masker_attention/2018-03-06:00:14/model.h5'
    #model = factory.getSaved('models/tmp/point_masker_shallow.h5')
    model.summary()

    #train_batch_generator = BatchGenerator.BboxBatchGenerator('data/train_ibug')
    #train_batch_generator = BatchGenerator.PointMaskBatchGenerator('data/train_ibug', factory.mask_side_len, val_split_perc=0.2)
    train_path = 'data/train_ibug' if ibug_version else 'data/train'
    val_path = 'data/test_ibug' if ibug_version else 'data/train'

    if ibug_version:
        with open(train_path + '/names.json') as fp:
            all_train_names = json.load(fp)
        with open(val_path + '/names.json') as fp:
            all_val_names = json.load(fp)
    else:
        with open(train_path + '/names.json') as fp:
            all_names = json.load(fp)
        val_split_ratio = 0.3
        num_val_samples = int(val_split_ratio * len(all_names))
        val_indices = np.random.randint(0, len(all_names), num_val_samples)
        all_val_names = [all_names[ind] for ind in val_indices]
        all_train_names = list(set(all_names) - set(all_val_names))
    
    """
    val_split_ratio = 0.2
    split_val = int(len(all_names) * val_split_ratio)
    all_val_names = all_names[:split_val]
    all_train_names = all_names[split_val:]
    """

    """
    train_batch_generator = BatchGenerator.PointMaskBatchGenerator(all_train_names, train_path, factory.mask_side_len, 
                                                                   augment_on_generate=True, ibug_version=ibug_version)
    val_batch_generator = BatchGenerator.PointMaskBatchGenerator(all_val_names, val_path, factory.mask_side_len, 
                                                                   augment_on_generate=False, ibug_version=ibug_version)
    """

    train_batch_generator = BatchGenerator.PointMaskCascadedBatchGenerator(all_train_names, train_path, factory.mask_side_len,
                                                                   factory.im_width, augment_on_generate=True, ibug_version=ibug_version)
    val_batch_generator = BatchGenerator.PointMaskCascadedBatchGenerator(all_val_names, val_path, factory.mask_side_len, 
                                                                   factory.im_width, augment_on_generate=False, ibug_version=ibug_version)
    #train_batch_generator = BatchGenerator.LineMaskBatchGenerator('data/train_ibug', 224)#factory.mask_side_len)

    """
    for sample in samples:
        inputs, _ = train_batch_generator.getPair(sample)
        im = inputs[0] #im = cv2.resize(inputs[0], (112, 112), interpolation=cv2.INTER_AREA)
        plt.imshow(im)
        plt.show()
        for i in range(1, len(inputs)):
            mask = inputs[i]
            plt.imshow(mask[:,:,0])
            plt.show()
            #helenUtils.visualizeMask(im, mask)
    """

    """inputs, _ = train_batch_generator.getPair(samples[0], flip_x=True)
    im = inputs[0]
    coords_masks = inputs[1]
    print np.sum(coords_masks[:,:,0])
    plt.imshow(im)
    plt.show()
    for i in range(12):
        coord_mask = 80.0 * coords_masks[:,:,i]
        coord_mask = cv2.resize(coord_mask, (224, 224), interpolation=cv2.INTER_LINEAR)
        helenUtils.visualizeMask(im, coord_mask)
    """
    


    #train_batch_generator = BatchGenerator.MaskAndBboxBatchGenerator('data/train_ibug', factory.mask_side_len)
    #train_batch_generator = BatchGenerator.PointsBatchGenerator('data/train_ibug')
    #test_batch_generator = BatchGenerator.MaskBatchGenerator('data/test', factory.coords_sparsity, read_all=True)
    #batch_generator = BatchGenerator.HeatmapBatchGenerator('data/train', factory.heatmap_side_len)
    
    if not train:
        #val_batch_generator = BatchGenerator.PointsBatchGenerator(all_val_names, val_path)
        #modelTests.testNormalizedDistanceError(model, val_batch_generator)
        modelTests.videoTest(model)
        #modelTests.tryPointMaskerDilatedOnSamples(model)
        #modelTests.tryPointMaskerCascadedOnSamples(model)
        #modelTests.tryPointMaskerVanilla(model, train_batch_generator)
        #trySavedFullyConnected(model, train_batch_generator)
        #tryLipMasker(model, train_batch_generator)
        #tryLipMaskerZoomed(model, train_batch_generator, samples)

    if train:
        epochs = 120
        tb_log_dir = model_folder + '/tensorboard/'
        tb_callback = keras.callbacks.TensorBoard(log_dir=tb_log_dir, 
                                                    histogram_freq=0,
                                                    write_grads=True,
                                                    write_graph=True, 
                                                    write_images=True)
        lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto', 
                                                    epsilon=0.0001, cooldown=0, min_lr=0)
        cp_callback = keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, 
                                                    save_weights_only=False, mode='auto', period=5)
        model.fit_generator(generator=train_batch_generator.generate(),
                            validation_data=val_batch_generator.getAllData(),  
                            steps_per_epoch=train_batch_generator.steps_per_epoch,
                            epochs=epochs,
                            callbacks=[tb_callback, cp_callback, lr_callback])
        print 'Finished training for: ' + str(epochs) + ' epochs, everything was saved to: ' + model_path
        
        #model.save('models/tmp/lip_fc.h5')
        #model.save('models/tmp/lip_masker_100.h5')
        #model.save('models/tmp/lip_masker_skip_100.h5')
        if notify_training_complete:
            from google.cloud import error_reporting
            client = error_reporting.Client()
            client.report('Training complete!')
    
    """
    for fname in os.listdir('downloads/samples'):
        im = scipy.misc.imread('downloads/samples/' + fname)
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC)

        # alpha channel needs to be cutoff
        if im.shape[2] > 3:
            im = im[:,:,:3]

        pred = np.squeeze(model.predict(np.array([im]), batch_size=1))
        expanded = utils.getExpandedBbox(pred, 0.7, 0.7)
        utils.visualizeBboxes(im, [224 * expanded])
    """

    #print getAvgTestError(model, test_batch_generator)
    #videoTestBboxModel(model)
    #videoTestBboxHaarCascade()
    #visualize_samples()