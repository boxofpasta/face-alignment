import sys
train = bool(int(sys.argv[1]))

import os
import time
import scipy.misc
import numpy as np
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

if __name__ == '__main__':
    #sess = K.get_session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #K.set_session(sess)
    notify_training_complete = True
    samples = ['100466187_1', '13602254_1', '2908549_1', '100032540_1', '1691766_1', '11564757_2', '110886318_1']
    
    factory = ModelFactory.ModelFactory()
    #model = factory.getSaved('models/tmp/test.h5')
    #model = factory.getSaved('models/lip_masker_100.h5')
    #model = factory.getLipMasker(alpha_1=1, alpha_2=1)
    #model = factory.getSaved('models/lip_fc_100.h5')
    #model = factory.getFullyConnected(alpha=1.0)
    #model.summary()
    #plot_model(model, to_file='models/lip_masker_skip_100.jpg')

    #model = factory.getSaved('models/lip_masker_rand_bbox_100.h5')
    #model = factory.getSaved('models/lip_masker_rand_bbox_fpn_100.h5')
    #model = factory.getLipMasker()
    #model_path = 'models/tmp/point_masker_small.h5'
    #model_path = 'models/tmp/point_masker_vanilla_no_skip.h5'
    model_name = 'point_masker_dilated'
    model_folder = 'models/' + model_name
    model_path = model_folder + '/model.h5'
    #model = factory.getPointMaskerSmall()
    #model = factory.getPointMaskerVanilla()
    #model = factory.getPointMaskerDilated()
    model = factory.getSaved(model_path)
    #model = factory.getSaved('models/tmp/point_masker_shallow.h5')
    #model = factory.getSaved(model_path)
    model.summary()
    #model = factory.getBboxRegressor()
    #model = factory.getFullyConnected(alpha=0.5)
    #model = factory.getBboxRegressor()
    #model = factory.getSaved('models/lip_masker_zoomed_100.h5')
    #model = factory.getSaved('models/lip_fc.h5')
    #model = factory.getSaved('models/lip_masker_zoomed_100.h5')
    #model = factory.getSaved('models/lip_masker_sep_100.h5')
    #model = factory.getSaved('models/lip_masker_050.h5')
    #train_batch_generator = BatchGenerator.BboxBatchGenerator('data/train_ibug')
    #train_batch_generator = BatchGenerator.PointMaskBatchGenerator('data/train_ibug', factory.mask_side_len, val_split_perc=0.2)
    path = 'data/train_ibug'
    with open(path + '/names.json') as fp:
        all_names = json.load(fp)

    val_split_ratio = 0.2
    split_val = int(len(all_names) * val_split_ratio)
    all_val_names = all_names[:split_val]
    all_train_names = all_names[split_val:]
    train_batch_generator = BatchGenerator.PointMaskVanillaBatchGenerator(all_train_names, path, factory.mask_side_len, 
                                                                          flip_x_augmentation=True)
    val_batch_generator = BatchGenerator.PointMaskVanillaBatchGenerator(all_val_names, path, factory.mask_side_len)
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
        val_batch_generator = BatchGenerator.PointsBatchGenerator(all_val_names, path)
        print modelTests.getNormalizedDistanceError(model, val_batch_generator)
        #modelTests.videoTest(model)
        #modelTests.tryPointMaskerDilatedOnSamples(model)
        #modelTests.tryPointMaskerVanilla(model, train_batch_generator)
        #trySavedFullyConnected(model, train_batch_generator)
        #tryLipMasker(model, train_batch_generator)
        #tryLipMaskerZoomed(model, train_batch_generator, samples)

    if train:
        total_epochs = 120
        epochs_before_saving = 30
        prev_epochs = 0
        cur_epoch = 0
        while cur_epoch < total_epochs:
            cur_num_epochs = min(total_epochs - cur_epoch, epochs_before_saving)
            tb_log_dir = model_folder + '/tensorboard/'
            tb_callback = keras.callbacks.TensorBoard(log_dir=tb_log_dir, 
                                                      histogram_freq=0, #cur_num_epochs, 
                                                      write_graph=True, 
                                                      write_images=True)
            model.fit_generator(generator=train_batch_generator.generate(),
                                validation_data=val_batch_generator.getAllData(),  
                                steps_per_epoch=train_batch_generator.steps_per_epoch,
                                epochs=cur_num_epochs,
                                callbacks=[tb_callback])
            model.save(model_path)
            cur_epoch += cur_num_epochs
            print 'Finished training for: ' + str(cur_epoch) + ' epochs, saving to: ' + model_path
        
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