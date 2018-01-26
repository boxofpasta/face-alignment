import os
import time
import scipy.misc
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
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
import keras.backend as K
#from keras.utils import plot_model
from keras.models import load_model
from keras.applications import mobilenet
from tensorflow.python import debug as tf_debug


def trySavedFullyConnected(model, batch_generator, sample_names):
    for sample_name in sample_names:
        im, label = batch_generator.getPair(sample_name)

        # alpha channel needs to be cutoff
        #if im.shape[2] > 3:
        #   im = im[:,:,:3]
        labels = np.array(model.predict(np.array([im]), batch_size=1))
        label = labels[0]
        label *= len(im)
        utils.visualizeCoords(im, label)

def tryLipMasker(model, batch_generator, sample_names):
    for sample_name in sample_names:
        im, labels = batch_generator.getPair(sample_name)
        labels = [np.array([label]) for label in labels]
        mask_gt = labels[1][0]
        bbox_gt = labels[0][0]
        #labes[1][0] = np.expand_dims(labels[1][0], axis=-1)
        x = [np.array([im])] + labels
        mask_loss, bbox_loss, bbox_coords, masks = model.predict(x, batch_size=1)
        #masks[0] = np.squeeze(masks[0])
        #masks[0] = np.squeeze(masks[0])
        #plt.imshow(masks[0])
        #plt.show()
        #utils.visualizeBboxes(im, bbox_coords)
        #masks[0] = utils.sigmoid(masks[0])
        
        c = np.zeros((56, 56, 3))
        mask_gt = helenUtils.getCropped(mask_gt, 224 * bbox_gt)
        mask_gt = cv2.resize(mask_gt, (56, 56), interpolation=cv2.INTER_AREA)
        c[:,:,0] = masks[0][:,:,0]
        c[:,:,1] = mask_gt
        plt.imshow(c)
        plt.show()
        #helenUtils.visualizeMask(im, masks[0], 224)
        #helenUtils.visualizeMask(im, masks[0], 56)
    
        #helenUtils.visualizeMask(im, mask_gts[0], 224)
        #helenUtils.visualizeMask(im, mask_gts[0], 28)

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

# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
def videoTestBboxModel(model):
    cap = cv2.VideoCapture(0)
    num_frames = 0
    start = time.clock()
    im_len = 224
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (im_len, im_len), interpolation=cv2.INTER_CUBIC)
        pred = im_len * np.squeeze(model.predict(np.array([frame]), batch_size=1))
        cv2.rectangle(frame, (pred[0],pred[1]), (pred[2],pred[3]), (0,0,255))

        # Display the resulting frame
        cv2.imshow('frame',frame)
        num_frames += 1
        if num_frames % 50 == 0:
            print 'Fps : ' + str(num_frames / (time.clock() - start))
            num_frames = 0
            start = time.clock()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def videoTestBboxHaarCascade():
    cap = cv2.VideoCapture(0)
    num_frames = 0
    start = time.clock()
    im_len = 224
    face_cascade = cv2.CascadeClassifier('downloads/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('downloads/haarcascades/haarcascade_eye.xml')
    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (im_len, im_len), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        num_frames += 1
        if num_frames % 50 == 0:
            print 'Fps : ' + str(num_frames / (time.clock() - start))
            num_frames = 0
            start = time.clock()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #sess = K.get_session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #K.set_session(sess)
    train = True
    notify_training_complete = True
    samples = ['100466187_1', '13602254_1', '2908549_1', '100032540_1', '1691766_1', '11564757_2', '110886318_1']
    
    #helenUtils.trySerializedFolder('data/train_ibug')
    #visualizeSamples('data/test')
    #model = get_saved_model('models/fully_connected_v2.h5')
    #print get_avg_test_error(model, 'data/test')
    #try_saved_model('models/fully_connected_v1.h5')
    #model = get_saved_model('models/tmp/fully_conv.h5')
    
    factory = ModelFactory.ModelFactory()
    #model = factory.getLipMasker(alpha=0.5)
    model = factory.getBboxRegressor()
    #model = factory.getFullyConnected(alpha=0.5)
    #model = factory.getBboxRegressor()
    #model = factory.getSaved('models/fully_connected_v1.h5')
    #model = factory.getSaved('models/lip_fc.h5')
    #model = factory.getSaved('models/lip_masker_050.h5')
    train_batch_generator = BatchGenerator.BboxBatchGenerator('data/train_ibug')
    #train_batch_generator = BatchGenerator.MaskBatchGenerator('data/train_ibug', factory.mask_side_len)
    #train_batch_generator = BatchGenerator.BatchGenerator('data/train_ibug')
    #test_batch_generator = BatchGenerator.MaskBatchGenerator('data/test', factory.coords_sparsity, read_all=True)
    #batch_generator = BatchGenerator.HeatmapBatchGenerator('data/train', factory.heatmap_side_len)
    
    if not train:
        #trySavedFullyConnected(model, train_batch_generator, samples)
        tryLipMasker(model, train_batch_generator, samples)

    if train:
        model.fit_generator(generator=train_batch_generator.generate(),
                            steps_per_epoch=train_batch_generator.steps_per_epoch,
                            epochs=240)

        #model.save('models/tmp/lip_fc.h5')
        model.save('models/tmp/lip_masker_sep_050.h5')
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