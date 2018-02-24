import utils.helenUtils as helenUtils
import utils.generalUtils as utils
import scipy.misc
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
#import dlib

def normalizeMask(mask):
    mask = np.maximum(mask - np.mean(mask), 0)
    return mask / np.max(mask)

def tryPointMaskerVanilla(model, batch_generator, sample_names=None):
    if sample_names == None:
        sample_names = batch_generator.all_names

    for sample_name in sample_names:
        inputs, _ = batch_generator.getPair(sample_name)
        X, Y = batch_generator.getBatchFromNames([sample_name])
        im = inputs[0]
        print_str = 'sample name: ' + sample_name 
        if batch_generator.isInValSet(sample_name):
            print_str += ', from val set'
        else:
            print_str += ', from train set'

        outputs = model.predict_on_batch(X)
        preds = np.squeeze(outputs[1])
        preds = utils.imSoftmax(preds)
        preds = np.moveaxis(preds, -1, 0)
        preds = [ normalizeMask(pred) for pred in preds]
        
        coords = utils.getCoordsFromPointMasks(preds, len(im[0]), len(im))
        utils.visualizeCoords(im, coords)
        print print_str
        
        """
        summed = 80 * np.sum(preds, axis=0)
        summed = cv2.resize(summed, (len(im), len(im[0])), interpolation=cv2.INTER_LINEAR)
        helenUtils.visualizeMask(im, summed)
        """

        """for i in range(3):
            pred = preds[i]
            label = labels[i]
            l = len(pred)
            c = np.zeros((l, l, 3))
            pred = np.maximum(pred - np.mean(pred), 0)
            c[:,:,0] = pred / np.max(pred)
            c[:,:,2] = label
            plt.imshow(c)
            plt.show()
            #plt.show()
            #plt.imshow(label)
            #plt.show()
        """

        """
        z2 = 1.0 / (1.0 + np.exp(-np.squeeze(s[3][0])))
        pred_coord_masks = s[-1][0]

        for i in range(68):
            d = np.zeros((56, 56, 3))
            d[:,:,0] = cv2.resize(coord_masks[:,:,i], (56, 56), interpolation=cv2.INTER_AREA)
            d[:,:,1] = pred_coord_masks[:,:,i]
            plt.imshow(d)
            plt.show()

        summed = np.squeeze(summed)
        
        plt.imshow(im)
        plt.show()

        c = np.zeros((112, 112, 3))
        c[:,:,0] = z2
        c[:,:,1] = summed
        plt.imshow(c)
        plt.show()
        """

def getNormalizedDistanceError(model, batch_generator):
    """
    Parameters
    ----------
    batch_generator: 
        Should be class PointsBatchGenerator.
    """
    X, Y = batch_generator.getAllData()
    X, Y = X[0], Y[0]
    overall_avg = 0.0

    for i in range(len(X)):
        im = X[i]
        leye_coords = helenUtils.getLeyeCenter(Y[i])
        reye_coords = helenUtils.getReyeCenter(Y[i])
        eye_dist = helenUtils.getEyeDistance(Y[i])
        lip_labels = helenUtils.getLipCoords(Y[i])
        lip_preds = np.array(getCoordsFromImage(model, im))
        lip_preds[:,0] /= float(len(im))
        lip_preds[:,1] /= float(len(im[0]))
        cur_avg = 0.0
        for j in range(0, len(lip_preds)):
            dist = np.linalg.norm(lip_preds[j] - (lip_labels[j] - 1.0 / len(im)))
            normalized = dist / eye_dist
            cur_avg += normalized
        cur_avg /= len(lip_preds)
        print 'eye to eye distance: ' + str(eye_dist)
        print 'avg error across all points: ' + str(cur_avg)
        all_coords = np.concatenate([len(im) * np.array(lip_preds), len(im) * np.array(lip_labels) - 1], axis=0)
        all_coords = np.concatenate([all_coords, [len(im) * leye_coords], [len(im) * reye_coords]], axis=0)
        pred_indices = np.arange(0, len(all_coords) / 2)
        utils.visualizeCoords(im, all_coords, pred_indices)
        overall_avg += cur_avg
    
    overall_avg /= len(X)
    return overall_avg


def getCoordsFromImage(model, im):
    im = im[:,:,0:3]
    im = cv2.resize(im, (224, 224))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    X = [np.array([im])]

    #_1, _2, _3, _4, low_res, high_res = model.predict(feed_inputs, batch_size=32)
    before = time.time()
    outputs = model.predict_on_batch(X)
    print 'inference time: ' + str(time.time() - before) + ' seconds'
    
    #labels = inputs[1]
    #labels = np.moveaxis(labels, -1, 0)
    preds = np.squeeze(outputs[0])
    preds = utils.imSoftmax(preds)
    preds = np.moveaxis(preds, -1, 0)
    preds = [ normalizeMask(pred) for pred in preds]
    
    coords = utils.getCoordsFromPointMasks(preds, len(im[0]), len(im))
    return coords

def tryPointMaskerDilatedOnSamples(model):
    png_ims = set([0, 2, 4, 8, 9])
    for i in range(0, 10):
        if i in png_ims:
            im = cv2.imread('downloads/samples/' + str(i) + '.png')
        else:
            im = cv2.imread('downloads/samples/' + str(i) + '.jpg')
        coords = getCoordsFromImage(model, im)
        utils.visualizeCoords(im, coords)

def tryPointMasker(model, batch_generator, sample_names=None):
    #if sample_names == None:
    sample_names = batch_generator.all_names

    for sample_name in sample_names:
        inputs, _ = batch_generator.getPair(sample_name)
        X, Y = batch_generator.getBatchFromNames([sample_name])
        im = inputs[0]
        print 'sample name: ' + sample_name

        #_1, _2, _3, _4, low_res, high_res = model.predict(feed_inputs, batch_size=32)
        outputs = model.predict_on_batch(X)
        plt.imshow(im)
        plt.show()
        
        labels = inputs[1]
        labels = np.moveaxis(labels, -1, 0)
        preds = np.squeeze(outputs[5])
        preds = np.moveaxis(preds, -1, 0)
        for i in range(3):
            pred = preds[i]
            label = labels[i]
            l = len(pred)
            c = np.zeros((l, l, 3))
            c[:,:,0] = pred
            c[:,:,1] = label
            plt.imshow(c)
            plt.show()
            #plt.show()
            #plt.imshow(label)
            #plt.show()

        """
        z2 = 1.0 / (1.0 + np.exp(-np.squeeze(s[3][0])))
        pred_coord_masks = s[-1][0]

        for i in range(68):
            d = np.zeros((56, 56, 3))
            d[:,:,0] = cv2.resize(coord_masks[:,:,i], (56, 56), interpolation=cv2.INTER_AREA)
            d[:,:,1] = pred_coord_masks[:,:,i]
            plt.imshow(d)
            plt.show()

        summed = np.squeeze(summed)
        
        plt.imshow(im)
        plt.show()

        c = np.zeros((112, 112, 3))
        c[:,:,0] = z2
        c[:,:,1] = summed
        plt.imshow(c)
        plt.show()
        """


def tryLipMasker(model, batch_generator, sample_names=None):
    if sample_names == None:
        sample_names = batch_generator.all_names

    for sample_name in sample_names:
        inputs, outputs = batch_generator.getPair(sample_name)
        mask_gt = inputs[2]
        feed_inputs = utils.transposeList([inputs])
        feed_inputs = [np.array(cur_input) for cur_input in feed_inputs]
        mask_loss, bbox_loss, bbox_coords, masks = model.predict(feed_inputs, batch_size=1)
        
        c = np.zeros((56, 56, 3))
        mask_gt_cropped = helenUtils.getCropped(mask_gt, 224 * bbox_coords[0])
        mask_gt_cropped = cv2.resize(mask_gt_cropped, (56, 56), interpolation=cv2.INTER_AREA)
        mask_pred = masks[0][:,:,0]
        mask_pred = 1.0 / (1.0 + np.exp(-mask_pred+0.5))
        b = bbox_coords[0]
        c[:,:,0] = mask_pred
        c[:,:,1] = mask_gt_cropped
        plt.imshow(c)
        plt.show()

        im = inputs[0]
        ax = plt.axes()
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        #print im.shape
        #print mask_pred.shape
        ax.imshow(im, extent=[0, 1, 0, 1])
        ax.imshow(50 * mask_pred, alpha=0.5, origin='upper', extent=[b[1], b[3], 1.0 - b[2], 1.0 - b[0]])
        plt.show()

        #helenUtils.visualizeMask(im, masks[0], 224)
        #helenUtils.visualizeMask(im, masks[0], 56)
    
        #helenUtils.visualizeMask(im, mask_gts[0], 224)
        #helenUtils.visualizeMask(im, mask_gts[0], 28)


def trySavedFullyConnected(model, batch_generator, sample_names=None):
    if sample_names == None:
        sample_names = batch_generator.all_names

    for sample_name in sample_names:
        inputs, outputs = batch_generator.getPair(sample_name)

        # alpha channel needs to be cutoff
        #if im.shape[2] > 3:
        #   im = im[:,:,:3]
        im = inputs[0]
        labels = np.array(model.predict(np.array(inputs), batch_size=1))
        label = labels[0]
        label *= len(im)
        utils.visualizeCoords(im, label)

def tryLipMaskerZoomed(model, batch_generator, sample_names):
    for sample_name in sample_names:
        inputs, outputs = batch_generator.getPair(sample_name)
        mask_gt = inputs[1]
        print mask_gt.shape
        mask_loss, masks = model.predict([inputs], batch_size=1)
        c = np.zeros((56, 56, 3))
        mask_gt = cv2.resize(mask_gt, (56, 56), interpolation=cv2.INTER_AREA)
        c[:,:,0] = masks[0][:,:,0]
        c[:,:,1] = mask_gt
        plt.imshow(c)
        plt.show()

"""
Video tests
"""
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

def videoTest(model):
    cap = cv2.VideoCapture(0)
    num_frames = 0
    start = time.clock()
    im_len = 640
    face_cascade = cv2.CascadeClassifier('downloads/haarcascades/haarcascade_frontalface_default.xml')
    predictor = dlib.shape_predictor('downloads/dlib/shape_predictor_68_face_landmarks.dat')
    
    #eye_cascade = cv2.CascadeClassifier('downloads/haarcascades/haarcascade_eye.xml')
    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_CUBIC)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,
                                              scaleFactor=1.05,  
                                              minNeighbors=5,  
                                              minSize=(100, 100),  
                                              )
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            preds = getCoordsFromImage(model, roi_color)
            for coord in preds:
                final_coord = (x + int(round(coord[1])), y + int(round(coord[0])))
                cv2.circle(frame, final_coord, 1, (0,255,0), thickness=1, lineType=8, shift=0)
        """

        #frame = frame[100:300, 140:540]
        #frame = frame[100:200, 100:200]
        #dlib_rect = dlib.rectangle(140, 100, len(frame[0]), len(frame)) 
        dlib_rect = dlib.rectangle(140, 0, 540, len(frame)) 
        detected_landmarks = predictor(frame, dlib_rect).parts()  
        landmarks = np.array([[p.x, p.y] for p in detected_landmarks])   
        
        min_x = np.min(landmarks[:,0])
        min_y = np.min(landmarks[:,1])
        max_x = np.max(landmarks[:,0])
        max_y = np.max(landmarks[:,1])
        cv2.rectangle(frame,(min_x,min_y),(max_x,max_y),(255,0,0),2)

        y, x = min_y, min_x
        h = max_y - min_y
        w = max_x - min_x
        print 'width and height: ' + str(w) + str(h)
        if w <= 0 or h <= 0 or x < 0 or x > len(frame[0]) or y < 0 or y > len(frame):
            continue

        for coord in landmarks:
            cv2.circle(frame, (int(coord[0]), int(coord[1])), 1, (0, 0, 255), thickness=1, lineType=8, shift=0)

        roi_color = frame[y:y+h, x:x+w]
        preds = getCoordsFromImage(model, roi_color)
        for coord in preds:
            final_coord = (x + int(round(w / 224.0 * coord[1])), y + int(h / 224.0 * round(coord[0])))
            cv2.circle(frame, final_coord, 1, (0,255,0), thickness=1, lineType=8, shift=0)

            #eyes = eye_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in eyes:
            #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

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