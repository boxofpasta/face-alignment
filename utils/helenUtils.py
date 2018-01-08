from __future__ import print_function
import os
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import generalUtils as utils
import json
import sys
import cv2
from matplotlib.patches import Circle


class DatasetProps:

    def __init__(self, im_extension, label_extension, im_path, label_path):
        """
        Parameters
        ----------
        im_extension: 
            image extension, e.g '.png'
        im_path: 
            path to folder with images
        """
        self.im_extension = im_extension
        self.label_extension = label_extension
        self.im_path = im_path
        self.label_path = label_path

def getNumCoords(coords_sparsity):
    return int(np.ceil(194.0 / coords_sparsity))

def getAllData(path):
    """
    Assumes that path/ims, path/labels folders exist (and contain .npy files).
    Assumes that path/names.json exists with names of all examples to run tests over.
    See functions below for implementations that serialize in this format.

    Returns
    -------
    (all_ims, all_labels) 
    """
    with open(path + '/names.json') as fp:
        names_set = set(json.load(fp))
    
    all_ims, all_labels = [], []
    for name in names_set:
        im = np.load(path + '/ims/' + name + '.npy')
        label = np.load(path + '/labels/' + name + '.npy')
        all_ims.append(im)
        all_labels.append(label)
    return (all_ims, all_labels)


def processData(props, targ_im_len, sample_names=None):
    """
    Parameters
    ----------
    props: 
        instance of DatasetProps. props.im_paths specifies the folders to read from.
    targ_im_len: 
        the target image width and height. If -1, won't resize or warp.
    sample_names: 
        specific samples to read from.

    Returns
    -------
    (ims, labels)
    """

    # train
    ims = read_images(props.im_path, props.im_extension, sample_names=sample_names)
    labels = readLabels(props.label_path, props.label_extension, sample_names=sample_names)

    if targ_im_len != -1:
        print('\n\nResizing samples ...')
        iter = 0
        for name in ims:
            if targ_im_len != -1:
                im, label = cropPair(ims[name], labels[name])
                im, label = resizePair(im, label, targ_im_len, targ_im_len)
                labels[name] = normalizeCoords(label, targ_im_len, targ_im_len)
                ims[name] = im
            utils.informProgress(iter, len(ims))
            iter += 1
        utils.informProgress(1,1)
    else:
        print('\n\nNo target dimension provided, not resizing.')

    return ims, labels
    #""" data centering """
    #all_ims = np.array(all_ims)
    #mean_im = np.average(all_ims, axis=0)
    #std_im = np.average(np.abs(all_ims - mean_im), axis=0)


def serializeData(ims, labels, npy_path):
    print('\n\nNormalizing data and serializing to disk ...')
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    if not os.path.exists(npy_path + '/ims'):
        os.makedirs(npy_path + '/ims')
    if not os.path.exists(npy_path + '/labels'):
        os.makedirs(npy_path + '/labels')
    try:
        # avoid overwriting data in json
        with open(npy_path + '/names.json') as fp:
            names_set = set(json.load(fp))
    except IOError or ValueError:
        names_set = set()
    iter = 0
    for name in ims:
        names_set.add(name)
        im = ims[name]
        np.save(npy_path + '/ims/' + name + '.npy', im)
        np.save(npy_path + '/labels/' + name + '.npy', labels[name])
        utils.informProgress(iter, len(ims))
        iter += 1

    utils.informProgress(1,1)
    print('\n')
    with open(npy_path + '/names.json', 'wb') as fp:
        json.dump(list(names_set), fp)


def normalizeCoords(coords, im_width, im_height):
    for coord in coords:
        coord[0] /= im_height
        coord[1] /= im_width
    return coords


def resizePair(im, label, targ_width, targ_height):
    cur_height = len(im)
    cur_width = len(im[0])
    scale_x = float(targ_width) / cur_width
    scale_y = float(targ_height) / cur_height
    for coords in label:
        coords[0] = (coords[0] - 1) * scale_x
        coords[1] = (coords[1] - 1) * scale_y
    resized = cv2.resize(im, (targ_height, targ_width), interpolation=cv2.INTER_CUBIC)
    return [resized, label]

def cropPair(im, label):
    label = np.reshape(label, (-1, 2))
    bbox = utils.getBbox(label)
    bbox = utils.getRandomlyExpandedBbox(bbox, 0.1, 0.3)
    label[:,0] -= bbox[0]
    label[:,1] -= bbox[1]
    l = int(max(0, bbox[0]))
    r = int(min(len(im[0]), bbox[2]+1))
    t = int(max(0, bbox[1]))
    b = int(min(len(im), bbox[3]+1))
    im = im[t:b, l:r]
    return [im, label]

def getOrdered(ims, labels):
    """
    Parameters
    ----------
    ims: 
        a dictionary of images
    labels:
        a dictionary of labels

    Returns
    -------
    [ims_arr, labels_arr], where ims_arr[i] corresponds to labels_arr[i]
    """
    ims_ordered, labels_ordered = [], []
    for name in ims:
        labels_ordered.append(labels[name])
        ims_ordered.append(ims[name])
    return [ims_ordered, labels_ordered]


def read_images(path, extension, sample_names=None):
    """
    Parameters
    ----------
    path: 
        path to folder
    extension: 
        file extension, e.g '.png'
    sample_names: 
        the samples in the folder to read from. If None, will read all of them.

    Returns
    -------
        dictionary of images, with key being filename without extension
    """

    print('Reading images ...')
    ims = {}

    if sample_names == None:
        sample_names = []
        for fname in os.listdir(path):
            if fname.endswith(extension):
                sample_names.append(fname[:-len(extension)])

    for i in range(0, len(sample_names)):
        ims[sample_names[i]] = scipy.misc.imread(path + '/' + sample_names[i] + extension)
        utils.informProgress(i, len(sample_names))
    utils.informProgress(1, 1)
    return ims


def readLabels(path, extension, sample_names):
    """
    Parameters
    ----------
    path: 
        path to folder
    extension: 
        file extension, e.g '.png'
    sample_names: 
        the samples in the folder to read from. If None, will read all of them.

    Returns
    -------
        dictionary of labels, with key being filename without extension
    """
    labels = {}

    if sample_names != None:
        allowed_samples = set(sample_names)

    for fname in os.listdir(path):
        if fname.endswith(extension):
            lines = open(path + '/' + fname).readlines()
            key = lines[0].strip()
            if sample_names == None or key in allowed_samples:
                cur_labels = []
                for i in range(1, len(lines)):
                    coords = []
                    for s in lines[i].split():
                        if utils.isNumber(s):
                            coords.append(float(s))
                    if len(coords) == 2:
                        cur_labels.append(coords)
                labels[key] = cur_labels
    return labels
