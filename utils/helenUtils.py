from __future__ import print_function
import os
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import generalUtils as utils
import json
import sys
import cv2
from PIL import Image
from matplotlib.patches import Circle
from matplotlib.patches import Polygon


class DatasetProps:

    def __init__(self, im_extension, coords_extension, im_path, coords_path):
        """
        Parameters
        ----------
        im_extension: 
            image extension, e.g '.png'
        im_path: 
            path to folder with images
        """
        self.im_extension = im_extension
        self.coords_extension = coords_extension
        self.im_path = im_path
        self.coords_path = coords_path

def getNumCoords(coords_sparsity):
    return int(np.ceil(194.0 / coords_sparsity))

def getAllData(path):
    """
    Assumes that path/ims, path/coords folders exist (and contain .npy files).
    Assumes that path/names.json exists with names of all examples to run tests over.
    See functions below for implementations that serialize in this format.

    Returns
    -------
    (all_ims, all_coords) 
    """
    with open(path + '/names.json') as fp:
        names_set = set(json.load(fp))
    
    all_ims, all_coords = [], []
    for name in names_set:
        im = np.load(path + '/ims/' + name + '.npy')
        coords = np.load(path + '/coords/' + name + '.npy')
        mask = np.load(path + '/masks/' + name + '.npy')
        all_ims.append(im)
        all_coords.append(label)

    return (all_ims, all_coords)


def processData(props, targ_im_len, sample_names=None, ibug_version=False):
    """
    Parameters
    ----------
    props: 
        Instance of DatasetProps. props.im_paths specifies the folders to read from.
    targ_im_len: 
        The target image width and height. If -1, won't resize or warp.
    sample_names: 
        Specific samples to read from.
    ibug_version: 
        Set to true if using data from https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/.
        otherwise the data should be from http://www.ifp.illinois.edu/~vuongle2/helen/.

    Returns
    -------
    (ims, labels)
    """

    all_ims = readImagesHelen(props.im_path, props.im_extension, sample_names=sample_names)
    all_coords = readCoordsHelen(props.coords_path, props.coords_extension, sample_names=sample_names, ibug_version=ibug_version)

    if targ_im_len != -1:
        print('\n\nResizing samples ...')
        iter = 0
        for name in all_ims:
            if targ_im_len != -1:
                im, coords = cropPair(all_ims[name], all_coords[name])
                im, coords = resizePair(im, coords, targ_im_len, targ_im_len)
                all_coords[name] = normalizeCoords(coords, targ_im_len, targ_im_len)
                all_ims[name] = im
            utils.informProgress(iter, len(all_ims))
            iter += 1
        utils.informProgress(1,1)
    else:
        print('\n\nNo target dimension provided, not resizing.')

    return all_ims, all_coords
    #""" data centering """
    #all_ims = np.array(all_ims)
    #mean_im = np.average(all_ims, axis=0)
    #std_im = np.average(np.abs(all_ims - mean_im), axis=0)


def serializeData(all_ims, all_coords, npy_path, ibug_version=False):
    print('\n\nNormalizing data and serializing to disk ...')
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    if not os.path.exists(npy_path + '/ims'):
        os.makedirs(npy_path + '/ims')
    if not os.path.exists(npy_path + '/coords'):
        os.makedirs(npy_path + '/coords')
    try:
        # avoid overwriting data in json
        with open(npy_path + '/names.json') as fp:
            names_set = set(json.load(fp))
    except IOError or ValueError:
        names_set = set()
    iter = 0
    for name in all_ims:
        names_set.add(name)
        im = all_ims[name]
        coords = all_coords[name]
        np.save(npy_path + '/ims/' + name + '.npy', im)
        np.save(npy_path + '/coords/' + name + '.npy', coords)
        utils.informProgress(iter, len(all_ims))
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

#def denormalizeCoords(coords, im_width, im_height):   

def resizePair(im, label, targ_width, targ_height):
    cur_height = len(im)
    cur_width = len(im[0])
    scale_x = float(targ_width) / cur_width
    scale_y = float(targ_height) / cur_height
    for coords in label:
        coords[0] = (coords[0] - 1) * scale_x
        coords[1] = (coords[1] - 1) * scale_y
    resized = cv2.resize(im, (targ_height, targ_width), interpolation=cv2.INTER_AREA)
    return [resized, label]

def cropPair(im, label):
    label = np.reshape(label, (-1, 2))
    bbox = utils.getBbox(label)
    bbox = utils.getRandomlyExpandedBbox(bbox, 0.03, 0.35)
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


def readImagesHelen(path, extension, sample_names=None):
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

def readCoordsHelen(path, extension, sample_names, ibug_version=False):
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
            if not ibug_version:
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
            else:
                key = fname[:-len(extension)]
                if sample_names == None or key in allowed_samples:
                    cur_labels = []
                    for i in range(3, len(lines)-1):
                        coords = [float(s) for s in lines[i].split()]
                        cur_labels.append(coords)
                    labels[key] = cur_labels
    return labels

def getLipCoords(coords):
    return coords[48:60]