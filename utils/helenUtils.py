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

def getNumCoords(coords_sparsity, ibug=True):
    num_points = 68 if ibug else 194
    return int(np.ceil(float(num_points) / coords_sparsity))

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

class ImagesReader:
    def __init__(self, path, extension, mem_limit=1000):
        """
        Parameters
        ----------
        mem_limit: 
            The maximum amount of images to read into RAM, in MB.
        """
        self.path = path
        self.extension = extension
        self.mem_limit = mem_limit
        self.all_names = getAllSampleNames(self.path, self.extension)
        self.cur = 0
        self.complete = False

    def reset(self):
        self.cur = 0
        self.complete = False

    def read(self, sample_names=None):
        """
        Reads images until either done or RAM limit exceeded (for a single invocation, not aware of others).
        
        Returns
        -------
        (ims, names). Each of them is a list.
        """
        cur_usage = 0
        ims = []
        names = []
        for i in range(self.cur, len(self.all_names)):
            cur_im = scipy.misc.imread(self.path + '/' + self.all_names[i] + self.extension)
            cur_usage += sys.getsizeof(cur_im)
            ims.append(cur_im)
            names.append(self.all_names[i])
            utils.informProgress(i, len(self.all_names))
            self.cur += 1
            if cur_usage >= self.mem_limit * 1024 * 1024:
                print('    RAM limit of ' + str(self.mem_limit) + 'MB has been exceeded for this invocation')
                break
        
        if self.cur >= len(self.all_names):
            self.complete = True
            utils.informProgress(1, 1)

        return ims, names



def processData(ims, coords, targ_im_len):
    """
    Parameters
    ----------
    Both ims and coords should be dicts. A key exists in ims iff it exists in coords.
    ims and coords will be modified by reference.

    """

    #ims = readImagesHelen(props.im_path, props.im_extension, sample_names=sample_names)
    #coords = readCoordsHelen(props.coords_path, props.coords_extension, sample_names=sample_names, ibug_version=ibug_version)

    if targ_im_len != -1:
        print('\n\nResizing samples ...')
        iter = 0
        for name in ims:
            if targ_im_len != -1:
                im, single_coords = cropPair(ims[name], coords[name])
                im, single_coords = resizePair(im, single_coords, targ_im_len, targ_im_len)
                coords[name] = normalizeCoords(single_coords, targ_im_len, targ_im_len)
                ims[name] = im
            utils.informProgress(iter, len(ims))
            iter += 1
        utils.informProgress(1,1)
    else:
        print('\n\nNo target dimension provided, not resizing.')

    return ims, coords
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
        coord[1] /= (im_height - 1)
        coord[0] /= (im_width - 1)
    return coords

#def denormalizeCoords(coords, im_width, im_height):   

def resizePair(im, label, targ_width, targ_height):
    cur_height = len(im)
    cur_width = len(im[0])
    scale_x = float(targ_width) / cur_width
    scale_y = float(targ_height) / cur_height
    for coords in label:
        coords[0] = coords[0] * scale_y
        coords[1] = coords[1] * scale_x
    resized = cv2.resize(im, (targ_height, targ_width), interpolation=cv2.INTER_AREA)
    return [resized, label]

def cropPair(im, label):
    label = np.reshape(label, (-1, 2))
    lip_coords = getLipCoords(label)
    #bbox = utils.getBbox(label)
    bbox = utils.getBbox(lip_coords)

    # randomly expand facebox
    #bbox = utils.getRandomlyExpandedBbox(bbox, 0.03, 0.35)
    bbox = utils.getRandomlyExpandedBbox(bbox, 0.10, 1.0)
    label[:,0] -= bbox[0]
    label[:,1] -= bbox[1]
    im = getCropped(im, bbox)
    return [im, label]

def getCropped(im, bbox):
    l = int(max(0, bbox[1]))
    r = int(min(len(im[0]), bbox[3]+1))
    t = int(max(0, bbox[0]))
    b = int(min(len(im), bbox[2]+1))
    return im[t:b, l:r]

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

def getAllSampleNames(path, extension):
    """
    Parameters
    ----------
    path: 
        Path to folder with all the images.
    
    Returns
    -------
        List of all sample names.
    """
    fnames_set = set()
    for fname in os.listdir(path):
        if fname.endswith(extension):
            fnames_set.add(fname[:-len(extension)])
    return list(fnames_set)


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

                            # want y, x because tensorflow likes it this way
                            cur_labels.append(list(reversed(coords)))
                    labels[key] = cur_labels
            else:
                key = fname[:-len(extension)]
                if sample_names == None or key in allowed_samples:
                    cur_labels = []
                    for i in range(3, len(lines)-1):
                        coords = [float(s) for s in lines[i].split()]

                        # want y, x because tensorflow likes it this way
                        cur_labels.append(list(reversed(coords)))
                    labels[key] = cur_labels
    return labels


def getLipCoords(coords):
    """
    Only works for the ibug annotated version currently.
    Parameters
    ----------
    coords: 
        Should have shape (num_coords, 2).
    """
    return coords[48:60]

def trySerializedSample(npy_path, name, targ_im_len):
    im = np.load(npy_path + '/ims/' + name + '.npy')
    if targ_im_len == -1:
        factor = 1
    else:
        factor = targ_im_len-1
    label = np.load(npy_path + '/coords/' + name + '.npy')
    label *= factor
    """
    mask = np.load(npy_path + '/masks/' + name + '.npy')
    mask = (80 * mask).astype(np.uint8)
    rem = 255 - im[:,:,1]
    im[:,:,1] += np.minimum(rem, mask)
    """
    reshaped_labels = np.reshape(label, (-1, 2))
    lip_coords = getLipCoords(reshaped_labels)
    shape = (targ_im_len, targ_im_len)
    mask = utils.getMask([lip_coords], shape, shape)
    mask = (80 * mask).astype(np.uint8)
    rem = 255 - im[:,:,1]
    im[:,:,1] += np.minimum(rem, mask)
    utils.visualizeCoords(im, label)

def visualizeMask(im, mask, targ_im_len=-1):
    if targ_im_len != -1:
        im_resize_method = cv2.INTER_CUBIC if targ_im_len > len(im) else cv2.INTER_AREA
        mask_resize_method = cv2.INTER_CUBIC if targ_im_len > len(mask) else cv2.INTER_AREA 
        im = cv2.resize(im, (targ_im_len, targ_im_len), interpolation=im_resize_method)
        mask = cv2.resize(mask, (targ_im_len, targ_im_len), interpolation=mask_resize_method)

    mask = (80 * mask).astype(np.uint8)
    rem = 255 - im[:,:,1]
    im[:,:,1] += np.minimum(rem, mask)
    plt.imshow(im)
    plt.show()

def trySerializedFolder(npy_path, targ_im_len):
    with open(npy_path + '/names.json') as fp:
        names = json.load(fp)
    for name in names:
        trySerializedSample(npy_path, name, targ_im_len)