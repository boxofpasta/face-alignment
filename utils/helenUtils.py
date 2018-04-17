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
from scipy.interpolate import interp1d
from skimage.draw import line_aa


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

def getNumCoords(coords_sparsity, ibug_version=True):
    num_points = 68 if ibug_version else 81
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


def reserializeFolderAsNpy(im_path, coords_path, im_extension, coords_extension, npy_path, targ_im_width, ibug_version=False):
    """
    Goes through im_path along with coords_path and saves them as npy arrays in npy_path
    """
    im_reader = ImagesReader(im_path, im_extension, 2000)

    # operating under the assumption that coords/annotations take up a negligible 
    # amount of memory relative to the images.
    while not im_reader.complete:
        print("\nReading images...")
        ims_list, names_list = im_reader.read()
        ims = utils.getDictFromLists(names_list, ims_list)
        coords = readCoordsHelen(coords_path, coords_extension, sample_names=names_list, ibug_version=ibug_version)
        ims, coords = processData(ims, coords, targ_im_width)
        serializeData(ims, coords, npy_path, ibug_version=ibug_version)


def processData(ims, coords, targ_im_width):
    """
    Parameters
    ----------
    Both ims and coords should be dicts. The keys that we process will be the intersection of the dictionaries.
    ims and coords will be modified by reference.

    """

    #ims = readImagesHelen(props.im_path, props.im_extension, sample_names=sample_names)
    #coords = readCoordsHelen(props.coords_path, props.coords_extension, sample_names=sample_names, ibug_version=ibug_version)

    if targ_im_width != -1:
        print('\n\nResizing samples ...')
        iter = 0
        names = utils.getKeysIntersection(ims, coords)
        for name in names:
            if targ_im_width != -1:
                im, single_coords = cropPair(ims[name], coords[name])
                h_w_ratio = float(len(im)) / len(im[0])
                im, single_coords = resizePair(im, single_coords, targ_im_width, targ_im_width * h_w_ratio)
                coords[name] = single_coords
                #coords[name] = normalizeCoords(single_coords, targ_im_width, targ_im_width * h_w_ratio)
                # check if grayscale
                if len(list(im.shape)) == 2 or im.shape[-1] == 1:
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
                elif len(list(im.shape)) != 3 or im.shape[-1] != 3:
                    raise ValueError('Unexpected image shape ' + str(im.shape))

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
    names = utils.getKeysIntersection(all_ims, all_coords)

    for name in names:
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
    label[:,0] *= scale_y
    label[:,1] *= scale_x
    """for coords in label:
        coords[0] = coords[0] * scale_y
        coords[1] = coords[1] * scale_x
    """
    resized = cv2.resize(im, (int(targ_width), int(targ_height)), interpolation=cv2.INTER_AREA)
    return [resized, label]

def cropPair(im, label):
    label = np.reshape(label, (-1, 2))
    #lip_coords = getLipCoords(label)
    bbox = utils.getBbox(label)
    bbox = utils.getExpandedBbox(bbox, 0.3, 0.3)
    bbox = utils.getClippedBbox(im, bbox)
    #bbox = utils.getBbox(lip_coords)

    # randomly expand facebox
    #bbox = utils.getRandomlyExpandedBbox(bbox, 0.03, 0.35)
    #bbox = utils.getRandomlyExpandedBbox(bbox, 0.10, 1.0)
    label[:,0] -= bbox[0]
    label[:,1] -= bbox[1]
    im = utils.getCropped(im, bbox)
    return [im, label]

def getLeyeCenter(coords, ibug_version=True):
    leye_coords = coords[36:42] if ibug_version else coords[59:69]
    return np.array([np.mean(leye_coords[:,0]), np.mean(leye_coords[:,1])])

def getReyeCenter(coords, ibug_version=True):
    reye_coords = coords[42:48] if ibug_version else coords[49:59]
    return np.array([np.mean(reye_coords[:,0]), np.mean(reye_coords[:,1])])

def getEyeDistance(coords, ibug_version=True):
    leye_center = getLeyeCenter(coords, ibug_version=ibug_version)
    reye_center = getReyeCenter(coords, ibug_version=ibug_version)
    return np.linalg.norm(leye_center - reye_center)

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
            key = fname[:-len(extension)]
            if sample_names == None or key in allowed_samples:
                cur_labels = []

                if ibug_version:
                    for i in range(3, len(lines)-1):
                        coords = [float(s) - 1.0 for s in lines[i].split()]

                        # want y, x because tensorflow likes it this way
                        cur_labels.append(list(reversed(coords)))
                    labels[key] = cur_labels
                else:
                    for i in range(3, len(lines)-3):
                        coords = [float(s) - 1.0 for s in lines[i].split()]

                        # want y, x because tensorflow likes it this way
                        cur_labels.append(list(reversed(coords)))

                    # facebox as the last 2 coordinates
                    vals = lines[-1].split()[1:]
                    cur_labels.append([vals[0], vals[1]])
                    cur_labels.append([vals[2], vals[3]])
                    cur_labels = np.array(cur_labels).astype(np.float32)
                    labels[key] = cur_labels
    return labels

def getLipCoords(coords, max_x, flip_x=False, ibug_version=True):
    """
    Parameters
    ----------
    coords: 
        Should have shape (num_coords, 2).
    flip_x: 
        If true, coordinates will correspond to image with with x-axis flipped. 
        Note that coordinate indices will still be following the same order as the original points.
    """
    if ibug_version:
        if flip_x:
            c = np.copy(coords[48:60])
            c[:,1] = float(max_x) - c[:,1]
            c[0], c[6] = c[6], c[0].copy()
            c[1], c[5] = c[5], c[1].copy()
            c[2], c[4] = c[4], c[2].copy()
            c[11], c[7] = c[7], c[11].copy()
            c[10], c[8] = c[8], c[10].copy()
            return c
        return coords[48:60]

    else:
        if flip_x:
            c = np.copy(coords[28:41])
            c[:,1] = float(max_x) - c[:,1]
            c[0], c[6] = c[6], c[0].copy()
            c[1], c[5] = c[5], c[1].copy()
            c[2], c[4] = c[4], c[2].copy()
            c[12], c[7] = c[7], c[12].copy()
            c[11], c[8] = c[8], c[11].copy()
            c[10], c[9] = c[9], c[10].copy()
            return c
        return coords[28:41]

def getLipLineMask(lip_coords, in_shape, out_shape):
    """
    Only works for the ibug annotated version currently.
    Parameters
    ----------
    in_shape: 
        shape of image to which lip_coords correspond to.
    out_shape: 
        shape of output image (we will resize).
    coords: 
        Should have shape (num_coords, 2).
    """

    # https://stackoverflow.com/questions/5283649/plot-smooth-line-with-pyplot
    lip_coords = np.array(lip_coords)
    x = in_shape[1] * lip_coords[:,1]
    y = in_shape[0] * lip_coords[:,0]
    pts = 40

    # p0 -> p6
    x_new = np.linspace(x[0], x[6], pts / 2)
    y_new_top = interp1d(x[0:7], y[0:7], kind='cubic')(x_new)

    # p6 -> p11 -> p0
    x_new = np.linspace(x[0], x[6], pts / 2)

    x_bot = np.flip(np.concatenate([x[6:], [x[0]]]), axis=0)
    y_bot = np.flip(np.concatenate([y[6:], [y[0]]]), axis=0)

    y_new_bot = interp1d(x_bot, y_bot, kind='cubic')(x_new)

    # draw into an image with many times as many pixels, as line_aa only works with integer coords
    mult = 1
    draw_shape = (in_shape[0] * mult, in_shape[1] * mult)

    x_new = np.rint(mult * np.array(x_new)).astype(int)
    y_new_top = np.rint(mult * np.array(y_new_top)).astype(int)
    y_new_bot = np.rint(mult * np.array(y_new_bot)).astype(int)
    img = np.zeros(draw_shape, dtype=np.uint8)

    # draw the anti-aliased lines
    for i in range(len(x_new)-1):

        # top
        rr, cc, val = line_aa(y_new_top[i], x_new[i], y_new_top[i+1], x_new[i+1])
        img[rr, cc] = val * 255

        # bot
        rr, cc, val = line_aa(y_new_bot[i], x_new[i], y_new_bot[i+1], x_new[i+1])
        img[rr, cc] = val * 255

    kernel_width = int(0.007 * len(img[0]))
    kernel_height = int(0.007 * len(img))
    kernel = np.ones((kernel_width, kernel_height))
    img = cv2.dilate(img, kernel, iterations=1)

    #rr, cc, val = line_aa(1, 1, 8, 4)
    #img[rr, cc] = val * 255
    #scipy.misc.imsave("out.png", img)

    img = img.astype(np.float32)
    img = cv2.resize(img, out_shape, interpolation=cv2.INTER_AREA)
    return img


def trySerializedSample(npy_path, name):
    im = np.load(npy_path + '/ims/' + name + '.npy')
    label = np.load(npy_path + '/coords/' + name + '.npy')
    utils.visualizeCoords(im, label)
    """
    if targ_im_len == -1:
        factor = 1
    else:
        factor = targ_im_len-1
    label *= factor
    reshaped_labels = np.reshape(label, (-1, 2))
    lip_coords = getLipCoords(reshaped_labels)
    shape = (targ_im_len, targ_im_len)
    mask = utils.getMask([lip_coords], shape, shape)
    mask = (80 * mask).astype(np.uint8)
    rem = 255 - im[:,:,1]
    im[:,:,1] += np.minimum(rem, mask)
    utils.visualizeCoords(im, label)
    """

def trySerializedFolder(npy_path, targ_im_len):
    with open(npy_path + '/names.json') as fp:
        names = json.load(fp)
    for name in names:
        trySerializedSample(npy_path, name, targ_im_len)