from __future__ import print_function
import os
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import utils
import json
from skimage.transform import resize
from matplotlib.patches import Circle


class DatasetProps:

    def __init__(self, im_extension, label_extension, im_paths, label_paths):
        self.im_extension = im_extension
        self.label_extension = label_extension
        self.im_paths = im_paths
        self.label_paths = label_paths


def save_data(props, npy_path, targ_im_len, append_to_names=False):
    """
    :param props: instance of DatasetProps.
    """

    # train
    ims = read_images(props.im_paths, props.im_extension)
    labels = read_labels(props.label_paths, props.label_extension)
    
    if append_to_names == True:
    	with open(npy_path + '/names.json') as fp:
	    names = json.load(fp)
    else:
    	names = []

    milestone = max(len(ims) / 10, 1)
    print('Resizing samples ...')

    all_ims = []
    iter = 0
    for name in ims:
        im, label = resize_pair(ims[name], labels[name], targ_im_len, targ_im_len)
        label = normalize_coords(label, targ_im_len, targ_im_len)
        ims[name] = im
        all_ims.append(im)
        if iter != 0 and iter % milestone == 0:
            print('    ' + str(100.0 * float(iter) / len(ims)) + '% complete')
        iter += 1

    print('COMPLETE\n')

    #""" data centering """
    #all_ims = np.array(all_ims)
    #mean_im = np.average(all_ims, axis=0)
    #std_im = np.average(np.abs(all_ims - mean_im), axis=0)

    print('Normalizing data and serializing to disk ...')
    iter = 0
    for name in ims:
        names.append(name)
        im = ims[name]
        np.save(npy_path + '/ims/' + name + '.npy', im)
        np.save(npy_path + '/labels/' + name + '.npy', labels[name])
        if iter != 0 and iter % milestone == 0:
            print('    ' + str(100.0 * float(iter) / len(ims)) + '% complete')
        iter += 1

    print('COMPLETE\n')

    with open(npy_path + '/names.json', 'wb') as fp:
        json.dump(names, fp)


def normalize_coords(coords, im_width, im_height):
    for coord in coords:
        coord[0] /= im_height
        coord[1] /= im_width
    return coords


def resize_pair(im, label, targ_width, targ_height):
    cur_height = len(im)
    cur_width = len(im[0])
    scale_x = float(targ_width) / cur_width
    scale_y = float(targ_height) / cur_height
    for coords in label:
        coords[0] *= scale_x
        coords[1] *= scale_y
    return [resize(im, (targ_height, targ_width)), label]


def get_ordered(ims, labels):
    """
    :param ims: a dictionary of images
    :param labels: a dictionary of labels

    :return: [ims_arr, labels_arr], where ims_arr[i] corresponds to labels_arr[i]
    """
    ims_ordered, labels_ordered = [], []
    for name in ims:
        labels_ordered.append(labels[name])
        ims_ordered.append(ims[name])
    return [ims_ordered, labels_ordered]


def read_images(paths, extension, max_images=-1):
    """
    :param paths: paths to all folders that we will read from
    :param extension: file extension, e.g '.png'
    :param max_images: the max. number of images to read into array before stopping. -1 means all of them.
    :returns: dictionary of images, with key being filename without extension
    """

    print('Reading images ...')
    ims = {}
    fpaths = []
    for path in paths:
        for fname in os.listdir(path):
            if fname.endswith(extension):
                fpaths.append([path, fname])
            if max_images != -1 and len(fpaths) >= max_images:
                break

    milestone = max(len(fpaths) / 10, 1)
    for i in range(0, len(fpaths)):
        path = fpaths[i][0]
        fname = fpaths[i][1]
        ims[fname[:-len(extension)]] = scipy.misc.imread(path + '/' + fname)
        if i != 0 and i % milestone == 0:
            print('    ' + str(100.0 * float(i) / len(fpaths)) + '% complete')
    print('COMPLETE\n')
    return ims


def read_labels(path, extension, max_labels=-1):
    """
    :param path: path to folder
    :param extension: file extension, e.g '.png'
    :param max_labels: the max. number of labels to read into array before stopping. -1 means all of them.
    :returns dictionary of labels, with key being filename without extension
    """
    labels = {}
    for fname in os.listdir(path):
        if fname.endswith(extension):
            cur_labels = []
            lines = open(path + '/' + fname).readlines()
            key = fname[:-len(extension)]
            for i in range(0, len(lines)):
                if i == 0:
                    key = lines[i].strip()
                else:
                    coords = []
                    for s in lines[i].split():
                        if utils.is_number(s):
                            coords.append(float(s))
                    if len(coords) == 2:
                        cur_labels.append(coords)
            labels[key] = cur_labels
        if max_labels != -1 and len(labels) >= max_labels:
            break
    return labels
