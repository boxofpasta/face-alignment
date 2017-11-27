from __future__ import print_function
import os
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import utils
import utils
import json
import sys
import cv2
from matplotlib.patches import Circle


class DatasetProps:

    def __init__(self, im_extension, label_extension, im_path, label_path):
        """
        :param im_extension: image extension, e.g '.png'
        :param im_path: path to folder with images
        """
        self.im_extension = im_extension
        self.label_extension = label_extension
        self.im_path = im_path
        self.label_path = label_path


def process_data(props, targ_im_len, sample_names=None):
    """
    :param props: instance of DatasetProps. props.im_paths specifies the folders to read from.
    :param targ_im_len: the target image width and height. If -1, won't resize or warp.
    :param sample_names: specific samples to read from.
    :returns (ims, labels)
    """

    # train
    ims = read_images(props.im_path, props.im_extension, sample_names=sample_names)
    labels = read_labels(props.label_path, props.label_extension, sample_names=sample_names)
    print('\n\nResizing samples ...')

    iter = 0
    for name in ims:
        if targ_im_len != -1:
            print('resizing lool')
            im, label = resize_pair(ims[name], labels[name], targ_im_len, targ_im_len)
            labels[name] = normalize_coords(label, targ_im_len, targ_im_len)
            ims[name] = im
        utils.inform_progress(iter, len(ims))
        iter += 1

    utils.inform_progress(1,1)
    return ims, labels
    #""" data centering """
    #all_ims = np.array(all_ims)
    #mean_im = np.average(all_ims, axis=0)
    #std_im = np.average(np.abs(all_ims - mean_im), axis=0)


def serialize_data(ims, labels, npy_path):
    print('\n\nNormalizing data and serializing to disk ...')
    try:
        # avoid overwriting data in json
        with open(npy_path + '/names.json') as fp:
            names_set = set(json.load(fp))
    except ValueError:
        names_set = set()
    iter = 0
    for name in ims:
        names_set.add(name)
        im = ims[name]
        np.save(npy_path + '/ims/' + name + '.npy', im)
        np.save(npy_path + '/labels/' + name + '.npy', labels[name])
        utils.inform_progress(iter, len(ims))
        iter += 1

    utils.inform_progress(1,1)
    with open(npy_path + '/names.json', 'wb') as fp:
        json.dump(list(names_set), fp)


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
    resized = cv2.resize(im, (targ_height, targ_width), interpolation=cv2.INTER_CUBIC)
    return [resized, label]


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


def read_images(path, extension, sample_names=None):
    """
    :param path: paths to folder that we're reading from
    :param extension: file extension, e.g '.png'
    :param sample_names: the samples in the folder to read from. If None, will read all of them.
    :returns: dictionary of images, with key being filename without extension
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
        utils.inform_progress(i, len(sample_names))
    utils.inform_progress(1, 1)
    return ims


def read_labels(path, extension, sample_names):
    """
    :param path: path to folder
    :param extension: file extension, e.g '.png'
    :param sample_names: the samples in the folder to read from. If None, will read all of them.
    :returns dictionary of labels, with key being filename without extension
    """
    labels = {}
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
                        if utils.is_number(s):
                            coords.append(float(s))
                    if len(coords) == 2:
                        cur_labels.append(coords)
                labels[key] = cur_labels
    return labels
