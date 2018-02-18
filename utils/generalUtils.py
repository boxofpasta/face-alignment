import os
import sys
import scipy.misc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Circle
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw
import cv2
import time
import matplotlib.lines as mlines

def transposeList(l):
    return map(list, zip(*l))

def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def getGaussians(num_means, num_elements, stddev=0.01):
    """
    Parameters
    ----------
    num_means: 
        Number of gaussian pdfs to store.
    num_elements: 
        Number of samples for each of the gaussian pdfs.
    
    Note that [start, end] for each gaussian for both means and elements = [0, 1]
    
    Returns
    -------
    Array of pdfs, e.g ret[0] is gaussian pdf with mean = 0. ret[-1] has mean = 1.0.
    """
    linspace = np.linspace(0, 1, num_elements)
    linspace_means = np.linspace(0, 1, num_means)
    gaussians = []
    for mean in linspace_means:
        pdf = norm.pdf(linspace, loc=mean, scale=stddev)
        gaussians.append(pdf)
    return gaussians

def coordsToHeatmapsFast(coords, pdfs):
    """
    Uses a table of pdfs (ideally shared b/w invocations) to speed up heatmap computation.
    
    Parameters
    ----------
    coords: 
        Numpy array of shape (num_coords, 2) representing points on the face. Coordinates should be normalized to be from [0, 1].

    pdf_map: 
        Array-like that stores pdfs for different means. 
        The distance of the means between adjacent array elements is 1.0 / (len(pdf) - 1).

    Returns
    -------
    Numpy array of shape (num_coords, l, l), where l = len(pdf[i]). Each point is represented by a heatmap with a single gaussian.
    """
    heatmaps = []
    for point in coords:
        i_x = int(np.rint(np.clip(point[1], 0, 1) * (len(pdfs) - 1)) ) 
        i_y = int(np.rint(np.clip(point[0], 0, 1) * (len(pdfs) - 1)))
        pdf_x = [pdfs[i_x]] 
        pdf_y = [pdfs[i_y]] 
        heatmap = np.transpose(pdf_y) * pdf_x
        heatmap /= np.sum(heatmap)
        heatmaps.append(heatmap)
    return heatmaps

def getDictFromLists(a, b):
    """
    Parameters
    ----------
    a: 
        The keys.
    b: 
        The values. Must be of the same length as b.
    """
    if len(a) != len(b):
        raise ValueError('Must be of the same length')
    c = {}
    for i in range(len(a)):
        c[a[i]] = b[i]
    return c

def coordsToHeatmaps(coords, elms_per_side, stddev=0.01):
    """
    Parameters
    ----------
    coords: 
        Numpy array of shape (194, 2) representing points on the face. Coordinates should be normalized to be from [0, 1].

    elms_per_side: 
        Heatmap has elms_per_side * elms_per_side elements.

    Returns
    -------
    Numpy array of shape (194, l, l). Each point is represented by a heatmap with a single gaussian.
    """
    heatmaps = []
    linspace = np.linspace(0, 1.0, elms_per_side)
    for point in coords:
        pdf_x = [norm.pdf(linspace, loc=point[0], scale=stddev)]
        pdf_y = [norm.pdf(linspace, loc=point[1], scale=stddev)]
        heatmap = np.transpose(pdf_y) * pdf_x
        heatmap /= np.sum(heatmap)
        heatmaps.append(heatmap)
    return heatmaps

def visualizeCoords(im, coords, special_indices=[]):
    """
    Parameters
    ----------
    im : 
        The 2D array image.
    coords : 
        The coordinates to draw e.g [[21, 10]].
    special_indices:
        An array of indices. Coordinates with these indices will have a red color
    """
    special_indices = set(special_indices)
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(im)
    radius = 0.003 * len(im)
    for i in range(0, len(coords)):
        x = coords[i][1]
        y = coords[i][0]
        if i in special_indices:
            circ = Circle((x, y), radius, color='red')
        else:
            circ = Circle((x, y), radius)
        ax.add_patch(circ)
    plt.show()

def getMask(polygons, src_dims, dst_dims):
    """
    https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
    Parameters
    ----------
    polygons: 
        List of polygons, where each polygon has shape (num_coords, 2).
    src_dims: 
        A 2-tuple of integers, specifying (height, width) of the image that coordinates in polygons are using
    dst_dims: 
        Same format as above. Output image will be resized to these dimensions.
    Returns
    -------
        Returns bitmask with all pixels within polygons being 1.0, and outside being 0.0.
    """
    # expand more for higher quality masks
    scale_factor = 4
    img = Image.new('L', (src_dims[0] * scale_factor, src_dims[1] * scale_factor), 0)

    for polygon in polygons:
        tupled = [tuple(reversed(tuple(scale_factor * coord))) for coord in polygon]
        ImageDraw.Draw(img).polygon(tupled, outline=1, fill=1)
    img = np.array(img).astype(float)
    return cv2.resize(img, dst_dims, interpolation=cv2.INTER_AREA)

def printTensorShape(tensor):
    print tensor.get_shape().as_list()

def getBbox(coords):
    """
    Returns
    -------
    Array with values ordered as such: [top, left, bottom, right]
    """
    coords = np.reshape(coords, (-1, 2))
    x_vals = coords[:,1]
    y_vals = coords[:,0]
    bbox = np.array([np.min(y_vals), np.min(x_vals), np.max(y_vals), np.max(x_vals)])
    return bbox

def getRandomlyExpandedBbox(bbox, ratio_low, ratio_high):
    """
    Internally uses getExpandedBbox after randomly sampling for a ratio (from a uniform distribution),
    done independently for width and height.
    
    Parameters
    ----------
    ratio_low: 
        Lower bound of the uniform distribution that we are sampling from.
    """
    ratio_x, ratio_y = np.random.uniform(ratio_low, ratio_high, 2)
    return getExpandedBbox(bbox, ratio_x, ratio_y)
    

def getExpandedBbox(bbox, ratio_x, ratio_y):
    """
    Parameters
    ----------
    bbox: 
        Order should be: left, top, right, bottom. Right > left, bottom > top.
    ratio_x: 
        E.g value of 1.0 means 100% of max(width, height) added as padding.
    """
    width = bbox[3] - bbox[1]
    height = bbox[2] - bbox[0]
    max_len = max(width, height)
    x_pad = 0.5 * ratio_x * max_len
    y_pad = 0.5 * ratio_y * max_len
    return np.array([
        bbox[0] - y_pad,
        bbox[1] - x_pad,
        bbox[2] + y_pad,
        bbox[3] + x_pad
    ])

def visualizeBboxes(im, boxes):
    """
    Parameters
    ----------
    boxes:
        Each box should have [left, top, right, bottom].
    """
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(im)
    for i in range(len(boxes)):
        color = (1.0, float(i) / len(boxes), 0.0)
        box = boxes[i]
        ax.plot([box[0], box[2]], [box[1], box[1]], '-', color=color)
        ax.plot([box[2], box[2]], [box[1], box[3]], '-', color=color)
        ax.plot([box[2], box[0]], [box[3], box[3]], '-', color=color)
        ax.plot([box[0], box[0]], [box[3], box[1]], '-', color=color)
    plt.show()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def imSoftmax(x):
    """
    Image should be of shape = (height, width, channels).
    This will take a per-channel softmax.
    Return value will be of the same shape as input.
    """
    channels = x.shape[2]
    height = x.shape[0]
    x = np.moveaxis(x, -1, 0)
    x = np.reshape(x, (channels, -1))

    x = [softmax(d) for d in x]
    x = np.reshape(x, (channels, height, -1))
    x = np.moveaxis(x, 0, -1)
    return x

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def informProgress(iter, total):
    milestone = max(total / 100, 1)
    if iter != 0 and iter % milestone == 0:
        sys.stdout.write('\r    ' + str(round(100.0 * float(iter) / total, 1)) + '% complete')
        sys.stdout.flush()