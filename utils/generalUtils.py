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
import tensorflow as tf
import matplotlib.lines as mlines


# https://stackoverflow.com/questions/18554012/intersecting-two-dictionaries-in-python
def getKeysIntersection(dict_a, dict_b):
    set_a = set(dict_a.keys())
    set_b = set(dict_b.keys())
    return set_a & set_b

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
    

def visualizeMask(im, mask, targ_im_len=-1):
    """
    mask values should all be between [0,1]
    """
    if targ_im_len != -1:
        im_resize_method = cv2.INTER_CUBIC if targ_im_len > len(im) else cv2.INTER_AREA
        mask_resize_method = cv2.INTER_CUBIC if targ_im_len > len(mask) else cv2.INTER_AREA 
        im = cv2.resize(im, (targ_im_len, targ_im_len), interpolation=im_resize_method)
        mask = cv2.resize(mask, (targ_im_len, targ_im_len), interpolation=mask_resize_method)

    mask = 180.0 * np.minimum(mask, 1)
    mask = mask.astype(np.uint8)
    rem = 255 - im[:,:,1]
    im[:,:,1] += np.minimum(rem, mask)
    #plt.imshow(mask)
    plt.imshow(im)
    plt.show()


def visualizeCoordMasks(im, masks):
    """
    Parameters
    ----------
    im : 
        The 2D array image.
    masks : 
        Has to be of shape (mask_side_len, mask_side_len, num_coords)
    """
    
    masks /= np.max(masks, axis=(0,1))
    summed = np.sum(masks, axis=-1)
    summed = cv2.resize(summed, (len(im), len(im[0])), interpolation=cv2.INTER_LINEAR)
    visualizeMask(im, summed)


def readCoords(path):
    """
    The file should contain 2 float values per-line, and space separated (i.g x y).
    """
    coords = []
    i = 0
    with open(path) as f:
        for line in f:
            coords.append([])
            line = line.strip()
            for number in line.split():
                coords[i].append(float(number))
            coords[i].reverse()
            i += 1
    return coords

def visualizeCoords(im, coords, special_indices=[], output_name=None):
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
    radius = 0.001 * len(im)
    for i in range(0, len(coords)):
        x = coords[i][1]
        y = coords[i][0]
        if i in special_indices:
            circ = Circle((x, y), radius, color='green')
        else:
            circ = Circle((x, y), radius, color='blue')
        ax.add_patch(circ)

    if output_name != None:
        plt.axis('off')
        plt.savefig(output_name)
    else:
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

def getRotationMatrix(angle):
    """
    angle should be in radians, and specifies ccw rotation.
    """
    row1 = [ np.cos(angle), -np.sin(angle) ] 
    row2 = [ np.sin(angle), np.cos(angle) ]
    return np.squeeze(np.array([row1, row2]))

def getRotatedPoints(points, center, angle):
    """
    For 2D points only.
    Angle should be in radians, and specifies ccw rotation.

    Parameters
    ----------
    points:         
        Points should be of shape (num_points, 2)
    center: 
        Of shape (2,)
    angle: 
        Angle should be in radians, and specifies ccw rotation.
    """
    rot = getRotationMatrix(angle)
    vecs = points - center
    rot_vecs = np.dot(rot, np.transpose(vecs))
    rot_vecs = np.transpose(rot_vecs)
    return np.array(center) + rot_vecs

def getSquareFromRect(_rect):
    """
    Parameters
    ----------
    _rect: 
        Array with values ordered as : [top, left, bottom, right]. Return value has the same format.
    Returns
    -------
    Rect will be padded on its smaller dimension so that it's a square.
    """
    rect = list(_rect)
    width = rect[3] - rect[1]
    height = rect[2] - rect[0]
    diff = abs(width - height)
    pad = diff / 2.0
    if width > height:
        rect[0] -= pad
        rect[2] += pad
    else:
        rect[1] -= pad
        rect[3] += pad
    return rect

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

def getShiftedBbox(_bbox, shifts):
    """
    Parameters
    ----------
    bbox: 
        Array with values ordered as such: [top, left, bottom, right] 
    shifts: 
        [y_shift, x_shift]
    """
    bbox = list(_bbox)
    bbox[0] += shifts[0]
    bbox[2] += shifts[0]
    bbox[1] += shifts[1]
    bbox[3] += shifts[1]
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

def getClippedBbox(im, bbox):
    l = int(max(0, bbox[1]))
    r = int(min(len(im[0]), bbox[3]+1))
    t = int(max(0, bbox[0]))
    b = int(min(len(im), bbox[2]+1))
    return [t, l, b, r]

def getCropped(im, bbox):
    t, l, b, r = getClippedBbox(im, bbox)
    return im[t:b, l:r]


def getExpandedBbox(bbox, ratio_x, ratio_y):
    """
    Parameters
    ----------
    bbox: 
        Order should be: top, left, bottom, right. Right > left, bottom > top.
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

def expandDimsRepeatedly(x, num_expansions, front=True):
    axis = 0 if front else -1
    for i in range(num_expansions):
        x = tf.expand_dims(x, axis=axis)
    return x

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

def getCoordsFromPointMasks(masks, targ_width, targ_height, mode = 'mean'):
    """
    Parameters
    ----------
    masks: 
        Should be of shape (num_coords, height, width)

    mode: 
        Method in which we extract indices from response maps. 'mean' or 'max' are valid.
    
    Returns
    -------
    Array of shape (num_coords, 2). Coordinates are wrt dimensions defined by targ_width and targ_height.
    """
    coords = []
    for coord_mask in masks:

        if mode == 'max':
            y_ind, x_ind = np.unravel_index(np.argmax(coord_mask), coord_mask.shape)
            coords.append([y_ind, x_ind])
        
        if mode == 'mean':
            # normalize mask
            coord_mask /= np.sum(coord_mask)

            # remove outliers
            max_val = np.max(coord_mask)
            coord_mask = np.where(coord_mask > 0.01 * max_val, coord_mask, 0.0)
            coord_mask /= np.sum(coord_mask)
            y_scale = targ_height / float(len(coord_mask))
            x_scale = targ_width / float(len(coord_mask[0]))

            x_inds = np.arange(0, len(coord_mask[0]))
            y_inds = np.arange(0, len(coord_mask))
            x_avg = x_scale * np.sum(np.array([x_inds]) * coord_mask)
            y_avg = y_scale * np.sum(np.transpose(np.array([y_inds])) * coord_mask)
            #plt.imshow(coord_mask)
            #plt.show()
            #print x_avg
            #print y_avg
            coords.append([y_avg, x_avg])
    
    return coords


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