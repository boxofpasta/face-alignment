import os
import sys
import scipy.misc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Circle

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
    Numpy array of shape (num_coords, l, l). Each point is represented by a heatmap with a single gaussian.
    """
    heatmaps = []
    for point in coords:
        i_x = int(np.rint(np.clip(point[0], 0, 1) * (len(pdfs) - 1)) ) 
        i_y = int(np.rint(np.clip(point[1], 0, 1) * (len(pdfs) - 1)))
        pdf_x = [pdfs[i_x]] 
        pdf_y = [pdfs[i_y]] 
        heatmap = np.transpose(pdf_y) * pdf_x
        heatmap /= np.sum(heatmap)
        heatmaps.append(heatmap)
    return heatmaps

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

def visualizeLabels(im, coords):
    """
    Parameters
    ----------
    im : 
        the 2D array image.
    coords : 
        the coordinates to draw e.g [[21, 10]].
    """
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(im)
    radius = 0.001 * len(im)
    for i in range(0, len(coords)):
        x = coords[i][0]
        y = coords[i][1]
        circ = Circle((x, y), radius)
        ax.add_patch(circ)

    plt.show()


def inform_progress(iter, total):
    milestone = max(total / 100, 1)
    if iter != 0 and iter % milestone == 0:
        sys.stdout.write('\r    ' + str(round(100.0 * float(iter) / total, 1)) + '% complete')
        sys.stdout.flush()