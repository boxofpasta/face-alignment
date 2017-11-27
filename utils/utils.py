import os
import sys
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def visualize_labels(im, coords):
    """
        :param im: the 2D array image
        :param coords: the coordinates to draw e.g [[21, 10]]
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
        sys.stdout.write('\r    ' + str(100.0 * float(iter) / total) + '% complete')
        sys.stdout.flush()