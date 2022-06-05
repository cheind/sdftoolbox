import numpy as np


def reorient_volume(x: np.ndarray):
    """Reorients the given volume for easier inspection.

    In particular the volume is reoriented, so that when printing the volume
    you see ij-slices (k from from front to back) and each ij-slice has the origin
    at lower left.
    """
    return np.flip(x.transpose((2, 1, 0)), 1)
