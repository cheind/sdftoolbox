import numpy as np


def reorient_volume(x: np.ndarray):
    return np.flip(x.transpose((2, 1, 0)), 1)
