import numpy as np


def print_volume_slices(x: np.ndarray):
    print(np.flip(x.transpose((2, 1, 0)), 1))
