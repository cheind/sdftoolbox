import numpy as np
from contextlib import contextmanager

float_dtype = np.float_


@contextmanager
def default_dtype(new_dtype: np.dtype):
    global float_dtype
    old_type = float_dtype
    try:
        float_dtype = new_dtype
        yield
    finally:
        float_dtype = old_type
