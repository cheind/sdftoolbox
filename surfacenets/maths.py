import numpy as np
from typing import TypeVar, Any

ShapeLike = TypeVar("ShapeLike", bound=Any)


def generalized_max(x: np.ndarray, axis: ShapeLike = None, alpha: float = np.inf):
    """Generalized maximum function.

    As `alpha` goes to infinity this function transforms from a smooth
    maximum to standard maximum. This function is frequently used by
    boolean CSG operations to avoid hard transitions.

    Based on the wikipedia article
    https://en.wikipedia.org/wiki/Smooth_maximum

    Params
        x: The input values to take the maximum over
        axis: Optional axis to perform operation on
        alpha: Defines the smoothness of the maximum approximation. Lower values
            give smoother results.

    Returns
        y: (smooth) maximum along axis.
    """
    if np.isfinite(alpha):
        xmax = np.max(x, axis=axis, keepdims=True)
        ex = np.exp(alpha * (x - xmax))
        smax = np.sum(x * ex, axis=axis) / np.sum(ex, axis=axis)
        return smax
    else:
        return np.max(x, axis=axis)


def hom(v, value=1, dtype: np.dtype = None):
    """Returns v as homogeneous vectors"""
    v = np.asanyarray(v, dtype=dtype)
    return np.insert(v, v.shape[-1], value, axis=-1)


def dehom(a, dtype: np.dtype = None):
    """Makes homogeneous vectors inhomogenious by dividing by the last element of the last axis"""
    a = np.asfarray(a, dtype=dtype)
    return a[..., :-1] / a[..., None, -1]


def translate(values: np.ndarray, dtype: np.dtype = None) -> np.ndarray:
    """Construct and return a translation matrix"""
    values = np.asfarray(values, dtype=dtype)
    dtype = dtype or values.dtype
    m = np.eye(4, dtype=dtype)
    m[:3, 3] = values
    return m


def scale(values: np.ndarray, dtype: np.dtype = None) -> np.ndarray:
    """Construct and return a scaling matrix"""
    values = np.asfarray(values, dtype=dtype)
    m = np.eye(4, dtype=dtype)
    m[[0, 1, 2], [0, 1, 2]] = values
    return m


def _skew(a):
    return np.array(
        [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=a.dtype
    )


def rotate(axis: np.ndarray, angle: float, dtype: np.dtype = None) -> np.ndarray:
    """Construct a rotation matrix given axis/angle pair."""
    # See
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    axis = np.asfarray(axis, dtype=dtype)
    sina = np.sin(angle)
    cosa = np.cos(angle)
    d = axis / np.linalg.norm(axis)
    R = (
        cosa * np.eye(3, dtype=axis.dtype)
        + sina * _skew(d)
        + (1 - cosa) * np.outer(d, d)
    )
    m = np.eye(4, dtype=axis.dtype)
    m[:3, :3] = R
    return m


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, 1000)
    xnx = np.stack((x, -x), -1)

    plt.plot(x, generalized_max(xnx, -1, 0.5), label=r"$\alpha$=0.5")
    plt.plot(x, generalized_max(xnx, -1, 1), label=r"$\alpha$=1.0")
    plt.plot(x, generalized_max(xnx, -1, 2), label=r"$\alpha$=2.0")
    plt.plot(x, generalized_max(xnx, -1, 4), label=r"$\alpha$=4.0")
    plt.plot(x, generalized_max(xnx, -1, 8), label=r"$\alpha$=4.0")
    plt.plot(x, generalized_max(xnx, -1), label=r"$\alpha$=$\inf$", color="k")
    plt.legend(
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        ncol=3,
    )
    plt.tight_layout()
    plt.show()
