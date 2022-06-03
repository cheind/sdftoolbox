import numpy as np
from typing import TypeVar, Any

ShapeLike = TypeVar("ShapeLike", bound=Any)


def generalized_max(
    x: np.ndarray, axis: ShapeLike = None, alpha: float = np.inf
):
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


def hom(v, value=1):
    """Returns v as homogeneous vectors by inserting one more element into the last axis
    the parameter value defines which value to insert (meaningful values would be 0 and 1)
    >>> homogenize([1, 2, 3]).tolist()
    [1, 2, 3, 1]
    >>> homogenize([1, 2, 3], 9).tolist()
    [1, 2, 3, 9]
    >>> homogenize([[1, 2], [3, 4]]).tolist()
    [[1, 2, 1], [3, 4, 1]]
    """
    v = np.asanyarray(v)
    return np.insert(v, v.shape[-1], value, axis=-1)


def dehom(a):
    """Makes homogeneous vectors inhomogenious by dividing by the last element in the last axis
    >>> dehomogenize([1, 2, 4, 2]).tolist()
    [0.5, 1.0, 2.0]
    >>> dehomogenize([[1, 2], [4, 4]]).tolist()
    [[0.5], [1.0]]
    """
    a = np.asfarray(a)
    return a[..., :-1] / a[..., None, -1]


def translate(values: np.ndarray) -> np.ndarray:
    """Construct and return a translation matrix"""
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = values
    return m


def scale(values: np.ndarray) -> np.ndarray:
    """Construct and return a scaling matrix"""
    m = np.eye(4, dtype=np.float32)
    m[[0, 1, 2], [0, 1, 2]] = values
    return m


def rotate(axis: np.ndarray, angle: float) -> np.ndarray:
    """Construct a rotation matrix around axis/angle pair."""
    axis = np.asarray(axis)
    sina = np.sin(angle)
    cosa = np.cos(angle)
    d = axis / np.linalg.norm(axis)
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(d, d) * (1.0 - cosa)
    ds = d * sina
    R += np.array(
        [
            [0.0, -ds[2], ds[1]],
            [ds[2], 0.0, -ds[0]],
            [-ds[1], ds[0], 0.0],
        ],
        dtype=axis.dtype,
    )
    m = np.eye(4, dtype=R.dtype)
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
