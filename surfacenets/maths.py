from turtle import Shape
import numpy as np
from typing import TypeVar, Any

ShapeLike = TypeVar("ShapeLike", bound=Any)


def generalized_max(x: np.ndarray, axis: ShapeLike = None, alpha=np.inf):
    if alpha < np.inf:
        xmax = np.max(x, axis=axis, keepdims=True)
        ex = np.exp(alpha * (x - xmax))
        smax = np.sum(x * ex, axis=axis) / np.sum(ex, axis=axis)
        return smax
    else:
        return np.max(x, axis=axis)


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
