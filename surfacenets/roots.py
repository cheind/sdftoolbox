"""Root finding methods for SDFs"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .sdfs import SDF


def directional_newton_roots(
    node: "SDF", x: np.ndarray, dirs: np.ndarray = None, max_steps: int = 10, eps=1e-5
) -> np.ndarray:
    """Direction Netwon method for root finding of scalar valued vector functions.

    Root finding on SDFs amounts to finding any point for which f(x)=0, f: R^3->R. Note
    that standard multivariate Newton methods do not apply, as the only consider the
    case f: R^N->R^N.

    Params:
        node: SDF root node
        x: (N,3) initial locations
        dirs: (N,3) fixed directions (optional). When not given, the directions are
            chosen to be the directions of gradient estimates.
        max_steps: max number of iterations
        eps: SDF value tolerance. locations within tolerance are excluded from
            further optimization.

    See:
    Levin, Yuri, and Adi Ben-Israel.
    "Directional Newton methods in n variables."
    Mathematics of Computation 71.237 (2002): 251-262.
    """
    x = np.atleast_2d(x).copy()

    for _ in range(max_steps):
        y = node.sample(x)
        mask = np.abs(y) > eps
        if np.sum(mask) == 0:
            break
        g = node.gradient(x[mask])
        if dirs is None:
            d = g / np.linalg.norm(g, axis=-1, keepdims=True)
        else:
            d = dirs[mask]
        dot = (g[:, None, :] @ d[..., None]).squeeze(-1)
        scaled_dir = (y[mask, None] / dot) * d
        x[mask] = x[mask] - scaled_dir
    return x
