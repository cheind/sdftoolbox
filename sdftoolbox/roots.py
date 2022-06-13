"""Root finding methods for SDFs"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .sdfs import SDF


def directional_newton_roots(
    node: "SDF", x: np.ndarray, dirs: np.ndarray = None, max_steps: int = 10, eps=1e-8
) -> np.ndarray:
    """Direction Netwon method for root finding of scalar valued vector functions.

    Root finding on SDFs amounts to finding any point for which f(x)=0, f: R^3->R. Note
    that standard multivariate Newton methods do not apply, as the only consider the
    case f: R^N->R^N.

    Note, if the directional derivative has zero length no progress is made. Keep in mind,
    for example when optimizing around the SDF of a box.

    Params:
        node: SDF root node
        x: (N,3) initial locations
        dirs: (N,3) or (3,) fixed directions (optional). When not given, the directions are
            chosen to be the directions of gradient estimates. When (3,) the same constant
            direction for all locations is assumed.
        max_steps: max number of iterations
        eps: SDF value tolerance. locations within tolerance are excluded from
            further optimization.

    Returns:
        x: (N,3) optimized locations

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
        elif dirs.ndim == 1:
            d = np.expand_dims(dirs, 0)
        else:
            d = dirs[mask]
        dot = (g[:, None, :] @ d[..., None]).squeeze(-1)
        noinfo = dot == 0.0  # gradient is orthogonal to direction
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = y[mask, None] / dot
            scale[noinfo] = 0
        x[mask] = x[mask] - scale * d
    return x


def bisect_roots(
    node: "SDF",
    a: np.ndarray,
    b: np.ndarray,
    x: np.ndarray = None,
    max_steps: int = 10,
    eps=1e-8,
    linear_interp: bool = False,
) -> np.ndarray:
    """Bisect method for root finding of a SDF.

    This method makes progress even in the case when the directional derivative along
    the line a/b is zero. Its similar to the effect of decreasing the edge length in
    grid sampling.

    For convenience, this method also takes an initial starting value array `x`, which
    is usually superfluous in bisection, but useful in our case.

    Params:
        node: SDF root node
        x: (N,3) initial locations
        a: (N, 3) endpoints of line segments
        b: (N, 3) endpoints of line segments
        max_steps: max number of iterations
        eps: SDF value tolerance. locations within SDF tolerance are excluded
            from further optimization.

    Returns:
        x: (N,3) optimized locations
    """

    def _select(a, b, x, sdf_a, sdf_b, sdf_x):
        new_a = a.copy()
        new_sdf_a = sdf_a.copy()
        mask = np.sign(sdf_a) == np.sign(sdf_x)
        new_a[mask] = b[mask]
        new_sdf_a[mask] = sdf_b[mask]

        return new_a, x, new_sdf_a, sdf_x

    # Initial segment choice
    sdf_a = node.sample(a)
    sdf_b = node.sample(b)

    if x is not None:
        sdf_x = node.sample(x)
        a, b, sdf_a, sdf_b = _select(a, b, x, sdf_a, sdf_b, sdf_x)

    for _ in range(max_steps):
        if not linear_interp:
            x = (a + b) * 0.5
        else:
            d = sdf_b - sdf_a
            d[d == 0.0] = 1e-8
            t = -sdf_a / d
            x = (1 - t[:, None]) * a + t[:, None] * b
        sdf_x = node.sample(x)

        if np.sum(np.abs(sdf_x) > eps) == 0:
            break
        a, b, sdf_a, sdf_b = _select(a, b, x, sdf_a, sdf_b, sdf_x)

    return x
