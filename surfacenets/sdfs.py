"""Signed distance function helpers.

Tools to create, manipulate and sample continuous signed distance functions in 3D. 
"""

import numpy as np
import abc

from . import maths


class SDF(abc.ABC):
    """Abstract base for a node in the signed distance function graph."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sample(x)

    @abc.abstractmethod
    def sample(self, x: np.ndarray) -> np.ndarray:
        """Samples the SDF at locations `x`.

        Params
            x: (...,3) array of sampling locations

        Returns
            v: (...) array of SDF values.
        """
        ...

    def merge(self, *others: list["SDF"], alpha: float = np.inf) -> "Union":
        return Union([self] + list(others), alpha=alpha)

    def intersect(self, *others: list["SDF"], alpha: float = np.inf) -> "Intersection":
        return Intersection([self] + list(others), alpha=alpha)

    def subtract(self, *others: list["SDF"], alpha: float = np.inf) -> "Difference":
        return Difference([self] + list(others), alpha=alpha)


class Transform(SDF):
    """Base for nodes with transforms"""

    def __init__(self, t_world_local: np.ndarray = None) -> None:
        if t_world_local is None:
            t_world_local = np.eye(4, dtype=np.float32)
        self._t_world_local = t_world_local
        self._t_local_world = np.linalg.inv(t_world_local)
        self._t_dirty = False

    @property
    def t_world_local(self) -> np.ndarray:
        return self._t_world_local

    @t_world_local.setter
    def t_world_local(self, m: np.ndarray):
        self._t_world_local = m
        self._t_dirty = True

    @property
    def t_local_world(self) -> np.ndarray:
        if self._t_dirty:
            self._t_local_world = np.linalg.inv(self.t_world_local)
            self._t_dirty = False
        return self._t_local_world

    def sample(self, x: np.ndarray) -> np.ndarray:
        return self.sample_local(self._to_local(x))

    @abc.abstractmethod
    def sample_local(self, x: np.ndarray) -> np.ndarray:
        """Samples the SDF at locations `x` in local space.

        Params
            x: (...,3) array of sampling locations

        Returns
            v: (...) array of SDF values.
        """
        ...

    def _to_local(self, x: np.ndarray) -> np.ndarray:
        return maths.dehom(maths.hom(x) @ self.t_local_world.T)


class Union(SDF):
    """Boolean union operation."""

    def __init__(self, sdfs: list[SDF], alpha: float = np.inf) -> None:
        if len(sdfs) == 0:
            raise ValueError("Need at least one SDF")
        self.children = sdfs
        self.alpha = alpha

    def sample(self, x: np.ndarray) -> np.ndarray:
        values = np.stack([c.sample(x) for c in self.children], 0)
        # min = -max(-values)
        return -maths.generalized_max(-values, 0, alpha=self.alpha)


class Intersection(SDF):
    """Boolean intersection operation."""

    def __init__(self, sdfs: list[SDF], alpha: float = np.inf) -> None:
        if len(sdfs) == 0:
            raise ValueError("Need at least one SDF")
        self.children = sdfs
        self.alpha = alpha

    def sample(self, x: np.ndarray) -> np.ndarray:
        values = np.stack([c.sample(x) for c in self.children], 0)
        return maths.generalized_max(values, 0, alpha=self.alpha)


class Difference(SDF):
    """Boolean difference operation."""

    def __init__(self, sdfs: list[SDF], alpha: float = np.inf) -> None:
        if len(sdfs) == 0:
            raise ValueError("Need at least one SDF")
        self.children = sdfs
        self.alpha = alpha

    def sample(self, x: np.ndarray) -> np.ndarray:
        values = np.stack([c.sample(x) for c in self.children], 0)
        if values.shape[0] > 1:
            values[1:] *= -1
        return maths.generalized_max(values, 0, alpha=self.alpha)


class Sphere(Transform):
    """The SDF of a sphere"""

    def __init__(self, center: np.ndarray, radius: float) -> None:
        super().__init__(maths.translate(center) @ maths.scale(radius))

    def sample_local(self, x: np.ndarray) -> np.ndarray:
        d2 = np.square(x).sum(-1)
        return d2 - 1.0


class Plane(SDF):
    """The SDF of plane"""

    def __init__(self, origin: np.ndarray = None, normal: np.ndarray = None) -> None:
        if origin is None:
            origin = np.zeros(3, dtype=np.float32)
        if normal is None:
            normal = np.array((0.0, 1.0, 0.0), dtype=np.float32)
        normal /= np.linalg.norm(normal, ord=2)
        self.origin = np.asarray(origin, dtype=np.float32)
        self.normal = np.asarray(normal, dtype=np.float32)

    def sample(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return np.dot(x - self.origin[..., :], self.normal)


if __name__ == "__main__":
    # https://0fps.net/2012/07/12/smooth-voxel-terrain-part-2/
    from skimage.measure import marching_cubes
    import matplotlib.pyplot as plt
    import time

    res = (40, 40, 40)
    min_corner = np.array([-2.0] * 3, dtype=np.float32)
    max_corner = np.array([2.0] * 3, dtype=np.float32)

    ranges = [
        np.linspace(min_corner[0], max_corner[0], res[0], dtype=np.float32),
        np.linspace(min_corner[1], max_corner[1], res[1], dtype=np.float32),
        np.linspace(min_corner[2], max_corner[2], res[2], dtype=np.float32),
    ]

    X, Y, Z = np.meshgrid(*ranges)
    xyz = np.stack((X, Y, Z), -1)
    spacing = (
        ranges[0][1] - ranges[0][0],
        ranges[1][1] - ranges[1][0],
        ranges[2][1] - ranges[2][0],
    )

    s1 = Sphere([0.0, 0.0, 0.0], 1.0)
    s2 = Sphere([1.0, 0.0, 0.0], 1.0)
    p1 = Plane(origin=[0.0, 0.0, -0.8], normal=(0.0, 0.0, 1.0))
    sdf = p1.merge(s1.subtract(s2, alpha=4), alpha=4)
    t0 = time.perf_counter()
    values = sdf(xyz)
    print("Eval SDF took", time.perf_counter() - t0, "secs")

    t0 = time.perf_counter()
    verts, faces, normals, _ = marching_cubes(values, 0.0, spacing=spacing)
    verts += min_corner[None, :]
    print("MC took", time.perf_counter() - t0, "secs")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        faces,
        verts[:, 2],
        cmap="Spectral",
        antialiased=True,
        linewidth=0,
    )
    ax.set_xlim(min_corner[0], max_corner[0])
    ax.set_ylim(min_corner[1], max_corner[1])
    ax.set_zlim(min_corner[2], max_corner[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect(
        (
            max_corner[0] - min_corner[0],
            max_corner[1] - min_corner[1],
            max_corner[2] - min_corner[2],
        )
    )

    plt.show()
