"""Signed distance function helpers.

Tools to create, manipulate and sample continuous signed distance functions in 3D.
"""
from argparse import ArgumentError
from typing import Callable, Literal
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

    def gradient(
        self,
        x: np.ndarray,
        h: float = 1e-5,
        normalize: bool = False,
        mode: Literal["central"] = "central",
    ) -> np.ndarray:
        """Returns derivatives of the SDF wrt. the input locations.

        Params:
            x: (...,3) array of sample locations
            h: step size for numeric approximation
            normalize: whether to return normalized gradients

        Returns:
            n: (...,3) array of gradients/normals
        """

        if mode == "central":
            offsets = (
                np.expand_dims(
                    np.eye(3, dtype=x.dtype),
                    np.arange(x.ndim - 1).tolist(),
                )
                * h
            )
            x = np.expand_dims(x, -2)
            fwd = self.sample(x + offsets)
            bwd = self.sample(x - offsets)

            g = (fwd - bwd) / (2 * h)
        else:
            raise ArgumentError("Unknown mode")

        if normalize:
            length = np.linalg.norm(g, axis=-1, keepdims=True)
            g = g / length
            g[~np.isfinite(g)] = 0.0

        return g

    def merge(self, *others: list["SDF"], alpha: float = np.inf) -> "Union":
        return Union([self] + list(others), alpha=alpha)

    def intersect(self, *others: list["SDF"], alpha: float = np.inf) -> "Intersection":
        return Intersection([self] + list(others), alpha=alpha)

    def subtract(self, *others: list["SDF"], alpha: float = np.inf) -> "Difference":
        return Difference([self] + list(others), alpha=alpha)


class Transform(SDF):
    """Base for nodes with transforms.

    Most of the primitives nodes are defined in terms of a canonical shape (unit sphere, xy-plane). The transform allows you to arbitrarily shift, rotate and scale them to your needs.

    When inheriting from Transform, you need to implement `sample_local` instead of `sample`.
    """

    def __init__(self, t_world_local: np.ndarray = None) -> None:
        if t_world_local is None:
            t_world_local = np.eye(4, dtype=np.float32)

        self._t_dirty = False
        self._t_scale: float = 1.0
        self.t_world_local = t_world_local

    @property
    def t_world_local(self) -> np.ndarray:
        return self._t_world_local

    @t_world_local.setter
    def t_world_local(self, m: np.ndarray):
        self._t_world_local = np.asarray(m).astype(np.float32)
        self._update_scale()
        self._t_dirty = True

    @property
    def t_local_world(self) -> np.ndarray:
        if self._t_dirty:
            self._t_local_world = np.linalg.inv(self.t_world_local)
            self._t_dirty = False
        return self._t_local_world

    def sample(self, x: np.ndarray) -> np.ndarray:
        return self.sample_local(self._to_local(x)) * self._t_scale

    def _update_scale(self):
        scales = np.linalg.norm(self._t_world_local[:3, :3], axis=0)
        if not np.allclose(scales, scales[0]):
            raise ValueError(
                "Only uniform scaling is supported. Anisotropic scaling"
                " destroys distance fields."
            )
        self._t_scale = scales[0]

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
    """(Smooth) Boolean union operation."""

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
    """(Smooth) Boolean intersection operation."""

    def __init__(self, sdfs: list[SDF], alpha: float = np.inf) -> None:
        if len(sdfs) == 0:
            raise ValueError("Need at least one SDF")
        self.children = sdfs
        self.alpha = alpha

    def sample(self, x: np.ndarray) -> np.ndarray:
        values = np.stack([c.sample(x) for c in self.children], 0)
        return maths.generalized_max(values, 0, alpha=self.alpha)


class Difference(SDF):
    """(Smooth) Boolean difference operation."""

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


class Displacement(SDF):
    """Displaces a SDF node by function modifier."""

    def __init__(self, node: SDF, dispfn: Callable[[np.ndarray], float]) -> None:
        self.dispfn = dispfn
        self.node = node

    def sample(self, x: np.ndarray) -> np.ndarray:
        node_values = self.node.sample(x)
        disp_values = self.dispfn(x)
        return node_values + disp_values


class Repetition(SDF):
    """Repeats a SDF node (in)finitely."""

    def __init__(
        self,
        node: SDF,
        periods: tuple[float, float, float] = (1, 1, 1),
        reps: tuple[int, int, int] = None,
    ) -> None:
        self.periods = np.array(periods).reshape(1, 1, 1, 3)
        self.node = node
        if reps is not None:
            self.reps = np.array(reps).reshape(1, 1, 1, 3)
            self.sample = self._repeat_finite
        else:
            self.sample = self._repeat_infinite

    def sample(self, x: np.ndarray) -> np.ndarray:
        pass  # set via __init__

    def _repeat_infinite(self, x: np.ndarray) -> np.ndarray:
        x = np.mod(x + 0.5 * self.periods, self.periods) - 0.5 * self.periods
        return self.node.sample(x)

    def _repeat_finite(self, x: np.ndarray) -> np.ndarray:
        x = x - self.periods * np.clip(np.round(x / self.periods), 0, self.reps - 1)
        return self.node.sample(x)


class Discretized(Transform):
    """Stores a discretized SDF.

    This node is useful when you wish to sample a continuous SDF for performance
    reasons.Internally, a anisotropic voxel grid is used to sample the SDF values
    and optionally gradients. For any query location, the SDF is then reconstructed
    by trilinear interpolation over the voxel grid.

    This class inherits from Transform, so you can adjust the position, orientation
    and uniform scale wrt. the SDF node to be sampled.

    Attributes:
        xyz: (I,J,K,3) array of sampling locations
        xyz_spacing: (3, ) spacing between two adjacent voxels
        xyz_sdf: (I,J,K) SDF values at sampling locations
        xyz_gradients: (I,J,K,3) optional gradients at sampling locations
    """

    def __init__(
        self,
        node: SDF,
        res: tuple[int, int, int] = (60, 60, 60),
        min_corner: tuple[float, float, float] = (-2, -2, -2),
        max_corner: tuple[float, float, float] = (2, 2, 2),
        t_world_local: np.ndarray = None,
        with_gradients: bool = False,
    ) -> None:
        """
        Params:
            node: the node representing the SDF to be discretized
            res: (3,) resolution of the voxel grid
            min_corner: (3,) minimum spatial sampling location
            max_corner: (3,) maximum spatial sampling location
            t_world_local: (4,4) optional transform
            with_gradients: Whether this class captures gradient or
                they are computed on the fly from voxel SDF values

        """
        super().__init__(t_world_local)
        self.res = np.array(res, dtype=np.int32)
        self.xyz, self.xyz_spacing = Discretized.sampling_coords(
            res, min_corner, max_corner
        )
        world_xyz = maths.dehom(maths.hom(self.xyz) @ self.t_world_local.T)
        self.xyz_sdf = node.sample(world_xyz)
        if with_gradients:
            self.xyz_gradients = node.gradient(world_xyz)
        else:
            self.xyz_gradients = None

    def sample_local(self, x: np.ndarray) -> np.ndarray:
        """Samples the discretized volume using trilinear interpolation.

        Params:
            x: (...,3) array of local coordinates

        Returns:
            sdf: (...) sdf values at given locations.
        """

        c = self._interp(self.xyz_sdf, x)
        return c.squeeze(-1)

    def gradient(
        self,
        x: np.ndarray,
        h: float = 0.00001,
        normalize: bool = False,
        mode: Literal["central"] = "central",
    ) -> np.ndarray:
        if self.xyz_gradients is not None:
            return self._interp(self.xyz_gradients, x)
        else:
            return super().gradient(x, h, normalize, mode)

    def _interp(self, vol: np.ndarray, x: np.ndarray) -> np.ndarray:
        P = x.shape[:-1]
        x = x.reshape(-1, 3)

        if vol.ndim == 3:
            vol = np.expand_dims(vol, -1)

        minc = np.expand_dims(self.xyz[0, 0, 0], 0)
        maxc = np.expand_dims(self.xyz[-1, -1, -1], 0)
        x = np.maximum(
            minc, np.minimum(maxc - 1e-8, x)
        )  # 1e-8 to always have sample point > x

        spacing = np.expand_dims(self.xyz_spacing, 0)
        xn = (x - minc) / spacing
        sijk = np.floor(xn).astype(np.int32)
        w = xn - sijk
        print(w.shape, vol.shape, sijk.shape)

        # See https://en.wikipedia.org/wiki/Trilinear_interpolation
        # i-diretion
        si, sj, sk = sijk.T
        c00 = vol[si, sj, sk] * (1 - w[..., 0:1]) + vol[si + 1, sj, sk] * w[..., 0:1]
        print((vol[si, sj, sk] * (1 - w[..., 0:1])).shape)
        c01 = (
            vol[si, sj, sk + 1] * (1 - w[..., 0:1])
            + vol[si + 1, sj, sk + 1] * w[..., 0:1]
        )
        c10 = (
            vol[si, sj + 1, sk] * (1 - w[..., 0:1])
            + vol[si + 1, sj + 1, sk] * w[..., 0:1]
        )
        c11 = (
            vol[si, sj + 1, sk + 1] * (1 - w[..., 0:1])
            + vol[si + 1, sj + 1, sk + 1] * w[..., 0:1]
        )
        # j-diretion
        c0 = c00 * (1 - w[..., 1:2]) + c10 * w[..., 1:2]
        c1 = c01 * (1 - w[..., 1:2]) + c11 * w[..., 1:2]
        # k-direction
        c = c0 * (1 - w[..., 2:3]) + c1 * w[..., 2:3]
        return c.reshape(P + (c.shape[-1],))

    @staticmethod
    def sampling_coords(
        res: tuple[int, int, int] = (60, 60, 60),
        min_corner: tuple[float, float, float] = (-1, -1, -1),
        max_corner: tuple[float, float, float] = (1, 1, 1),
        dtype=np.float32,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates volumentric sampling locations.

        Params:
            res: resolution for each axis
            min_corner: bounds for the sampling volume
            max_corner: bounds for the sampling volume

        Returns:
            xyz: (I,J,K,3) array of sampling locations
            spacing: (3,) the spatial spacing between two voxels
        """

        ranges = [
            np.linspace(min_corner[0], max_corner[0], res[0], dtype=dtype),
            np.linspace(min_corner[1], max_corner[1], res[1], dtype=dtype),
            np.linspace(min_corner[2], max_corner[2], res[2], dtype=dtype),
        ]

        X, Y, Z = np.meshgrid(*ranges, indexing="ij")
        xyz = np.stack((X, Y, Z), -1)
        spacing = np.array(
            [
                ranges[0][1] - ranges[0][0],
                ranges[1][1] - ranges[1][0],
                ranges[2][1] - ranges[2][0],
            ],
            dtype=dtype,
        )
        return xyz, spacing


class Sphere(Transform):
    """The SDF of a unit sphere

    Use the transform properties to adjust the shape and position.
    """

    def sample_local(self, x: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(x, axis=-1)
        return d - 1.0

    @staticmethod
    def create(
        center: np.ndarray = (0.0, 0.0, 0.0),
        radius: float = 1.0,
    ) -> "Sphere":
        """Creates a sphere from center and radius."""
        t = maths.translate(center) @ maths.scale(radius)
        return Sphere(t)


class Plane(Transform):
    """A plane parallel to xy-plane through origin.

    Use the transform properties to adjust the shape and position.
    """

    def sample_local(self, x: np.ndarray) -> np.ndarray:
        return x[..., -1]

    @staticmethod
    def create(
        origin: np.ndarray = (0, 0, 0), normal: np.ndarray = (0, 0, 1)
    ) -> "Plane":
        """Creates a plane from a point and normal direction."""
        normal = np.asarray(normal, dtype=np.float32)
        origin = np.asarray(origin, dtype=np.float32)
        normal /= np.linalg.norm(normal)
        # Need to find a rotation that alignes canonical frame's z-axis
        # with normal.
        z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        d = np.dot(z, normal)
        if d == 1.0:
            t = np.eye(4)
        elif d == -1.0:
            t = maths.rotate([1.0, 0.0, 0.0], np.pi)
        else:
            p = np.cross(normal, z)
            a = np.arccos(normal[-1])
            t = maths.rotate(p, -a)
        t[:3, 3] = origin
        return Plane(t)


class Box(Transform):
    """A three-dimensional box centered at origin with side-length two.

    Use the transform properties to adjust the shape and position.
    """

    def __init__(
        self,
        lengths: tuple[float, float, float] = (1.0, 1.0, 1.0),
        t_world_local: np.ndarray = None,
    ) -> None:
        super().__init__(t_world_local)
        self.half_lengths = np.asarray(lengths, dtype=np.float32) * 0.5

    def sample_local(self, x: np.ndarray) -> np.ndarray:
        a = np.abs(x) - self.half_lengths
        return np.linalg.norm(np.maximum(a, 0), axis=-1) + np.minimum(
            np.max(a, axis=-1), 0
        )

    @staticmethod
    def create(lengths: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> "Box":
        return Box(lengths)


if __name__ == "__main__":
    scene = Sphere.create()
    vol = Discretized(
        scene, res=(64, 64, 64), min_corner=(-1, -1, -1), max_corner=(1, 1, 1)
    )

    print(vol.sample(np.array([[[0.0, 0.0, 0.0]]])))
