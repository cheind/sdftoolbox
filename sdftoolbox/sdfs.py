"""Signed distance function helpers.

Tools to create, manipulate and sample continuous signed distance functions in 3D.
"""
from argparse import ArgumentError
from typing import Callable, Literal
import numpy as np
import abc

from . import maths
from .types import float_dtype
from .grid import Grid


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
        h: float = 1e-8,
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
                * 0.5
            )
            x = np.expand_dims(x, -2)
            fwd = self.sample(x + offsets)
            bwd = self.sample(x - offsets)

            g = (fwd - bwd) / h
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

    def transform(
        self,
        trans: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        scale: float = 1.0,
    ) -> "Transform":
        return Transform(self, Transform.create_transform(trans, rot, scale))


class Transform(SDF):
    """Base for nodes with transforms.

    Most of the primitives nodes are defined in terms of a canonical shape
    (unit sphere, xy-plane). The transform allows you to shift,
    rotate and isotropically scale them to your needs.
    """

    def __init__(self, node: SDF, t_world_local: np.ndarray = None) -> None:
        if t_world_local is None:
            t_world_local = np.eye(4, dtype=float_dtype)

        self._t_scale: float = 1.0
        self.node = node
        self.t_world_local = t_world_local

    @property
    def t_world_local(self) -> np.ndarray:
        return self._t_world_local

    @t_world_local.setter
    def t_world_local(self, m: np.ndarray):
        self._t_world_local = np.asfarray(m, dtype=float_dtype)
        self._update_scale()
        self._update_inv()

    @property
    def t_local_world(self) -> np.ndarray:
        return self._t_local_world

    def sample(self, x: np.ndarray) -> np.ndarray:
        return self.node.sample(self._to_local(x)) * self._t_scale

    def _update_inv(self):
        self._t_local_world = np.linalg.inv(self.t_world_local)

    def _update_scale(self):
        scales = np.linalg.norm(self._t_world_local[:3, :3], axis=0)
        if not np.allclose(scales, scales[0]):
            raise ValueError(
                "Only uniform scaling is supported. Anisotropic scaling"
                " destroys distance fields."
            )
        self._t_scale = scales[0]

    def _to_local(self, x: np.ndarray) -> np.ndarray:
        return maths.dehom(maths.hom(x) @ self.t_local_world.T)

    def transform(
        self,
        trans: tuple[float, float, float] = (0, 0, 0),
        rot: tuple[float, float, float, float] = (1, 0, 0, 0),
        scale: float = 1,
    ) -> "Transform":
        t = Transform.create_transform(trans, rot, scale)
        self.t_world_local = t @ self.t_world_local
        return self

    @staticmethod
    def create_transform(
        trans=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), scale: float = 1.0
    ) -> np.ndarray:
        t = (
            maths.translate(trans)
            @ maths.rotate(rot[:3], rot[-1])
            @ maths.scale((scale, scale, scale))
        )
        return t


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

    def __init__(self, node: SDF, dispfn: Callable[[np.ndarray], np.ndarray]) -> None:
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
        self.periods = np.asfarray(periods, dtype=float_dtype).reshape(1, 1, 1, 3)
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


class Discretized(SDF):
    """Stores a discretized SDF.

    For any query location, the SDF is then reconstructed
    by trilinear interpolation over the voxel grid. This class inherits
    from Transform, so you can adjust the position, orientation and
    uniform scale wrt. the SDF node to be sampled.

    Attributes:
        grid: Grid (I,J,K) holds grid sampling locations
        sdf_values: (I,J,K) SDF values at sampling locations
    """

    def __init__(
        self,
        grid: Grid,
        sdf_values: np.ndarray,
    ) -> None:
        """
        Params:
            grid: local sampling coordinates
            sdf_valus: (I,J,K) signed distance values
        """
        self.grid = grid
        self.sdf_values = sdf_values

    def sample(self, x: np.ndarray) -> np.ndarray:
        """Samples the discretized volume using trilinear interpolation.

        Params:
            x: (...,3) array of local coordinates

        Returns:
            sdf: (...) sdf values at given locations.
        """

        c = self._interp(self.sdf_values, x)
        return c.squeeze(-1)

    def _interp(self, vol: np.ndarray, x: np.ndarray) -> np.ndarray:
        P = x.shape[:-1]
        x = x.reshape(-1, 3)

        if vol.ndim == 3:
            vol = np.expand_dims(vol, -1)

        minc = np.expand_dims(self.grid.min_corner, 0)
        maxc = np.expand_dims(self.grid.max_corner, 0)
        x = np.maximum(
            minc, np.minimum(maxc - 1e-8, x)
        )  # 1e-8 to always have sample point > x

        spacing = np.expand_dims(self.grid.spacing, 0)
        xn = (x - minc) / spacing
        sijk = np.floor(xn).astype(np.int32)
        w = xn - sijk

        # See https://en.wikipedia.org/wiki/Trilinear_interpolation
        # i-diretion
        si, sj, sk = sijk.T
        c00 = vol[si, sj, sk] * (1 - w[..., 0:1]) + vol[si + 1, sj, sk] * w[..., 0:1]
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


class Sphere(SDF):
    """The SDF of a unit sphere."""

    def sample(self, x: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(x, axis=-1)
        return d - 1.0

    @staticmethod
    def create(
        center: np.ndarray = (0.0, 0.0, 0.0),
        radius: float = 1.0,
    ) -> Transform:
        """Creates a sphere from center and radius."""
        t = maths.translate(center) @ maths.scale(radius)
        return Transform(Sphere(), t_world_local=t)


class Plane(SDF):
    """A plane parallel to xy-plane through origin."""

    def sample(self, x: np.ndarray) -> np.ndarray:
        return x[..., -1]

    @staticmethod
    def create(
        origin: tuple[float, float, float] = (0, 0, 0),
        normal: tuple[float, float, float] = (0, 0, 1),
    ) -> Transform:
        """Creates a plane from a point and normal direction."""
        normal = np.asfarray(normal, dtype=float_dtype)
        origin = np.asfarray(origin, dtype=float_dtype)
        normal /= np.linalg.norm(normal)
        # Need to find a rotation that alignes canonical frame's z-axis
        # with normal.
        z = np.array([0.0, 0.0, 1.0], dtype=normal.dtype)
        d = np.dot(z, normal)
        if d == 1.0:
            t = np.eye(4, dtype=normal.dtype)
        elif d == -1.0:
            t = maths.rotate([1.0, 0.0, 0.0], np.pi)
        else:
            p = np.cross(normal, z)
            a = np.arccos(normal[-1])
            t = maths.rotate(p, -a)
        t[:3, 3] = origin
        return Transform(Plane(), t_world_local=t)


class Box(SDF):
    """An axis aligned bounding box centered at origin."""

    def __init__(
        self,
        lengths: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.half_lengths = np.asfarray(lengths, dtype=float_dtype) * 0.5

    @staticmethod
    def create(lengths: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> "Box":
        """Creates a box from given lengths."""
        return Box(lengths)

    def sample(self, x: np.ndarray) -> np.ndarray:
        a = np.abs(x) - self.half_lengths
        return np.linalg.norm(np.maximum(a, 0), axis=-1) + np.minimum(
            np.max(a, axis=-1), 0
        )
