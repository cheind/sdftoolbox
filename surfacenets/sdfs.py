"""Signed distance function helpers.

Tools to create, manipulate and sample continuous signed distance functions in 3D.
"""
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
        mode: Literal["forward", "complex"] = "forward",
    ) -> np.ndarray:
        """Returns derivatives of the SDF wrt. the input locations.

        Params:
            x: (N,3) array of sample locations
            h: step size for numeric approximation
            mode: method to use for computation

        Returns:
            n: (N,3) array of gradients
        """

        pass

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
        self.t_world_local = t_world_local

    @property
    def t_world_local(self) -> np.ndarray:
        return self._t_world_local

    @t_world_local.setter
    def t_world_local(self, m: np.ndarray):
        self._t_world_local = np.asarray(m).astype(np.float32)
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


class Sphere(Transform):
    """The SDF of a unit sphere

    Use the transform properties to adjust the shape and position.
    """

    def sample_local(self, x: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(x, axis=-1, ord=2)
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

    def sample_local(self, x: np.ndarray) -> np.ndarray:
        a = np.abs(x) - 1
        return np.linalg.norm(np.maximum(a, 0), axis=-1) + np.minimum(
            np.maximum(a[..., 0], np.maximum(a[..., 1], a[..., 2])), 0
        )

    @staticmethod
    def create(lengths: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> "Box":
        s = np.asarray(lengths, dtype=np.float32)
        s = s * 0.5
        return Box(maths.scale(s))
