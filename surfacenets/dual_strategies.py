import numpy as np
import abc
from typing import Literal, Optional, TYPE_CHECKING, Protocol

from .types import float_dtype

if TYPE_CHECKING:
    from .grid import Grid
    from .sdfs import SDF


class DualVertexStrategy(abc.ABC):
    """Base class for methods that compute vertex locations.

    In dual methods one vertex per active voxel is required.
    A voxel is active if at least one of 12 edges belonging
    to the voxel is intersected by the surface boundary.
    """

    @abc.abstractmethod
    def find_vertex_locations(
        self,
        active_voxels: np.ndarray,
        edge_coords: np.ndarray,
        node: "SDF",
        grid: "Grid",
    ) -> np.ndarray:
        pass


class MidpointVertexStrategy(DualVertexStrategy):
    """Computes vertex locations based on voxel centers.
    This results in Minecraft-like reconstructions.
    """

    def find_vertex_locations(
        self,
        active_voxels: np.ndarray,
        edge_coords: np.ndarray,
        node: "SDF",
        grid: "Grid",
    ) -> np.ndarray:
        del node, edge_coords
        sijk = grid.unravel_nd(active_voxels, grid.padded_shape)
        return sijk + np.array([[0.5, 0.5, 0.5]], dtype=float_dtype)


class NaiveSurfaceNetVertexStrategy(DualVertexStrategy):
    """Computes vertex locations based on averaging edge intersection points.

    Each vertex location is chosen to be the average of intersection points
    of all active edges that belong to a voxel.

    References:
    - Gibson, Sarah FF. "Constrained elastic surface nets:
    Generating smooth surfaces from binary segmented data."
    Springer, Berlin, Heidelberg, 1998.
    - https://0fps.net/2012/07/12/smooth-voxel-terrain-part-2/
    """

    def find_vertex_locations(
        self,
        active_voxels: np.ndarray,
        edge_coords: np.ndarray,
        node: "SDF",
        grid: "Grid",
    ) -> np.ndarray:
        del node
        active_voxel_edges = grid.find_voxel_edges(active_voxels)  # (M,12)
        e = edge_coords[active_voxel_edges]  # (M,12,3)
        return np.nanmean(e, 1)  # (M,3)


class DualContouringVertexStrategy(DualVertexStrategy):
    """Computes vertex locations based on dual-contouring strategy.

    This method additionally requires intersection normals. The idea
    is to find (for each voxel) the location that agrees 'best' with
    all intersection points/normals from from surrounding active edges.

    Each interseciton point/normal consitutes a plane in 3d. A location
    agrees with it when it lies on (close-to) the plane. A location
    that agrees with all planes is considered the best.

    In practice, this can be solved by linear least squares. Consider
    an isect location p and isect normal n. Then, we'd like to find
    x, such that

        n^T(x-p) = 0

    which we can rearrange as follows

        n^Tx -n^Tp = 0
        n^Tx = n^Tp

    and concatenate (for multiple p, n pairs as) into

        Ax = b

    which we solve using least squares. However, one need to ensures that
    resulting locations lie within voxels. We do this by biasing the
    linear system to the naive SurfaceNets solution using additional
    equations of the form

        ei^Tx = bias[0]
        ej^Tx = bias[1]
        ek^Tx = bias[2]

    wher e(i,j,k) are the canonical unit vectors.

    References:
    - Ju, Tao, et al. "Dual contouring of hermite data."
    Proceedings of the 29th annual conference on Computer
    graphics and interactive techniques. 2002.
    """

    def __init__(
        self,
        bias_mode: Literal["always", "failed", "disabled"] = "always",
        bias_strength: float = 1e-3,
    ):
        assert bias_mode in ["always", "failed", "disabled"]
        self.bias_strength = bias_strength
        self.sqrt_bias_strength = np.sqrt(self.bias_strength)
        self.bias_mode = bias_mode

    def find_vertex_locations(
        self,
        active_voxels: np.ndarray,
        edge_coords: np.ndarray,
        node: "SDF",
        grid: "Grid",
    ) -> np.ndarray:
        sijk = grid.unravel_nd(active_voxels, grid.padded_shape)  # (M,3)
        active_voxel_edges = grid.find_voxel_edges(active_voxels)  # (M,12)
        points = edge_coords[active_voxel_edges]  # (M,12,3)
        normals = node.gradient(grid.grid_to_data(points))  # (M,12,3)
        if self.bias_mode != "disabled":
            bias_verts = NaiveSurfaceNetVertexStrategy().find_vertex_locations(
                active_voxels, edge_coords, node, grid
            )
        else:
            bias_verts = [None] * len(points)

        # Consider a batched variant using block-diagonal matrices
        verts = []
        bias_always = self.bias_mode == "always"
        bias_failed = self.bias_mode == "failed"
        for off, p, n, bias in zip(sijk, points, normals, bias_verts):
            # off: (3,), p: (12,3), n: (12,3), bias: (3,)
            q = p - off[None, :]  # [0,1) range in each dim
            mask = np.isfinite(q).all(-1)  # Skip non-active voxel edges

            # Try to solve unbiased
            x = self._solve_lst(
                q[mask], n[mask], bias=(bias - off) if bias_always else None
            )
            if bias_failed and ((x < 0.0).any() or (x > 1.0).any()):
                x = self._solve_lst(q[mask], n[mask], bias=(bias - off))
            x = np.clip(x, 0.0, 1.0)
            verts.append(x + off)
        return np.array(verts, dtype=float_dtype)

    def _solve_lst(
        self, q: np.ndarray, n: np.ndarray, bias: Optional[np.ndarray]
    ) -> np.ndarray:
        A = n
        b = (q[:, None, :] @ n[..., None]).reshape(-1)
        if bias is not None and self.bias_strength > 0.0:
            C = np.eye(3, dtype=A.dtype) * np.sqrt(self.bias_strength)
            d = bias * np.sqrt(self.bias_strength)
            A = np.concatenate((A, C), 0)
            b = np.concatenate((b, d), 0)
        x, res, rank, _ = np.linalg.lstsq(A.astype(float), b.astype(float), rcond=None)
        return x.astype(q.dtype)
