import numpy as np
import abc
import dataclasses
from typing import Optional

from .topology import VoxelTopology
from .sdfs import SDF


@dataclasses.dataclass
class SurfaceContext:
    top: VoxelTopology
    active_voxels: np.ndarray
    edge_coords: np.ndarray
    pass


class DualVertexStrategy(abc.ABC):
    @abc.abstractmethod
    def find_vertex_locations(self, ctx: SurfaceContext) -> np.ndarray:
        pass


class MidpointStrategy(DualVertexStrategy):
    """Computes vertex locations based on voxel centers.
    This results in Minecraft-like reconstructions.
    """

    def find_vertex_locations(self, ctx: SurfaceContext) -> np.ndarray:
        sijk = ctx.top.unravel_nd(ctx.active_voxels, ctx.top.sample_shape)
        return sijk + np.array([[0.5, 0.5, 0.5]], dtype=np.float32)


class NaiveSurfaceNetStrategy(DualVertexStrategy):
    """Computes vertex locations based on averaging edge intersection points.

    Each vertex location is chosen to be the average of intersection points
    of all active edges that belong to a voxel.

    References:
    - Gibson, Sarah FF. "Constrained elastic surface nets:
    Generating smooth surfaces from binary segmented data."
    Springer, Berlin, Heidelberg, 1998.
    - https://0fps.net/2012/07/12/smooth-voxel-terrain-part-2/
    """

    def find_vertex_locations(self, ctx: SurfaceContext) -> np.ndarray:
        active_voxel_edges = ctx.top.find_voxel_edges(ctx.active_voxels)  # (M,12)
        e = ctx.edge_coords[active_voxel_edges]  # (M,12,3)
        return np.nanmean(e, 1)


class DualContouringStrategy(DualVertexStrategy):
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
        node: SDF,
        spacing: tuple[float, float, float],
        min_corner: np.ndarray,
        bias_strength: float = 1e-3,
    ):
        self.node = node
        self.bias_strength = bias_strength
        self.sqrt_bias_strength = np.sqrt(self.bias_strength)
        self.spacing = spacing
        self.min_corner = min_corner

    def find_vertex_locations(self, ctx: SurfaceContext) -> np.ndarray:
        sijk = ctx.top.unravel_nd(ctx.active_voxels, ctx.top.sample_shape)  # (M,3)
        active_voxel_edges = ctx.top.find_voxel_edges(ctx.active_voxels)  # (M,12)
        points = ctx.edge_coords[active_voxel_edges]  # (M,12,3)
        normals = self.node.gradient(self._to_data(points), normalize=True)  # (M,12,3)
        if self.bias_strength > 0:
            bias_verts = NaiveSurfaceNetStrategy().find_vertex_locations(ctx)
        else:
            bias_verts = [None] * len(points)

        # Consider a batched variant using block-diagonal matrices
        verts = []
        for off, p, n, bias in zip(sijk, points, normals, bias_verts):
            # off: (3,), p: (12,3), n: (12,3), bias: (3,)
            q = p - off[None, :]  # [0,1) range in each dim
            mask = np.isfinite(q).all(-1)  # Skip non-active voxel edges

            # Try to solve unbiased
            x = self._solve_lst(q[mask], n[mask], bias=None)
            failed = (x < 0.0).any() or (x > 1.0).any()
            if failed and bias is not None:
                # If failed, we try a biased solution
                x = self._solve_lst(q[mask], n[mask], bias=(bias - off))
            x = np.clip(x, 0.0, 1.0)
            verts.append(x + off)
        return np.array(verts, dtype=np.float32)

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
        x, res, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        return x

    def _to_data(self, x: np.ndarray) -> np.ndarray:
        return (x - (1, 1, 1)) * self.spacing + self.min_corner