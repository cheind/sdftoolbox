import logging
import time
from typing import TYPE_CHECKING, Union
from dataclasses import dataclass

import numpy as np

from .dual_strategies import LinearEdgeStrategy, NaiveSurfaceNetVertexStrategy
from .mesh import triangulate_quads

if TYPE_CHECKING:
    from .dual_strategies import DualEdgeStrategy, DualVertexStrategy
    from .grid import Grid
    from .sdfs import SDF


_logger = logging.getLogger("surfacenets")


@dataclass
class DebugInfo:
    edges_active_mask: np.ndarray
    edges_isect_coords: np.ndarray


def dual_isosurface(
    node: "SDF",
    grid: "Grid",
    edge_strategy: "DualEdgeStrategy" = None,
    vertex_strategy: "DualVertexStrategy" = None,
    triangulate: bool = False,
    return_debug_info: bool = False,
    vertex_relaxation_percent: float = 0.1,
) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, DebugInfo]]:
    """A vectorized dual iso-surface extraction algorithm for signed distance fields.

    This implementation approximates the relaxation based vertex placement
    method of [1] using a `naive` [2] average. This method is fully vectorized.

    This method does not compute surface normals, see `surfacenets.normals`
    for details.

    Params:
        node: the root node of the SDF. If you already have discretized SDF values in
            grid like fashion, wrap them using sdfs.Discretized.
        grid: (I,J,K) spatial sampling locations
        edge_strategy: Defines how edge/surface boundary intersection are determined.
            If not specified defaults to LinearEdgeStrategy.
        vertex_strategy: Defines how vertices are placed inside of voxels. If not
            specified defaults to NaiveSurfaceNetVertexStrategy.
        triangulate: When true, returns triangles instead of quadliterals.
        vertex_relaxation_percent: Edge intersection values outside of [0,1) will
            be tolerated up to this percentage. Increasing this value allows for
            more accurate shapes when the resolution of the grid is low and multiple
            vertices per cell are required.
        return_debug_info: Whether to return additional intermediate results

    Returns:
        verts: (N,3) array of vertices
        faces: (M,F) index array of faces into vertices. For quadliterals F is 4,
            otherwise 3.
        debug: instance of DebugInfo when return_debug_info is True.

    References:
        Gibson, S. F. F. (1999). Constrained elastic surfacenets: Generating smooth
            models from binary segmented data.
    """
    t0 = time.perf_counter()
    # Defaults
    if vertex_strategy is None:
        vertex_strategy = NaiveSurfaceNetVertexStrategy()
    if edge_strategy is None:
        edge_strategy = LinearEdgeStrategy()

    # First, we pad the sample volume on each outer boundary single (nan) value to
    # avoid having to deal with most out-of-bounds issues.
    padded_sdf_values = np.pad(
        node.sample(grid.xyz),
        ((0, 1), (0, 1), (0, 1)),
        mode="constant",
        constant_values=np.nan,
    )
    _logger.debug(f"After padding; elapsed {time.perf_counter() - t0:.4f} secs")

    # 1. Step - Active Edges
    # We find the set of active edges that cross the surface boundary. Note,
    # edges are defined in terms of originating voxel index + a number {0,1,2}
    # that defines the positive canonical edge direction. Only considering forward edges
    # has the advantage of not having to deal with no duplicate edges. Also, through
    # padding we can ensure that no edges will be missed on the volume boundary.
    # In the code below, we perform one iteration per axis. Compared to full
    # vectorization, this avoids fancy indexing the SDF volume and has performance
    # advantages.

    edges_active_mask = np.zeros((grid.num_edges,), dtype=bool)
    edges_flip_mask = np.zeros((grid.num_edges,), dtype=bool)
    edges_isect_coords = np.full(
        (grid.num_edges, 3), np.nan, dtype=padded_sdf_values.dtype
    )

    # Get all possible edge source locations
    sijk = grid.get_all_source_vertices()  # (N,3)
    si, sj, sk = sijk.T
    sdf_src = padded_sdf_values[si, sj, sk]  # (N,)

    _logger.debug(f"After initialization; elapsed {time.perf_counter() - t0:.4f} secs")

    # For each axis
    for aidx, off in enumerate(np.eye(3, dtype=np.int32)):
        # Compute the edge target locations and fetch SDF values
        tijk = sijk + off[None, :]
        ti, tj, tk = tijk.T
        sdf_dst = padded_sdf_values[ti, tj, tk]

        # By intermediate value theorem for continuous functions if the sign of src
        # and dst is different, there must be a root enclosed. We also avoid edges
        # with NaNs, that might occur at boundaries as induced by potential SDF padding.
        src_sign = np.sign(sdf_src)
        dst_sign = np.sign(sdf_dst)
        active = np.logical_and(src_sign != dst_sign, np.isfinite(sdf_dst))

        # Just like in MC, we compute a parametric value t for each edge that
        # tells use where the surface boundary intersects the edge.
        t = edge_strategy.find_edge_intersections(
            sijk[active],
            sdf_src[active],
            tijk[active],
            sdf_dst[active],
            aidx,
            off,
            node,
            grid,
        )
        # Compute the floating point grid coords of intersection
        isect_coords = sijk[active] + off[None, :] * t[:, None]
        need_flip = (sdf_dst[active] - sdf_src[active]) < 0.0

        # We store the partial axis results in the global arrays in interleaved
        # fashion. We do this, to comply with np.unravel_index/np.ravel_multi_index
        # that are used internally by the grid module.
        edges_active_mask[aidx::3] = active
        edges_flip_mask[aidx::3][active] = need_flip
        edges_isect_coords[aidx::3][active] = isect_coords

    _logger.debug(f"After active edges; elapsed {time.perf_counter() - t0:.4f} secs")

    # 2. Step - Tesselation
    # Each active edge gives rise to a quad that is formed by the final vertices of the
    # 4 voxels sharing the edge. In this implementation we consider only those quads
    # where a full neighborhood exists - i.e non of the adjacent voxels is in the
    # padding area.
    active_edges = np.where(edges_active_mask)[0]  # (A,)
    active_quads, complete_mask = grid.find_voxels_sharing_edge(active_edges)  # (A,4)
    active_edges = active_edges[complete_mask]
    active_quads = active_quads[complete_mask]
    _logger.debug(f"After finding quads; elapsed {time.perf_counter() - t0:.4f} secs")

    # The active quad indices are are ordered ccw when looking from the positive
    # active edge direction. In case the sign difference is negative between edge
    # start and end, we need to reverse the indices to maintain a correct ccw
    # winding order.
    active_edges_flip = edges_flip_mask[active_edges]
    active_quads[active_edges_flip] = np.flip(active_quads[active_edges_flip], -1)
    _logger.debug(
        f"After correcting quads; elapsed {time.perf_counter() - t0:.4f} secs"
    )

    # Voxel indices are not unique, since any active voxel will be part in more than one
    # quad. However, each active voxel should give rise to only one vertex. To
    # avoid duplicate computations, we compute the set of unique active voxels. Bonus:
    # the inverse array is already the flattened final face array.
    active_voxels, faces = np.unique(active_quads, return_inverse=True)  # (M,)

    # Step 3. Vertex locations
    # For each active voxel, we need to find one vertex location. The
    # method todo that depennds on `vertex_placement_mode`. No matter which method
    # is selected, we expect the returned coordinates to be in voxel space.
    grid_verts = vertex_strategy.find_vertex_locations(
        active_voxels, edges_isect_coords, node, grid
    )
    # Clip vertices to voxel bounds allowing for a relaxation tolerance.
    grid_ijk = grid.unravel_nd(active_voxels, grid.padded_shape)
    grid_verts = (
        np.clip(
            grid_verts - grid_ijk,
            0.0 - vertex_relaxation_percent,
            1.0 + vertex_relaxation_percent,
        )
        + grid_ijk
    )

    # Finally, we need to account for the padded voxels and scale them to
    # data dimensions
    verts = grid.grid_to_data(grid_verts)
    _logger.debug(
        f"After vertex computation; elapsed {time.perf_counter() - t0:.4f} secs"
    )

    # 4. Step - Postprocessing
    # In case triangulation is required, we simply split each quad into two
    # triangles. Since the vertex order in faces is ccw, that's easy too.
    faces = faces.reshape(-1, 4)
    if triangulate:
        faces = triangulate_quads(faces)
        _logger.debug(
            f"After triangulation; elapsed {time.perf_counter() - t0:.4f} secs"
        )
    _logger.info(f"Finished after {time.perf_counter() - t0:.4f} secs")
    _logger.info(f"Found {len(verts)} vertices and {len(faces)} faces")

    if return_debug_info:
        return verts, faces, DebugInfo(edges_active_mask, edges_isect_coords)
    else:
        return verts, faces
