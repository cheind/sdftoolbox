import logging
import time
from typing import Literal

import numpy as np

from .tesselation import triangulate_quads
from .topology import VoxelTopology

_logger = logging.getLogger("surfacenets")


def compute_vertices_midpoint(
    top: VoxelTopology, active_voxels: np.ndarray, edge_coords: np.ndarray
) -> np.ndarray:
    """Computes vertex locations based on voxel centers.
    This results in Minecraft-like reconstructions.
    """
    sijk = top.unravel_nd(active_voxels, top.sample_shape)
    return sijk + np.array([[0.5, 0.5, 0.5]], dtype=np.float32)


def compute_vertices_surfacenets_naive(
    top: VoxelTopology, active_voxels: np.ndarray, edge_coords: np.ndarray
) -> np.ndarray:
    """Computes vertex locations based on averaging edge intersection points.

    Each vertex location is chosen to be the average of intersection points
    of all active edges that belong to a voxel.
    """
    active_voxel_edges = top.find_voxel_edges(active_voxels)  # (M,12)
    e = edge_coords[active_voxel_edges]  # (M,12,3)
    return np.nanmean(e, 1)


def surface_nets(
    sdf_values: np.ndarray,
    spacing: tuple[float, float, float],
    vertex_placement_mode: Literal["midpoint", "naive"] = "naive",
    triangulate: bool = False,
):
    """SurfaceNet algorithm for isosurface extraction from discrete signed
    distance fields.

    This implementation approximates the relaxation based vertex placement
    method of [1] using a `naive` [2] average. This method is fully vectorized.

    This method does not compute surface normals, see `surfacenets.normals`
    for details.

    Params:
        sdf_values: (I,J,K) array if SDF values at sample locations
        spacing: The spatial step size in each dimension
        vertex_placement_mode: Defines how vertices are placed inside of voxels. Use
            `naive` (default) for a good approximation, or `midpoint` to get Minecraft like
            box reconstructions.
        triangulate: When true, returns triangles instead of quadliterals.

    Returns:
        verts: (N,3) array of vertices
        faces: (M,F) index array of faces into vertices. For quadliterals F is 4, otherwise 3.

    References:
        1. Gibson, S. F. F. (1999). Constrained elastic surfacenets: Generating smooth models from binary segmented data.
        1. Naive SurfaceNets: https://0fps.net/2012/07/12/smooth-voxel-terrain-part-2/
    """
    t0 = time.perf_counter()
    # Sanity checks
    assert vertex_placement_mode in ["midpoint", "naive"]
    assert sdf_values.ndim == 3
    spacing = np.asarray(spacing, dtype=np.float32)

    # First, we pad the sample volume on each side with a single (nan) value to
    # avoid having to deal with most out-of-bounds issues.
    sdf_values = np.pad(
        sdf_values,
        ((1, 1), (1, 1), (1, 1)),
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

    # We construct a topology helper to deal with indices and neighborhoods.
    top = VoxelTopology(sdf_values.shape)

    edges_active_mask = np.zeros((top.num_edges,), dtype=bool)
    edges_flip_mask = np.zeros((top.num_edges,), dtype=bool)
    edges_isect_coords = np.full(
        (top.num_edges, 3), np.nan, dtype=sdf_values.dtype
    )

    # Get all possible edge source locations
    sijk = top.get_all_source_vertices()  # (N,3)
    si, sj, sk = sijk.T
    sdf_src = sdf_values[si, sj, sk]  # (N,)

    _logger.debug(
        f"After initialization; elapsed {time.perf_counter() - t0:.4f} secs"
    )

    # For each axis
    for aidx, off in enumerate(np.eye(3, dtype=np.int32)):
        # Compute the edge target locations and fetch SDF values
        tijk = sijk + off[None, :]
        ti, tj, tk = tijk.T
        sdf_dst = sdf_values[ti, tj, tk]

        # Just like in MC, we compute a parametric value t for each edge that
        # tells use where the surface boundary intersects the edge. We assume
        # the surface behaves linearily close to an edge and the edge crossing
        # value t can be approximated by linear equation. Note, active edges
        # have t value in [0,1].
        sdf_diff = sdf_dst - sdf_src
        sdf_diff[sdf_diff == 0] = 1e-8
        t = -sdf_src / sdf_diff
        active = np.logical_and(t >= 0, t < 1.0)  # t==1 is t==0 for next edge
        t[~active] = np.nan

        #
        if vertex_placement_mode == "midpoint":
            t[active] = 0.5

        active_t = t[active, None]
        isect = (1 - active_t) * sijk[active] + active_t * tijk[active]

        # We store the partial axis results in the global arrays in interleaved
        # fashion. We do this, to comply with np.unravel_index/np.ravel_multi_index
        # that are used internally by the topology module.
        edges_active_mask[aidx::3] = active
        edges_flip_mask[aidx::3][active] = sdf_diff[active] < 0.0
        edges_isect_coords[aidx::3][active] = isect

    _logger.debug(
        f"After active edges; elapsed {time.perf_counter() - t0:.4f} secs"
    )

    # 2. Step - Tesselation
    # Each active edge gives rise to a quad that is formed by the final vertices of the
    # 4 voxels sharing the edge. In this implementation we consider only those quads
    # where a full neighborhood exists - i.e non of the adjacent voxels is in the
    # padding area.
    active_edges = np.where(edges_active_mask)[0]  # (A,)
    active_quads, complete_mask = top.find_voxels_sharing_edge(
        active_edges
    )  # (A,4)
    active_edges = active_edges[complete_mask]
    active_quads = active_quads[complete_mask]
    _logger.debug(
        f"After finding quads; elapsed {time.perf_counter() - t0:.4f} secs"
    )

    # The active quad indices are are ordered ccw when looking from the positive
    # active edge direction. In case the sign difference is negative between edge
    # start and end, we need to reverse the indices to maintain a correct ccw
    # winding order.
    active_edges_flip = edges_flip_mask[active_edges]
    active_quads[active_edges_flip] = np.flip(
        active_quads[active_edges_flip], -1
    )
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
    if vertex_placement_mode == "midpoint":
        verts = compute_vertices_midpoint(
            top, active_voxels, edges_isect_coords
        )
    elif vertex_placement_mode == "naive":
        verts = compute_vertices_surfacenets_naive(
            top, active_voxels, edges_isect_coords
        )
    # Finally, we need to account for the padded voxels and scale them to
    # data dimensions
    verts = (verts - (1, 1, 1)) * spacing
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
    return verts, faces
