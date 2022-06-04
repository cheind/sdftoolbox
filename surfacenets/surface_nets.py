from typing import Literal

import numpy as np

from .topology import VoxelTopology


def surface_nets(
    sdf_values: np.ndarray,
    spacing: tuple[float, float, float],
    vertex_placement_mode: Literal["midpoint", "naive"] = "naive",
    triangulate: bool = False,
):
    """SurfaceNet algorithm for isosurface extraction from discrete signed distance fields.

    This implementation approximates the relaxation based vertex placement method of [1] using a `naive` [2] average. This method is fully vectorized.

    This method does not compute surface normals. Use `surfacenets.normals` instead.

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

    # We construct a topology helper to deal with indices and neighborhoods.
    top = VoxelTopology(sdf_values.shape)

    # 1. Step - Active Edges
    # We find the set of active edges that cross the surface boundary. Note,
    # edges are defined in terms of originating voxel index + a number {0,1,2}
    # that defines the positive canonical edge direction. Hence, no duplicate edges
    # are generated and no missing edges (as long as we pad the positive axes with
    # a single element as done above).

    # Get all edge endpoints in terms of voxel coords.
    sijk, tijk = top.find_edge_vertices(range(top.num_edges), ravel=False)
    si, sj, sk = sijk.T
    ti, tj, tk = tijk.T

    # Just like in MC, we compute a parametric value t for each edge that
    # tells use where the surface boundary intersects the edge. We assume
    # the surface behaves linearily close to an edge and the edge crossing
    # value t can be approximated by linear equation. Note, active edges
    # have t value in [0,1].
    with np.errstate(divide="ignore", invalid="ignore"):
        sdf_diff = sdf_values[ti, tj, tk] - sdf_values[si, sj, sk]
        t = -sdf_values[si, sj, sk] / sdf_diff
    active_edge_mask = np.logical_and(t >= 0, t <= 1.0)

    # Vertex placements are chosen by averaging intersection points
    # of active edges belonging to a voxel. In case we wish to get Minecraft
    # like results, we can simply set the the intersection to the midpoint
    # of an edge.
    if vertex_placement_mode == "midpoint":
        t[:] = 0.5
    t[~active_edge_mask] = np.nan

    # Compute the edge intersection points for all edges (also non-active
    # ones to avoid index headaches.)
    edge_isect = (1 - t[:, None]) * sijk + t[:, None] * tijk

    active_edges = np.where(active_edge_mask)[0]  # (A,)

    # 2. Step - Tesselation
    # Each active edge gives rise to a quad that is formed by the final vertices of the
    # 4 voxels sharing the edge. In this implementation we consider only those quads
    # where a full neighborhood exists - i.e non of the adjacent voxels is in the
    # padding area.
    active_quads, complete_mask = top.find_voxels_sharing_edge(active_edges)  # (A,4)
    active_edges = active_edges[complete_mask]
    active_quads = active_quads[complete_mask]

    # The active quad indices are are ordered ccw when looking from the positive
    # active edge direction. In case the sign difference is negative between edge
    # start and end, we need to reverse the indices to maintain a correct ccw
    # winding order.
    flip_mask = sdf_diff[active_edges] < 0.0
    active_quads[flip_mask] = np.flip(active_quads[flip_mask], -1)

    # Voxel indices are not unique, since an active voxel will be part in more than one
    # quad. However, each active voxel will give rise to only one vertex. We
    # avoid duplicate computations, by computing the set of unique active voxels. Bonus:
    # the inverse array is already the flattened final face array.
    active_voxels, faces = np.unique(active_quads, return_inverse=True)  # (N,)

    # For each active voxel, we find the 12 constituting edges and then compute the
    # vertex location as the average of the edge intersection points of active edges.
    # Note, non-active edges have a nan intersection point. We use np.nanmean to compute
    # the mean only of finite values. Also note, we need take care of vertex locations
    # being shifted by one voxel length due to padding.
    active_voxel_edges = top.find_voxel_edges(active_voxels)  # (N,12)
    e = edge_isect[active_voxel_edges]  # (N,12,3)
    verts = (np.nanmean(e, 1) - (1, 1, 1)) * spacing  # (M,3)

    # 3. Step - Postprocessing
    # In case triangulation is required, we simply split each quad into two triangles.
    # Since the vertex order in faces is ccw, that's easy too.
    faces = faces.reshape(-1, 4)
    if triangulate:
        tris = np.empty((faces.shape[0], 2, 3), dtype=faces.dtype)
        tris[:, 0, :] = faces[:, [0, 1, 2]]
        tris[:, 1, :] = faces[:, [0, 2, 3]]
        faces = tris.reshape(-1, 3)

    return verts, faces
