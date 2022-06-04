from typing import Literal

import numpy as np

from .topology import VoxelTopology


def surface_nets(
    sdf_values: np.ndarray,
    spacing: tuple[float, float, float],
    vertex_placement_mode: Literal["midpoint", "naive"] = "naive",
    triangulate: bool = False,
):
    """Generate surface net triangulation

    Params
        sdf_values: (I,J,K) array if SDF values at sample locations
    """
    assert vertex_placement_mode in ["midpoint", "naive"]
    spacing = np.asarray(spacing, dtype=np.float32)

    sdf_values = np.pad(
        sdf_values,
        ((1, 1), (1, 1), (1, 1)),
        mode="constant",
        constant_values=np.nan,
    )

    top = VoxelTopology(sdf_values.shape)
    sijk, tijk = top.find_edge_vertices(range(top.num_edges), ravel=False)
    si, sj, sk = sijk.T
    ti, tj, tk = tijk.T

    # Find the edge intersection points along the edge using
    # linear interpolation just like in MC.
    with np.errstate(divide="ignore", invalid="ignore"):
        sdf_diff = sdf_values[ti, tj, tk] - sdf_values[si, sj, sk]
        t = -sdf_values[si, sj, sk] / sdf_diff
    active_edge_mask = np.logical_and(t >= 0, t <= 1.0)
    t[~active_edge_mask] = np.nan
    if vertex_placement_mode == "midpoint":
        t[:] = 0.5

    edge_isect = (1 - t[:, None]) * sijk + t[:, None] * tijk

    active_edges = np.where(active_edge_mask)[0]  # (A,)
    active_quads, complete_mask = top.find_voxels_sharing_edge(active_edges)  # (A,4)
    active_edges = active_edges[complete_mask]
    active_quads = active_quads[complete_mask]
    flip_mask = sdf_diff[active_edges] < 0.0
    active_quads[flip_mask] = np.flip(active_quads[flip_mask], -1)
    active_voxels, faces = np.unique(active_quads, return_inverse=True)  # (M,)

    # Compute vertices
    active_voxel_edges = top.find_voxel_edges(active_voxels)  # (M,12)
    e = edge_isect[active_voxel_edges]  # (M,12,3)
    verts = (np.nanmean(e, 1) - (1, 1, 1)) * spacing  # (M,3) subtract padding again

    faces = faces.reshape(-1, 4)
    if triangulate:
        tris = np.empty((faces.shape[0], 2, 3), dtype=faces.dtype)
        tris[:, 0, :] = faces[:, [0, 1, 2]]
        tris[:, 1, :] = faces[:, [0, 2, 3]]
        faces = tris.reshape(-1, 3)

    return verts, faces
