import numpy as np
from .topology import VolumeTopology2, VolumeTopology3
from .topology2 import VoxelTopology

VOXEL_EDGES = np.array(
    (
        ((0, 0, 0), (1, 0, 0)),  # Front face loop
        ((1, 0, 0), (1, 0, 1)),
        ((1, 0, 1), (0, 0, 1)),
        ((0, 0, 1), (0, 0, 0)),
        ((0, 1, 0), (1, 1, 0)),  # Back face loop
        ((1, 1, 0), (1, 1, 1)),
        ((1, 1, 1), (0, 1, 1)),
        ((0, 1, 1), (0, 1, 0)),
        ((0, 0, 0), (0, 1, 0)),  # Connecting edges
        ((1, 0, 0), (1, 1, 0)),
        ((1, 0, 1), (1, 1, 1)),
        ((0, 0, 1), (0, 1, 1)),
    ),
    dtype=np.int32,
)


def surface_net(
    sdf_values: np.ndarray, xyz: np.ndarray, vertex_placement_mode="midpoint"
):
    """Generate surface net triangulation

    Params
        sdf_values: (X,Y,Z) array if SDF values at sample locations
        xyz: (X,Y,Z,3) array of sampling locations

    """

    assert len(sdf_values.shape) == 3

    # The number of corners
    X, Y, Z = sdf_values.shape
    # The number of voxels
    VX, VY, VZ = X - 1, Y - 1, Z - 1
    # Generate all voxel indices
    voxids = np.stack(
        np.meshgrid(range(VX), range(VY), range(VZ), indexing="ij"), -1
    ).astype(np.int32)

    # Compute voxel edge values as (VX,VY,VZ,12,2) array
    edge_sdfs = np.empty((VX, VY, VZ, 12, 2), dtype=np.float32)
    for eidx, (source_offset, target_offset) in enumerate(VOXEL_EDGES):
        sidx = voxids + source_offset
        tidx = voxids + target_offset
        edge_sdfs[..., eidx, 0] = sdf_values[
            sidx[..., 0], sidx[..., 1], sidx[..., 2]
        ]
        edge_sdfs[..., eidx, 1] = sdf_values[
            tidx[..., 0], tidx[..., 1], tidx[..., 2]
        ]

    # Find the voxels through which the surface passes.
    # We do this by finding voxels that have at least one SDF corner value
    # that is different from the rest.
    signs = np.sign(edge_sdfs)
    selected_edges = abs(signs.sum(-1)) != 2  # (VX,VY,VZ,12)
    selected_voxels = selected_edges.any(-1)  # (VX,VY,VZ)

    # Compute the intersection points within the voxel for all selected voxels
    if vertex_placement_mode == "midpoint":
        # Assume constant spacing, so to simplify midpoint computation
        spacing = xyz[1, 1, 1] - xyz[0, 0, 0]
        x, y, z = np.where(selected_voxels)
        verts = xyz[x, y, z] + spacing[None, :] * 0.5
    elif vertex_placement_mode == "naive":
        # We compute the vertex locations per voxel as the mean of the voxel
        # edge-surface intersections.
        edge_isects = np.empty(
            (VX, VY, VZ, 12, 3), dtype=np.float32
        )  # TODO: should be sparse, but how to accumulate over voxels then?
        x, y, z, e = np.where(selected_edges)

        # Compute surface intersection points along selected edges. We assume
        # that the surface is smooth and linear close to an edge. Hence, we can
        # find the linear interpolation value t [0,1] along the selected edge
        # for which SDF should be 0. We then use t to mix start and end point
        # of edges to compute the intersection point. Finally, we determine a
        # single vertex per selected voxel as the mean of all intersection points
        # corresponding to an voxel edge.
        t = -edge_sdfs[x, y, z, e, 0] / (
            edge_sdfs[x, y, z, e, 1] - edge_sdfs[x, y, z, e, 0]
        )
        source_offsets = VOXEL_EDGES[e, 0]
        target_offsets = VOXEL_EDGES[e, 1]

        # Compute all the start/end vertices for each selected edge
        vsource = xyz[
            x + source_offsets[:, 0],
            y + source_offsets[:, 1],
            z + source_offsets[:, 2],
        ]
        vtarget = xyz[
            x + target_offsets[:, 0],
            y + target_offsets[:, 1],
            z + target_offsets[:, 2],
        ]
        # Find the intersection points using linear interpolation
        isects = (1 - t[:, None]) * vsource + t[:, None] * vtarget

        # Write back to dense array for accumulation
        edge_isects[x, y, z, e] = isects
        num_edges = selected_edges.sum(-1, keepdims=True)
        # Compute vertex locations as mean over voxel edge intersection points
        verts = (
            edge_isects[selected_voxels].sum(-2) / num_edges[selected_voxels]
        )
        return verts

    return verts


def surface_net2(
    sdf_values: np.ndarray,
    spacing: tuple[float, float, float],
    vertex_placement_mode="midpoint",
):
    """Generate surface net triangulation

    Params
        sdf_values: (X,Y,Z) array if SDF values at sample locations
        xyz: (X,Y,Z,3) array of sampling locations

    """
    assert vertex_placement_mode in ["midpoint", "naive"]
    spacing = np.asarray(spacing, dtype=np.float32)

    top = VolumeTopology2(sdf_values.shape)
    ravelled_sdf = sdf_values.reshape(-1)

    # Compute the set of active edges, which are those edges that
    # cross the surface boundary.
    sdf_signs = np.sign(ravelled_sdf)
    sidx, tidx = top.edge_endpoints(
        top.edges[top.valid_edge_mask], return_ravelled=True
    )
    active_mask = np.zeros(len(top.edges), dtype=bool)
    active_mask[top.valid_edge_mask] = sdf_signs[sidx] != sdf_signs[tidx]
    active_edges = top.edges[active_mask]

    # Compute the intersection points (vertices) for those voxels that
    # share at least one active edge.
    active_quads = top.edge_neighbors(active_edges)
    active_voxels = np.unique(active_quads.reshape(-1))

    if vertex_placement_mode == "midpoint":
        active_ijk = top.unravel_nd(active_voxels, top.voxel_shape)
        verts = active_ijk * spacing[None, :] + spacing[None, :] * 0.5
        return verts
    elif vertex_placement_mode == "naive":
        # Compute parametric interpolation value surface boundary
        # crossing edges, assuming surfaces are linear close to edges.
        t = -ravelled_sdf[sidx] / (ravelled_sdf[tidx] - ravelled_sdf[sidx])
        sijk = top.unravel_nd(sidx, top.sample_shape)
        tijk = top.unravel_nd(tidx, top.sample_shape)
        # Find the edge intersection points along the edge using
        # linear interpolation just like in MC.
        edge_isects = (1 - t[:, None]) * sijk + t[:, None] * tijk
        print(len(edge_isects))

        # For all active voxels, we compute the 12 edges constructing it
        voxel_edges = top.voxel_edges(active_voxels)

        # To compute the vertices, we average

    #
    #     active_voxel_edges = top.voxel_edges(active_voxels)  # (A,12,4)

    #     #     isects = (1 - t[:, None]) * vsource + t[:, None] * vtarget
    #     print(active_voxel_edges.shape)

    #    t = -edge_sdfs[x, y, z, e, 0] / (
    #         edge_sdfs[x, y, z, e, 1] - edge_sdfs[x, y, z, e, 0]
    #     )
    #     source_offsets = VOXEL_EDGES[e, 0]
    #     target_offsets = VOXEL_EDGES[e, 1]

    #     # Compute all the start/end vertices for each selected edge
    #     vsource = xyz[
    #         x + source_offsets[:, 0], y + source_offsets[:, 1], z + source_offsets[:, 2]
    #     ]
    #     vtarget = xyz[
    #         x + target_offsets[:, 0], y + target_offsets[:, 1], z + target_offsets[:, 2]
    #     ]
    #     # Find the intersection points using linear interpolation
    #     isects = (1 - t[:, None]) * vsource + t[:, None] * vtarget

    #     # Write back to dense array for accumulation
    #     edge_isects[x, y, z, e] = isects
    #     num_edges = selected_edges.sum(-1, keepdims=True)
    #     # Compute vertex locations as mean over voxel edge intersection points
    #     verts = edge_isects[selected_voxels].sum(-2) / num_edges[selected_voxels]

    # print(sum(active_mask))

    # # The number of corners
    # X, Y, Z = sdf_values.shape
    # # The number of voxels
    # VX, VY, VZ = X - 1, Y - 1, Z - 1
    # # Generate all voxel indices
    # voxids = np.stack(
    #     np.meshgrid(range(VX), range(VY), range(VZ), indexing="ij"), -1
    # ).astype(np.int32)

    # # Compute voxel edge values as (VX,VY,VZ,12,2) array
    # edge_sdfs = np.empty((VX, VY, VZ, 12, 2), dtype=np.float32)
    # for eidx, (source_offset, target_offset) in enumerate(VOXEL_EDGES):
    #     sidx = voxids + source_offset
    #     tidx = voxids + target_offset
    #     edge_sdfs[..., eidx, 0] = sdf_values[sidx[..., 0], sidx[..., 1], sidx[..., 2]]
    #     edge_sdfs[..., eidx, 1] = sdf_values[tidx[..., 0], tidx[..., 1], tidx[..., 2]]

    # # Find the voxels through which the surface passes.
    # # We do this by finding voxels that have at least one SDF corner value
    # # that is different from the rest.
    # signs = np.sign(edge_sdfs)
    # selected_edges = abs(signs.sum(-1)) != 2  # (VX,VY,VZ,12)
    # selected_voxels = selected_edges.any(-1)  # (VX,VY,VZ)

    # # Compute the intersection points within the voxel for all selected voxels
    # if vertex_placement_mode == "midpoint":
    #     # Assume constant spacing, so to simplify midpoint computation
    #     spacing = xyz[1, 1, 1] - xyz[0, 0, 0]
    #     x, y, z = np.where(selected_voxels)
    #     verts = xyz[x, y, z] + spacing[None, :] * 0.5
    # elif vertex_placement_mode == "naive":
    #     # We compute the vertex locations per voxel as the mean of the voxel
    #     # edge-surface intersections.
    #     edge_isects = np.empty(
    #         (VX, VY, VZ, 12, 3), dtype=np.float32
    #     )  # TODO: should be sparse, but how to accumulate over voxels then?
    #     x, y, z, e = np.where(selected_edges)

    #     # Compute surface intersection points along selected edges. We assume
    #     # that the surface is smooth and linear close to an edge. Hence, we can
    #     # find the linear interpolation value t [0,1] along the selected edge
    #     # for which SDF should be 0. We then use t to mix start and end point
    #     # of edges to compute the intersection point. Finally, we determine a
    #     # single vertex per selected voxel as the mean of all intersection points
    #     # corresponding to an voxel edge.
    #     t = -edge_sdfs[x, y, z, e, 0] / (
    #         edge_sdfs[x, y, z, e, 1] - edge_sdfs[x, y, z, e, 0]
    #     )
    #     source_offsets = VOXEL_EDGES[e, 0]
    #     target_offsets = VOXEL_EDGES[e, 1]

    #     # Compute all the start/end vertices for each selected edge
    #     vsource = xyz[
    #         x + source_offsets[:, 0], y + source_offsets[:, 1], z + source_offsets[:, 2]
    #     ]
    #     vtarget = xyz[
    #         x + target_offsets[:, 0], y + target_offsets[:, 1], z + target_offsets[:, 2]
    #     ]
    #     # Find the intersection points using linear interpolation
    #     isects = (1 - t[:, None]) * vsource + t[:, None] * vtarget

    #     # Write back to dense array for accumulation
    #     edge_isects[x, y, z, e] = isects
    #     num_edges = selected_edges.sum(-1, keepdims=True)
    #     # Compute vertex locations as mean over voxel edge intersection points
    #     verts = edge_isects[selected_voxels].sum(-2) / num_edges[selected_voxels]
    #     return verts

    # return verts


def surface_net3(
    sdf_values: np.ndarray,
    spacing: tuple[float, float, float],
    vertex_placement_mode="midpoint",
):
    """Generate surface net triangulation

    Params
        sdf_values: (I,J,K) array if SDF values at sample locations

    """
    assert vertex_placement_mode in ["midpoint", "naive"]
    spacing = np.asarray(spacing, dtype=np.float32)

    top = VoxelTopology(sdf_values.shape)

    sdf_values = np.pad(sdf_values, ((1, 1), (1, 1), (1, 1)), mode="edge")

    sijk, tijk = top.find_edge_vertices(top.edge_ids, ravel=False)
    # active_edge_mask = sdf_signs[si, sj, sk] != sdf_signs[ti, tj, tk]

    si, sj, sk = sijk.T
    ti, tj, tk = tijk.T

    # Find the edge intersection points along the edge using
    # linear interpolation just like in MC.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -sdf_values[si, sj, sk] / (
            sdf_values[ti, tj, tk] - sdf_values[si, sj, sk]
        )
    active_edge_mask = np.logical_and(t >= 0, t <= 1.0)
    t[~active_edge_mask] = np.nan

    edge_isect = (1 - t[:, None]) * sijk + t[:, None] * tijk

    active_edges = top.edge_ids[active_edge_mask]
    active_quads = top.find_voxels_sharing_edge(active_edges)
    active_voxels = np.unique(active_quads.reshape(-1))
    active_voxel_edges = top.find_voxel_edges(active_voxels)

    e = edge_isect[active_voxel_edges.reshape(-1)].reshape(-1, 12, 3)
    verts = np.nanmean(e, 1) * spacing - spacing  # subtract padding again
    return verts


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage.measure import marching_cubes
    from . import sdfs, plots
    import time

    res = (40, 40, 40)
    min_corner = np.array([-2.0] * 3, dtype=np.float32)
    max_corner = np.array([2.0] * 3, dtype=np.float32)

    ranges = [
        np.linspace(min_corner[0], max_corner[0], res[0], dtype=np.float32),
        np.linspace(min_corner[1], max_corner[1], res[1], dtype=np.float32),
        np.linspace(min_corner[2], max_corner[2], res[2], dtype=np.float32),
    ]

    X, Y, Z = np.meshgrid(*ranges, indexing="ij")
    xyz = np.stack((X, Y, Z), -1)
    spacing = (
        ranges[0][1] - ranges[0][0],
        ranges[1][1] - ranges[1][0],
        ranges[2][1] - ranges[2][0],
    )

    s1 = sdfs.Sphere([-0.5, 0.0, 0.0], 1.0)
    s2 = sdfs.Sphere([0.5, 0.0, 0.0], 1.0)
    s = s1.merge(s2)
    values = s(xyz)

    # verts, faces, normals, _ = marching_cubes(values, 0.0, spacing=spacing)
    # verts += min_corner[None, :]

    fig, ax = plots.create_figure()
    # plots.plot_mesh(verts, faces, ax)

    t0 = time.perf_counter()
    verts = surface_net3(s(xyz), spacing, vertex_placement_mode="naive")
    verts += min_corner[None, :]
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=5)
    print("Surface-nets took", time.perf_counter() - t0, "secs")

    # t0 = time.perf_counter()
    # verts = surface_net(s1(xyz), xyz, vertex_placement_mode="naive")
    # ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=5)
    # print("Surface-nets took", time.perf_counter() - t0, "secs")

    # verts = surface_net(s1(xyz), xyz, vertex_placement_mode="midpoint")
    # ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=5)

    plots.setup_axes(ax, min_corner, max_corner)
    plt.show()
