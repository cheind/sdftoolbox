import numpy as np

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
    edges = np.empty((VX, VY, VZ, 12, 2), dtype=np.float32)
    for eidx, (source_offset, target_offset) in enumerate(VOXEL_EDGES):
        sidx = voxids + source_offset
        tidx = voxids + target_offset
        edges[..., eidx, 0] = sdf_values[sidx[..., 0], sidx[..., 1], sidx[..., 2]]
        edges[..., eidx, 1] = sdf_values[tidx[..., 0], tidx[..., 1], tidx[..., 2]]

    # Find the voxels through which the surface passes.
    # We do this by finding voxels that have at least one SDF corner value
    # that is different from the rest.
    signs = np.sign(edges)
    selected_edges = abs(signs.sum(-1)) != 2  # (VX,VY,VZ,12)
    selected_voxels = selected_edges.any(-1)  # (VX,VY,VZ)

    # Compute the intersection points within the voxel for all selected voxels
    if vertex_placement_mode == "midpoint":
        # Assume constant resultion
        spacing = xyz[1, 1, 1] - xyz[0, 0, 0]
        x, y, z = np.where(selected_voxels)
        verts = xyz[x, y, z] + spacing[None, :] * 0.5
    else:
        edge_isects = np.zeros((VX, VY, VZ, 12, 3), dtype=np.float32)
        x, y, z, e = np.where(selected_edges)
        t = -edges[x, y, z, e, 0] / (edges[x, y, z, e, 1] - edges[x, y, z, e, 0])
        source_offsets = VOXEL_EDGES[e][:, 0]
        target_offsets = VOXEL_EDGES[e][:, 1]

        vsource = xyz[
            x + source_offsets[:, 0], y + source_offsets[:, 1], z + source_offsets[:, 2]
        ]
        vtarget = xyz[
            x + target_offsets[:, 0], y + target_offsets[:, 1], z + target_offsets[:, 2]
        ]
        isects = (1 - t[:, None]) * vsource + t[:, None] * vtarget
        edge_isects[x, y, z, e] = isects
        sum_edges = selected_edges.sum(-1, keepdims=True)
        verts = edge_isects[selected_voxels].sum(-2) / sum_edges[selected_voxels]
        return verts

    return verts


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage.measure import marching_cubes
    from . import sdfs, plots

    res = (40, 40, 40)
    min_corner = np.array([-2.0] * 3, dtype=np.float32)
    max_corner = np.array([2.0] * 3, dtype=np.float32)

    ranges = [
        np.linspace(min_corner[0], max_corner[0], res[0], dtype=np.float32),
        np.linspace(min_corner[1], max_corner[1], res[1], dtype=np.float32),
        np.linspace(min_corner[2], max_corner[2], res[2], dtype=np.float32),
    ]

    X, Y, Z = np.meshgrid(*ranges)
    xyz = np.stack((X, Y, Z), -1)
    spacing = (
        ranges[0][1] - ranges[0][0],
        ranges[1][1] - ranges[1][0],
        ranges[2][1] - ranges[2][0],
    )

    s1 = sdfs.Sphere([0.0, 0.0, 0.0], 1.0)
    values = s1(xyz)

    verts, faces, normals, _ = marching_cubes(values, 0.0, spacing=spacing)
    verts += min_corner[None, :]

    fig, ax = plots.create_figure()
    # plots.plot_mesh(verts, faces, ax)

    verts = surface_net(s1(xyz), xyz, vertex_placement_mode="naive")
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2])

    # verts = surface_net(s1(xyz), xyz, vertex_placement_mode="midpoint")
    # ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2])

    plots.setup_axes(ax, min_corner, max_corner)
    plt.show()
