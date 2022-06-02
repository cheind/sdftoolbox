import numpy as np
from .topology2 import VoxelTopology


def surface_nets(
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
    sijk, tijk = top.find_edge_vertices(range(top.num_edges), ravel=False)

    si, sj, sk = sijk.T
    ti, tj, tk = tijk.T

    # Find the edge intersection points along the edge using
    # linear interpolation just like in MC.
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -sdf_values[si, sj, sk] / (sdf_values[ti, tj, tk] - sdf_values[si, sj, sk])
    active_edge_mask = np.logical_and(t >= 0, t <= 1.0)
    t[~active_edge_mask] = np.nan
    if vertex_placement_mode == "midpoint":
        t[:] = 0.5

    edge_isect = (1 - t[:, None]) * sijk + t[:, None] * tijk

    active_edges = np.where(active_edge_mask)[0]
    active_quads = top.find_voxels_sharing_edge(active_edges)
    active_voxels = np.unique(active_quads.reshape(-1))
    active_voxel_edges = top.find_voxel_edges(active_voxels)

    e = edge_isect[active_voxel_edges.reshape(-1)].reshape(-1, 12, 3)
    verts = (np.nanmean(e, 1) - (1, 1, 1)) * spacing  # subtract padding again
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
    sdf = s(xyz)
    print("SDF sampling took", time.perf_counter() - t0, "secs")

    t0 = time.perf_counter()
    verts = surface_nets(sdf, spacing, vertex_placement_mode="midpoint")
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
