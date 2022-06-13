"""Compute and render surface normals"""

import matplotlib.pyplot as plt
import numpy as np
import sdftoolbox


def get_rotated_box(rot):
    scene = sdftoolbox.sdfs.Box((1.1, 1.1, 1.1))
    grid = sdftoolbox.Grid((3, 3, 3))

    # Generate mesh
    verts, faces = sdftoolbox.dual_isosurface(
        scene,
        grid,
        vertex_strategy=sdftoolbox.DualContouringVertexStrategy(),
        edge_strategy=sdftoolbox.BisectionEdgeStrategy(),
        triangulate=False,
    )

    t = sdftoolbox.maths.rotate(rot[:3], rot[3])

    verts = sdftoolbox.maths.dehom(sdftoolbox.maths.hom(verts) @ t.T)
    return verts, faces


def main():

    # Setup the scene
    rot = (1.0, 1.0, 1.0, np.pi / 4)
    scene = sdftoolbox.sdfs.Box((1.1, 1.1, 1.1)).transform(rot=rot)
    grid = sdftoolbox.Grid((3, 3, 3))

    # Generate mesh
    verts, faces, debug = sdftoolbox.dual_isosurface(
        scene,
        grid,
        # vertex_strategy=sn.DualContouringVertexStrategy(),
        # edge_strategy=sn.BisectionEdgeStrategy(max_steps=50),
        triangulate=False,
        return_debug_info=True,
    )

    # Compute normals
    face_normals = sdftoolbox.mesh.compute_face_normals(verts, faces)
    vert_normals = scene.gradient(verts, normalize=True)
    # Alternatively via averaging face normals
    # vert_normals = sn.mesh.compute_vertex_normals(verts, faces, face_normals)

    # Plot mesh+normals
    fig, ax = sdftoolbox.plotting.create_mesh_figure(
        verts, faces
    )  # face_normals, vert_normals)

    v, f = get_rotated_box(rot)
    sdftoolbox.plotting.plot_mesh(ax, v, f, alpha=0.1, color="gray")

    isect = grid.grid_to_data(debug.edges_isect_coords[debug.edges_active_mask])
    isect_n = scene.gradient(isect, normalize=True)

    active_src, active_dst = grid.find_edge_vertices(
        np.where(debug.edges_active_mask)[0], ravel=False
    )
    print(active_src, active_dst)
    active_src = grid.grid_to_data(active_src)
    active_dst = grid.grid_to_data(active_dst)
    sdftoolbox.plotting.plot_edges(
        ax, active_src, active_dst, color="yellow", linewidth=0.5
    )
    sdftoolbox.plotting.plot_normals(ax, isect, isect_n, "yellow")

    sdftoolbox.plotting.plot_samples(ax, grid.xyz, scene.sample(grid.xyz))
    sdftoolbox.plotting.setup_axes(ax, grid.min_corner, grid.max_corner)
    # sn.plotting.generate_rotation_gif("normals.gif", fig, ax)
    plt.show()


if __name__ == "__main__":
    main()
