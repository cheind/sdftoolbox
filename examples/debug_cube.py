"""Compute and render surface normals"""

import matplotlib.pyplot as plt
import surfacenets as sn

import numpy as np


def get_rotated_box(rot):
    scene = sn.sdfs.Box((1.1, 1.1, 1.1))
    grid = sn.Grid((3, 3, 3))

    # Generate mesh
    verts, faces = sn.dual_isosurface(
        scene,
        grid,
        vertex_strategy=sn.DualContouringVertexStrategy(),
        edge_strategy=sn.BisectionEdgeStrategy(),
        triangulate=False,
    )

    t = sn.maths.rotate(rot[:3], rot[3])

    verts = sn.maths.dehom(sn.maths.hom(verts) @ t.T)
    return verts, faces


def main():

    # Setup the scene
    scene = sn.sdfs.Box((1.1, 1.1, 1.1)).transform(rot=(1.0, 1.0, 0, np.pi / 4))
    grid = sn.Grid((32, 32, 32))

    # Generate mesh
    verts, faces, debug = sn.dual_isosurface(
        scene,
        grid,
        vertex_strategy=sn.DualContouringVertexStrategy(),
        # edge_strategy=sn.NewtonEdgeStrategy(max_steps=50),
        triangulate=False,
        return_debug_info=True,
    )

    # Compute normals
    face_normals = sn.mesh.compute_face_normals(verts, faces)
    vert_normals = scene.gradient(verts, normalize=True)
    # Alternatively via averaging face normals
    # vert_normals = sn.mesh.compute_vertex_normals(verts, faces, face_normals)

    # Plot mesh+normals
    fig, ax = sn.plotting.create_mesh_figure(
        verts, faces
    )  # face_normals, vert_normals)

    # v, f = get_rotated_box((1.0, 1.0, 0, np.pi / 4))
    # sn.plotting.plot_mesh(ax, v, f, alpha=0.5)

    isect = grid.grid_to_data(debug.edges_isect_coords[debug.edges_active_mask])
    isect_n = scene.gradient(isect, normalize=True)
    # sn.plotting.plot_normals(ax, isect, isect_n, "yellow")

    # sn.plotting.plot_samples(ax, grid.xyz, scene.sample(grid.xyz))
    sn.plotting.setup_axes(ax, grid.min_corner, grid.max_corner)
    # sn.plotting.generate_rotation_gif("normals.gif", fig, ax)
    plt.show()


if __name__ == "__main__":
    main()
