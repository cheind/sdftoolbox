"""Compute and render surface normals"""

import matplotlib.pyplot as plt
import sdftoolbox


def main():

    # Setup the scene
    scene = sdftoolbox.sdfs.Box((1.1, 1.1, 1.1))
    grid = sdftoolbox.Grid((5, 5, 5))

    # Generate mesh
    verts, faces = sdftoolbox.dual_isosurface(
        scene,
        grid,
        vertex_strategy=sdftoolbox.DualContouringVertexStrategy(),
        # strategy=sn.NaiveSurfaceNetStrategy(),
        triangulate=False,
    )

    # Compute normals
    face_normals = sdftoolbox.mesh.compute_face_normals(verts, faces)
    vert_normals = scene.gradient(verts, normalize=True)
    # Alternatively via averaging face normals
    # vert_normals = sn.mesh.compute_vertex_normals(verts, faces, face_normals)

    # Plot mesh+normals
    fig, ax = sdftoolbox.plotting.create_mesh_figure(
        verts, faces, face_normals, vert_normals
    )
    sdftoolbox.plotting.plot_samples(ax, grid.xyz, scene.sample(grid.xyz))
    sdftoolbox.plotting.setup_axes(ax, grid.min_corner, grid.max_corner)
    # sn.plotting.generate_rotation_gif("normals.gif", fig, ax)
    plt.show()


if __name__ == "__main__":
    main()
