"""Compute and render surface normals"""

import matplotlib.pyplot as plt
import surfacenets as sn


def main():

    # Setup the scene
    scene = sn.sdfs.Box((1.1, 1.1, 1.1))
    grid = sn.Grid((5, 5, 5))

    # Generate mesh
    sdfv = scene.sample(grid.xyz)
    verts, faces = sn.dual_isosurface(
        sdfv,
        grid,
        strategy=sn.DualContouringStrategy(
            scene,
        ),
        # strategy=sn.NaiveSurfaceNetStrategy(),
        triangulate=False,
    )

    # Compute normals
    face_normals = sn.normals.compute_face_normals(verts, faces)
    vert_normals = scene.gradient(verts, normalize=True)
    # Alternatively via averaging face normals
    # vert_normals = sn.normals.compute_vertex_normals(verts, faces, face_normals)

    # Plot mesh+normals
    fig, ax = sn.plotting.create_mesh_figure(verts, faces, face_normals, vert_normals)
    sn.plotting.plot_samples(ax, grid.xyz, sdfv)
    # sn.plotting.generate_rotation_gif("normals.gif", fig, ax)
    plt.show()


if __name__ == "__main__":
    main()
