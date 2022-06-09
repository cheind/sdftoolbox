"""Compute and render surface normals"""

import matplotlib.pyplot as plt
import surfacenets as sn


def main():

    # Setup the scene
    scene = sn.sdfs.Box((1.1, 1.1, 1.1))
    scene = sn.sdfs.Transform(scene, sn.maths.rotate([1, 0, 0], 0))
    xyz, spacing = sn.sdfs.Discretized.sampling_coords(
        res=(4, 4, 4), min_corner=(-1, -1, -1), max_corner=(1, 1, 1)
    )

    # Generate mesh
    sdfv = scene.sample(xyz)
    verts, faces = sn.dual_isosurface(
        sdfv,
        spacing=spacing,
        strategy=sn.DualContouringStrategy(
            scene,
            spacing=spacing,
            min_corner=xyz[0, 0, 0],
            bias_strength=1e-5,
        ),
        # strategy=sn.NaiveSurfaceNetStrategy(),
        triangulate=False,
    )
    verts += xyz[0, 0, 0]

    # Compute normals
    face_normals = sn.normals.compute_face_normals(verts, faces)
    vert_normals = scene.gradient(verts, normalize=True)
    # Alternatively via averaging face normals
    # vert_normals = sn.normals.compute_vertex_normals(verts, faces, face_normals)

    # Plot mesh+normals
    fig, ax = sn.plotting.create_mesh_figure(verts, faces, face_normals, vert_normals)
    sn.plotting.plot_samples(ax, xyz, sdfv)
    sn.plotting.generate_rotation_gif("normals.gif", fig, ax)
    plt.show()


if __name__ == "__main__":
    main()
