"""Introductory example to surfacenet usage"""
import numpy as np


def main():

    # Main import
    import surfacenets as sn

    # Setup the scene
    scene = sn.sdfs.Box(
        # t_world_local=sn.maths.rotate([1.0, 1.0, 1.0], np.pi / 4)
    )
    # Generate the sampling locations. Here we use the default params
    xyz, spacing = sn.sdfs.Discretized.sampling_coords(res=(3, 3, 3))

    # Evaluate the SDF
    sdfv = scene.sample(xyz)
    normals = scene.gradient(xyz, normalize=True)

    # Extract the surface using quadliterals
    verts, faces = sn.surface_nets(
        sdfv,
        spacing=spacing,
        normals=normals,
        vertex_placement_mode="dual-contouring",
        triangulate=False,
    )
    verts += xyz[0, 0, 0]

    # Visualize
    import matplotlib.pyplot as plt

    fig, ax = sn.plotting.create_mesh_figure(verts, faces)
    plt.show()


if __name__ == "__main__":
    main()
