"""Introductory example to surfacenet usage"""
import numpy as np


def main():

    # Main import
    import surfacenets as sn

    # Setup the scene
    scene = sn.sdfs.Box.create((0.6, 0.6, 0.6))  #
    # Generate the sampling locations. Here we use the default params
    xyz, spacing = sn.sdfs.Discretized.sampling_coords(res=(40, 40, 40))

    # Evaluate the SDF
    sdfv = scene.sample(xyz)

    # Extract the surface using quadliterals
    verts, faces = sn.dual_isosurface(
        sdfv,
        spacing=spacing,
        strategy=sn.DualContouringStrategy(
            scene, spacing=spacing, min_corner=xyz[0, 0, 0], bias_strength=1e-3
        ),
        # strategy=sn.NaiveSurfaceNetStrategy(),
        triangulate=False,
    )
    print("#faces", len(faces))
    verts += xyz[0, 0, 0]

    # Visualize
    import matplotlib.pyplot as plt

    fig, ax = sn.plotting.create_mesh_figure(
        verts, faces, fig_kwargs={"proj_type": "persp"}
    )
    # sn.plotting.plot_samples(ax, xyz)
    plt.show()


if __name__ == "__main__":
    main()
