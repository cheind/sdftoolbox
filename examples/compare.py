"""Compares surface nets to other methods.

This code compares the result of surface nets to marching cubes.
We use the MC implementation from scikit-image, which is required
to run this example.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

import surfacenets as sn


def main():

    scene = sn.sdfs.Sphere.create([0, 0, 0], 1.0)
    scene = sn.sdfs.Displacement(
        scene, lambda xyz: 0.3 * np.sin(10 * xyz).prod(-1)
    )

    # Define the sampling locations. Here we use the default params
    xyz, spacing = sn.sample_volume()

    # Evaluate the SDF
    t0 = time.perf_counter()
    sdfv = scene.sample(xyz)
    print(f"SDF sampling took {time.perf_counter() - t0:.3f} secs")

    t0 = time.perf_counter()
    verts_sn, faces_sn = sn.surface_nets(
        sdfv,
        spacing,
        vertex_placement_mode="naive",
        triangulate=False,
    )
    verts_sn += xyz[0, 0, 0]
    print(f"SurfaceNets took {time.perf_counter() - t0:.3f} secs")

    t0 = time.perf_counter()
    verts_mc, faces_mc, _, _ = marching_cubes(
        sdfv,
        0.0,
        spacing=spacing,
    )
    verts_mc += xyz[0, 0, 0]
    print(f"MarchingCubes took {time.perf_counter() - t0:.3f} secs")

    plt.style.use("dark_background")
    minc = verts_mc.min(0)
    maxc = verts_mc.max(0)
    fig, (ax0, ax1) = sn.plotting.create_split_figure(sync=True)
    sn.plotting.plot_mesh(ax0, verts_sn, faces_sn)
    sn.plotting.plot_mesh(ax1, verts_mc, faces_mc)
    sn.plotting.setup_axes(ax0, minc, maxc)
    sn.plotting.setup_axes(ax1, minc, maxc)
    plt.show()


if __name__ == "__main__":
    main()
