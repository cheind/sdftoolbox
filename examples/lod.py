"""Successively increasing the level of detail"""

import matplotlib.pyplot as plt
import numpy as np

# Main import
import surfacenets as sn


def main():

    # Setup the scene
    scene = sn.sdfs.Sphere.create(center=(0, 0, 0), radius=1.0)

    fig, ax = sn.plotting.create_figure(fig_aspect=9 / 16, proj_type="persp")

    max_corner = np.array([-np.inf, -np.inf, -np.inf])
    min_corner = np.array([np.inf, np.inf, np.inf])

    # Note, the resolution is chosen such that stepping in powers of 2
    # always contains the endpoint. This is important, since the sampling
    # bounds are close the surface of the sphere.
    xyz, spacing = sn.sample_volume(
        res=(65, 65, 65),
        min_corner=(-1.1, -1.1, -1.1),
        max_corner=(1.1, 1.1, 1.1),
    )
    sdfv = scene.sample(xyz)

    for idx in range(1, 6):
        step = 2**idx
        verts, faces = sn.surface_nets(
            sdfv[::step, ::step, ::step],
            spacing=spacing * step,
            vertex_placement_mode="naive",
            triangulate=False,
        )
        verts += xyz[0, 0, 0] + (idx * 3, 0, 0)
        max_corner = np.maximum(verts.max(0), max_corner)
        min_corner = np.minimum(verts.min(0), min_corner)
        sn.plotting.plot_mesh(ax, verts, faces)
    sn.plotting.setup_axes(ax, min_corner, max_corner, num_grid=0)

    plt.show()


if __name__ == "__main__":
    main()