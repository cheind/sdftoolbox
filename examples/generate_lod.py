"""Successively increasing the level of detail"""

import matplotlib.pyplot as plt
import numpy as np

import sdftoolbox


def main():

    # Setup the scene
    scene = sdftoolbox.sdfs.Sphere.create(center=(0, 0, 0), radius=1.0)

    fig, ax = sdftoolbox.plotting.create_figure(fig_aspect=9 / 16, proj_type="persp")

    max_corner = np.array([-np.inf, -np.inf, -np.inf])
    min_corner = np.array([np.inf, np.inf, np.inf])

    # Note, the resolution is chosen such that stepping in powers of 2
    # always contains the endpoint. This is important, since the sampling
    # bounds are close the surface of the sphere.
    grid = sdftoolbox.Grid(
        res=(65, 65, 65),
        min_corner=(-1.1, -1.1, -1.1),
        max_corner=(1.1, 1.1, 1.1),
    )

    for idx in range(1, 6):
        step = 2**idx
        verts, faces = sdftoolbox.dual_isosurface(
            scene,
            grid.subsample(step),
            triangulate=False,
        )
        # The lower the resolution the higher the chance of violating the
        # linearity assumptions when determining edge intersections. Here we
        # improve by reprojecting vertices onto the SDF. This also counterfights
        # shrinkage induced by vertex placement strategies.
        verts = sdftoolbox.mesh.project_vertices(scene, verts)
        verts += (idx * 3, 0, 0)
        max_corner = np.maximum(verts.max(0), max_corner)
        min_corner = np.minimum(verts.min(0), min_corner)
        sdftoolbox.plotting.plot_mesh(ax, verts, faces)
    sdftoolbox.plotting.setup_axes(ax, min_corner, max_corner, num_grid=0)

    plt.show()


if __name__ == "__main__":
    main()
