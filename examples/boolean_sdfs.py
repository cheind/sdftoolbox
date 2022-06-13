"""Compute and render surface normals"""

import matplotlib.pyplot as plt
import sdftoolbox
import numpy as np


def extract(scene: sdftoolbox.sdfs.SDF, grid: sdftoolbox.Grid):

    verts, faces = sdftoolbox.dual_isosurface(
        scene,
        grid,
        vertex_strategy=sdftoolbox.DualContouringVertexStrategy(),
        triangulate=False,
    )
    return verts, faces


def main():

    fig, ax = sdftoolbox.plotting.create_figure(fig_aspect=9 / 16, proj_type="persp")
    max_corner = np.array([-np.inf, -np.inf, -np.inf])
    min_corner = np.array([np.inf, np.inf, np.inf])

    box = sdftoolbox.sdfs.Box.create((1, 2, 0.5))
    sphere = sdftoolbox.sdfs.Sphere.create(radius=0.4)
    grid = sdftoolbox.Grid(
        res=(40, 40, 40), min_corner=(-1.2, -1.2, -1.2), max_corner=(1.2, 1.2, 1.2)
    )

    # Union
    scene = box.merge(sphere, alpha=np.inf)
    verts, faces = extract(scene, grid)
    sdftoolbox.plotting.plot_mesh(ax, verts, faces)
    max_corner = np.maximum(verts.max(0), max_corner)
    min_corner = np.minimum(verts.min(0), min_corner)

    # Intersection
    scene = box.intersect(sphere, alpha=np.inf)
    verts, faces = extract(scene, grid)
    verts += (1.5, 0.0, 0.0)
    sdftoolbox.plotting.plot_mesh(ax, verts, faces)
    max_corner = np.maximum(verts.max(0), max_corner)
    min_corner = np.minimum(verts.min(0), min_corner)

    # Difference
    scene = box.subtract(sphere, alpha=np.inf)
    verts, faces = extract(scene, grid)
    verts += (3.0, 0.0, 0.0)
    sdftoolbox.plotting.plot_mesh(ax, verts, faces)
    max_corner = np.maximum(verts.max(0), max_corner)
    min_corner = np.minimum(verts.min(0), min_corner)

    sdftoolbox.plotting.setup_axes(ax, min_corner, max_corner)
    plt.show()


if __name__ == "__main__":
    main()
