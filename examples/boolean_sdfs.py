"""Compute and render surface normals"""

import matplotlib.pyplot as plt
import surfacenets as sn
import numpy as np


def extract(scene: sn.sdfs.SDF):
    xyz, spacing = sn.sample_volume(res=(60, 60, 60))
    sdfv = scene.sample(xyz)

    verts, faces = sn.surface_nets(
        sdfv,
        spacing=spacing,
        vertex_placement_mode="naive",
        triangulate=False,
    )
    verts += xyz[0, 0, 0]
    return verts, faces


def main():

    fig, ax = sn.plotting.create_figure(fig_aspect=9 / 16, proj_type="persp")
    max_corner = np.array([-np.inf, -np.inf, -np.inf])
    min_corner = np.array([np.inf, np.inf, np.inf])

    box = sn.sdfs.Box.create((1, 2, 0.5))
    sphere = sn.sdfs.Sphere.create(radius=0.4)

    # Union
    scene = box.merge(sphere, alpha=np.inf)
    verts, faces = extract(scene)
    sn.plotting.plot_mesh(ax, verts, faces)
    max_corner = np.maximum(verts.max(0), max_corner)
    min_corner = np.minimum(verts.min(0), min_corner)

    # Intersection
    scene = box.intersect(sphere, alpha=np.inf)
    verts, faces = extract(scene)
    verts += (1.5, 0.0, 0.0)
    sn.plotting.plot_mesh(ax, verts, faces)
    max_corner = np.maximum(verts.max(0), max_corner)
    min_corner = np.minimum(verts.min(0), min_corner)

    # Difference
    scene = box.subtract(sphere, alpha=np.inf)
    verts, faces = extract(scene)
    verts += (3.0, 0.0, 0.0)
    sn.plotting.plot_mesh(ax, verts, faces)
    max_corner = np.maximum(verts.max(0), max_corner)
    min_corner = np.minimum(verts.min(0), min_corner)

    sn.plotting.setup_axes(ax, min_corner, max_corner)
    plt.show()


if __name__ == "__main__":
    main()
