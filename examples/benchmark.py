"""Benchmarking different methods"""
import logging
import numpy as np

import surfacenets as sn


def main():
    logging.basicConfig(level=logging.DEBUG)

    scene = sn.sdfs.Union(
        [
            sn.sdfs.Sphere.create(center=(0, 0, 0), radius=0.5),
            sn.sdfs.Sphere.create(center=(0, 0, 0.6), radius=0.3),
            sn.sdfs.Sphere.create(center=(0, 0, 1.0), radius=0.2),
        ],
        alpha=8,
    )

    xyz, spacing = sn.sample_volume(res=(100, 100, 100))
    sdfv = scene.sample(xyz).astype(np.float32)
    verts, faces = sn.surface_nets(
        sdfv,
        spacing=spacing,
        vertex_placement_mode="naive",
        triangulate=False,
    )
    verts += xyz[0, 0, 0]


if __name__ == "__main__":
    main()
