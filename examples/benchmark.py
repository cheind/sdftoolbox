"""Benchmarking different methods"""
import logging
import numpy as np

import sdftoolbox as sdftoolbox


def main():
    logging.basicConfig(level=logging.DEBUG)

    scene = sdftoolbox.sdfs.Union(
        [
            sdftoolbox.sdfs.Sphere.create(center=(0, 0, 0), radius=0.5),
            sdftoolbox.sdfs.Sphere.create(center=(0, 0, 0.6), radius=0.3),
            sdftoolbox.sdfs.Sphere.create(center=(0, 0, 1.0), radius=0.2),
        ],
        alpha=8,
    )

    xyz, spacing = sdftoolbox.sdfs.Discretized.sampling_coords(res=(100, 100, 100))
    sdfv = scene.sample(xyz).astype(np.float32)
    verts, faces = sdftoolbox.surface_nets(
        sdfv,
        spacing=spacing,
        vertex_placement_mode="naive",
        triangulate=False,
    )
    verts += xyz[0, 0, 0]


if __name__ == "__main__":
    main()
