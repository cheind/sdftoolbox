"""Compute and render surface normals"""

import matplotlib.pyplot as plt
import surfacenets as sn
import numpy as np


def main():

    # Setup the scene
    scene = sn.sdfs.Box.create()
    # scene.t_world_local = sn.maths.rotate([1, 1, 1], 0.78)
    xyz, spacing = sn.sdfs.Discretized.sampling_coords(res=(10, 10, 10))

    # Generate mesh
    sdfv = scene.sample(xyz)
    verts, faces = sn.dual_isosurface(
        sdfv,
        spacing=spacing,
        strategy=sn.NaiveSurfaceNetStrategy(),
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
    plt.show()


if __name__ == "__main__":
    main()
