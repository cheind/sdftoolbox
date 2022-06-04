"""Compute and render surface normals"""

import matplotlib.pyplot as plt
import surfacenets as sn


def main():

    # Setup the scene
    scene = sn.sdfs.Sphere.create()
    xyz, spacing = sn.sample_volume(res=(10, 10, 10))
    sdfv = scene.sample(xyz)

    verts, faces = sn.surface_nets(
        sdfv,
        spacing=spacing,
        vertex_placement_mode="naive",
        triangulate=False,
    )
    verts += xyz[0, 0, 0]

    # Compute normals
    face_normals = sn.normals.compute_face_normals(verts, faces)
    vert_normals = sn.normals.compute_vertex_normals(verts, faces, face_normals)

    # Plot mesh+normals
    fig, ax = sn.plotting.create_mesh_figure(verts, faces, face_normals, vert_normals)
    plt.show()


if __name__ == "__main__":
    main()
