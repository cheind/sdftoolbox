from distutils.fancy_getopt import fancy_getopt
from tkinter.tix import Tree
import numpy as np


def compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Computes face normals for the given mesh."""
    xyz = verts[faces]  # (N,F,3)
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, -1] - xyz[:, 0], axis=-1)  # (N,3)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals


def compute_vertex_normals(
    verts: np.ndarray, faces: np.ndarray, face_normals: np.ndarray
) -> np.ndarray:
    # Repeat face normal for each face vertex
    vertex_normals = np.zeros_like(verts)
    vertex_counts = np.zeros((verts.shape[0]), dtype=verts.dtype)

    for f, fn in zip(faces, face_normals):
        vertex_normals[f] += fn
        vertex_counts[f] += 1

    vertex_normals /= vertex_counts.reshape(-1, 1)
    return vertex_normals


if __name__ == "__main__":
    from .surface_nets import surface_nets
    from .sampling import sample_volume
    from .sdfs import Plane, Sphere
    from . import plotting

    # p = Plane.create(origin=(0.1, 0.1, 0.1))
    p = Sphere.create()
    xyz, spacing = sample_volume(res=(3, 3, 3))
    sdf = p.sample(xyz)

    verts, faces = surface_nets(sdf, spacing)
    face_normals = compute_face_normals(verts, faces)
    print(face_normals)

    print(compute_vertex_normals(verts, faces, face_normals))
