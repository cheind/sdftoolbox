from typing import TYPE_CHECKING

import numpy as np

from .roots import directional_newton_roots

if TYPE_CHECKING:
    from .sdfs import SDF


def compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Computes face normals for the given mesh.

    This assumes that faces are ordered ccw when viewed from the face normal.
    Note that vertices of quads are not guaranteed to be coplanar and hence normals
    depend on which vertices are chosen. This implementation uses always the the first,
    the second and the last vertex to estimate a normal.

    Params:
        verts: (N,3) array of vertices
        faces: (M,F) array of faces with F=3 for triangles and F=4 for quads

    Returns:
        normals: (M,3) array of face normals
    """
    xyz = verts[faces]  # (N,F,3)
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, -1] - xyz[:, 0], axis=-1)  # (N,3)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals


def compute_vertex_normals(
    verts: np.ndarray, faces: np.ndarray, face_normals: np.ndarray
) -> np.ndarray:
    """Computes vertex normals for the given mesh.

    Each vertex normal is the average of the adjacent face normals.

    Params:
        verts: (N,3) array of vertices
        faces: (M,F) array of faces with F=3 for triangles and F=4 for quads
        face_normals: (M,3) array of face normals

    Returns:
        normals: (N,3) array of vertex normals
    """
    # Repeat face normal for each face vertex
    vertex_normals = np.zeros_like(verts)
    vertex_counts = np.zeros((verts.shape[0]), dtype=int)

    for f, fn in zip(faces, face_normals):
        vertex_normals[f] += fn
        vertex_counts[f] += 1

    vertex_normals /= vertex_counts.reshape(-1, 1)
    return vertex_normals


def triangulate_quads(quads: np.ndarray) -> np.ndarray:
    """Triangulates a quadliteral mesh.

    Assumes CCW winding order.

    Params:
        quads: (M,4) array of quadliterals

    Returns:
        tris: (M*2,3) array of triangles
    """
    tris = np.empty((quads.shape[0], 2, 3), dtype=quads.dtype)
    tris[:, 0, :] = quads[:, [0, 1, 2]]
    tris[:, 1, :] = quads[:, [0, 2, 3]]
    return tris.reshape(-1, 3)


def project_vertices(node: "SDF", verts: np.ndarray, **newton_kwargs):
    """Projects vertices onto the surface."""
    return directional_newton_roots(node, verts, **newton_kwargs)
