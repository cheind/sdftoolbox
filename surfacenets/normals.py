import numpy as np


def compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Computes face normals for the given mesh.

    This assumes that faces are ordered ccw when viewed from the face normal.

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
    vertex_counts = np.zeros((verts.shape[0]), dtype=verts.dtype)

    for f, fn in zip(faces, face_normals):
        vertex_normals[f] += fn
        vertex_counts[f] += 1

    vertex_normals /= vertex_counts.reshape(-1, 1)
    return vertex_normals
