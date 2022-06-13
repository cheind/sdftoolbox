import numpy as np
import os
from pathlib import Path

from .mesh import compute_face_normals, triangulate_quads


def export_stl(
    fname: Path, verts: np.ndarray, faces: np.ndarray, face_normals: np.ndarray = None
):
    """Export mesh to (ASCII) STL format.

    Params:
        fname: output path
        verts: (N,3) array of vertices
        faces: (M,F) array of faces with F=3 for triangles and F=4 for quads
        face_normals: (M,3) array of face normals (optional.)
    """
    if faces.shape[-1] == 4:
        faces = triangulate_quads(faces)
        face_normals = None
    if face_normals is None:
        face_normals = compute_face_normals(verts, faces)
    sep = os.linesep
    with open(fname, "w") as fd:
        fd.write("solid surfacenets" + sep)
        for tri, n in zip(faces, face_normals):
            fd.write(f"facet normal {n[0]:.4e} {n[1]:.4e} {n[2]:.4e}" + sep)
            fd.write("outer loop" + sep)
            for v in verts[tri]:
                fd.write(f"vertex {v[0]:.4e} {v[1]:.4e} {v[2]:.4e}" + sep)
            fd.write("endloop" + sep)
            fd.write("endfacet" + sep)
        fd.write("endsolid surfacenets" + sep)
