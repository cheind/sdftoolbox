import numpy as np
import os
from pathlib import Path


def export_stl(fname: Path, verts: np.ndarray, faces: np.ndarray):
    assert faces.shape[-1] == 3, "STL requires triangles"
    sep = os.linesep
    with open(fname, "w") as fd:
        fd.write("solid surfacenets" + sep)
        for tri in faces:
            a, b, c = verts[tri]
            n = np.cross(b - a, c - a)
            n /= np.linalg.norm(n)
            fd.write(f"facet normal {n[0]:.4e} {n[1]:.4e} {n[2]:.4e}" + sep)
            fd.write("outer loop" + sep)
            for v in [a, b, c]:
                fd.write(f"vertex {v[0]:.4e} {v[1]:.4e} {v[2]:.4e}" + sep)

            fd.write("endloop" + sep)
            fd.write("endfacet" + sep)
        fd.write("endsolid surfacenets" + sep)
