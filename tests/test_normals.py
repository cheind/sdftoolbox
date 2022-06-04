import numpy as np
import surfacenets as sn
from numpy.testing import assert_allclose


def test_plane_normals():
    def gen_normals(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scene = sn.sdfs.Plane.create((0.01, 0.01, 0.01), normal=n)
        xyz, spacing = sn.sample_volume(res=(3, 3, 3))
        sdfv = scene.sample(xyz)

        # Extract the surface using quadliterals
        verts, faces = sn.surface_nets(
            sdfv,
            spacing=spacing,
            vertex_placement_mode="naive",
            triangulate=False,
        )
        verts += xyz[0, 0, 0]
        face_normals = sn.normals.compute_face_normals(verts, faces)
        vert_normals = sn.normals.compute_vertex_normals(verts, faces, face_normals)
        return face_normals, vert_normals

    for n in np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]):
        fn, vn = gen_normals(n=n)
        assert np.allclose(fn, [n], atol=1e-5)
        assert np.allclose(vn, [n], atol=1e-5)

        fn, vn = gen_normals(n=-n)
        assert np.allclose(fn, [-n], atol=1e-5)
        assert np.allclose(vn, [-n], atol=1e-5)