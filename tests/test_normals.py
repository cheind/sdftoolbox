import numpy as np
import surfacenets as sn


def test_plane_normals():
    def gen_normals(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scene = sn.sdfs.Plane.create((0.01, 0.01, 0.01), normal=n)
        grid = sn.Grid(res=(3, 3, 3))
        sdfv = scene.sample(grid.xyz)

        # Extract the surface using quadliterals
        verts, faces = sn.dual_isosurface(
            sdfv,
            grid,
            strategy=sn.NaiveSurfaceNetStrategy(),
            triangulate=False,
        )
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
