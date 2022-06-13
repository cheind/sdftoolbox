import numpy as np
import sdftoolbox


def test_plane_normals():
    def gen_normals(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scene = sdftoolbox.sdfs.Plane.create((0.01, 0.01, 0.01), normal=n)
        grid = sdftoolbox.Grid(res=(3, 3, 3))

        # Extract the surface using quadliterals
        verts, faces = sdftoolbox.dual_isosurface(
            scene,
            grid,
            vertex_strategy=sdftoolbox.NaiveSurfaceNetVertexStrategy(),
            triangulate=False,
        )
        face_normals = sdftoolbox.mesh.compute_face_normals(verts, faces)
        vert_normals = sdftoolbox.mesh.compute_vertex_normals(
            verts, faces, face_normals
        )
        return face_normals, vert_normals

    for n in np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]):
        fn, vn = gen_normals(n=n)
        assert np.allclose(fn, [n], atol=1e-5)
        assert np.allclose(vn, [n], atol=1e-5)

        fn, vn = gen_normals(n=-n)
        assert np.allclose(fn, [-n], atol=1e-5)
        assert np.allclose(vn, [-n], atol=1e-5)
