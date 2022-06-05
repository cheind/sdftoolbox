from multiprocessing.sharedctypes import Value
import numpy as np
import surfacenets as sn
import pytest


def test_anisotropic_scaling_not_supported():
    scene = sn.sdfs.Sphere.create(radius=2)

    with pytest.raises(ValueError):
        scene.t_world_local = np.diag([1.0, 2.0, 3.0])


def test_sphere():
    scene = sn.sdfs.Sphere.create(radius=2)

    sdfv = scene.sample(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    assert np.allclose(sdfv, [-2.0, -1.0, 0.0])


def test_plane():
    scene = sn.sdfs.Plane.create(origin=(1.0, 1.0, 1.0), normal=(1.0, 0.0, 0.0))

    sdfv = scene.sample(np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert np.allclose(sdfv, [-1.0, -2.0, 0.0], atol=1e-5)


def test_gradients():
    scene = sn.sdfs.Sphere.create(radius=1)
    xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    ng = scene.gradient(xyz)
    analytic = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    g = analytic(xyz)
    assert np.allclose(ng, g, atol=1e-3)
