import numpy as np
import sdftoolbox
import pytest


def test_anisotropic_scaling_not_supported():
    scene = sdftoolbox.sdfs.Sphere.create(radius=2)

    with pytest.raises(ValueError):
        scene.t_world_local = np.diag([1.0, 2.0, 3.0])


def test_sphere():
    scene = sdftoolbox.sdfs.Sphere.create(radius=2)

    sdfv = scene.sample(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    assert np.allclose(sdfv, [-2.0, -1.0, 0.0])


def test_plane():
    scene = sdftoolbox.sdfs.Plane.create(origin=(1.0, 1.0, 1.0), normal=(1.0, 0.0, 0.0))

    sdfv = scene.sample(np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert np.allclose(sdfv, [-1.0, -2.0, 0.0], atol=1e-5)


def test_box():
    scene = sdftoolbox.sdfs.Box((1, 1, 1))

    sdfv = scene.sample(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-0.5, -0.5, -0.5],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
            ]
        )
    )
    assert np.allclose(sdfv, [-0.5, 0.0, 0.5, 0.0, 0.0, 0.8660254], atol=1e-3)

    scene = sdftoolbox.sdfs.Box((1.1, 1.1, 1.1))
    sdfv = scene.sample(
        np.array(
            [
                [-1.0, 0.0, 0.0],
                [-0.333333333333, 0.0, 0.0],
            ]
        )
    )
    assert np.allclose(sdfv, [0.45, -0.216666667], atol=1e-3)

    scene = sdftoolbox.sdfs.Box((1.1, 1.1, 1.1))
    scene = sdftoolbox.sdfs.Transform(
        scene, sdftoolbox.maths.rotate([1, 0, 0], np.pi / 4)
    )
    sdfv = scene.sample(
        np.array(
            [
                [-1.0, 0.0, 0.0],
                [-0.333333333333, 0.0, 0.0],
            ]
        )
    )
    assert np.allclose(sdfv, [0.45, -0.216666667], atol=1e-3)


def test_root():
    scene = sdftoolbox.sdfs.Box((1.1, 1.1, 1.1))
    scene = sdftoolbox.sdfs.Transform(
        scene, sdftoolbox.maths.rotate([1, 0, 0], np.pi / 4)
    )
    xyz = np.array(
        [
            [-1.0, -0.333333333333, -0.333333333333],
            [-0.333333333333, -0.333333333333, -0.333333333333],
        ]
    )
    sdfv = scene.sample(xyz)
    assert np.allclose(sdfv, [0.45, -0.07859548], atol=1e-3)

    x0 = xyz[1] + (-0.07859548, 0.0, 0.0)
    print(scene.sample([x0]))
    print(scene.gradient(x0.reshape(1, 3)))


def test_gradients():
    scene = sdftoolbox.sdfs.Sphere.create(radius=1)

    def analytic(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    g = analytic(xyz)

    g_central = scene.gradient(xyz, mode="central")
    assert np.allclose(g_central, g, atol=1e-3)


def test_discretized():
    scene = sdftoolbox.sdfs.Sphere.create(radius=1)
    grid = sdftoolbox.Grid(res=(20, 20, 20))
    sdf_scene = scene.sample(grid.xyz)
    disc = sdftoolbox.sdfs.Discretized(grid, sdf_scene)
    sdf_disc = disc.sample(grid.xyz)
    assert np.allclose(sdf_scene, sdf_disc)

    # Shift coords test
    scene = sdftoolbox.sdfs.Plane.create()
    grid = sdftoolbox.Grid(res=(3, 3, 3), min_corner=(-1, -1, -1), max_corner=(1, 1, 1))
    sdf_scene = scene.sample(grid.xyz)
    disc = sdftoolbox.sdfs.Discretized(grid, sdf_scene)

    from sdftoolbox.utils import reorient_volume

    sdf_disc = reorient_volume(disc.sample(grid.xyz + (0.0, 0.0, 0.5)))
    assert np.allclose(sdf_disc[0], -0.5)
    assert np.allclose(sdf_disc[1], 0.5)
    assert np.allclose(sdf_disc[2], 1.0, atol=1e-5)  # out of
