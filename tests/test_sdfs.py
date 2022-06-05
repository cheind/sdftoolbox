def test_anisotropic_scaling_not_supported():
    scene = sn.sdfs.Sphere.create(radius=2)

    with pytest.raises(ValueError):
        scene.t_world_local = np.diag([1.0, 2.0, 3.0])

