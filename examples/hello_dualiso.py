"""Introductory example to surfacenet usage"""


def main():

    # Main import
    import surfacenets as sn

    # Setup a snowman-scene
    snowman = sn.sdfs.Union(
        [
            sn.sdfs.Sphere.create(center=(0, 0, 0), radius=0.4),
            sn.sdfs.Sphere.create(center=(0, 0, 0.45), radius=0.3),
            sn.sdfs.Sphere.create(center=(0, 0, 0.8), radius=0.2),
        ],
    )
    family = sn.sdfs.Union(
        [
            snowman.transform(trans=(-0.75, 0.0, 0.0)),
            snowman.transform(trans=(0.0, -0.3, 0.0), scale=0.8),
            snowman.transform(trans=(0.75, 0.0, 0.0), scale=0.6),
        ]
    )
    scene = sn.sdfs.Difference(
        [
            family,
            sn.sdfs.Plane().transform(trans=(0, 0, -0.2)),
        ]
    )

    # Generate the sampling locations. Here we use the default params
    grid = sn.Grid(
        res=(65, 65, 65),
        min_corner=(-1.5, -1.5, -1.5),
        max_corner=(1.5, 1.5, 1.5),
    )

    # Extract the surface using dual contouring
    verts, faces = sn.dual_isosurface(
        scene,
        grid,
        strategy=sn.NaiveSurfaceNetStrategy(),
        triangulate=False,
    )

    sdfs_after = scene.sample(verts)
    import matplotlib.pyplot as plt

    plt.hist(sdfs_after, bins=50)
    plt.show()
    print(sdfs_after.min(), sdfs_after.max())

    # Export
    sn.io.export_stl("surfacenets.stl", verts, faces)

    # Visualize
    import matplotlib.pyplot as plt

    # plt.style.use("dark_background")
    fig, ax = sn.plotting.create_mesh_figure(verts, faces)
    plt.show()


if __name__ == "__main__":
    main()
