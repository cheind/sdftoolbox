"""Introductory example to surfacenet usage"""


def main():

    # Main import
    import surfacenets as sn

    # Setup a snowman-scene
    scene = sn.sdfs.Union(
        [
            sn.sdfs.Sphere.create(center=(0, 0, 0), radius=0.4),
            sn.sdfs.Sphere.create(center=(0, 0, 0.45), radius=0.3),
            sn.sdfs.Sphere.create(center=(0, 0, 0.8), radius=0.2),
        ],
        alpha=8,
    )
    # Generate the sampling locations. Here we use the default params
    grid = sn.Grid(res=(32, 32, 32))

    # Extract the surface using dual contouring
    verts, faces = sn.dual_isosurface(
        scene,
        grid,
        strategy=sn.DualContouringStrategy(),
        triangulate=False,
    )

    # Export
    sn.io.export_stl("surfacenets.stl", verts, faces)

    # Visualize
    import matplotlib.pyplot as plt

    # plt.style.use("dark_background")
    fig, ax = sn.plotting.create_mesh_figure(verts, faces)
    plt.show()


if __name__ == "__main__":
    main()
