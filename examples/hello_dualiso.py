"""Introductory example to surfacenet usage"""


def main():

    # Main import
    import sdftoolbox

    # Setup a snowman-scene
    snowman = sdftoolbox.sdfs.Union(
        [
            sdftoolbox.sdfs.Sphere.create(center=(0, 0, 0), radius=0.4),
            sdftoolbox.sdfs.Sphere.create(center=(0, 0, 0.45), radius=0.3),
            sdftoolbox.sdfs.Sphere.create(center=(0, 0, 0.8), radius=0.2),
        ],
    )
    family = sdftoolbox.sdfs.Union(
        [
            snowman.transform(trans=(-0.75, 0.0, 0.0)),
            snowman.transform(trans=(0.0, -0.3, 0.0), scale=0.8),
            snowman.transform(trans=(0.75, 0.0, 0.0), scale=0.6),
        ]
    )
    scene = sdftoolbox.sdfs.Difference(
        [
            family,
            sdftoolbox.sdfs.Plane().transform(trans=(0, 0, -0.2)),
        ]
    )

    # Generate the sampling locations. Here we use the default params
    grid = sdftoolbox.Grid(
        res=(65, 65, 65),
        min_corner=(-1.5, -1.5, -1.5),
        max_corner=(1.5, 1.5, 1.5),
    )

    # Extract the surface using dual contouring
    verts, faces = sdftoolbox.dual_isosurface(
        scene,
        grid,
        vertex_strategy=sdftoolbox.NaiveSurfaceNetVertexStrategy(),
        triangulate=False,
    )

    # Export
    sdftoolbox.io.export_stl("surfacenets.stl", verts, faces)

    # Visualize
    import matplotlib.pyplot as plt

    # plt.style.use("dark_background")
    fig, ax = sdftoolbox.plotting.create_mesh_figure(verts, faces)
    plt.show()


if __name__ == "__main__":
    main()
