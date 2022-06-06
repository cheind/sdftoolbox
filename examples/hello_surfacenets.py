"""Introductory example to surfacenet usage"""


def main():

    # Main import
    import surfacenets as sn

    # Setup the scene
    scene = sn.sdfs.Union(
        [
            sn.sdfs.Sphere.create(center=(0, 0, 0), radius=0.5),
            sn.sdfs.Sphere.create(center=(0, 0, 0.6), radius=0.3),
            sn.sdfs.Sphere.create(center=(0, 0, 1.0), radius=0.2),
        ],
        alpha=8,
    )
    # Generate the sampling locations. Here we use the default params
    xyz, spacing = sn.sdfs.Discretized.sampling_coords()

    # Sample SDF
    sdfv = scene.sample(xyz)

    # Extract the surface using quadliterals
    verts, faces = sn.surface_nets(
        sdfv,
        spacing=spacing,
        vertex_placement_mode="naive",
        triangulate=False,
    )
    verts += xyz[0, 0, 0]

    # Export
    sn.io.export_stl("surfacenets.stl", verts, faces)

    # Visualize
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, ax = sn.plotting.create_mesh_figure(verts, faces)
    plt.show()


if __name__ == "__main__":
    main()
