import logging


def main():
    logging.basicConfig(level=logging.INFO)

    # Main import
    import sdftoolbox

    # Generate the sampling locations. Here we use the default params
    res = [64] * 3
    th = 0.0
    drange = 4.0
    fpath = "density_A.png"
    grid = sdftoolbox.Grid(
        res=res,
        min_corner=(-1.5, -1.5, -1.5),
        max_corner=(1.5, 1.5, 1.5),
    )

    sdfvalues = sdftoolbox.io.import_volume_from_density_image(
        fpath,
        res,
        drange,
        th,
    )

    scene = sdftoolbox.sdfs.Discretized(grid, sdfvalues)

    # Extract the surface using dual contouring
    verts, faces = sdftoolbox.dual_isosurface(
        scene,
        grid,
        vertex_strategy=sdftoolbox.DualContouringVertexStrategy(),
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
