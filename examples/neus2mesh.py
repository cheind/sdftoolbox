import logging


def main():
    logging.basicConfig(level=logging.INFO)

    # Main import
    import sdftoolbox

    # Generate the sampling locations. Here we use the default params
    res = [256] * 3
    drange = 4.0
    fpath = "density_e.png"
    grid = sdftoolbox.Grid(
        res=res,
        min_corner=(-1.5, -1.5, -1.5),
        max_corner=(1.5, 1.5, 1.5),
    )

    sdfvalues = sdftoolbox.io.import_volume_from_density_image(
        fpath, res, drange, flip=True
    )

    scene = sdftoolbox.sdfs.Discretized(grid, sdfvalues)

    # Extract the surface using dual contouring
    verts, faces = sdftoolbox.dual_isosurface(
        scene,
        grid,
        triangulate=False,
    )

    # Export
    sdftoolbox.io.export_stl("dual.stl", verts, faces)

    # from skimage.measure import marching_cubes

    # verts_mc, faces_mc, _, _ = marching_cubes(
    #     sdfvalues,
    #     0.0,
    #     spacing=grid.spacing,
    # )
    # verts_mc += grid.min_corner
    # sdftoolbox.io.export_stl("mc.stl", verts_mc, faces_mc)

    # Visualize
    # import matplotlib.pyplot as plt

    # # plt.style.use("dark_background")
    # fig, ax = sdftoolbox.plotting.create_mesh_figure(verts, faces)
    # plt.show()


if __name__ == "__main__":
    main()
