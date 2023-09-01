"""Convert stored SDF volumes to meshes.

Loads a 'density image' as an SDF volume and triangulates
it using dual contouring.

Density images are generated by tools like

 - instant-ngp https://github.com/NVlabs/instant-ngp 
 - NeuS/S2 https://github.com/19reborn/NeuS2/

by sampling the learnt SDF network at grid locations. The 
resulting SDF values mapped to flat image space, converted
to intensity values and then saved as an ordinary image.

This tool reverses the above process to reconstruct a 3D 
grid of SDF values and then applies the contouring algorithm
to generate the final mesh.
"""

import logging
import argparse

import sdftoolbox

_logger = logging.getLogger("sdftoolbox")


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        help="grid resolution in each direction",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--spacing",
        type=float,
        help=(
            "grid spacing between voxels in each direction. Affects the scaling of the"
            " resulting mesh."
        ),
    )
    parser.add_argument(
        "-t",
        "--sdf-truncation",
        type=float,
        help="+/- sdf range encoded in pixel intensity",
        default=4.0,
    )
    parser.add_argument(
        "-o",
        "--offset",
        type=float,
        help="grid offset in each direction. Affects the origin of the resulting mesh.",
        default=0.0,
    )
    parser.add_argument(
        "--sdf-flip",
        action="store_true",
        help=(
            "flip sdf values. Needed for Instant-NGP but not for NeuS. When set"
            " incorrectly, resulting mesh has wrongly ordered triangles"
        ),
    )
    parser.add_argument(
        "--output", help="Path to resulting mesh (.STL)", default="dual.stl"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show the resulting mesh using matplotlib. Slow for large meshes.",
    )
    parser.add_argument(
        "fpath",
        help="Path to density image (.PNG)",
    )
    parser.add_argument(
        "--vstrategy", choices=["naive", "dual", "mid"], default="naive"
    )

    args = parser.parse_args()
    if args.spacing is None:
        args.spacing = 1.0 / args.resolution

    res = [args.resolution] * 3

    # Load volumentric SDF values from density image atlas
    sdfvalues = sdftoolbox.io.import_volume_from_density_image(
        fname=args.fpath, res=res, density_range=args.sdf_truncation, flip=args.sdf_flip
    )

    # Define the sampling locations associated with SDF values
    grid = sdftoolbox.Grid(
        res=res,
        min_corner=[args.offset] * 3,
        max_corner=[args.offset + args.spacing * args.resolution] * 3,
    )

    # Wrap grid and SDF values i a discrectized SDF volume
    scene = sdftoolbox.sdfs.Discretized(grid, sdfvalues)

    vstrategy = {
        "naive": sdftoolbox.NaiveSurfaceNetVertexStrategy,
        "dual": sdftoolbox.DualContouringVertexStrategy,
        "mid": sdftoolbox.MidpointVertexStrategy,
    }[args.vstrategy]()
    _logger.info(f"Using {vstrategy.__class__.__name__} strategy")

    # Extract the surface using dual contouring
    verts, faces = sdftoolbox.dual_isosurface(
        scene,
        grid,
        triangulate=False,
        vertex_strategy=vstrategy,
        edge_strategy=sdftoolbox.LinearEdgeStrategy(),
    )

    # Export
    _logger.info(f"Saving to {args.output}")
    sdftoolbox.io.export_stl(args.output, verts, faces)

    if args.show:
        import matplotlib.pyplot as plt

        fig, ax = sdftoolbox.plotting.create_mesh_figure(verts, faces)
        plt.show()


if __name__ == "__main__":
    main()