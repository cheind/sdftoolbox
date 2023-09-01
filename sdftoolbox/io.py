import os
import logging
from pathlib import Path

import imageio
import numpy as np

from .mesh import compute_face_normals, triangulate_quads

_logger = logging.getLogger("sdftoolbox")


def export_stl(
    fname: Path, verts: np.ndarray, faces: np.ndarray, face_normals: np.ndarray = None
):
    """Export mesh to (ASCII) STL format.

    Params:
        fname: output path
        verts: (N,3) array of vertices
        faces: (M,F) array of faces with F=3 for triangles and F=4 for quads
        face_normals: (M,3) array of face normals (optional.)
    """
    if faces.shape[-1] == 4:
        faces = triangulate_quads(faces)
        face_normals = None
    if face_normals is None:
        face_normals = compute_face_normals(verts, faces)
    sep = os.linesep
    with open(fname, "w") as fd:
        fd.write("solid sdftoolbox" + sep)
        for tri, n in zip(faces, face_normals):
            fd.write(f"facet normal {n[0]:.4e} {n[1]:.4e} {n[2]:.4e}" + sep)
            fd.write("outer loop" + sep)
            for v in verts[tri]:
                fd.write(f"vertex {v[0]:.4e} {v[1]:.4e} {v[2]:.4e}" + sep)
            fd.write("endloop" + sep)
            fd.write("endfacet" + sep)
        fd.write("endsolid sdftoolbox" + sep)


def import_volume_from_density_image(
    fname: Path,
    res: tuple[int, int, int],
    density_range: float = 4.0,
    flip: bool = False,
) -> np.ndarray:
    """Loads SDF values from a density 2D image.

    Both, Instant-NGP and NeuS, provide methods to export grid sampled SDF values
    as a single PNG image. Grid positions are uniquely mapped to 2D pixel locations
    in the target image and the SDF values are transformed to intensity values by
    a lossy linear transform

        intensity = int(clip((density-threshold)*scale + 128.5, 0, 255.0))

    where

        scale = 128.0 / density_range

    Note, these equations seem to be slightly wrong. Disregarding the threshold,
    a linear mapping

        y = k*x + d

    from [-range,range] to [0,255] should use d=127.5, k=127.5/range.

    Also note, the `threshold` defines the new zero crossing to differentiate inside
    and outside and should hence not be applied when reading the density image to
    generate sdfs.

    Additionally, when loading instant-ngp files we need to flip the resulting
    sdf values, since in instant-ngp higher density values (positive sdf) represent
    the inside.

    Params:
        fname: input path
        res: grid resolution in x,y,z directions
        density_range: range (+/-) of sdf values mapped to 0..255
        flip: Flip SDF values (i.e change inside/outside). Needed for instant-ngp but
            not for NeuS2

    Returns:
        sdfvalues: (I,J,K) array of SDF values
    See:
        https://github.com/NVlabs/instant-ngp/blob/7d5e858bba5885bbc593fc65e337e4410b992bef/src/marching_cubes.cu#L958

    """
    scale = 128.0 / density_range

    # Load the intensity values as image
    I = np.asarray(imageio.v2.imread(fname)).astype(np.float32)
    # Convert back to 'density' which is SDF in our case
    # See comment in docs for more info
    D = (I - 128.5) / scale
    if flip:
        D *= -1.0

    # Convert pixel coordinates to grid coordinates
    U, V = np.meshgrid(
        np.arange(I.shape[1], dtype=int), np.arange(I.shape[0], dtype=int)
    )
    ndown = int(np.sqrt(res[2]))
    nacross = int(res[2] + ndown - 1) // ndown

    X = U % res[0]
    Y = V % res[1]
    Z = (U // res[0] + (V // res[1]) * nacross).astype(int)
    mask = Z < res[2]
    _logger.info(
        f"Intensity range {I[mask].min()},{I[mask].max()} mapped to SDF range"
        f" {D[mask].min():.3f},{D[mask].max():.3f}"
    )

    # Map 2D -> 3D
    sdf = np.zeros(res)
    sdf[
        X[mask].flatten(),
        res[1] - Y[mask].flatten() - 1,  # adjust to be y-up like NeuS/Instant-NGP
        Z[mask].flatten(),
    ] = D[V[mask].flatten(), U[mask].flatten()]
    return sdf
