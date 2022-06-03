import numpy as np


def sample_volume(
    res: tuple[int, int, int] = (60, 60, 60),
    min_corner: tuple[float, float, float] = (-2, -2, -2),
    max_corner: tuple[float, float, float] = (2, 2, 2),
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates volumentric sampling locations.

    Params:
        res: resolution for each axis
        min_corner: bounds for the sampling volume
        max_corner: bounds for the sampling volume

    Returns:
        xyz: (I,J,K,3) array of sampling locations
        spacing: (3,) the spatial spacing between two voxels
    """
    ranges = [
        np.linspace(min_corner[0], max_corner[0], res[0], dtype=dtype),
        np.linspace(min_corner[1], max_corner[1], res[1], dtype=dtype),
        np.linspace(min_corner[2], max_corner[2], res[2], dtype=dtype),
    ]

    X, Y, Z = np.meshgrid(*ranges, indexing="ij")
    xyz = np.stack((X, Y, Z), -1)
    spacing = (
        ranges[0][1] - ranges[0][0],
        ranges[1][1] - ranges[1][0],
        ranges[2][1] - ranges[2][0],
    )
    return xyz, spacing
