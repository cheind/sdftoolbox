import numpy as np

from .types import float_dtype


class Grid:
    def __init__(
        self,
        res: tuple[int, int, int] = (33, 33, 33),
        min_corner: tuple[float, float, float] = (-1.0, -1.0, -1.0),
        max_corner: tuple[float, float, float] = (1.0, 1.0, 1.0),
        xyz: np.ndarray = None,
    ):
        if xyz is None:
            xyz = Grid.sampling_coords(res, min_corner, max_corner)
        self.xyz = xyz

    @property
    def spacing(self):
        """The spatial step size in each dimension"""
        return self.xyz[1, 1, 1] - self.xyz[0, 0, 0]

    @property
    def min_corner(self):
        """Minimum sampling point"""
        return self.xyz[0, 0, 0]

    @property
    def max_corner(self):
        """Maximum sampling corner"""
        return self.xyz[-1, -1, -1]

    @property
    def shape(self):
        """Shape of the grid"""
        return self.xyz.shape

    def subsample(self, step: int) -> "Grid":
        """Subsample the grid using every nth sample point."""
        return Grid(xyz=self.xyz[::step, ::step, ::step])

    @staticmethod
    def sampling_coords(
        res: tuple[int, int, int],
        min_corner: tuple[float, float, float],
        max_corner: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates volumentric sampling locations.

        Params:
            res: resolution for each axis
            min_corner: bounds for the sampling volume
            max_corner: bounds for the sampling volume
            dtype: floating point data type of result

        Returns:
            xyz: (I,J,K,3) array of sampling locations
            spacing: (3,) the spatial spacing between two voxels
        """

        ranges = [
            np.linspace(min_corner[0], max_corner[0], res[0], dtype=float_dtype),
            np.linspace(min_corner[1], max_corner[1], res[1], dtype=float_dtype),
            np.linspace(min_corner[2], max_corner[2], res[2], dtype=float_dtype),
        ]

        X, Y, Z = np.meshgrid(*ranges, indexing="ij")
        xyz = np.stack((X, Y, Z), -1)
        return xyz
