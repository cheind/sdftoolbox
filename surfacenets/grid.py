import numpy as np

from .types import float_dtype


class Grid:
    """A 3D sampling grid

    This class provides helper methods to determine sampling locations
    and methods to traverse the topology of grids in vectorized fashion.
    """

    """Offsets used to compute the edge indices for a single voxel."""
    VOXEL_EDGE_OFFSETS = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [1, 0, 0, 1],
            [1, 0, 0, 2],
            [0, 1, 0, 0],
            [0, 1, 0, 2],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 2],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ],
        dtype=np.int32,
    ).reshape(1, 12, 4)

    """Offsets for computing voxel indices neighboring a given edge.
    The ordering is such that voxel indices are CCW when looking from
    positive edge dir, always starting with maximum voxel index.
    """
    EDGE_VOXEL_OFFSETS = np.array(
        [
            [  # i
                [0, 0, 0],
                [0, -1, 0],
                [0, -1, -1],
                [0, 0, -1],
            ],
            [  # j
                [0, 0, 0],
                [0, 0, -1],
                [-1, 0, -1],
                [-1, 0, 0],
            ],
            [  # k
                [0, 0, 0],
                [-1, 0, 0],
                [-1, -1, 0],
                [0, -1, 0],
            ],
        ],
        dtype=np.int32,
    )

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
        self.padded_shape = (
            self.xyz.shape[0] + 1,
            self.xyz.shape[1] + 1,
            self.xyz.shape[2] + 1,
        )
        self.edge_shape = self.xyz.shape[:3] + (3,)
        self.num_edges = np.prod(self.edge_shape)

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
        return self.xyz.shape[:3]

    def subsample(self, step: int) -> "Grid":
        """Subsample the grid using every nth sample point."""
        return Grid(xyz=self.xyz[::step, ::step, ::step])

    def ravel_nd(self, nd_indices: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """Convert multi-dimensional indices to a flat indices."""
        return np.ravel_multi_index(list(nd_indices.T), dims=shape)

    def unravel_nd(self, indices: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """Convert flat indices back to a multi-dimensional indices."""
        ur = np.unravel_index(indices, shape)
        return np.stack(ur, -1)

    def find_edge_vertices(
        self, edges: np.ndarray, ravel: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find start/end voxel indices for the given edges.

        Params:
            edges: (N,) or (N,4) array of edge indices
            ravel: Whether to return voxel as flat indices or nd indices

        Returns:
            s: (N,) or (N,3) array of source voxel indices for each edge
            t: (N,) or (N,3) array of target voxel indices for each edge
        """
        edges = np.asarray(edges, dtype=np.int32)
        if edges.ndim == 1:
            edges = self.unravel_nd(edges, self.edge_shape)
        offs = np.eye(3, dtype=np.int32)
        src = edges[:, :3]
        dst = src + offs[edges[:, -1]]
        if ravel:
            src = self.ravel_nd(src, self.padded_shape)
            dst = self.ravel_nd(dst, self.padded_shape)
        return src, dst

    def get_all_edge_vertices(self, ravel: bool = True):
        """Find start/end voxels for all possible edges.

        This method is quite a bit faster than

            find_edge_vertices(range(num_edges))

        because it avoids unravelling. In the current implementation
        this method is is not used in favor of `get_all_source_vertices`,
        however I leave it here for reference.

        Params:
            ravel: Whether to return voxel as flat indices or nd indices.

        Returns:
            s: (N,) or (N,3) array of source voxel indices for each edge
            t: (N,) or (N,3) array of target voxel indices for each edge
        """
        I, J, K = self.edge_shape[:3]
        sijk = (
            np.stack(
                np.meshgrid(
                    np.arange(I, dtype=np.int32),
                    np.arange(J, dtype=np.int32),
                    np.arange(K, dtype=np.int32),
                    indexing="ij",
                ),
                -1,
            )
            .reshape(-1, 3)
            .repeat(3, 0)
            .reshape(-1, 3, 3)
        )
        tijk = sijk + np.eye(3, dtype=np.int32).reshape(1, 3, 3)
        tijk = tijk.reshape(-1, 3)
        sijk = sijk.reshape(-1, 3)
        if ravel:
            sijk = self.ravel_nd(sijk, self.padded_shape)
            tijk = self.ravel_nd(tijk, self.padded_shape)
        return sijk, tijk

    def get_all_source_vertices(self):
        """Find all edge start voxel indices

        Similar to `get_all_edge_vertices` but does not compute
        target voxel indices also does not repeat (x3) the source
        voxel indices for each possible edge direction.

        Returns:
            s: (N,3) array of source voxel indices for each possible edge
        """
        I, J, K = self.edge_shape[:3]
        sijk = np.stack(
            np.meshgrid(
                np.arange(I, dtype=np.int32),
                np.arange(J, dtype=np.int32),
                np.arange(K, dtype=np.int32),
                indexing="ij",
            ),
            -1,
        ).reshape(-1, 3)
        return sijk

    def find_voxels_sharing_edge(
        self, edges: np.ndarray, ravel: bool = True
    ) -> np.ndarray:
        """Returns all voxel neighbors sharing the given edge in ccw order.

        Params:
            edges: (N,) or (N,4) array of edge indices
            ravel: Whether to return voxel as flat indices or nd indices

        Returns:
            v: (N,4) or (N,4,3) of voxel indices in ccw order when viewed from
                position edge direction.
        """
        edges = np.asarray(edges, dtype=np.int32)
        if edges.ndim == 1:
            edges = self.unravel_nd(edges, self.edge_shape)
        voxels = edges[..., :3]
        elabels = edges[..., -1]

        neighbors = (
            np.expand_dims(voxels, -2) + Grid.EDGE_VOXEL_OFFSETS[elabels]
        )  # (N,4,3)

        edge_mask = (neighbors >= 0) & (neighbors < np.array(self.shape) - 1)
        edge_mask = edge_mask.all(-1).all(-1)  # All edges that have 4 valid neighbors
        neighbors[~edge_mask] = 0

        if ravel:
            neighbors = self.ravel_nd(
                neighbors.reshape(-1, 3), self.padded_shape
            ).reshape(-1, 4)
        return neighbors, edge_mask

    def find_voxel_edges(self, voxels: np.ndarray, ravel: bool = True) -> np.ndarray:
        """Finds all edges for the given voxels.

        Params:
            voxels: (N,) or (N,3) voxel indices.
            ravel: Whether to return voxel as flat indices or nd indices

        Returns:
            edges: (N,12) or (N,12,4) edge indices.
        """
        voxels = np.asarray(voxels, dtype=np.int32)
        if voxels.ndim == 1:
            voxels = self.unravel_nd(voxels, self.padded_shape)
        N = voxels.shape[0]

        voxels = np.expand_dims(
            np.concatenate((voxels, np.zeros((N, 1), dtype=np.int32)), -1), -2
        )
        edges = voxels + Grid.VOXEL_EDGE_OFFSETS
        if ravel:
            edges = self.ravel_nd(edges.reshape(-1, 4), self.edge_shape).reshape(-1, 12)
        return edges

    def grid_to_data(self, x: np.ndarray) -> np.ndarray:
        """Convert coordinates in grid space to data space."""
        return x * self.spacing + self.min_corner

    def data_to_grid(self, x: np.ndarray) -> np.ndarray:
        """Convert coordinates in data space to grid space."""
        return (x - self.min_corner) / self.spacing
