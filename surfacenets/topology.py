import numpy as np


class VolumeTopology:
    """Helpers for accesing voxel volume topology in vectorized form."""

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

    EDGE_VOXEL_OFFSETS = np.array(
        [
            [
                [0, 0, 0],
                [0, -1, 0],
                [0, -1, -1],
                [0, 0, -1],
            ],
            [
                [0, 0, 0],
                [-1, 0, 0],
                [-1, 0, -1],
                [0, 0, -1],
            ],
            [
                [0, 0, 0],
                [-1, 0, 0],
                [-1, -1, 0],
                [0, -1, 0],
            ],
        ],
        dtype=np.int32,
    )

    def __init__(self, sample_shape: tuple[int, int, int]) -> None:
        self.sample_shape = sample_shape
        self.voxel_shape = (
            sample_shape[0] - 1,
            sample_shape[1] - 1,
            sample_shape[2] - 1,
        )
        self.sample_indices = self._create_voxel_indices(self.sample_shape)
        self.voxel_indices = self.sample_indices[:-1, :-1, :-1]
        self.edge_indices = self._create_edge_indices()

    def _create_voxel_indices(self, voxel_shape: tuple[int, int, int]):
        voxel_indices = np.stack(
            np.meshgrid(
                list(range(voxel_shape[0])),
                list(range(voxel_shape[1])),
                list(range(voxel_shape[2])),
                indexing="ij",
            ),
            -1,
        ).astype(np.int32)
        return voxel_indices

    def _create_edge_indices(self):
        # extend voxel indices by one in each direction (after)
        I, J, K = self.voxel_shape
        V = np.prod(self.sample_shape)
        E = 3
        sample_indices = self.sample_indices.reshape(-1, 3).repeat(E, axis=0)
        edge_ids = np.tile(np.array([0, 1, 2], dtype=np.int32), V).reshape(-1, 1)
        edges = np.concatenate((sample_indices, edge_ids), -1)
        mask = np.logical_and(edges[:, 0] >= I, edges[:, -1] == 0)
        mask |= np.logical_and(edges[:, 1] >= J, edges[:, -1] == 1)
        mask |= np.logical_and(edges[:, 2] >= K, edges[:, -1] == 2)
        return edges[~mask]

    def edge_sources(self, edges: np.ndarray) -> np.ndarray:
        return edges[..., :3]

    def edge_targets(self, edges: np.ndarray) -> np.ndarray:
        offs = np.eye(3, dtype=np.int32)
        return edges[..., :3] + offs[edges[..., -1]]

    def voxel_edges(self, voxels: np.ndarray) -> np.ndarray:
        """Returns all edge indices for the given voxels

        Params:
            voxels: (*,3) index array

        Returns:
            edges: (*,12) index array
        """
        P = voxels.shape[:-1]

        voxels = np.expand_dims(
            np.concatenate((voxels, np.zeros(P + (1,), dtype=np.int32)), -1), -2
        )
        edges = voxels + VolumeTopology.VOXEL_EDGE_OFFSETS
        return edges

    def edge_neighbors(self, edges: np.ndarray) -> np.ndarray:
        """Returns all voxel neighbors sharing the given edge.

        Params:
            edges: (*,4) index array

        Returns:
            voxels: (*,4,3) voxel index array. 4 voxel indices (i,j,k) per
                edge.
            mask: (*, 4) boolean mask to marking valid voxels
        """
        edges = np.asarray(edges, dtype=np.int32)
        voxels = edges[..., :3]
        elabels = edges[..., -1]

        neighbors = (
            np.expand_dims(voxels, -2) + VolumeTopology.EDGE_VOXEL_OFFSETS[elabels]
        )
        mask = np.logical_and(
            neighbors >= 0, neighbors <= np.array(self.voxel_shape)
        ).all(-1)
        return neighbors, mask

    def unique_voxels(self, voxels: np.ndarray) -> np.ndarray:
        # map voxels to unique ids
        voxels = np.asarray(voxels, dtype=np.int32)
        voxels_flat = voxels.reshape(-1, 3)
        ids = np.ravel_multi_index(
            [voxels_flat[:, 0], voxels_flat[:, 1], voxels_flat[:, 2]], self.voxel_shape
        )
        ids = np.unique(ids)
        voxels_flat = np.unravel_index(ids, self.voxel_shape)
        return np.stack(voxels_flat, -1)


if __name__ == "__main__":
    import time

    t0 = time.perf_counter()
    top = VolumeTopology((3, 3, 3))
    print(time.perf_counter() - t0)

    edges = top.edge_indices
    sources = top.edge_sources(edges)
    targets = top.edge_targets(edges)

    xyz = np.stack(np.meshgrid(np.arange(3), np.arange(3), np.arange(3)), -1)

    print(xyz[sources[:, 0], sources[:, 1], sources[:, 2]])
    print(xyz[targets[:, 0], targets[:, 1], targets[:, 2]])

    ve = top.voxel_edges(np.array([[1, 0, 0]]))
    s = top.edge_sources(ve)
    print(sources.shape)
    t = top.edge_targets(ve)

    for sxyz, txyz in zip(s[0], t[0]):
        print(sxyz, txyz)

    print(top.edge_neighbors(np.array([[0, 0, 0, 1]], dtype=np.int32)))
