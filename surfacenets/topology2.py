from random import sample


import numpy as np


class VoxelTopology:
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

    # ccw along +edge dir, always starting at max voxel
    EDGE_VOXEL_OFFSETS = np.array(
        [
            [  # i
                [0, 0, 0],
                [0, 0, -1],
                [0, -1, -1],
                [0, -1, 0],
            ],
            [  # j
                [0, 0, 0],
                [-1, 0, 0],
                [-1, 0, -1],
                [0, 0, -1],
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

    def __init__(
        self, sample_shape: tuple[int, int, int], padding: int = 1
    ) -> None:
        self.sample_shape = sample_shape
        self.padding = padding
        self.edge_shape = (
            sample_shape[0] - padding,
            sample_shape[1] - padding,
            sample_shape[2] - padding,
            3,
        )
        self.num_edges = np.prod(self.edge_shape)

    def ravel_nd(
        self, nd_indices: np.ndarray, shape: tuple[int, ...]
    ) -> np.ndarray:
        return np.ravel_multi_index(list(nd_indices.T), dims=shape)

    def unravel_nd(
        self, indices: np.ndarray, shape: tuple[int, ...]
    ) -> np.ndarray:
        ur = np.unravel_index(indices, shape)
        return np.stack(ur, -1)

    def find_edge_vertices(
        self, edges: np.ndarray, ravel: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        edges = np.asarray(edges, dtype=np.int32)
        if edges.ndim == 1:
            edges = self.unravel_nd(edges, self.edge_shape)
        offs = np.eye(3, dtype=np.int32)
        src = edges[..., :3]
        dst = src + offs[edges[..., -1]]
        if ravel:
            src = self.ravel_nd(src, self.sample_shape)
            dst = self.ravel_nd(dst, self.sample_shape)
        return src, dst

    def find_voxels_sharing_edge(
        self, edges: np.ndarray, ravel: bool = True
    ) -> np.ndarray:
        """Returns all voxel neighbors sharing the given edge."""
        edges = np.asarray(edges, dtype=np.int32)
        if edges.ndim == 1:
            edges = self.unravel_nd(edges, self.edge_shape)
        voxels = edges[..., :3]
        elabels = edges[..., -1]

        neighbors = (
            np.expand_dims(voxels, -2)
            + VoxelTopology.EDGE_VOXEL_OFFSETS[elabels]
        )  # (N,4,3)

        mask = (neighbors >= self.padding) & (
            neighbors < np.array(self.sample_shape) - 2 * self.padding
        )
        mask = mask.all(-1).all(-1)
        neighbors = neighbors[mask]

        if ravel:
            neighbors = self.ravel_nd(
                neighbors.reshape(-1, 3), self.sample_shape
            ).reshape(-1, 4)
        return neighbors

    def find_voxel_edges(
        self, voxels: np.ndarray, ravel: bool = True
    ) -> np.ndarray:
        """Returns all edges for the given voxels

        Params:
            voxels: (N,) or (N,3) index array

        Returns:
            edges: (N,12) edge index array
        """
        voxels = np.asarray(voxels, dtype=np.int32)
        if voxels.ndim == 1:
            voxels = self.unravel_nd(voxels, self.sample_shape)
        N = voxels.shape[0]

        voxels = np.expand_dims(
            np.concatenate((voxels, np.zeros((N, 1), dtype=np.int32)), -1), -2
        )
        edges = voxels + VoxelTopology.VOXEL_EDGE_OFFSETS
        if ravel:
            edges = self.ravel_nd(
                edges.reshape(-1, 4), self.edge_shape
            ).reshape(-1, 12)
        return edges


if __name__ == "__main__":
    top = VoxelTopology((2, 2, 2))

    xyz = np.zeros((4, 4, 4), dtype=np.int32)

    print(top.unravel_nd([0, 1, 2, 3], top.edge_shape))

    print(top.find_edge_vertices([0, 1, 2, 3], ravel=False))

    sijk, tijk = top.find_edge_vertices(top.edge_ids, ravel=False)
    si, sj, sk = sijk.T
    ti, tj, tk = tijk.T

    xyz[si, sj, sk] = 1
    xyz[ti, tj, tk] = 1

    print(top.find_edge_vertices([0], ravel=False))
    vijk = top.find_voxels_sharing_edge([0], ravel=False).squeeze(0)
    vi, vj, vk = vijk.T
    print(vijk)
    xyz[:] = 0
    xyz[vi, vj, vk] = 1

    print(np.flip(xyz.transpose((2, 1, 0)), 1))

    vijk = top.find_voxels_sharing_edge(top.edge_ids, ravel=False).reshape(
        -1, 3
    )
    vi, vj, vk = vijk.T
    xyz[:] = 0
    xyz[vi, vj, vk] = 1

    print(np.flip(xyz.transpose((2, 1, 0)), 1))

    print(top.find_voxel_edges([[1, 1, 1]], ravel=False))

    # print(xyz.transpose((2, 0, 1)))
    # print(xyz.transpose((2, 0, 1)))

    # edge_ijke = top.unravel_nd(top.edge_ids, top.edge_shape)

    # print(top.num_edges)

    # sidx, tidx = top.find_edge_vertices(top.edge_ids)
    # print(sidx[0], top.unravel_nd(sidx[0], top.ext_sample_shape))
    # print(tidx[0], top.unravel_nd(tidx[0], top.ext_sample_shape))

    # # print(top.unravel_nd(sidx[25], top.sample_shape))
    # # print(top.unravel_nd(tidx[25], top.sample_shape))

    # print(top.find_voxels_sharing_edge([0], ravel=False))
    # print(top.find_voxel_edges([0], ravel=False))

    # print(sidx[5], tidx[5])
