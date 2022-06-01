import numpy as np
from numpy.testing import assert_allclose
from surfacenets.topology import VolumeTopology


def test_correct_shapes():
    t = VolumeTopology((3, 3, 3))
    assert t.sample_shape == (3, 3, 3)
    assert t.voxel_shape == (2, 2, 2)


def test_correct_voxel_indices():
    t = VolumeTopology((4, 4, 4))
    assert t.voxel_indices.shape == (3, 3, 3, 3)

    assert_allclose(t.voxel_indices[0, 0, 0], (0, 0, 0))
    assert_allclose(t.voxel_indices[0, 1, 0], (0, 1, 0))
    assert_allclose(t.voxel_indices[0, 1, 1], (0, 1, 1))


def test_correct_edge_shape():
    t = VolumeTopology((4, 4, 4))

    # The number of unique edges (in R^3) is
    # the number of sample points, Si*Sj*Sk, times 3 (canonical edges) minus
    # the non-existing boundary edges. For each direction i,j,k we have
    # Sj*Sk, Si*Sk, Si*Sj such boundary edges.
    assert t.edge_indices.shape == (
        np.prod(t.sample_shape) * 3
        - np.sum(
            [
                t.sample_shape[(i + 1) % 3] * t.sample_shape[(i + 2) % 3]
                for i in range(3)
            ]
        ),
        4,
    )


def test_correct_edge_indices():
    t = VolumeTopology((3, 3, 3))

    # No duplicate edge indices
    edge_set = {tuple(e.tolist()) for e in t.edge_indices}
    assert len(edge_set) == len(t.edge_indices)

    # No edge out of bounds
    low = (t.edge_indices[:, :3] >= 0).all()
    high = (t.edge_indices[:, :3] < (3, 3, 3)).all()
    assert low and high


def test_source_target_indices():
    t = VolumeTopology((3, 3, 3))
    e = np.array(
        [
            [0, 0, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 2],
        ]
    )
    sources = t.edge_sources(e)
    targets = t.edge_targets(e)
    assert_allclose(sources, e[:, :3])
    assert_allclose(
        targets,
        [
            [0, 1, 0],
            [2, 1, 1],
            [1, 2, 1],
            [1, 1, 2],
        ],
    )


def test_correct_edge_neighbors():
    t = VolumeTopology((3, 3, 3))
    ns, mask = t.edge_neighbors([[0, 0, 0, 1], [1, 1, 1, 0], [2, 2, 2, 0]])
    assert ns.shape == (3, 4, 3)
    assert mask.shape == (3, 4)

    assert_allclose(
        ns,
        [
            [
                [0, 0, 0],
                [-1, 0, 0],
                [-1, 0, -1],
                [0, 0, -1],
            ],
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 0, 0],
                [1, 1, 0],
            ],
            [
                [2, 2, 2],
                [2, 1, 2],
                [2, 1, 1],
                [2, 2, 1],
            ],
        ],
    )

    assert_allclose(
        mask,
        [
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
    )


def test_correct_unique_voxels():
    t = VolumeTopology((4, 4, 4))
    uv = t.unique_voxels(
        [
            [0, 1, 0],
            [0, 2, 1],
            [0, 2, 1],
            [0, 0, 0],
        ]
    )
    assert_allclose(
        uv,
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 2, 1],
        ],
    )
