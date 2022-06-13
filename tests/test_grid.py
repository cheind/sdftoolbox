import numpy as np
from numpy.testing import assert_allclose
from sdftoolbox import Grid
from sdftoolbox.utils import reorient_volume


def test_correct_shapes():
    g = Grid((2, 2, 2))
    assert g.shape == (2, 2, 2)
    assert g.padded_shape == (3, 3, 3)
    assert g.num_edges == np.prod((2, 2, 2, 3))
    assert g.edge_shape == (2, 2, 2, 3)


def test_find_edge_vertices():
    g = Grid((2, 2, 2))
    test = np.zeros(g.padded_shape, dtype=np.int32)

    sijk, tijk = g.find_edge_vertices(range(g.num_edges), ravel=False)
    si, sj, sk = sijk.T
    ti, tj, tk = tijk.T

    test[si, sj, sk] = 1

    assert_allclose(
        reorient_volume(test),
        np.array(
            [
                [  # first ij slice (origin lower left)
                    [0, 0, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                ],
                [
                    [0, 0, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            ]
        ),
    )

    test[:] = 0
    test[ti, tj, tk] = 1
    # print(repr(reorient_volume(test)))
    assert_allclose(
        reorient_volume(test),
        np.array(
            [
                [  # first ij slice (origin lower left)
                    [1, 1, 0],
                    [1, 1, 1],
                    [0, 1, 1],
                ],
                [
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                ],
                [
                    [0, 0, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                ],
            ]
        ),
    )


# def test_correct_voxel_indices():
#     t = VolumeTopology((4, 4, 4))
#     assert t.voxel_indices.shape == (3, 3, 3, 3)

#     assert_allclose(t.voxel_indices[0, 0, 0], (0, 0, 0))
#     assert_allclose(t.voxel_indices[0, 1, 0], (0, 1, 0))
#     assert_allclose(t.voxel_indices[0, 1, 1], (0, 1, 1))


# def test_correct_edge_indices():
#     t = VolumeTopology((3, 3, 3))

#     # No duplicate edge indices
#     edge_set = {tuple(e.tolist()) for e in t.edge_indices}
#     assert len(edge_set) == len(t.edge_indices)

#     # No edge out of bounds
#     low = (t.edge_indices[:, :3] >= 0).all()
#     high = (t.edge_indices[:, :3] < (3, 3, 3)).all()
#     assert low and high


# def test_source_target_indices():
#     t = VolumeTopology((3, 3, 3))
#     e = np.array(
#         [
#             [0, 0, 0, 1],
#             [1, 1, 1, 0],
#             [1, 1, 1, 1],
#             [1, 1, 1, 2],
#         ]
#     )
#     sources = t.edge_sources(e)
#     targets = t.edge_targets(e)
#     assert_allclose(sources, e[:, :3])
#     assert_allclose(
#         targets,
#         [
#             [0, 1, 0],
#             [2, 1, 1],
#             [1, 2, 1],
#             [1, 1, 2],
#         ],
#     )


# def test_correct_edge_neighbors():
#     t = VolumeTopology((3, 3, 3))
#     ns, mask = t.edge_neighbors([[0, 0, 0, 1], [1, 1, 1, 0], [2, 2, 2, 0]])
#     assert ns.shape == (3, 4, 3)
#     assert mask.shape == (3, 4)

#     assert_allclose(
#         ns,
#         [
#             [
#                 [0, 0, 0],
#                 [-1, 0, 0],
#                 [-1, 0, -1],
#                 [0, 0, -1],
#             ],
#             [
#                 [1, 1, 1],
#                 [1, 0, 1],
#                 [1, 0, 0],
#                 [1, 1, 0],
#             ],
#             [
#                 [2, 2, 2],
#                 [2, 1, 2],
#                 [2, 1, 1],
#                 [2, 2, 1],
#             ],
#         ],
#     )

#     assert_allclose(
#         mask,
#         [
#             [1, 0, 0, 0],
#             [1, 1, 1, 1],
#             [1, 1, 1, 1],
#         ],
#     )


# def test_correct_unique_voxels():
#     t = VolumeTopology((4, 4, 4))
#     uv = t.unique_voxels(
#         [
#             [0, 1, 0],
#             [0, 2, 1],
#             [0, 2, 1],
#             [0, 0, 0],
#         ]
#     )
#     assert_allclose(
#         uv,
#         [
#             [0, 0, 0],
#             [0, 1, 0],
#             [0, 2, 1],
#         ],
#     )
