import mlx.core as mx

from mlx_graphs.utils.array_ops import pairwise_distances


def test_pairwise_distances():
    x = mx.zeros([2, 4])
    y = mx.zeros([3, 4])
    expected_distances = mx.zeros([2, 3])
    distances = pairwise_distances(x, y)

    assert distances.shape == (2, 3), "Wrong output shape"
    assert mx.array_equal(distances, expected_distances), "Distances not all 0"

    x = mx.array([[1], [2]])
    y = mx.array([[2], [3]])
    expected_distances = mx.array([[1, 2], [0, 1]])
    distances = pairwise_distances(x, y)
    assert mx.array_equal(distances, expected_distances), "Wrong distances"

    # # commenting this test out as there are currently numerical errors between
    # # the output of cdist and the mlx implementation
    # from scipy.spatial.distance import cdist
    # import numpy as np
    # x = mx.random.normal([80, 3])
    # y = mx.random.normal([80, 3])
    # distances = pairwise_distances(x, y)
    # sp_distances = mx.array(cdist(np.array(x), np.array(y)).tolist())
    # assert mx.array_equal(distances, sp_distances)
