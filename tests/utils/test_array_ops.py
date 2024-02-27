import mlx.core as mx
import numpy as np

from mlx_graphs.utils.array_ops import pairwise_distances


def test_pairwise_distances():
    from scipy.spatial.distance import cdist

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

    x = mx.random.normal([80, 3])
    y = mx.random.normal([80, 3])
    distances = pairwise_distances(x, y)
    sp_distances = mx.array(cdist(np.array(x), np.array(y)).tolist())
    for i in range(10):
        print(distances[i] - sp_distances[i])
    assert mx.array_equal(distances, sp_distances)
