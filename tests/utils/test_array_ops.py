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
