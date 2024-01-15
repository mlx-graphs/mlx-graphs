import mlx.core as mx
from mlx_graphs.utils.topology import sort_edge_index, sort_edge_index_and_features


def test_sort_edge_index():
    edge_index = mx.array([[2, 1, 0, 2, 0, 2], [1, 0, 2, 0, 1, 2]])
    expected_edge_index = mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])
    edge_features = mx.array([5, 3, 2, 4, 1, 6])
    expected_edge_features = mx.array([1, 2, 3, 4, 5, 6])
    s, idx = sort_edge_index(edge_index)
    assert mx.array_equal(expected_edge_index, s)
    assert mx.array_equal(expected_edge_features, edge_features[idx])


def test_sort_edge_index_and_features():
    edge_index = mx.array([[2, 1, 0, 2, 0, 2], [1, 0, 2, 0, 1, 2]])
    expected_edge_index = mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])
    edge_features = mx.array([5, 3, 2, 4, 1, 6])
    expected_edge_features = mx.array([1, 2, 3, 4, 5, 6])
    s, f = sort_edge_index_and_features(edge_index, edge_features)
    assert mx.array_equal(expected_edge_index, s)
    assert mx.array_equal(expected_edge_features, f)
    # test 2d edge_features
    edge_features = mx.array([[5, 3, 2, 4, 1, 6], [2, 4, 5, 3, 6, 1]])
    expected_edge_features = mx.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    s, f = sort_edge_index_and_features(edge_index, edge_features)
    assert mx.array_equal(expected_edge_index, s)
    assert mx.array_equal(expected_edge_features, f)
