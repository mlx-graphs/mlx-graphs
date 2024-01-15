import mlx.core as mx
from mlx_graphs.utils.topology import is_undirected, is_directed


def test_is_undirected():
    edge_index = mx.array([[2, 1, 0, 2, 0, 2], [1, 0, 2, 0, 1, 2]])
    assert is_undirected(edge_index) is False

    edge_index = mx.array([[2, 1, 0, 2, 0, 2, 1], [1, 0, 2, 0, 1, 2, 2]])
    assert is_undirected(edge_index) is True

    edge_features = mx.array([5, 1, 2, 2, 1, 6, 5])
    assert is_undirected(edge_index, edge_features) is True

    edge_features = mx.array([5, 1, 2, 2, 1, 6, 1])
    assert is_undirected(edge_index, edge_features) is False


def test_is_directed():
    edge_index = mx.array([[2, 1, 0, 2, 0, 2], [1, 0, 2, 0, 1, 2]])
    assert is_directed(edge_index) is True

    edge_index = mx.array([[2, 1, 0, 2, 0, 2, 1], [1, 0, 2, 0, 1, 2, 2]])
    assert is_directed(edge_index) is False

    edge_features = mx.array([5, 1, 2, 2, 1, 6, 5])
    assert is_directed(edge_index, edge_features) is False

    edge_features = mx.array([5, 1, 2, 2, 1, 6, 1])
    assert is_directed(edge_index, edge_features) is True
