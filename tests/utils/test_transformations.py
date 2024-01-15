import mlx.core as mx
from mlx_graphs.utils.transformations import (
    to_edge_index,
    to_sparse_adjacency_matrix,
    check_adjacency_matrix,
    to_adjacency_matrix,
)
import pytest


@pytest.mark.parametrize(
    "x, expected_exception",
    [
        (mx.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), None),  # Valid matrix
        (mx.array([[0, 1, 0], [1, 0, 1]]), ValueError),  # Non-square matrix
        (mx.array([0, 1, 0, 1, 0, 1]), ValueError),  # 1D matrix
        (mx.array([[[0]]]), ValueError),  # 3D matrix
    ],
)
def test_check_adjacency_matrix(x, expected_exception):
    @check_adjacency_matrix
    def foo(adjacency_matrix):
        return True

    if expected_exception:
        with pytest.raises(expected_exception):
            foo(adjacency_matrix=x)
    else:
        assert (
            foo(adjacency_matrix=x) is True
        ), "Input with valid adjacency matrix failed"


@pytest.mark.parametrize(
    "dtype",
    [
        (mx.uint8),
        (mx.uint16),
        (mx.uint32),
        (mx.uint64),
        (mx.int8),
        (mx.int16),
        (mx.int32),
        (mx.int64),
    ],
)
def test_to_edge_index_dtype(dtype):
    matrix = mx.array([[0, 1], [3, 0]])
    edge_index = to_edge_index(matrix, dtype=dtype)
    assert edge_index.dtype == dtype, "dtype of returned array incorrect"


def test_to_edge_index():
    matrix = mx.array([[0, 1, 2], [3, 0, 0], [5, 1, 2]])
    edge_index = to_edge_index(matrix)
    assert edge_index.dtype == mx.uint32, "Default dtype of returned array != uint32"

    expected_output = mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])
    assert mx.array_equal(
        edge_index, expected_output
    ), "Incorrectly computed edge index"


def test_to_sparse_adjacency_matrix():
    matrix = mx.array([[0, 1, 2], [3, 0, 0], [5, 1, 2]])
    edge_index, edge_features = to_sparse_adjacency_matrix(matrix)

    expected_index = mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])
    assert mx.array_equal(edge_index, expected_index), "Incorrect computed edge index"
    expected_features = mx.array([1, 2, 3, 5, 1, 2])
    assert mx.array_equal(edge_features, expected_features), "Incorrect edge features"


def test_to_adjacency_matrix():
    edge_index = mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])
    edge_features = mx.array([1, 2, 3, 5, 1, 2])

    # 3 nodes
    num_nodes = 3
    expected_binary_matrix = mx.array([[0, 1, 1], [1, 0, 0], [1, 1, 1]])
    adj_matrix = to_adjacency_matrix(edge_index, num_nodes)
    assert mx.array_equal(
        expected_binary_matrix, adj_matrix
    ), "Incorrect conversion to adjacency matrix"
    expected_matrix = mx.array([[0, 1, 2], [3, 0, 0], [5, 1, 2]])
    weighted_adj_matrix = to_adjacency_matrix(edge_index, num_nodes, edge_features)
    assert mx.array_equal(
        expected_matrix, weighted_adj_matrix
    ), "Incorrect conversion to weighted adjacency matrix"

    # 4 nodes (extra padding)
    num_nodes = 4
    expected_binary_matrix = mx.array(
        [[0, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]]
    )
    adj_matrix = to_adjacency_matrix(edge_index, num_nodes)
    assert mx.array_equal(
        expected_binary_matrix, adj_matrix
    ), "Incorrect conversion to adjacency matrix"
    expected_matrix = mx.array([[0, 1, 2, 0], [3, 0, 0, 0], [5, 1, 2, 0], [0, 0, 0, 0]])
    weighted_adj_matrix = to_adjacency_matrix(edge_index, num_nodes, edge_features)
    assert mx.array_equal(
        expected_matrix, weighted_adj_matrix
    ), "Incorrect conversion to weighted adjacency matrix"

    # 2 nodes (expect error as there are 3 in index)
    with pytest.raises(ValueError):
        to_adjacency_matrix(edge_index, num_nodes=2)

    # non 2D edge_index
    with pytest.raises(ValueError):
        to_adjacency_matrix(edge_index.reshape([1, 2, 6]), 3)

    # column edge_index
    with pytest.raises(ValueError):
        to_adjacency_matrix(edge_index.transpose(), 3)

    # more features than edges
    with pytest.raises(ValueError):
        edge_features = mx.array([1, 2, 3, 5, 1, 2, 5])
        to_adjacency_matrix(edge_index, 3, edge_features)

    # less features than edges
    with pytest.raises(ValueError):
        edge_features = mx.array([1, 2, 3, 5, 1])
        to_adjacency_matrix(edge_index, 3, edge_features)
