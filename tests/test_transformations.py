import mlx.core as mx
from mlx_graphs.utils.transformations import (
    to_edge_index,
    to_sparse_adjacency_matrix,
    check_adjacency_matrix,
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
        assert foo(adjacency_matrix=x) is True


def test_to_edge_index():
    matrix = mx.array([[0, 1, 2], [3, 0, 0], [5, 1, 2]])
    edge_index = to_edge_index(matrix)
    assert edge_index.dtype == matrix.dtype

    expected_output = mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])
    assert mx.array_equal(edge_index, expected_output)


def test_to_sparse_adjacency_matrix():
    matrix = mx.array([[0, 1, 2], [3, 0, 0], [5, 1, 2]])
    edge_index, edge_features = to_sparse_adjacency_matrix(matrix)
    assert edge_index.dtype == matrix.dtype
    assert edge_features.dtype == matrix.dtype

    expected_index = mx.array([[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 2]])
    assert mx.array_equal(edge_index, expected_index)
    expected_features = mx.array([1, 2, 3, 5, 1, 2])
    assert mx.array_equal(edge_features, expected_features)
