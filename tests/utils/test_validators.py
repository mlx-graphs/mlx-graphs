import mlx.core as mx
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.utils.validators import (
    validate_adjacency_matrix,
    validate_edge_index,
    validate_edge_index_and_features,
    validate_list_of_graph_data,
)


@pytest.mark.parametrize(
    "x, expected_exception",
    [
        (mx.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), None),  # Valid matrix
        (mx.array([[0, 1, 0], [1, 0, 1]]), ValueError),  # Non-square matrix
        (mx.array([0, 1, 0, 1, 0, 1]), ValueError),  # 1D matrix
        (mx.array([[[0]]]), ValueError),  # 3D matrix
    ],
)
def test_validate_adjacency_matrix(x, expected_exception):
    @validate_adjacency_matrix
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
    "x, expected_exception",
    [
        (mx.array([[0, 1, 2, 3], [1, 2, 3, 0]]), None),  # Valid edge_index
        (mx.array([0, 1, 0, 1, 0, 1]), ValueError),  # non 2d edge_index
        (
            mx.array([[0, 1, 2, 3], [1, 2, 3, 0], [1, 2, 3, 4]]),
            ValueError,
        ),  # more than 2 rows
    ],
)
def test_validate_edge_index(x, expected_exception):
    @validate_edge_index
    def foo(edge_index):
        return True

    if expected_exception:
        with pytest.raises(expected_exception):
            foo(edge_index=x)
    else:
        assert foo(edge_index=x) is True, "Input with valid edge index failed"


@pytest.mark.parametrize(
    "x, f, expected_exception",
    [
        (
            mx.array([[0, 1, 2], [1, 2, 3]]),
            mx.array([1, 2, 3]),
            None,
        ),  # Valid index and features
        (
            mx.array([[0, 1, 2], [1, 2, 3]]),
            mx.array([[1, 2, 3], [1, 2, 3]]).T,
            None,
        ),  # Valid index and features
        (
            mx.array([[0, 1, 2], [1, 2, 3]]),
            mx.array([1, 2]),
            ValueError,
        ),  # less features than edges
        (
            mx.array([[0, 1, 2], [1, 2, 3]]),
            mx.array([1, 2, 3, 4]),
            ValueError,
        ),  # more features than edges
    ],
)
def test_validate_edge_index_and_features(x, f, expected_exception):
    @validate_edge_index_and_features
    def foo(edge_index, edge_features):
        return True

    if expected_exception:
        with pytest.raises(expected_exception):
            foo(edge_index=x, edge_features=f)
    else:
        assert (
            foo(edge_index=x, edge_features=f) is True
        ), "Input with valid edge_index and features failed"


@pytest.mark.parametrize(
    "x, expected_exception",
    [
        ([GraphData()], None),  # ok
        ([1, 2, 3], ValueError),  # not list of GraphData
        ([GraphData(), 1], ValueError),  # list with spurious items
        (
            [
                GraphData(edge_index=mx.array([[0], [1]])),
                GraphData(node_features=mx.array([[1]])),
            ],
            ValueError,
        ),  # GraphData with different attributes
    ],
)
def test_validate_list_of_graph_data(x, expected_exception):
    @validate_list_of_graph_data
    def foo(graph_list):
        return True

    if expected_exception:
        with pytest.raises(expected_exception):
            foo(graph_list=x)
    else:
        assert foo(graph_list=x) is True, "Input with valid edge index failed"
