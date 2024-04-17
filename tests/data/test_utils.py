import mlx.core as mx
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.data.utils import (
    validate_list_of_graph_data,
)


@pytest.mark.parametrize(
    "x, expected_exception",
    [
        ([GraphData(edge_index=mx.array([[0], [1]]))], None),  # ok
        ([1, 2, 3], ValueError),  # not list of GraphData
        (
            [GraphData(edge_index=mx.array([[0], [1]])), 1],
            ValueError,
        ),  # list with spurious items
        (
            [
                GraphData(edge_index=mx.array([[0], [1]])),
                GraphData(
                    edge_index=mx.array([[0], [0]]), node_features=mx.array([[1]])
                ),
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
