import mlx.core as mx
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.sampler.neighbor_sampler import sampler


def test_sampler():
    # Test for batch_size 1 and 2 seed nodes
    edge_index = mx.array([[0, 0, 1, 1, 2, 3, 4], [2, 3, 3, 4, 5, 6, 7]])
    node_features = mx.array([[1], [1], [1], [1], [1], [1], [1], [1]])
    first_subgraph_edge_index = mx.array([[0, 0, 2, 3], [2, 3, 5, 6]])
    second_subgraph_edge_index = mx.array([[1, 1, 3, 4], [3, 4, 6, 7]])
    graph = GraphData(edge_index=edge_index, node_features=node_features)
    subgraphs = sampler(
        graph=graph, input_nodes=[0, 1], num_neighbors=[2, 1], batch_size=1
    )

    assert len(subgraphs) == 2, "Incorrect length of subgraphs"
    assert mx.array_equal(
        mx.sort(subgraphs[0].edge_index), mx.sort(first_subgraph_edge_index)
    ), "Incorrect sampling"
    assert mx.array_equal(
        mx.sort(subgraphs[1].edge_index), mx.sort(second_subgraph_edge_index)
    ), "Incorrect sampling"

    # Test for batch_size 2 and 2 seed nodes
    edge_index = mx.array([[0, 0, 1, 1, 2, 3, 4], [2, 3, 3, 4, 5, 6, 7]])
    node_features = mx.array([[1], [1], [1], [1], [1], [1], [1], [1]])
    subgraph_edge_index = mx.array([[0, 0, 2, 3, 1, 1, 3, 4], [2, 3, 5, 6, 3, 4, 6, 7]])
    graph = GraphData(edge_index=edge_index, node_features=node_features)
    subgraphs = sampler(
        graph=graph, input_nodes=[0, 1], num_neighbors=[2, 1], batch_size=2
    )
    assert len(subgraphs) == 1, "Incorrect length of subgraphs"
    assert mx.array_equal(
        mx.sort(subgraphs[0].edge_index), mx.sort(subgraph_edge_index)
    ), " Incorrect sampling"

    # test for incorrect inputs
    edge_index = mx.array([[0, 0, 1, 1, 2, 3, 4], [2, 3, 3, 4, 5, 6, 7]])
    node_features = mx.array([[1], [1], [1], [1], [1], [1], [1], [1]])
    graph = GraphData(edge_index=edge_index, node_features=node_features)
    with pytest.raises(ValueError):
        subgraphs = sampler(
            graph=graph, input_nodes=[0, 1], num_neighbors=[2, 1], batch_size=0
        )
    with pytest.raises(ValueError):
        subgraphs = sampler(
            graph=graph, input_nodes=[], num_neighbors=[2, 1], batch_size=0
        )
    with pytest.raises(ValueError):
        subgraphs = sampler(
            graph=graph, input_nodes=[0, 1], num_neighbors=[], batch_size=1
        )
    with pytest.raises(ValueError):
        subgraphs = sampler(
            graph=edge_index, input_nodes=[0, 1], num_neighbors=[2, 1], batch_size=1
        )
