import mlx.core as mx

from mlx_graphs.data import GraphData
from mlx_graphs.sampler.neighbor_sampler import sampler


def test_sampler():
    edge_index = mx.array([[0, 0, 1, 1, 2, 3, 4], [2, 3, 3, 4, 5, 6, 7]])
    node_features = mx.array([[1], [1], [1], [1], [1], [1], [1], [1]])
    first_subgraph_edge_index = mx.array([[0, 0, 2, 3], [2, 3, 5, 6]])
    graph = GraphData(edge_index=edge_index, node_features=node_features)
    subgraphs = sampler(
        graph=graph, input_nodes=[0, 1], num_neighbors=[2, 1], batch_size=1
    )

    assert mx.array_equal(
        mx.sort(subgraphs[0].edge_index), mx.sort(first_subgraph_edge_index)
    )
