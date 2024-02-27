import importlib.util

import mlx.core as mx
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.utils.convert import to_networkx

networkx_spec = importlib.util.find_spec("networkx")
networkx_installed = networkx_spec is not None


@pytest.mark.skipif(not networkx_installed, reason="networkx is not installed")
def test_to_networkx():
    # GraphData with edge_index and node_features
    edge_index = mx.array([[0, 0, 1, 1, 2, 2, 3], [0, 1, 0, 2, 1, 3, 2]])
    node_features = mx.array([[1], [1], [1], [1]])

    graph = GraphData(node_features=node_features, edge_index=edge_index)

    networkx_graph = to_networkx(graph)

    assert networkx_graph.number_of_nodes() == 4
    assert networkx_graph.number_of_edges() == len(edge_index[0])

    networkx_graph_no_self_loops = to_networkx(graph, remove_self_loops=True)
    assert networkx_graph_no_self_loops.number_of_edges() == len(edge_index[0]) - 1

    # Without node features
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 0]]))
    networkx_graph = to_networkx(graph)

    assert networkx_graph.number_of_nodes() == 2
    assert networkx_graph.number_of_edges() == 2

    # Empty GraphData
    graph = GraphData()
    networkx_graph = to_networkx(graph)

    assert networkx_graph.number_of_nodes() == 0
    assert networkx_graph.number_of_edges() == 0
