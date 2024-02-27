import importlib.util

import mlx.core as mx
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.utils.convert import to_networkx

networkx_spec = importlib.util.find_spec("networkx")
networkx_installed = networkx_spec is not None


@pytest.mark.skipif(not networkx_installed, reason="networkx is not installed")
def test_to_networkx():
    edge_index = mx.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    node_features = mx.array([[1], [1], [1], [1]])

    graph = GraphData(node_features=node_features, edge_index=edge_index)

    networkx_graph = to_networkx(graph)
    print(networkx_graph.number_of_nodes())

    assert networkx_graph.number_of_nodes() == 4
    assert networkx_graph.number_of_edges() == len(edge_index[0])
