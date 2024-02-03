import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv import GCNConv
from mlx_graphs.utils.topology import get_num_hops, is_directed, is_undirected


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


def test_get_num_hops():
    class GNN_two_hops(nn.Module):
        def __init__(self):
            super(GNN_two_hops, self).__init__()

            self.conv1 = GCNConv(4, 16)
            self.conv2 = GCNConv(16, 16)
            self.lin = nn.Linear(16, 2)

        def __call__(self, edge_index: mx.array, node_features: mx.array) -> mx.array:
            x = nn.relu(self.conv1(node_features, edge_index))
            x = self.conv2(node_features, edge_index)
            return self.lin(x)

    class GNN_no_hops(nn.Module):
        def __init__(self):
            super(GNN_no_hops, self).__init__()
            self.lin = nn.Linear(16, 2)

        def __call__(self, edge_index: mx.array, node_features: mx.array) -> mx.array:
            return self.lin(node_features)

    assert get_num_hops(GNN_two_hops()) == 2, "get_num_hops failed"
    assert get_num_hops(GNN_no_hops()) == 0, "get_num_hops failed"
