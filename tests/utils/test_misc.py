import mlx.core as mx

from mlx_graphs.utils import get_num_hops
import mlx.nn as nn
from mlx_graphs.nn import GCNConv


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
