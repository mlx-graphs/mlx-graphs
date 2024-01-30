import mlx.core as mx

from mlx_graphs.utils import get_num_hops
import mlx.nn as nn
from mlx_graphs.nn import GCNConv


def test_get_num_hops():
    class GCN(nn.Module):
        def __init__(self):
            super(GCN, self).__init__()

            self.conv1 = GCNConv(4, 16)
            self.conv2 = GCNConv(16, 16)
            self.lin = nn.Linear(16, 2)

        def __call__(self, edge_index: mx.array, node_features: mx.array) -> mx.array:
            x = nn.relu(self.conv1(node_features, edge_index))
            x = self.conv2(node_features, edge_index)
            return self.lin(x)

    assert get_num_hops(GCN()) == 2, "get_num_hops failed"
