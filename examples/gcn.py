import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv.gcn_conv import GCNConv


class GCN(nn.Module):
    r"""Graph Convolutional Network implementation [1]

    Args:
        x_dim (int): size of input node features
        h_dim (int): size of hidden node embeddings
        bias (bool): whether to use bias in the node projection

    References:
        [1] Kipf et al. Semi-Supervised Classification with Graph Convolutional Networks.
        https://arxiv.org/abs/1609.02907
    """

    def __init__(
        self,
        x_dim: int,
        h_dim: int,
        out_dim: int,
        nb_layers: int = 2,
        dropout: float = 0.5,
        bias: bool = True,
    ):
        super(GCN, self).__init__()

        layer_sizes = [x_dim] + [h_dim] * nb_layers + [out_dim]
        self.gcn_layers = [
            GCNConv(in_dim, out_dim, bias)
            for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x: mx.array, edge_index: mx.array) -> mx.array:
        for layer in self.gcn_layers[:-1]:
            x = nn.relu(layer(x, edge_index))
            x = self.dropout(x)

        x = self.gcn_layers[-1](x, edge_index)
        return x
