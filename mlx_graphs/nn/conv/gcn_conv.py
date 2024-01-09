from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_graphs.nn.message_passing import MessagePassing


class GCNConv(MessagePassing):
    r"""Applies a GCN convolution over input node features.

    Args:
        x_dim (int): size of input node features
        h_dim (int): size of hidden node embeddings
        bias (bool): whether to use bias in the node projection
    """

    def __init__(
        self,
        x_dim: int,
        h_dim: int,
        bias: bool = True,
    ):
        super(GCNConv, self).__init__(aggr="add")
        self.linear = nn.Linear(x_dim, h_dim, bias)

    def __call__(
        self, x: mx.array, edge_index: mx.array, normalize: bool = True, **kwargs: Any
    ) -> mx.array:
        assert edge_index.shape[0] == 2, "edge_index must have shape (2, num_edges)"
        assert edge_index[1].size > 0, "'col' component of edge_index should not be empty"

        x = self.linear(x)

        row, col = edge_index

        # Compute node degree normalization for the mean aggregation.
        norm: mx.array = None
        if normalize:
            deg = self._degree(col, x.shape[0])
            deg_inv_sqrt = deg ** (-0.5)
            # NOTE : need boolean indexing in order to zero out inf values 
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        else:
            norm = mx.ones_like(row)

        # Compute messages and aggregate them with sum and norm.
        x = self.propagate(x=x, edge_index=edge_index, edge_weight=norm)

        return x

    def message(
        self, x_i: mx.array, x_j: mx.array, edge_weight: mx.array=None, **kwargs: Any
    ) -> mx.array:
        return x_i if edge_weight is None else edge_weight.reshape(-1, 1) * x_i

    def _degree(self, index: mx.array, num_edges: int) -> mx.array:
        out = mx.zeros((num_edges,))
        one = mx.ones((index.shape[0],), dtype=out.dtype)

        return mx.scatter_add(out, index, one.reshape(-1, 1), 0)
