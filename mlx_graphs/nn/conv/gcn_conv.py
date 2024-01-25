from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.utils import degree


class GCNConv(MessagePassing):
    r"""Applies a GCN convolution over input node features.

    Args:
        node_features_dim (int): size of input node features
        out_features_dim (int): size of output node embeddings
        bias (bool): whether to use bias in the node projection
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        bias: bool = True,
    ):
        super(GCNConv, self).__init__(aggr="add")
        self.linear = nn.Linear(node_features_dim, out_features_dim, bias)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        normalize: bool = True,
        **kwargs: Any,
    ) -> mx.array:
        assert edge_index.shape[0] == 2, "edge_index must have shape (2, num_edges)"
        assert (
            edge_index[1].size > 0
        ), "'col' component of edge_index should not be empty"

        node_features = self.linear(node_features)

        row, col = edge_index

        # Compute node degree normalization for the mean aggregation.
        norm: mx.array = None
        if normalize:
            deg = degree(col, node_features.shape[0])
            deg_inv_sqrt = deg ** (-0.5)
            # NOTE : need boolean indexing in order to zero out inf values
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        else:
            norm = mx.ones_like(row)

        # Compute messages and aggregate them with sum and norm.
        node_features = self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_weight": norm},
        )

        return node_features

    def message(
        self,
        src_features: mx.array,
        dst_features: mx.array,
        edge_weight: mx.array = None,
        **kwargs: Any,
    ) -> mx.array:
        return (
            src_features
            if edge_weight is None
            else edge_weight.reshape(-1, 1) * src_features
        )
