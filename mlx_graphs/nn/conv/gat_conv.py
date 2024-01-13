from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.nn as nn

from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.utils import glorot_init, scatter_softmax, gather_src_dst


class GATConv(MessagePassing):
    r"""Graph Attention Network convolution layer.

    Args:
        x_dim (int): size of input node features
        h_dim (int): size of hidden node embeddings
        heads (int): number of attention heads
        concat (bool): whether to use concat of heads or mean reduction
        bias (bool): whether to use bias in the node projection
        negative_slope (float): slope for the leaky relu
        dropout (float): probability p for dropout
    """

    def __init__(
        self,
        x_dim: int,
        h_dim: int,
        heads: int = 1,
        concat: bool = True,
        bias: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_features_dim: Optional[int] = None,
    ):
        super(GATConv, self).__init__(aggr="add")

        self.h_dim = h_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        
        # NOTE: Check to add glorot_init within the Linear layer
        self.lin_proj = nn.Linear(x_dim, heads * h_dim, bias=False)

        self.att_src = glorot_init((1, heads, h_dim))
        self.att_dst = glorot_init((1, heads, h_dim))

        if bias:
            bias_shape = (heads * h_dim) if concat else (h_dim)
            self.bias = mx.zeros(bias_shape)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

        if edge_features_dim is not None:

            # NOTE: Check to add glorot_init within the Linear layer
            self.edge_lin_proj = nn.Linear(edge_features_dim, heads * h_dim, bias=False)
            self.edge_att = glorot_init((1, heads, h_dim))


    def __call__(
        self, x: mx.array, edge_index: mx.array, edge_features: Optional[mx.array] = None, **kwargs: Any
    ) -> mx.array:
        H, C = self.heads, self.h_dim
        
        x_src = x_dst = self.lin_proj(x).reshape(-1, H, C)

        alpha_src = (x_src * self.att_src).sum(-1)
        alpha_dst = (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        alpha_src, alpha_dst = gather_src_dst((alpha_src, alpha_dst), edge_index)
        dst_idx = edge_index[1]

        h = self.propagate(
            x=(x_src, x_dst),
            edge_index=edge_index,
            message_kwargs={
                "alpha_src": alpha_src,
                "alpha_dst": alpha_dst,
                "index": dst_idx,
                "edge_features": edge_features,
            }
        )

        if self.concat:
            h = h.reshape(-1, self.heads * self.h_dim)
        else:
            h = mx.mean(h, axis=1)

        if "bias" in self:
            h = h + self.bias

        return h

    def message(
        self,
        x_i: mx.array,
        x_j: mx.array,
        alpha_src: mx.array=None,
        alpha_dst: mx.array=None,
        index: mx.array=None,
        edge_features: Optional[mx.array]=None,
    ) -> mx.array:

        alpha = alpha_src + alpha_dst

        if edge_features is not None:
            alpha_edge = self._compute_alpha_edge_features(edge_features)
            alpha = alpha + alpha_edge

        num_nodes = self.node_dim[0]

        alpha = nn.leaky_relu(alpha, self.negative_slope)
        alpha = scatter_softmax(alpha, index, num_nodes)
        
        if "dropout" in self:
            alpha = self.dropout(alpha)

        return mx.expand_dims(alpha, -1) * x_i


    def _compute_alpha_edge_features(self, edge_features: mx.array):
        assert "edge_lin_proj" in self and "edge_att" in self, \
            "Using edge features, GATConv layer should be provided argument `edge_features_dim`."

        if edge_features.ndim == 1:
            edge_features = edge_features.reshape(-1, 1)
        
        edge_features = self.edge_lin_proj(edge_features)
        edge_features = edge_features.reshape(-1, self.heads, self.h_dim)
        alpha_edge = (edge_features * self.edge_att).sum(axis=-1)

        return alpha_edge
