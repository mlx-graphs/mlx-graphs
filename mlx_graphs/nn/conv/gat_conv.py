from typing import Any

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
            bias_shape = (heads * h_dim) if concat else (heads)
            self.bias = mx.zeros(bias_shape)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)


    def __call__(
        self, x: mx.array, edge_index: mx.array, normalize: bool = True, **kwargs: Any
    ) -> mx.array:
        H, C = self.heads, self.h_dim
        
        x_src = x_dst = self.lin_proj(x).reshape(-1, H, C)

        alpha_src = (x_src * self.att_src).sum(-1)
        alpha_dst = (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        dst_idx = edge_index[1]
        alpha = self._edge_update(alpha_src, alpha_dst, dst_idx)

        h = self.propagate((x_src, x_dst), edge_index, alpha=alpha)

        if self.concat:
            h = h.reshape(-1, self.heads * self.h_dim)
        else:
            h = h.mean(dim=1)

        if "bias" in self:
            h = h + self.bias

        return h

    def message(
        self, x_i: mx.array, x_j: mx.array, alpha: mx.array=None, **kwargs: Any
    ) -> mx.array:
        return alpha.reshape(*alpha.shape, 1) * x_i

    def _edge_update(self, alpha_j: mx.array, alpha_i: mx.array, index: mx.array):
        # TODO: replace num_nodes by max_nodes, verify it works for GCN
        # max_nodes = index.max().item() + 1
        alpha = alpha_j + alpha_i
        num_nodes = alpha.shape[0]

        alpha = nn.leaky_relu(alpha, self.negative_slope)
        alpha = scatter_softmax(alpha, index, num_nodes)
        
        if "dropout" in self:
            alpha = self.dropout(alpha)

        return alpha
