from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.utils import get_src_dst_features, scatter


class GATv2Conv(MessagePassing):
    """Graph Attention Network convolution layer with dynamic attention.

    Modification of GATConv based off of "How Attentive are Graph Attention Networks"

    Args:
        node_features_dim: Size of input node features
        out_features_dim: Size of output node embeddings
        heads: Number of attention heads
        concat: Whether to use concat of heads or mean reduction
        bias: Whether to use bias in the node projection
        negative_slope: Slope for the leaky relu
        dropout: Probability p for dropout
        edge_features_dim: Size of edge features

    Example:

    .. code-block:: python

        conv = GATv2Conv(16, 32, heads=2, concat=True)
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))

        h = conv(edge_index, node_features)

        >>> h.shape
        [5, 64]

    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        heads: int = 1,
        concat: bool = True,
        bias: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_features_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(GATv2Conv, self).__init__(**kwargs)

        self.out_features_dim = out_features_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        # Weights
        self.W_src = Linear(node_features_dim, heads * out_features_dim, bias=False)
        self.W_dst = Linear(node_features_dim, heads * out_features_dim, bias=False)

        # Attention is applied in message() during message passing stage.
        glorot_init = nn.init.glorot_uniform()
        self.att = glorot_init(mx.zeros((1, heads, out_features_dim)))

        if bias:
            bias_shape = (heads * out_features_dim) if concat else (out_features_dim)
            self.bias = mx.zeros(bias_shape)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

        if edge_features_dim is not None:
            self.edge_lin_proj = Linear(
                edge_features_dim, heads * out_features_dim, bias=False
            )
            self.edge_att = glorot_init(mx.zeros((1, heads, out_features_dim)))

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of GATConv.

        Args:
            edge_index: input edge index of shape `[2, num_edges]`
            node_features: input node features
            edge_features: edge features

        Returns:
            mx.array: computed node embeddings
        """
        H, C = self.heads, self.out_features_dim

        src_feats = self.W_src(node_features).reshape(-1, H, C)
        dst_feats = self.W_dst(node_features).reshape(-1, H, C)

        src, dst = get_src_dst_features(edge_index, (src_feats, dst_feats))
        dst_idx = edge_index[1]

        node_features = self.propagate(
            node_features=(src_feats, dst_feats),
            edge_index=edge_index,
            message_kwargs={
                "src": src,
                "dst": dst,
                "index": dst_idx,
                "edge_features": edge_features,
            },
        )

        if self.concat:
            node_features = node_features.reshape(
                -1, self.heads * self.out_features_dim
            )
        else:
            node_features = mx.mean(node_features, axis=1)

        if "bias" in self:
            node_features = node_features + self.bias

        return node_features

    def message(
        self,
        src_features: mx.array,
        dst_features: mx.array,
        src: mx.array,
        dst: mx.array,
        index: mx.array,
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        # Collect alpha before applying non-linearity
        alpha = src + dst
        if edge_features is not None:
            alpha_edge = self._compute_edge_features(edge_features)
            alpha = alpha + alpha_edge

        alpha = nn.leaky_relu(alpha, self.negative_slope)
        # Apply attention after non-linearity to get dynamic attention
        alpha = (alpha * self.att).sum(-1)

        alpha = scatter(alpha, index, self.num_nodes, aggr="softmax")

        if "dropout" in self:
            alpha = self.dropout(alpha)

        return mx.expand_dims(alpha, -1) * src_features

    def _compute_edge_features(self, edge_features: mx.array):
        assert all(layer in self for layer in ["edge_lin_proj", "edge_att"]), """Using
        edge features, GATConv layer should be provided argument
        `edge_features_dim`."""

        if edge_features.ndim == 1:
            edge_features = edge_features.reshape(-1, 1)

        edge_features = self.edge_lin_proj(edge_features)
        edge_features = edge_features.reshape(-1, self.heads, self.out_features_dim)

        return edge_features
