from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.kernels.gatv2_edge_scores import _reference_scores, gatv2_edge_scores
from mlx_graphs.nn.linear import Linear, glorot
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing
from mlx_graphs.utils import scatter


@fused_message_passing
class GATv2Conv(MessagePassing):
    """Graph Attention Network convolution layer with dynamic attention.

    Modification of GATConv based off of `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper.

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
        self.att = glorot((1, heads, out_features_dim))

        if bias:
            bias_shape = (heads * out_features_dim) if concat else (out_features_dim)
            self.bias = mx.zeros(bias_shape)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

        if edge_features_dim is not None:
            self.edge_lin_proj = Linear(
                edge_features_dim, heads * out_features_dim, bias=False
            )
            self.edge_att = glorot((1, heads, out_features_dim))

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
        dst_idx = edge_index[1]

        src_feats = self.W_src(node_features).reshape(-1, H, C)
        dst_feats = self.W_dst(node_features).reshape(-1, H, C)

        edge_feats = (
            self._compute_edge_features(edge_features)
            if edge_features is not None
            else None
        )
        # Tie the Metal score path to the existing fused message-passing mode.
        score = (
            gatv2_edge_scores
            if self._can_fuse(edge_index, src_feats, None, None)
            else _reference_scores
        )
        alpha = score(
            src_feats, dst_feats, self.att, edge_index, self.negative_slope, edge_feats
        )
        alpha = scatter(alpha, dst_idx, src_feats.shape[0], aggr="softmax")

        if "dropout" in self:
            alpha = self.dropout(alpha)

        # Weighted sum of neighbor features, one weight per (edge, head).
        node_features = self.propagate(
            node_features=src_feats,
            edge_index=edge_index,
            message_kwargs={"edge_weights": alpha},
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

    def _compute_edge_features(self, edge_features: mx.array):
        assert all(layer in self for layer in ["edge_lin_proj", "edge_att"]), """Using
        edge features, GATConv layer should be provided argument
        `edge_features_dim`."""

        if edge_features.ndim == 1:
            edge_features = edge_features.reshape(-1, 1)

        edge_features = self.edge_lin_proj(edge_features)
        edge_features = edge_features.reshape(-1, self.heads, self.out_features_dim)

        return edge_features
