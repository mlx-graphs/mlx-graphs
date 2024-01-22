from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.nn.linear import Linear
from mlx_graphs.utils import glorot_init, scatter, get_src_dst_features


class GATConv(MessagePassing):
    r"""Graph Attention Network convolution layer.

    Args:
        node_features_dim (int): Size of input node features
        out_features_dim (int): Size of output node embeddings
        heads (int): Number of attention heads
        concat (bool): Whether to use concat of heads or mean reduction
        bias (bool): Whether to use bias in the node projection
        negative_slope (float): Slope for the leaky relu
        dropout (float): Probability p for dropout
        edge_features_dim (int, optional): Size of edge features

    Example:

    .. code-block:: python

        conv = GATConv(16, 32, heads=2, concat=True)
        x = mx.random.uniform(low=0, high=1, shape=(5, 16))
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])

        h = conv(x, edge_index)
        h.shape
        >>> [5, 64]
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
    ):
        super(GATConv, self).__init__(aggr="add")

        self.out_features_dim = out_features_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.lin_proj = Linear(node_features_dim, heads * out_features_dim, bias=False)

        self.att_src = glorot_init((1, heads, out_features_dim))
        self.att_dst = glorot_init((1, heads, out_features_dim))

        if bias:
            bias_shape = (heads * out_features_dim) if concat else (out_features_dim)
            self.bias = mx.zeros(bias_shape)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

        if edge_features_dim is not None:
            self.edge_lin_proj = Linear(
                edge_features_dim, heads * out_features_dim, bias=False
            )
            self.edge_att = glorot_init((1, heads, out_features_dim))

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Computes the forward pass of GATConv.

        Args:
            node_features (mx.array): input node features
            edge_index (mx.array): input edge index of shape (2, |E|)
            edge_features (optional mx.array) edge features of shape (2, |E|)

        Returns:
            mx.array: computed node embeddings
        """
        H, C = self.heads, self.out_features_dim

        src_feats = dst_feats = self.lin_proj(node_features).reshape(-1, H, C)

        alpha_src = (src_feats * self.att_src).sum(-1)
        alpha_dst = (dst_feats * self.att_dst).sum(-1)

        alpha_src, alpha_dst = get_src_dst_features(edge_index, (alpha_src, alpha_dst))
        dst_idx = edge_index[1]

        node_features = self.propagate(
            node_features=(src_feats, dst_feats),
            edge_index=edge_index,
            message_kwargs={
                "alpha_src": alpha_src,
                "alpha_dst": alpha_dst,
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
        alpha_src: mx.array = None,
        alpha_dst: mx.array = None,
        index: mx.array = None,
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Computes a message for each edge in the graph following GAT's propagation rule.

        Parameters:
            src_features (mx.array): Features of the source nodes.
            dst_features (mx.array): Features of the destination nodes (not used in this function but included for compatibility).
            alpha_src (mx.array, optional): Precomputed attention values for the source nodes.
            alpha_dst (mx.array, optional): Precomputed attention values for the destination nodes.
            index (mx.array, optional): 1D array with indices of either src or dst nodes to compute softmax.
            edge_features (Optional[mx.array], optional): Features of the edges in the graph.

        Returns:
            mx.array: The computed messages for each edge in the graph.
        """
        alpha = alpha_src + alpha_dst

        if edge_features is not None:
            alpha_edge = self._compute_alpha_edge_features(edge_features)
            alpha = alpha + alpha_edge

        alpha = nn.leaky_relu(alpha, self.negative_slope)
        alpha = scatter(alpha, index, self.num_nodes, aggr="softmax")

        if "dropout" in self:
            alpha = self.dropout(alpha)

        return mx.expand_dims(alpha, -1) * src_features

    def _compute_alpha_edge_features(self, edge_features: mx.array):
        assert all(
            layer in self for layer in ["edge_lin_proj", "edge_att"]
        ), "Using edge features, GATConv layer should be provided argument `edge_features_dim`."

        if edge_features.ndim == 1:
            edge_features = edge_features.reshape(-1, 1)

        edge_features = self.edge_lin_proj(edge_features)
        edge_features = edge_features.reshape(-1, self.heads, self.out_features_dim)
        alpha_edge = (edge_features * self.edge_att).sum(axis=-1)

        return alpha_edge
