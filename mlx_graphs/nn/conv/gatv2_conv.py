from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv._norm import SelfLoopFillValue, attention_self_loops
from mlx_graphs.nn.linear import Linear, glorot
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing
from mlx_graphs.utils import scatter


@fused_message_passing
class GATv2Conv(MessagePassing):
    """Graph Attention Network convolution layer with dynamic attention.

    Modification of GATConv based off of `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper.

    The layer also works on bipartite graphs (e.g. ``author -> paper`` in a
    heterogeneous graph): pass ``node_features`` as a ``(src_features,
    dst_features)`` tuple and, if the two sides have different feature
    sizes, ``node_features_dim`` as a ``(src_dim, dst_dim)`` tuple. The output
    has one row per destination node.

    Args:
        node_features_dim: Size of input node features, or a
            ``(src_dim, dst_dim)`` tuple for bipartite graphs
        out_features_dim: Size of output node embeddings
        heads: Number of attention heads
        concat: Whether to use concat of heads or mean reduction
        bias: Whether to use bias in the node projection
        negative_slope: Slope for the leaky relu
        dropout: Probability p for dropout
        edge_features_dim: Size of edge features
        add_self_loops: Whether to add a self-loop to every node, as in PyG
            (existing self-loops are replaced by it). Not supported on
            bipartite ``(src_features, dst_features)`` inputs. Default:
            ``True``
        fill_value: Edge features of the added self-loops, if
            ``edge_features`` are given: a constant, or the reduction
            (``"mean"``, ``"add"``, ``"max"``, ``"min"`` or ``"mul"``) of the
            features of the incoming edges of each node. Default: ``"mean"``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import GATv2Conv

        conv = GATv2Conv(16, 32, heads=2, concat=True)
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))

        h = conv(edge_index, node_features)

        >>> h.shape
        (5, 64)

    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        out_features_dim: int,
        heads: int = 1,
        concat: bool = True,
        bias: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_features_dim: Optional[int] = None,
        add_self_loops: bool = True,
        fill_value: SelfLoopFillValue = "mean",
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(GATv2Conv, self).__init__(**kwargs)

        self.out_features_dim = out_features_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        # read by to_hetero, which rejects self-loops on bipartite edge types
        self._add_self_loops = add_self_loops
        self.fill_value = fill_value

        # Weights
        if isinstance(node_features_dim, int):
            src_dim = dst_dim = node_features_dim
        else:
            src_dim, dst_dim = node_features_dim
        self.W_src = Linear(src_dim, heads * out_features_dim, bias=False)
        self.W_dst = Linear(dst_dim, heads * out_features_dim, bias=False)

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
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of GATConv.

        Args:
            edge_index: input edge index of shape `[2, num_edges]`
            node_features: input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs
            edge_features: edge features

        Returns:
            mx.array: computed node embeddings
        """
        H, C = self.heads, self.out_features_dim

        if isinstance(node_features, tuple):
            src_x, dst_x = node_features
        else:
            src_x = dst_x = node_features
        if self._add_self_loops:
            if dst_x is not src_x:
                raise ValueError(
                    f"{type(self).__name__} cannot add self-loops on a bipartite "
                    "graph ((src_features, dst_features) input): set "
                    "add_self_loops=False."
                )
            edge_index, edge_features, removed_loops = attention_self_loops(
                edge_index, src_x.shape[0], edge_features, self.fill_value
            )
        src_idx, dst_idx = edge_index
        src_feats = self.W_src(src_x).reshape(-1, H, C)
        dst_feats = self.W_dst(dst_x).reshape(-1, H, C)

        # Dynamic attention: alpha = att . leaky_relu(z), with one z per edge,
        # z = W_src x_j + W_dst x_i (+ edge features). Writing
        # leaky_relu(z) = slope * z + (1 - slope) * relu(z) leaves a term that
        # is linear in z, computed per node and gathered as [num_edges, heads]
        # scalars, and a relu term, whose backward pass is cheaper than the
        # leaky relu's on the large [num_edges, heads, channels] array.
        z = src_feats[src_idx] + dst_feats[dst_idx]
        linear = (src_feats * self.att).sum(-1)[src_idx] + (dst_feats * self.att).sum(
            -1
        )[dst_idx]
        if edge_features is not None:
            edge_feats = self._compute_edge_features(edge_features)
            z = z + edge_feats
            linear = linear + (edge_feats * self.att).sum(-1)

        slope = self.negative_slope
        alpha = slope * linear + (1 - slope) * (mx.maximum(z, 0) * self.att).sum(-1)
        if self._add_self_loops:
            # the input self-loops are replaced by the added ones
            alpha = mx.where(removed_loops[:, None], -float("inf"), alpha)
        alpha = scatter(alpha, dst_idx, dst_feats.shape[0], aggr="softmax")

        if "dropout" in self:
            alpha = self.dropout(alpha)

        # Weighted sum of neighbor features, one weight per (edge, head).
        node_features = self.propagate(
            node_features=(src_feats, dst_feats),
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
