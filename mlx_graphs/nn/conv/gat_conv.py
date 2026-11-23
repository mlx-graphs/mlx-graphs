from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv._norm import (
    SelfLoopFillValue,
    attention_leaky_relu,
    attention_self_loops,
)
from mlx_graphs.nn.linear import Linear, glorot
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


@fused_message_passing
class GATConv(MessagePassing):
    r"""Graph Attention Network convolution layer from `"Graph Attention
    Networks" <https://arxiv.org/abs/1710.10903>`_ paper.

    .. math::
        \mathbf{h}_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}
        \mathbf{x}_j, \qquad
        \alpha_{ij} = \mathrm{softmax}_j \left( \mathrm{LeakyReLU} \left(
        \mathbf{a}_{s}^{\top} \mathbf{W} \mathbf{x}_j +
        \mathbf{a}_{d}^{\top} \mathbf{W} \mathbf{x}_i \right) \right)

    The attention coefficients are computed per edge, but the weighted sum
    over neighbors runs as one sparse x dense matmul per head, so the
    ``[num_edges, heads, out_features_dim]`` message array is never built.

    The layer also works on bipartite graphs (e.g. ``author -> paper`` in a
    heterogeneous graph): pass ``node_features`` as a ``(src_features,
    dst_features)`` tuple. With an integer ``node_features_dim`` both sides
    share one projection, as in PyG; with a ``(src_dim, dst_dim)`` tuple they
    get separate projections ``lin_src_proj`` and ``lin_dst_proj``. The
    output has one row per destination node.

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
        from mlx_graphs.nn import GATConv

        conv = GATConv(16, 32, heads=2, concat=True)
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
        super(GATConv, self).__init__(**kwargs)

        self.out_features_dim = out_features_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        # read by to_hetero, which rejects self-loops on bipartite edge types
        self._add_self_loops = add_self_loops
        self.fill_value = fill_value

        if isinstance(node_features_dim, int):
            self.lin_proj = Linear(
                node_features_dim, heads * out_features_dim, bias=False
            )
        else:
            src_dim, dst_dim = node_features_dim
            self.lin_src_proj = Linear(src_dim, heads * out_features_dim, bias=False)
            self.lin_dst_proj = Linear(dst_dim, heads * out_features_dim, bias=False)

        self.att_src = glorot((1, heads, out_features_dim))
        self.att_dst = glorot((1, heads, out_features_dim))

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
            mx.array: computed node embeddings, one row per destination node
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
        if "lin_proj" in self:
            features = self.lin_proj(src_x).reshape(-1, H, C)
            dst_features = (
                features if dst_x is src_x else self.lin_proj(dst_x).reshape(-1, H, C)
            )
        else:
            features = self.lin_src_proj(src_x).reshape(-1, H, C)
            dst_features = self.lin_dst_proj(dst_x).reshape(-1, H, C)

        # Attention logits are a sum of one term per source node and one per
        # destination node, so they are computed per node and only gathered
        # per edge as [num_edges, heads] scalars.
        alpha_src = (features * self.att_src).sum(-1)
        alpha_dst = (dst_features * self.att_dst).sum(-1)
        alpha = alpha_src[src_idx] + alpha_dst[dst_idx]

        if edge_features is not None:
            alpha = alpha + self._compute_alpha_edge_features(edge_features)

        alpha = attention_leaky_relu(alpha, self.negative_slope)
        if self._add_self_loops:
            # the input self-loops are replaced by the added ones
            alpha = mx.where(removed_loops[:, None], -float("inf"), alpha)
        node_features = self._softmax_aggregate(
            edge_index=edge_index,
            logits=alpha,
            node_features=(
                features if dst_features is features else (features, dst_features)
            ),
            dropout=self.dropout if "dropout" in self else None,
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

    def _compute_alpha_edge_features(self, edge_features: mx.array):
        assert all(layer in self for layer in ["edge_lin_proj", "edge_att"]), """Using
        edge features, GATConv layer should be provided argument
        `edge_features_dim`."""

        if edge_features.ndim == 1:
            edge_features = edge_features.reshape(-1, 1)

        edge_features = self.edge_lin_proj(edge_features)
        edge_features = edge_features.reshape(-1, self.heads, self.out_features_dim)
        alpha_edge = (edge_features * self.edge_att).sum(axis=-1)

        return alpha_edge
