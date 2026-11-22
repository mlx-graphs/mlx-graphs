import math
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import Linear  # PyTorch's default initialization, as in PyG

from mlx_graphs.nn.message_passing import (
    MessagePassing,
    edge_scores,
    fused_message_passing,
)
from mlx_graphs.utils import scatter


# message() reduces to the default edge_weights * x_j without edge features
@fused_message_passing(allow_message_override=True)
class TransformerConv(MessagePassing):
    r"""Graph transformer convolution layer from `"Masked Label Prediction: Unified
    Message Passing Model for Semi-Supervised Classification"
    <https://arxiv.org/abs/2009.03509>`_ paper.

    .. math::
        \mathbf{h}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_j,

    where the attention coefficients :math:`\alpha_{i,j}` are computed with
    multi-head dot-product attention:

    .. math::
        \alpha_{i,j} = \textrm{softmax}_{j \in \mathcal{N}(i)} \left(
        \frac{(\mathbf{W}_3 \mathbf{x}_i)^{\top} (\mathbf{W}_4 \mathbf{x}_j)}
        {\sqrt{d}} \right)

    With edge features :math:`\mathbf{e}_{ij}` (``edge_features_dim`` set), these
    are projected and added to both the keys and the values:

    .. math::
        \mathbf{h}_i = \mathbf{W}_1 \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left(
        \mathbf{W}_2 \mathbf{x}_j + \mathbf{W}_5 \mathbf{e}_{ij} \right),
        \quad
        \alpha_{i,j} = \textrm{softmax}_{j \in \mathcal{N}(i)} \left(
        \frac{(\mathbf{W}_3 \mathbf{x}_i)^{\top}
        (\mathbf{W}_4 \mathbf{x}_j + \mathbf{W}_5 \mathbf{e}_{ij})}{\sqrt{d}} \right)

    With ``beta=True``, the skip connection and the aggregated messages
    :math:`\mathbf{m}_i` are combined with a learned gate instead of a sum:

    .. math::
        \mathbf{h}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i + (1 - \beta_i) \mathbf{m}_i,
        \quad
        \beta_i = \textrm{sigmoid}(\mathbf{w}_6^{\top} [ \mathbf{W}_1 \mathbf{x}_i,
        \mathbf{m}_i, \mathbf{W}_1 \mathbf{x}_i - \mathbf{m}_i ])

    The layer also works on bipartite graphs: pass ``node_features`` as a
    ``(src_features, dst_features)`` tuple and, if the two sets have different
    feature sizes, ``node_features_dim`` as a ``(src_dim, dst_dim)`` tuple. The
    output has one row per destination node.

    Args:
        node_features_dim: Size of input node features, or a ``(src_dim, dst_dim)``
            tuple for bipartite graphs
        out_features_dim: Size of output node embeddings (per head)
        heads: Number of attention heads. Default: ``1``
        concat: Whether to concatenate the heads (output size
            ``heads * out_features_dim``) or average them (output size
            ``out_features_dim``). Default: ``True``
        beta: Whether to combine the skip connection and the aggregated messages
            with a learned gate. Only used if ``root_weight`` is ``True``.
            Default: ``False``
        dropout: Dropout probability applied on the attention coefficients.
            Default: ``0.0``
        edge_features_dim: Size of edge features. If ``None``, the layer does not
            use edge features. Default: ``None``
        bias: Whether to use bias in the key, query, value and skip projections.
            Default: ``True``
        root_weight: Whether to add the transformed features of the node itself
            (skip connection) to the output. Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.conv.transformer_conv import TransformerConv

        conv = TransformerConv(16, 32, heads=2, edge_features_dim=4)
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))
        edge_features = mx.random.uniform(low=0, high=1, shape=(5, 4))

        h = conv(edge_index, node_features, edge_features)

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
        beta: bool = False,
        dropout: float = 0.0,
        edge_features_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(TransformerConv, self).__init__(**kwargs)

        if isinstance(node_features_dim, int):
            src_dim = dst_dim = node_features_dim
        else:
            src_dim, dst_dim = node_features_dim

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.heads = heads
        self.concat = concat
        self.root_weight = root_weight
        self.beta = beta and root_weight
        self.edge_features_dim = edge_features_dim

        # Keys and values come from source nodes, queries from destination nodes.
        self.lin_key = Linear(src_dim, heads * out_features_dim, bias=bias)
        self.lin_query = Linear(dst_dim, heads * out_features_dim, bias=bias)
        self.lin_value = Linear(src_dim, heads * out_features_dim, bias=bias)

        if edge_features_dim is not None:
            self.lin_edge = Linear(
                edge_features_dim, heads * out_features_dim, bias=False
            )

        if root_weight:
            skip_dim = heads * out_features_dim if concat else out_features_dim
            self.lin_skip = Linear(dst_dim, skip_dim, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * skip_dim, 1, bias=False)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of TransformerConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs
            edge_features: Edge features of shape `[num_edges, edge_features_dim]`.
                Required if (and only if) the layer was built with
                ``edge_features_dim``. Default: ``None``

        Returns:
            mx.array: The computed node embeddings, one row per destination node
        """
        H, C = self.heads, self.out_features_dim

        if isinstance(node_features, tuple):
            src_features, dst_features = node_features
        else:
            src_features = dst_features = node_features

        if ("lin_edge" in self) != (edge_features is not None):
            raise ValueError(
                "`edge_features` must be passed if and only if the layer was "
                "created with `edge_features_dim`."
            )
        if edge_features is not None:
            if edge_features.ndim == 1:
                edge_features = edge_features.reshape(-1, 1)
            edge_features = self.lin_edge(edge_features).reshape(-1, H, C)

        query = self.lin_query(dst_features).reshape(-1, H, C)
        key = self.lin_key(src_features).reshape(-1, H, C)
        value = self.lin_value(src_features).reshape(-1, H, C)

        src_idx, dst_idx = edge_index[0], edge_index[1]

        num_dst = query.shape[0]
        scale = 1.0 / math.sqrt(C)

        # Attention logits q_i . k_j / sqrt(C), one per (edge, head). With a
        # recent mlx_cluster this never gathers [num_edges, heads, C] queries
        # and keys (see edge_scores).
        alpha = edge_scores(dst_idx, src_idx, query, key, scale)
        if edge_features is not None:
            # keys are k_j + e_ij; the edge term is per edge already
            alpha = alpha + scale * (query[dst_idx] * edge_features).sum(axis=-1)
        alpha = scatter(alpha, dst_idx, num_dst, aggr="softmax")

        if "dropout" in self:
            alpha = self.dropout(alpha)

        # Weighted sum of the values, one weight per (edge, head). Without edge
        # features this is a fused SpMM per head, as in GATConv; with them the
        # values are v_j + e_ij and messages go through message().
        # shape: [num_dst_nodes, heads, out_features_dim]
        out = self.propagate(
            edge_index=edge_index,
            node_features=(value, query),
            message_kwargs={"edge_weights": alpha, "edge_features": edge_features},
        )

        if self.concat:
            out = out.reshape(-1, H * C)
        else:
            out = mx.mean(out, axis=1)

        if self.root_weight:
            skip = self.lin_skip(dst_features)
            if self.beta:
                beta = mx.sigmoid(
                    self.lin_beta(mx.concatenate([out, skip, out - skip], axis=-1))
                )
                out = beta * skip + (1 - beta) * out
            else:
                out = out + skip

        return out

    def message(
        self,
        src_features: mx.array,
        dst_features: mx.array,
        edge_weights: mx.array,
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes one attention-weighted message per edge.

        Only used on the gather-scatter path (with edge features, or without
        the fused kernels); without edge features it is the default
        ``edge_weights * x_j`` that the fused SpMM computes.

        Args:
            src_features: Values of the source nodes,
                shape `[num_edges, heads, out_features_dim]`
            dst_features: Queries of the destination nodes (unused)
            edge_weights: Attention coefficients, shape `[num_edges, heads]`
            edge_features: Projected edge features,
                shape `[num_edges, heads, out_features_dim]`

        Returns:
            mx.array: The messages, shape `[num_edges, heads, out_features_dim]`
        """
        value = src_features
        if edge_features is not None:
            value = value + edge_features
        return mx.expand_dims(edge_weights, -1) * value
