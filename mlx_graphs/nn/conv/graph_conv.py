from typing import Optional, Tuple, Union

import mlx.core as mx
from mlx.nn import Linear  # PyTorch's default initialization, as in PyG

from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


@fused_message_passing
class GraphConv(MessagePassing):
    r"""Graph convolution layer from `"Weisfeiler and Leman Go Neural: Higher-order
    Graph Neural Networks" <https://arxiv.org/abs/1810.02244>`_ paper.

    .. math::
        \mathbf{h}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \bigoplus_{j \in \mathcal{N}(i)} e_{j,i} \cdot \mathbf{x}_j,

    where :math:`\mathbf{x}_i` represents the input features of node :math:`i`,
    :math:`e_{j,i}` is the weight of the edge from node :math:`j` to node
    :math:`i` (1 if no weights are given), :math:`\bigoplus` denotes the
    aggregation function, set by default to `add` and :math:`\mathbf{h}_i` is
    the computed embedding of node :math:`i`.

    The layer also works on bipartite graphs: pass ``node_features`` as a
    ``(src_features, dst_features)`` tuple and, if the two sets have different
    feature sizes, ``node_features_dim`` as a ``(src_dim, dst_dim)`` tuple. The
    output has one row per destination node.

    Args:
        node_features_dim: Size of input node features, or a ``(src_dim, dst_dim)``
            tuple for bipartite graphs whose source and destination nodes have
            different feature sizes
        out_features_dim: Size of output node embeddings
        bias: Whether to add a bias to the output
        **kwargs: Arguments of :class:`MessagePassing`, e.g. ``aggr``
            (default: ``"add"``)

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import GraphConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.ones((5, 16))

        conv = GraphConv(16, 32)
        h = conv(edge_index, node_features)
        >>> h.shape
        (5, 32)

        # Weighted edges and another aggregation
        conv = GraphConv(16, 32, aggr="mean")
        h = conv(edge_index, node_features, edge_weights=mx.ones((5,)))

        # Bipartite: 5 authors (16 features) -> 2 papers (8 features)
        conv = GraphConv((16, 8), 32)
        edge_index = mx.array([[0, 1, 4], [0, 0, 1]])  # author -> paper
        h_paper = conv(edge_index, (mx.ones((5, 16)), mx.ones((2, 8))))
        >>> h_paper.shape
        (2, 32)
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        out_features_dim: int,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(GraphConv, self).__init__(**kwargs)

        if isinstance(node_features_dim, int):
            src_dim = dst_dim = node_features_dim
        else:
            src_dim, dst_dim = node_features_dim

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.src_features_dim = src_dim
        self.dst_features_dim = dst_dim

        # Same layout as PyG: `neigh_proj` is `lin_rel` (with the bias),
        # `self_proj` is `lin_root`.
        self.neigh_proj = Linear(src_dim, out_features_dim, bias=bias)
        self.self_proj = Linear(dst_dim, out_features_dim, bias=False)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of GraphConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs
            edge_weights: Optional scalar weight per edge, of shape `[num_edges]`
                or `[num_edges, 1]`. Default: ``None``

        Returns:
            mx.array: The computed node embeddings, one row per destination node
        """
        if isinstance(node_features, tuple):
            src_features, dst_features = node_features
        else:
            src_features = dst_features = node_features

        # Sum and mean aggregation commute with the weight matrix, so aggregate
        # the smaller of the two feature sizes (as SAGEConv does). The bias is
        # added after aggregation in both cases.
        project_first = (
            self.aggr in ("add", "mean")
            and self.src_features_dim > self.out_features_dim
        )
        if project_first:
            src_features = src_features @ self.neigh_proj.weight.T

        neigh_features = self.propagate(
            edge_index=edge_index,
            node_features=(src_features, dst_features),
            message_kwargs={"edge_weights": edge_weights},
        )

        if not project_first:
            neigh_features = neigh_features @ self.neigh_proj.weight.T
        if "bias" in self.neigh_proj:
            neigh_features = neigh_features + self.neigh_proj.bias

        return neigh_features + self.self_proj(dst_features)
