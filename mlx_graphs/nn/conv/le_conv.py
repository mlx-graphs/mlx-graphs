from typing import Optional, Tuple, Union

import mlx.core as mx
from mlx.nn import Linear  # PyTorch's default initialization, as in PyG

from mlx_graphs.nn.message_passing import MessagePassing


class LEConv(MessagePassing):
    r"""Local extremum graph convolution layer from the `"ASAP: Adaptive
    Structure Aware Pooling for Learning Hierarchical Graph Representations"
    <https://arxiv.org/abs/1911.07979>`_ paper.

    .. math::
        \mathbf{h}_i = \mathbf{\Theta}_3 \mathbf{x}_i + \sum_{j \in
        \mathcal{N}(i)} e_{j,i} (\mathbf{\Theta}_1 \mathbf{x}_j -
        \mathbf{\Theta}_2 \mathbf{x}_i)

    where :math:`e_{j,i}` is the weight of the edge from node :math:`j` to node
    :math:`i` (1 if no weights are given). :math:`\mathbf{\Theta}_1`,
    :math:`\mathbf{\Theta}_2` and :math:`\mathbf{\Theta}_3` are the ``lin1``,
    ``lin2`` and ``lin3`` projections (``lin2`` has no bias).

    Args:
        node_features_dim: Size of input node features, or a ``(src_dim,
            dst_dim)`` tuple for bipartite graphs
        out_features_dim: Size of output node embeddings
        bias: Whether to use biases in the projections. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`, e.g. ``aggr``
            (default: ``"add"``)

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import LEConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [1, 0, 3, 2, 0]])
        node_features = mx.random.normal((5, 16))

        conv = LEConv(16, 32)
        h = conv(edge_index, node_features)
        >>> h.shape
        (5, 32)

    References:
        [1] `Ranjan et al. ASAP: Adaptive Structure Aware Pooling for Learning
        Hierarchical Graph Representations. <https://arxiv.org/abs/1911.07979>`_
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
        super(LEConv, self).__init__(**kwargs)

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim

        if isinstance(node_features_dim, int):
            src_dim = dst_dim = node_features_dim
        else:
            src_dim, dst_dim = node_features_dim

        # Same names as PyG.
        self.lin1 = Linear(src_dim, out_features_dim, bias=bias)
        self.lin2 = Linear(dst_dim, out_features_dim, bias=False)
        self.lin3 = Linear(dst_dim, out_features_dim, bias=bias)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of LEConv.

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

        out = self.propagate(
            edge_index=edge_index,
            node_features=(self.lin1(src_features), self.lin2(dst_features)),
            message_kwargs={"edge_weights": edge_weights},
        )
        return out + self.lin3(dst_features)

    def message(
        self,
        src_features: mx.array,
        dst_features: mx.array,
        edge_weights: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        out = src_features - dst_features
        return out if edge_weights is None else out * edge_weights.reshape(-1, 1)
