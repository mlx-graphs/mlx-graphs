from typing import Callable, Optional, Tuple, Union

import mlx.core as mx
from mlx.nn import Linear  # PyTorch's default initialization, as in PyG

from mlx_graphs.nn.message_passing import MessagePassing


class ResGatedGraphConv(MessagePassing):
    r"""Residual gated graph convolution layer from the `"Residual Gated Graph
    ConvNets" <https://arxiv.org/abs/1711.07553>`_ paper.

    .. math::
        \mathbf{h}_i = \mathbf{W}_1 \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)}
        \eta_{i,j} \odot \mathbf{W}_2 \mathbf{x}_j + \mathbf{b}

    where the gate :math:`\eta_{i,j}` is defined as

    .. math::
        \eta_{i,j} = \sigma(\mathbf{W}_3 \mathbf{x}_i + \mathbf{W}_4
        \mathbf{x}_j)

    with :math:`\sigma` the sigmoid function. With edge features
    (``edge_features_dim`` set), the features of the edge from :math:`j` to
    :math:`i` are concatenated to :math:`\mathbf{x}_i` and :math:`\mathbf{x}_j`
    before the projections :math:`\mathbf{W}_2`, :math:`\mathbf{W}_3` and
    :math:`\mathbf{W}_4`.

    Args:
        node_features_dim: Size of input node features, or a ``(src_dim,
            dst_dim)`` tuple for bipartite graphs
        out_features_dim: Size of output node embeddings
        act: Gating function :math:`\sigma`. Default: :func:`mlx.core.sigmoid`
        edge_features_dim: Size of the edge features, if any. Default: ``None``
        root_weight: Whether to add the transformed root node features
            :math:`\mathbf{W}_1 \mathbf{x}_i`. Default: ``True``
        bias: Whether to add a bias to the output. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`, e.g. ``aggr``
            (default: ``"add"``)

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import ResGatedGraphConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [1, 0, 3, 2, 0]])
        node_features = mx.random.normal((5, 16))

        conv = ResGatedGraphConv(16, 32)
        h = conv(edge_index, node_features)
        >>> h.shape
        (5, 32)

        conv = ResGatedGraphConv(16, 32, edge_features_dim=4)
        h = conv(edge_index, node_features, mx.random.normal((5, 4)))

    References:
        [1] `Bresson and Laurent. Residual Gated Graph ConvNets.
        <https://arxiv.org/abs/1711.07553>`_
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        out_features_dim: int,
        act: Optional[Callable[[mx.array], mx.array]] = mx.sigmoid,
        edge_features_dim: Optional[int] = None,
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(ResGatedGraphConv, self).__init__(**kwargs)

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.act = act
        self.edge_features_dim = edge_features_dim
        self.root_weight = root_weight

        if isinstance(node_features_dim, int):
            src_dim = dst_dim = node_features_dim
        else:
            src_dim, dst_dim = node_features_dim
        edge_dim = edge_features_dim if edge_features_dim is not None else 0

        # Same names as PyG.
        self.lin_key = Linear(dst_dim + edge_dim, out_features_dim)
        self.lin_query = Linear(src_dim + edge_dim, out_features_dim)
        self.lin_value = Linear(src_dim + edge_dim, out_features_dim)
        if root_weight:
            self.lin_skip = Linear(dst_dim, out_features_dim, bias=False)
        if bias:
            self.bias = mx.zeros((out_features_dim,))

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of ResGatedGraphConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs
            edge_features: Edge features of shape `[num_edges, edge_features_dim]`,
                required if and only if ``edge_features_dim`` is set.
                Default: ``None``

        Returns:
            mx.array: The computed node embeddings, one row per destination node
        """
        if (edge_features is not None) != (self.edge_features_dim is not None):
            raise ValueError(
                "`edge_features` must be given if and only if the layer was "
                "created with `edge_features_dim`."
            )
        if isinstance(node_features, tuple):
            src_features, dst_features = node_features
        else:
            src_features = dst_features = node_features

        if edge_features is None:
            # Project the nodes once; queries and values travel together.
            src = mx.concatenate(
                [self.lin_query(src_features), self.lin_value(src_features)], axis=-1
            )
            dst = self.lin_key(dst_features)
        else:
            src, dst = src_features, dst_features

        out = self.propagate(
            edge_index=edge_index,
            node_features=(src, dst),
            message_kwargs={"edge_features": edge_features},
        )

        if self.root_weight:
            out = out + self.lin_skip(dst_features)
        if "bias" in self:
            out = out + self.bias
        return out

    def message(
        self,
        src_features: mx.array,
        dst_features: mx.array,
        edge_features: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        if edge_features is None:
            q_j, v_j = mx.split(src_features, 2, axis=-1)
            k_i = dst_features
        else:
            k_i = self.lin_key(mx.concatenate([dst_features, edge_features], -1))
            q_j = self.lin_query(mx.concatenate([src_features, edge_features], -1))
            v_j = self.lin_value(mx.concatenate([src_features, edge_features], -1))
        gate = k_i + q_j
        if self.act is not None:
            gate = self.act(gate)
        return gate * v_j
