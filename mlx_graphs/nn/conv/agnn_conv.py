import mlx.core as mx

from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing
from mlx_graphs.utils import scatter
from mlx_graphs.utils.array_ops import p_norm


@fused_message_passing
class AGNNConv(MessagePassing):
    r"""Graph attentional propagation layer from the `"Attention-based Graph
    Neural Network for Semi-Supervised Learning"
    <https://arxiv.org/abs/1803.03735>`_ paper.

    .. math::
        \mathbf{H} = \mathbf{P} \mathbf{X},

    where the propagation matrix :math:`\mathbf{P}` is computed as

    .. math::
        P_{i,j} = \frac{\exp( \beta \cdot \cos(\mathbf{x}_i, \mathbf{x}_j))}
        {\sum_{k \in \mathcal{N}(i)\cup \{ i \}} \exp( \beta \cdot
        \cos(\mathbf{x}_i, \mathbf{x}_k))}

    with a (trainable) temperature :math:`\beta`. The layer keeps the size of
    the node features.

    Args:
        requires_grad: Whether :math:`\beta` is trainable (it is fixed to 1
            otherwise). Default: ``True``
        add_self_loops: Whether to replace the self-loops of the graph by one
            self-loop per node. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import AGNNConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [1, 0, 3, 2, 0]])
        node_features = mx.random.normal((5, 16))

        conv = AGNNConv()
        h = conv(edge_index, node_features)
        >>> h.shape
        (5, 16)

    References:
        [1] `Thekumparampil et al. Attention-based Graph Neural Network for
        Semi-Supervised Learning. <https://arxiv.org/abs/1803.03735>`_
    """

    def __init__(
        self, requires_grad: bool = True, add_self_loops: bool = True, **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(AGNNConv, self).__init__(**kwargs)

        self.requires_grad = requires_grad
        self._add_self_loops = add_self_loops
        self.beta = mx.ones((1,))
        if not requires_grad:
            self.freeze(keys=["beta"], recurse=False)

    def __call__(self, edge_index: mx.array, node_features: mx.array) -> mx.array:
        """Computes the forward pass of AGNNConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features

        Returns:
            mx.array: The computed node embeddings
        """
        num_nodes = node_features.shape[0]
        src, dst = edge_index[0], edge_index[1]
        # Cosine similarity, with PyG's F.normalize epsilon.
        # p_norm: zero gradient for nodes whose features are all zero
        norm = p_norm(node_features, axis=-1, keepdims=True)
        x_norm = node_features / mx.maximum(norm, 1e-12)
        alpha = self.beta * (x_norm[src] * x_norm[dst]).sum(-1)

        if self._add_self_loops:
            # Existing self-loops are masked out of the softmax (instead of
            # being removed, which would need a data-dependent shape), and one
            # self-loop per node is appended.
            alpha = mx.where(src == dst, -float("inf"), alpha)
            loop_alpha = self.beta * (x_norm * x_norm).sum(-1)
            alpha = mx.concatenate([alpha, loop_alpha])
            loop = mx.arange(num_nodes, dtype=edge_index.dtype)
            edge_index = mx.concatenate([edge_index, mx.stack([loop, loop])], axis=1)

        alpha = scatter(alpha, edge_index[1], num_nodes, aggr="softmax")

        return self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_weights": alpha},
        )
