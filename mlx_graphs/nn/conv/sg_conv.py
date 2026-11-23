from typing import Optional

import mlx.core as mx
from mlx.nn import Linear  # PyTorch's default initialization, as in PyG

from mlx_graphs.nn.conv._norm import cached_gcn_norm
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


@fused_message_passing
class SGConv(MessagePassing):
    r"""Simple graph convolution layer from `"Simplifying Graph Convolutional
    Networks" <https://arxiv.org/abs/1902.07153>`_ paper.

    .. math::
        \mathbf{H} = {\left( \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{W},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` is the (weighted)
    adjacency matrix with self-loops and :math:`\mathbf{\hat{D}}` its diagonal
    degree matrix.

    Args:
        node_features_dim: Size of input node features
        out_features_dim: Size of output node embeddings
        K: Number of propagation steps
        cached: If ``True``, the propagated features :math:`{\left(
            \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}
            \right)}^K \mathbf{X}` are computed on the first call only and
            reused afterwards: later calls ignore their graph and features.
            Only for transductive learning, where both stay the same. Call
            :meth:`clear_cache` to compute them again.
        add_self_loops: Whether to add a self-loop to every node that has none
        bias: Whether to use bias in the node projection
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import SGConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.ones((5, 16))

        conv = SGConv(16, 32, K=2)
        h = conv(edge_index, node_features)
        >>> h.shape
        (5, 32)

        # Full-batch training on one graph: propagate once, then only train W
        conv = SGConv(16, 32, K=2, cached=True)
        h = conv(edge_index, node_features)
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        K: int = 1,
        cached: bool = False,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(SGConv, self).__init__(**kwargs)

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.K = K
        self.cached = cached
        self._add_self_loops = add_self_loops
        # The leading underscore keeps the cache out of the module parameters.
        self._cached_features: Optional[mx.array] = None

        self.linear = Linear(node_features_dim, out_features_dim, bias=bias)

    def clear_cache(self) -> None:
        """Drops the propagated features stored when ``cached=True``."""
        self._cached_features = None

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of SGConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            edge_weights: Optional scalar weight per edge, of shape `[num_edges]`
                or `[num_edges, 1]`. Default: ``None``

        Returns:
            mx.array: The computed node embeddings
        """
        if self.cached and self._cached_features is not None:
            return self.linear(self._cached_features)

        edge_index, norm = cached_gcn_norm(
            edge_index,
            node_features.shape[0],
            edge_weights,
            add_self_loops=self._add_self_loops,
        )

        # Propagation and the weight matrix commute, so propagate the smaller
        # of the two feature sizes (the bias is added after propagation in
        # both cases). The cache has to hold the unprojected features.
        project_first = (
            not self.cached and self.node_features_dim > self.out_features_dim
        )
        if project_first:
            node_features = node_features @ self.linear.weight.T

        for _ in range(self.K):
            node_features = self.propagate(
                edge_index=edge_index,
                node_features=node_features,
                message_kwargs={"edge_weights": norm},
            )

        if project_first:
            if "bias" in self.linear:
                node_features = node_features + self.linear.bias
            return node_features

        if self.cached:
            node_features = mx.stop_gradient(node_features)
            self._cached_features = node_features

        return self.linear(node_features)
