from typing import Any, Optional

import mlx.core as mx

from mlx_graphs.nn.conv._norm import cached_gcn_norm
from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


@fused_message_passing
class GCNConv(MessagePassing):
    r"""Graph convolution layer from `"Semi-Supervised Classification with Graph
    Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_ paper.

    .. math::
        \mathbf{H} = \hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}}
        \hat{\mathbf{D}}^{-1/2} \mathbf{X} \mathbf{W} + \mathbf{b}

    where :math:`\hat{\mathbf{A}}` is the (optionally weighted) adjacency matrix,
    with self-loops if ``add_self_loops=True``, and :math:`\hat{\mathbf{D}}` its
    degree matrix. A self-loop of weight 1 is added to the nodes that do not
    have one; existing self-loops keep their weight.

    The normalization of a graph without edge weights is cached and reused as
    long as the same ``edge_index`` array is passed, which is the common case
    in full-batch training. Build a new array instead of modifying an
    ``edge_index`` in place between calls, or call
    :func:`~mlx_graphs.nn.clear_norm_cache`.

    Args:
        node_features_dim: size of input node features
        out_features_dim: size of output node embeddings
        bias: whether to add a bias to the output
        add_self_loops: whether to add a self-loop for each node

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import GCNConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))

        conv = GCNConv(16, 32, add_self_loops=True)
        h = conv(edge_index, node_features)

        >>> h.shape
        (5, 32)
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        bias: bool = True,
        add_self_loops: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(GCNConv, self).__init__(**kwargs)

        self.linear = Linear(node_features_dim, out_features_dim, bias=False)
        if bias:
            self.bias = mx.zeros((out_features_dim,))
        self._add_self_loops = add_self_loops

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> mx.array:
        """Computes the forward pass of GCNConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            edge_weights: Optional weight of each edge. Default: ``None``
            normalize: Whether to apply the symmetric degree normalization.
                Default: ``True``

        Returns:
            The computed node embeddings
        """
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")

        node_features = self.linear(node_features)
        num_nodes = node_features.shape[0]

        if not normalize:
            norm = None if edge_weights is None else edge_weights.reshape(-1)
            if self._add_self_loops:
                loops = mx.arange(num_nodes, dtype=edge_index.dtype)
                edge_index = mx.concatenate(
                    [edge_index, mx.stack([loops, loops])], axis=1
                )
                if norm is not None:
                    norm = mx.concatenate([norm, mx.ones((num_nodes,), norm.dtype)])
        else:
            edge_index, norm = cached_gcn_norm(
                edge_index, num_nodes, edge_weights, self._add_self_loops
            )

        # Compute messages and aggregate them with sum and norm.
        node_features = self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_weights": norm},
        )

        if "bias" in self:
            node_features = node_features + self.bias

        return node_features
