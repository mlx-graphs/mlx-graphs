from typing import Optional

import mlx.core as mx

from mlx_graphs.nn.conv._norm import cached_gcn_norm
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


@fused_message_passing
class APPNP(MessagePassing):
    r"""Approximate personalized propagation of neural predictions layer from
    `"Predict then Propagate: Graph Neural Networks meet Personalized PageRank"
    <https://arxiv.org/abs/1810.05997>`_ paper.

    .. math::
        \mathbf{X}^{(0)} &= \mathbf{X}

        \mathbf{X}^{(k)} &= (1 - \alpha) \mathbf{\hat{D}}^{-1/2}
        \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \mathbf{X}^{(k-1)} + \alpha
        \mathbf{X}^{(0)}

        \mathbf{H} &= \mathbf{X}^{(K)},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` is the (weighted)
    adjacency matrix with self-loops and :math:`\mathbf{\hat{D}}` its diagonal
    degree matrix.

    The layer has no parameters. It is usually applied to the output of an MLP.

    Args:
        K: Number of propagation steps
        alpha: Teleport probability :math:`\alpha`
        dropout: Probability to drop each edge of the normalized adjacency
            matrix. A new set of edges is dropped at every propagation step,
            and only in training mode.
        add_self_loops: Whether to add a self-loop to every node that has none
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_graphs.nn import APPNP

        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.ones((5, 16))

        mlp = nn.Linear(16, 32)
        prop = APPNP(K=10, alpha=0.1)
        h = prop(edge_index, mlp(node_features))
        >>> h.shape
        (5, 32)
    """

    def __init__(
        self,
        K: int,
        alpha: float,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(APPNP, self).__init__(**kwargs)

        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self._add_self_loops = add_self_loops

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of APPNP.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            edge_weights: Optional scalar weight per edge, of shape `[num_edges]`
                or `[num_edges, 1]`. Default: ``None``

        Returns:
            mx.array: The propagated node features, same shape as ``node_features``
        """
        edge_index, norm = cached_gcn_norm(
            edge_index,
            node_features.shape[0],
            edge_weights,
            add_self_loops=self._add_self_loops,
        )

        keep_prob = 1.0 - self.dropout
        initial_features = node_features
        for _ in range(self.K):
            step_norm = norm
            if self.training and self.dropout > 0:
                keep = mx.random.bernoulli(keep_prob, norm.shape)
                step_norm = norm * keep / keep_prob

            node_features = self.propagate(
                edge_index=edge_index,
                node_features=node_features,
                message_kwargs={"edge_weights": step_norm},
            )
            node_features = (
                1 - self.alpha
            ) * node_features + self.alpha * initial_features

        return node_features
