from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv._norm import cached_gcn_norm
from mlx_graphs.nn.linear import glorot
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


@fused_message_passing
class ARMAConv(MessagePassing):
    r"""ARMA graph convolution layer from the `"Graph Neural Networks with
    Convolutional ARMA Filters" <https://arxiv.org/abs/1901.01343>`_ paper.

    .. math::
        \mathbf{H} = \frac{1}{K} \sum_{k=1}^K \mathbf{X}_k^{(T)},

    where each of the :math:`K` stacks is computed recursively by

    .. math::
        \mathbf{X}_k^{(t+1)} = \sigma \left( \mathbf{\hat{L}}
        \mathbf{X}_k^{(t)} \mathbf{W} + \mathbf{X}^{(0)} \mathbf{V} \right),

    with :math:`\mathbf{\hat{L}} = \mathbf{I} - \mathbf{L} = \mathbf{D}^{-1/2}
    \mathbf{A} \mathbf{D}^{-1/2}` (no self-loops are added) and
    :math:`\mathbf{X}_k^{(0)} = \mathbf{X}`.

    Args:
        node_features_dim: Size of input node features
        out_features_dim: Size of output node embeddings
        num_stacks: Number of parallel stacks :math:`K`. Default: ``1``
        num_layers: Number of layers :math:`T` of each stack. Default: ``1``
        shared_weights: If ``True``, all layers of a stack share their weights.
            Default: ``False``
        act: Activation function :math:`\sigma` applied after each layer, or
            ``None``. Default: :func:`mlx.nn.relu`
        dropout: Dropout probability of the skip connection
            :math:`\mathbf{X}^{(0)}`, applied in training mode. Default: ``0``
        bias: Whether to add a bias after each layer. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import ARMAConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [1, 0, 3, 2, 0]])
        node_features = mx.random.normal((5, 16))

        conv = ARMAConv(16, 32, num_stacks=2, num_layers=3)
        h = conv(edge_index, node_features)
        >>> h.shape
        (5, 32)

    References:
        [1] `Bianchi et al. Graph Neural Networks with Convolutional ARMA
        Filters. <https://arxiv.org/abs/1901.01343>`_
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        num_stacks: int = 1,
        num_layers: int = 1,
        shared_weights: bool = False,
        act: Optional[Callable[[mx.array], mx.array]] = nn.relu,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(ARMAConv, self).__init__(**kwargs)

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.num_stacks = num_stacks
        self.num_layers = num_layers
        self.shared_weights = shared_weights
        self.act = act

        K, F_in, F_out = num_stacks, node_features_dim, out_features_dim
        T = 1 if shared_weights else num_layers

        # Same shapes as PyG.
        self.weight = glorot((max(1, T - 1), K, F_out, F_out))
        self.init_weight = glorot((K, F_in, F_out))
        self.root_weight = glorot((T, K, F_in, F_out))
        if bias:
            self.bias = mx.zeros((T, K, 1, F_out))
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of ARMAConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            edge_weights: Optional scalar weight per edge, of shape `[num_edges]`
                or `[num_edges, 1]`. Default: ``None``

        Returns:
            mx.array: The computed node embeddings
        """
        num_nodes = node_features.shape[0]
        K, F_out = self.num_stacks, self.out_features_dim
        edge_index, norm = cached_gcn_norm(
            edge_index, num_nodes, edge_weights, add_self_loops=False
        )

        # Stacks are kept in a [num_nodes, K, F] layout, and flattened to
        # [num_nodes, K * F] for message passing.
        out = node_features
        for t in range(self.num_layers):
            if t == 0:
                out = mx.einsum("nf,kfo->nko", out, self.init_weight)
            else:
                weight = self.weight[0 if self.shared_weights else t - 1]
                out = mx.einsum("nkf,kfo->nko", out, weight)

            out = self.propagate(
                edge_index=edge_index,
                node_features=out.reshape(num_nodes, K * F_out),
                message_kwargs={"edge_weights": norm},
            ).reshape(num_nodes, K, F_out)

            root = node_features
            if "dropout" in self:
                root = self.dropout(root)
            root_weight = self.root_weight[0 if self.shared_weights else t]
            out = out + mx.einsum("nf,kfo->nko", root, root_weight)

            if "bias" in self:
                out = out + self.bias[0 if self.shared_weights else t][:, 0]
            if self.act is not None:
                out = self.act(out)

        return out.mean(axis=1)
