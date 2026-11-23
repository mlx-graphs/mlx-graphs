from typing import Optional

import mlx.core as mx

from mlx_graphs.nn.conv._norm import cached_gcn_norm
from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


@fused_message_passing
class TAGConv(MessagePassing):
    r"""Topology adaptive graph convolution layer from `"Topology Adaptive Graph
    Convolutional Networks" <https://arxiv.org/abs/1710.10370>`_ paper.

    .. math::
        \mathbf{H} = \sum_{k=0}^{K} \left( \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \right)^k \mathbf{X} \mathbf{W}_k + \mathbf{b},

    where :math:`\mathbf{A}` is the (weighted) adjacency matrix and
    :math:`\mathbf{D}` its diagonal degree matrix. No self-loops are added.

    Args:
        node_features_dim: Size of input node features
        out_features_dim: Size of output node embeddings
        K: Number of hops :math:`K`
        bias: Whether to add a bias to the output
        normalize: Whether to apply the symmetric normalization. If ``False``,
            the adjacency matrix (with ``edge_weights``, if given) is used as is.
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import TAGConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.ones((5, 16))

        conv = TAGConv(16, 32, K=3)
        h = conv(edge_index, node_features)
        >>> h.shape
        (5, 32)
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        K: int = 3,
        bias: bool = True,
        normalize: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(TAGConv, self).__init__(**kwargs)

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.K = K
        self.normalize = normalize

        self.lins = [
            Linear(node_features_dim, out_features_dim, bias=False)
            for _ in range(K + 1)
        ]
        if bias:
            self.bias = mx.zeros((out_features_dim,))

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of TAGConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            edge_weights: Optional scalar weight per edge, of shape `[num_edges]`
                or `[num_edges, 1]`. Default: ``None``

        Returns:
            mx.array: The computed node embeddings
        """
        if self.normalize:
            edge_index, edge_weights = cached_gcn_norm(
                edge_index,
                node_features.shape[0],
                edge_weights,
                add_self_loops=False,
            )

        out = self.lins[0](node_features)
        for lin in self.lins[1:]:
            node_features = self.propagate(
                edge_index=edge_index,
                node_features=node_features,
                message_kwargs={"edge_weights": edge_weights},
            )
            out = out + lin(node_features)

        if "bias" in self:
            out = out + self.bias

        return out
