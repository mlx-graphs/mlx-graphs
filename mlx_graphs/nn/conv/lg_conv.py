from typing import Optional

import mlx.core as mx

from mlx_graphs.nn.conv._norm import cached_gcn_norm
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


@fused_message_passing
class LGConv(MessagePassing):
    r"""Light graph convolution from the `"LightGCN: Simplifying and Powering
    Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.

    .. math::
        \mathbf{h}_i = \sum_{j \in \mathcal{N}(i)} \frac{e_{j,i}}{\sqrt{\deg(i)
        \deg(j)}} \mathbf{x}_j

    where :math:`e_{j,i}` is the weight of the edge from node :math:`j` to node
    :math:`i` (1 if no weights are given). The layer has no parameters and does
    not add self-loops.

    Args:
        normalize: Whether to apply the symmetric degree normalization.
            Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import LGConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [1, 0, 3, 2, 0]])
        node_features = mx.random.normal((5, 16))

        conv = LGConv()
        h = conv(edge_index, node_features)
        >>> h.shape
        (5, 16)

    References:
        [1] `He et al. LightGCN: Simplifying and Powering Graph Convolution
        Network for Recommendation. <https://arxiv.org/abs/2002.02126>`_
    """

    def __init__(self, normalize: bool = True, **kwargs):
        kwargs.setdefault("aggr", "add")
        super(LGConv, self).__init__(**kwargs)
        self.normalize = normalize

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of LGConv.

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
                edge_index, node_features.shape[0], edge_weights, add_self_loops=False
            )
        return self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_weights": edge_weights},
        )
