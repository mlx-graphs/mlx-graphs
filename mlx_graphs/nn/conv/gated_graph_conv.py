import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


@fused_message_passing
class GatedGraphConv(MessagePassing):
    r"""Gated graph convolution layer from the `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper.

    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

        \mathbf{m}_i^{(l+1)} &= \bigoplus_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        \mathbf{\Theta} \cdot \mathbf{h}_j^{(l)}

        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})

    up to :math:`\mathbf{h}_i^{(L)}`, where :math:`e_{j,i}` is the weight of the
    edge from node :math:`j` to node :math:`i` (1 if no weights are given).
    Input features smaller than ``out_features_dim`` are zero-padded.

    Args:
        out_features_dim: Size of output node embeddings (input features must
            not be larger)
        num_layers: Number of message passing steps :math:`L`
        aggr: Aggregation :math:`\bigoplus`. Default: ``"add"``
        bias: Whether the GRU cell uses biases. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import GatedGraphConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [1, 0, 3, 2, 0]])
        node_features = mx.random.normal((5, 16))

        conv = GatedGraphConv(32, num_layers=3)
        h = conv(edge_index, node_features)
        >>> h.shape
        (5, 32)

    References:
        [1] `Li et al. Gated Graph Sequence Neural Networks.
        <https://arxiv.org/abs/1511.05493>`_
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        out_features_dim: int,
        num_layers: int,
        aggr: str = "add",
        bias: bool = True,
        **kwargs,
    ):
        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_features_dim = out_features_dim
        self.num_layers = num_layers

        bound = 1.0 / math.sqrt(out_features_dim)
        # Applied as `h @ weight[l]`, as in PyG.
        self.weight = mx.random.uniform(
            -bound, bound, (num_layers, out_features_dim, out_features_dim)
        )
        # Same gates as torch.nn.GRUCell (with bias_ih and the r/z parts of
        # bias_hh folded into `rnn.b`, and the n part of bias_hh in `rnn.bhn`).
        self.rnn = nn.GRU(out_features_dim, out_features_dim, bias=bias)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of GatedGraphConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features, with at most
                ``out_features_dim`` features
            edge_weights: Optional scalar weight per edge, of shape `[num_edges]`
                or `[num_edges, 1]`. Default: ``None``

        Returns:
            mx.array: The computed node embeddings
        """
        in_dim = node_features.shape[-1]
        if in_dim > self.out_features_dim:
            raise ValueError(
                "The number of input features is not allowed to be larger than "
                "`out_features_dim`."
            )
        h = node_features
        if in_dim < self.out_features_dim:
            padding = mx.zeros(
                (h.shape[0], self.out_features_dim - in_dim), dtype=h.dtype
            )
            h = mx.concatenate([h, padding], axis=1)

        for i in range(self.num_layers):
            m = self.propagate(
                edge_index=edge_index,
                node_features=h @ self.weight[i],
                message_kwargs={"edge_weights": edge_weights},
            )
            # One GRU step: input m, hidden state h.
            h = self.rnn(m[:, None, :], h)[:, 0, :]

        return h
