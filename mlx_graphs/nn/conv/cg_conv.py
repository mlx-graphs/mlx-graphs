from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import Linear  # PyTorch's default initialization, as in PyG

from mlx_graphs.nn.message_passing import MessagePassing


class CGConv(MessagePassing):
    r"""Crystal graph convolution layer from the `"Crystal Graph Convolutional
    Neural Networks for an Accurate and Interpretable Prediction of Material
    Properties" <https://arxiv.org/abs/1710.10324>`_ paper.

    .. math::
        \mathbf{h}_i = \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)}
        \sigma \left( \mathbf{z}_{i,j} \mathbf{W}_f + \mathbf{b}_f \right)
        \odot g \left( \mathbf{z}_{i,j} \mathbf{W}_s + \mathbf{b}_s \right)

    where :math:`\mathbf{z}_{i,j} = [ \mathbf{x}_i, \mathbf{x}_j,
    \mathbf{e}_{j,i} ]` concatenates the features of the destination node, the
    source node and (optionally) the edge, :math:`\sigma` is the sigmoid and
    :math:`g` the softplus function.

    Args:
        node_features_dim: Size of input node features, or a ``(src_dim,
            dst_dim)`` tuple for bipartite graphs. The output has ``dst_dim``
            features.
        edge_features_dim: Size of the edge features. Default: ``0``
        aggr: Aggregation function. Default: ``"add"``
        batch_norm: Whether to apply batch normalization to the aggregated
            messages. Default: ``False``
        bias: Whether the two projections use a bias. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import CGConv

        edge_index = mx.array([[0, 1, 2, 3, 4], [1, 0, 3, 2, 0]])
        node_features = mx.random.normal((5, 16))
        edge_features = mx.random.normal((5, 4))

        conv = CGConv(16, edge_features_dim=4)
        h = conv(edge_index, node_features, edge_features)
        >>> h.shape
        (5, 16)

    References:
        [1] `Xie and Grossman. Crystal Graph Convolutional Neural Networks for
        an Accurate and Interpretable Prediction of Material Properties.
        <https://arxiv.org/abs/1710.10324>`_
    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        edge_features_dim: int = 0,
        aggr: str = "add",
        batch_norm: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super(CGConv, self).__init__(aggr=aggr, **kwargs)

        self.node_features_dim = node_features_dim
        self.edge_features_dim = edge_features_dim

        if isinstance(node_features_dim, int):
            src_dim = dst_dim = node_features_dim
        else:
            src_dim, dst_dim = node_features_dim

        in_dim = src_dim + dst_dim + edge_features_dim
        self.lin_f = Linear(in_dim, dst_dim, bias=bias)
        self.lin_s = Linear(in_dim, dst_dim, bias=bias)
        if batch_norm:
            self.bn = nn.BatchNorm(dst_dim)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of CGConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs
            edge_features: Optional edge features of shape
                `[num_edges, edge_features_dim]`. Default: ``None``

        Returns:
            mx.array: The computed node embeddings, one row per destination node
        """
        if not isinstance(node_features, tuple):
            node_features = (node_features, node_features)

        out = self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_features": edge_features},
        )
        if "bn" in self:
            out = self.bn(out)
        return out + node_features[1]

    def message(
        self,
        src_features: mx.array,
        dst_features: mx.array,
        edge_features: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        z = [dst_features, src_features]
        if edge_features is not None:
            z.append(edge_features)
        # MLX 0.32 can crash (segfault) when a compiled function, such as
        # nn.softplus or nn.relu, runs on the output of an empty matmul:
        # without edges, the gated projection is run on a single dummy row instead.
        num_edges = z[0].shape[0]
        z = mx.concatenate(z, axis=-1)
        if num_edges == 0:
            z = mx.zeros((1, z.shape[1]), dtype=z.dtype)
        out = mx.sigmoid(self.lin_f(z)) * nn.softplus(self.lin_s(z))
        return out[:num_edges]
