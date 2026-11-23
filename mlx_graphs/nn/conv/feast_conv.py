from typing import Tuple, Union

import mlx.core as mx
from mlx.nn import Linear  # uniform initialization, as PyG's

from mlx_graphs.nn.conv.point_conv import _set_self_loops
from mlx_graphs.nn.message_passing import MessagePassing


class FeaStConv(MessagePassing):
    r"""The (translation-invariant) feature-steered convolutional operator
    from the `"FeaStNet: Feature-Steered Graph Convolutions for 3D Shape
    Analysis" <https://arxiv.org/abs/1706.05206>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|}
        \sum_{j \in \mathcal{N}(i)} \sum_{h=1}^H
        q_h(\mathbf{x}_i, \mathbf{x}_j) \mathbf{W}_h \mathbf{x}_j

    with :math:`q_h(\mathbf{x}_i, \mathbf{x}_j) = \mathrm{softmax}_j
    (\mathbf{u}_h^{\top}(\mathbf{x}_j - \mathbf{x}_i) + c_h)`, where :math:`H`
    denotes the number of attention heads, and :math:`\mathbf{W}_h`,
    :math:`\mathbf{u}_h` and :math:`c_h` are trainable parameters. The softmax
    is taken over the heads, as in PyG.

    Args:
        node_features_dim: Size of the input features.
        out_features_dim: Size of the output features.
        heads: Number of attention heads :math:`H`. Default: ``1``
        add_self_loops: If ``True``, removes the self-loops of the graph and
            adds one for each node. Default: ``True``
        bias: Whether to add a bias. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing` (``aggr="mean"`` by
            default).

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import FeaStConv

        edge_index = mx.array([[0, 1, 2, 3], [1, 0, 3, 2]])
        x = mx.random.normal((4, 16))
        conv = FeaStConv(16, 32, heads=4)
        >>> conv(edge_index, x).shape
        (4, 32)
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        heads: int = 1,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "mean")
        super().__init__(**kwargs)
        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.heads = heads
        self.add_self_loops = add_self_loops
        self.lin = Linear(node_features_dim, heads * out_features_dim, bias=False)
        self.u = Linear(node_features_dim, heads, bias=False)
        self.c = 0.1 * mx.random.normal((heads,))
        if bias:
            self.bias = 0.1 * mx.random.normal((out_features_dim,))

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
    ) -> mx.array:
        """Computes the forward pass of FeaStConv.

        Args:
            edge_index: Edges of shape ``[2, num_edges]``.
            node_features: Node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs.

        Returns:
            The features of the destination nodes.
        """
        if not isinstance(node_features, tuple):
            node_features = (node_features, node_features)
        if self.add_self_loops:
            edge_index = _set_self_loops(edge_index, node_features[1].shape[0])
        out = self.propagate(edge_index=edge_index, node_features=node_features)
        if "bias" in self:
            out = out + self.bias
        return out

    def message(
        self, src_features: mx.array, dst_features: mx.array, **kwargs
    ) -> mx.array:
        q = mx.softmax(self.u(src_features - dst_features) + self.c, axis=1)
        x_j = self.lin(src_features).reshape(src_features.shape[0], self.heads, -1)
        return (x_j * q[..., None]).sum(axis=1)
