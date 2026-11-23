from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.message_passing import MessagePassing


class EdgeConv(MessagePassing):
    r"""Edge convolution layer from `"Dynamic Graph CNN for Learning on Point Clouds"
    <https://arxiv.org/abs/1801.07829>`_ paper.

    .. math::
        \mathbf{h}_i = \bigoplus_{j \in \mathcal{N}(i)}
        \text{MLP} \left( [ \mathbf{x}_i, \, \mathbf{x}_j - \mathbf{x}_i ] \right),

    where :math:`\mathbf{x}_i` represents the input features of node :math:`i`,
    :math:`[\cdot, \cdot]` is the concatenation, :math:`\text{MLP}` denotes a custom
    neural network provided by the user and :math:`\bigoplus` denotes the aggregation
    function, set by default to `max`. Nodes without incoming edges get a zero
    embedding.

    The layer also works on bipartite graphs: pass ``node_features`` as a
    ``(src_features, dst_features)`` tuple (both with the same feature size). The
    output has one row per destination node.

    Args:
        mlp: Callable :class:`mlx.nn.Module` applied on each edge, mapping the
            concatenated features of shape `[num_edges, 2 * node_features_dim]` to
            shape `[num_edges, out_features_dim]`
        **kwargs: Arguments of :class:`MessagePassing`, e.g. ``aggr`` (``"max"`` by
            default, or ``"add"``, ``"mean"``, ``"min"``)

    Example:

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_graphs.nn.conv.edge_conv import EdgeConv

        mlp = nn.Sequential(
            nn.Linear(2 * 16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        conv = EdgeConv(mlp)

        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))

        h = conv(edge_index, node_features)

        >>> h.shape
        (5, 32)
    """

    def __init__(self, mlp: nn.Module, **kwargs):
        kwargs.setdefault("aggr", "max")
        super(EdgeConv, self).__init__(**kwargs)

        self.mlp = mlp

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
    ) -> mx.array:
        """Computes the forward pass of EdgeConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs

        Returns:
            mx.array: The computed node embeddings, one row per destination node
        """
        return self.propagate(edge_index=edge_index, node_features=node_features)

    def message(
        self, src_features: mx.array, dst_features: mx.array, **kwargs
    ) -> mx.array:
        """Computes ``mlp([x_i, x_j - x_i])`` for every edge ``j -> i``.

        Args:
            src_features: Features of the source nodes :math:`\\mathbf{x}_j`
            dst_features: Features of the destination nodes :math:`\\mathbf{x}_i`

        Returns:
            mx.array: The messages, shape `[num_edges, out_features_dim]`
        """
        # MLX 0.32 can crash (segfault) when a compiled function, such as
        # nn.softplus or nn.relu, runs on the output of an empty matmul:
        # without edges, the MLP is run on a single dummy row instead.
        features = mx.concatenate([dst_features, src_features - dst_features], axis=-1)
        num_edges = features.shape[0]
        if num_edges == 0:
            features = mx.zeros((1, *features.shape[1:]), dtype=features.dtype)
        return self.mlp(features)[:num_edges]
