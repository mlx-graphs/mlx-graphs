import mlx.nn as nn

from mlx_graphs.nn.conv.edge_conv import EdgeConv
from mlx_graphs.nn.models.basic_gnn import MLP, BasicGNN


class EdgeCNN(BasicGNN):
    """The Dynamic Graph CNN model from the `"Dynamic Graph CNN for Learning
    on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper, a stack of
    :class:`~mlx_graphs.nn.EdgeConv` layers, each with a two-layer MLP
    mapping ``2 * in_dim`` features to ``out_dim``.

    The graph is fixed (given by ``edge_index``), not recomputed from the
    embeddings at every layer.

    Args:
        node_features_dim: Size of the input node features.
        hidden_features_dim: Size of the hidden node embeddings.
        num_layers: Number of message passing layers.
        out_features_dim: Size of the output. Default: ``None``
        dropout: Dropout probability. Default: ``0.0``
        act: Activation between layers and inside the MLPs of the
            convolutions (as PyG). Default: ``"relu"``
        norm: Normalization between layers and inside the MLPs of the
            convolutions (as PyG). Default: ``None``
        jk: Jumping Knowledge mode. Default: ``None``
        **kwargs: Additional arguments of :class:`~mlx_graphs.nn.EdgeConv`
            (e.g. ``aggr="mean"``; ``"max"`` by default).

    Example:

    .. code-block:: python

        from mlx_graphs.nn import EdgeCNN

        model = EdgeCNN(3, 64, num_layers=3, out_features_dim=10)
        out = model(edge_index, node_features)
    """

    def init_conv(
        self, node_features_dim: int, out_features_dim: int, is_last: bool, **kwargs
    ) -> nn.Module:
        # As PyG, the inner MLPs use the activation and normalization of the
        # model.
        mlp = MLP(
            [2 * node_features_dim, out_features_dim, out_features_dim],
            act=self.act,
            norm=self._norm_name,
        )
        return EdgeConv(mlp, **kwargs)
