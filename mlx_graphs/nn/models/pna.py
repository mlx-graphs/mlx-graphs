from mlx_graphs.nn.conv.pna_conv import PNAConv
from mlx_graphs.nn.models.basic_gnn import BasicGNN


class PNA(BasicGNN):
    """Principal Neighbourhood Aggregation network from `"Principal
    Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_, a stack of
    :class:`~mlx_graphs.nn.PNAConv` layers.

    Args:
        node_features_dim: Size of the input node features.
        hidden_features_dim: Size of the hidden node embeddings.
        num_layers: Number of message passing layers.
        out_features_dim: Size of the output. Default: ``None``
        dropout: Dropout probability. Default: ``0.0``
        act: Activation between layers. Default: ``"relu"``
        norm: Normalization between layers. Default: ``None``
        jk: Jumping Knowledge mode. Default: ``None``
        **kwargs: Arguments of :class:`~mlx_graphs.nn.PNAConv`; ``aggregators``,
            ``scalers`` and ``deg`` are required (e.g. also ``towers`` or
            ``edge_features_dim``).

    Example:

    .. code-block:: python

        from mlx_graphs.nn import PNA, PNAConv

        deg = PNAConv.get_degree_histogram(train_dataset)
        model = PNA(
            dataset.num_node_features,
            64,
            num_layers=3,
            out_features_dim=dataset.num_graph_classes,
            aggregators=["mean", "min", "max", "std"],
            scalers=["identity", "amplification", "attenuation"],
            deg=deg,
        )
        h = model(graph.edge_index, graph.node_features)
    """

    supports_edge_features = True

    def init_conv(self, node_features_dim, out_features_dim, is_last, **kwargs):
        return PNAConv(node_features_dim, out_features_dim, **kwargs)
