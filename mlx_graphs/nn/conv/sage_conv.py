from typing import Optional, Tuple, Union

import mlx.core as mx

from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing


class SAGEConv(MessagePassing):
    r"""GraphSAGE convolution layer from `"Inductive Representation Learning on Large
    Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

    .. math::
        \mathbf{h}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \bigoplus_{j \in \mathcal{N}(i)} \mathbf{x}_j,

    where :math:`\mathbf{x}_i` represents the input features of node :math:`i`,
    :math:`\bigoplus` denotes the aggregation function, set by default to `mean` and
    :math:`\mathbf{h}_i` is the computed embedding of node :math:`i`.

    The layer also works on bipartite graphs, where messages flow from one set
    of nodes to another (e.g. ``author -> paper`` in a heterogeneous graph):
    pass ``node_features`` as a ``(src_features, dst_features)`` tuple and, if
    the two sets have different feature sizes, ``node_features_dim`` as a
    ``(src_dim, dst_dim)`` tuple. The output has one row per destination node.

    Args:
        node_features_dim: Size of input node features, or a ``(src_dim, dst_dim)``
            tuple for bipartite graphs whose source and destination nodes have
            different feature sizes
        out_features_dim: Size of output node embeddings
        bias: Whether to use bias in the node projection

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.data.data import GraphData
        from mlx_graphs.nn import SAGEConv

        graph = GraphData(
            edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]]),
            node_features = mx.ones((5, 16)),
        )

        conv = SAGEConv(16, 32)
        h = conv(graph.edge_index, graph.node_features)

        >>> h
        array([[1.65429, -0.376169, 1.04172, ..., -0.919106, 1.42576, 0.490938],
            [1.65429, -0.376169, 1.04172, ..., -0.919106, 1.42576, 0.490938],
            [1.05823, -0.295776, 0.075439, ..., -0.104383, 0.031947, -0.351897],
            [1.65429, -0.376169, 1.04172, ..., -0.919106, 1.42576, 0.490938],
            [1.05823, -0.295776, 0.075439, ..., -0.104383, 0.031947, -0.351897]],
            dtype=float32)

        # Bipartite: 5 authors (16 features) -> 2 papers (8 features)
        conv = SAGEConv((16, 8), 32)
        edge_index = mx.array([[0, 1, 4], [0, 0, 1]])  # author -> paper
        h_paper = conv(edge_index, (mx.ones((5, 16)), mx.ones((2, 8))))
        >>> h_paper.shape
        (2, 32)
    """

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        out_features_dim: int,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "mean")
        super(SAGEConv, self).__init__(**kwargs)

        if isinstance(node_features_dim, int):
            src_dim = dst_dim = node_features_dim
        else:
            src_dim, dst_dim = node_features_dim

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.src_features_dim = src_dim
        self.dst_features_dim = dst_dim

        # Neighbor messages come from source nodes, the self term from
        # destination nodes.
        self.neigh_proj = Linear(src_dim, out_features_dim, bias=False)
        self.self_proj = Linear(dst_dim, out_features_dim, bias=bias)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of SAGEConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs
            edge_weights: Edge weights leveraged in message passing. Default: ``None``

        Returns:
            mx.array: The computed node embeddings, one row per destination node
        """
        if isinstance(node_features, tuple):
            src_features, dst_features = node_features
        else:
            src_features = dst_features = node_features

        # We follow DGL's way by applying projection on the smaller feature dimension
        linear_before_mp = self.src_features_dim > self.out_features_dim

        if linear_before_mp:
            neigh_features = self.propagate(
                edge_index=edge_index,
                node_features=(self.neigh_proj(src_features), dst_features),
                message_kwargs={"edge_weights": edge_weights},
            )
        else:
            neigh_features = self.propagate(
                edge_index=edge_index,
                node_features=(src_features, dst_features),
                message_kwargs={"edge_weights": edge_weights},
            )
            neigh_features = self.neigh_proj(neigh_features)

        out_features = self.self_proj(dst_features) + neigh_features

        return out_features
