from typing import Optional

import mlx.core as mx

from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing


class SAGEConv(MessagePassing):
    r"""GraphSAGE convolution layer from `"Inductive Representation Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

    .. math::
        \mathbf{h}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \bigoplus_{j \in \mathcal{N}(i)} \mathbf{x}_j,

    where :math:`\mathbf{x}_i` represents the input features of node :math:`i`, :math:`\bigoplus`
    denotes the aggregation function, set by default to `mean` and :math:`\mathbf{h}_i`
    is the computed embedding of node :math:`i`.

    Args:
        node_features_dim: Size of input node features
        out_features_dim: Size of output node embeddings
        bias: Whether to use bias in the node projection

    Example:

    .. code-block:: python

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
            [1.05823, -0.295776, 0.075439, ..., -0.104383, 0.031947, -0.351897]], dtype=float32)
    """

    def __init__(
        self, node_features_dim: int, out_features_dim: int, bias: bool = True, **kwargs
    ):
        kwargs.setdefault("aggr", "mean")
        super(SAGEConv, self).__init__(**kwargs)

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim

        self.neigh_proj = Linear(node_features_dim, out_features_dim, bias=False)
        self.self_proj = Linear(node_features_dim, out_features_dim, bias=bias)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of SAGEConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            edge_weights: Edge weights leveraged in message passing. Default: ``None``

        Returns:
            mx.array: The computed node embeddings
        """

        # We follow DGL's way here by applying projection on the smaller feature dimension
        linear_before_mp = self.node_features_dim > self.out_features_dim

        if linear_before_mp:
            neigh_features = self.neigh_proj(node_features)
            neigh_features = self.propagate(
                edge_index=edge_index,
                node_features=neigh_features,
                message_kwargs={"edge_weights": edge_weights},
            )
        else:
            neigh_features = self.propagate(
                edge_index=edge_index,
                node_features=node_features,
                message_kwargs={"edge_weights": edge_weights},
            )
            neigh_features = self.neigh_proj(neigh_features)

        out_features = self.self_proj(node_features) + neigh_features

        return out_features
