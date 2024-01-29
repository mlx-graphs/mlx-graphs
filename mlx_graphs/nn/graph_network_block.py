from typing import Optional

import mlx.core as mx
from mlx.nn.layers.base import Module


class GraphNetworkBlock(Module):
    """Implements a generic Graph Network block as defined in [1].

    A Graph Network block takes as input a graph with N nodes and E edges and
    returns a graph with the same topology.

    The input graph can have
        - ``node_features``: features associated with each node in the graph, provided as an array of size [N, F_N]
        - ``edge_features``: features associated with each edge in the graph, provided as an array of size [E, F_E]
        - ``graph_features``: features associated to the graph itself, of size [F_U]

    The topology of the graph is specified as an `edge_index`, an array of size [2, E],
    containing the source and destination nodes of each edge as column vectors.
    A Graph Network block is initialized by providing node, edge and global models (all
    optional), that are used to update node, edge and global features (if present).
    Depending on which models are provided and how they are implemented, the Graph
    Network block acts as a flexible ``meta-layer`` that can be used to implement other
    architectures, like message-passing networks, deep sets, relation networks and more
    (see [1]).

    For a usage example see `here <https://github.com/TristanBilot/mlx-graphs/blob/main/examples/graph_network_block.py>`_.

    Args:
        node_model: a callable Module which updates
            a graph's node features
        edge_model: a callable Module which updates
            a graph's edge features
        graph_model: a callable Module which updates
            a graph's global features

    References:
        [1] `Battaglia et al. Relational Inductive Biases, Deep Learning, and Graph Networks. <https://arxiv.org/pdf/1806.01261.pdf>`_

    """

    def __init__(
        self,
        node_model: Optional[Module] = None,
        edge_model: Optional[Module] = None,
        graph_model: Optional[Module] = None,
    ):
        super().__init__()
        self.node_model = node_model
        self.edge_model = edge_model
        self.graph_model = graph_model

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Optional[mx.array] = None,
        edge_features: Optional[mx.array] = None,
        graph_features: Optional[mx.array] = None,
    ) -> tuple[Optional[mx.array], Optional[mx.array], Optional[mx.array]]:
        """Forward pass of the Graph Network block

        Args:
            edge_index: array of size [2, E], where each column contains the source
                and destination nodes of an edge.
            node_features: features associated with each node in the
                graph, provided as an array of size [N, F_N]
            edge_features: features associated with each edge in the
                graph, provided as an array of size [E, F_E]
            graph_features: features associated to the graph itself,
                of size [F_U]

        Returns:
            The tuple of updated node, edge and global attributes.
        """
        if self.edge_model:
            edge_features = self.edge_model(
                edge_index, node_features, edge_features, graph_features
            )

        if self.node_model:
            node_features = self.node_model(
                edge_index, node_features, edge_features, graph_features
            )

        if self.graph_model:
            graph_features = self.graph_model(
                edge_index, node_features, edge_features, graph_features
            )

        return node_features, edge_features, graph_features
