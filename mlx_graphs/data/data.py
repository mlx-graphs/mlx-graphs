from typing import Optional

import mlx.core as mx


class Data:
    """
    Represents a graph data object with optional features and labels.

    Args:
        edge_index (mlx.core.array, optional): edge index representing the topology of
            the graph, with shape [2, num_edges].
        node_features (mlx.core.array, optional): Array of shape [num_nodes, num_node_features]
            containing the features of each node.
        edge_features (mlx.core.array, optional): Array of shape [num_edges, num_edge_features]
            containing the features of each edge.
        graph_features (mlx.core.array, optional): Array of shape [num_graph_features]
            containing the global features of the graph.
        node_labels (mlx.core.array, optional): Array of shape [num_nodes, num_node_labels]
            containing the labels of each node.
        edge_labels (mlx.core.array, optional): Array of shape [num_edges, num_edge_labels]
            containing the labels of each edge.
        graph_labels (mlx.core.array, optional): Array of shape [num_graph_labels]
            containing the global labels of the graph.
        **kwargs: Additional keyword arguments to store any other custom attributes.
    """

    def __init__(
        self,
        edge_index: Optional[mx.array] = None,
        node_features: Optional[mx.array] = None,
        edge_features: Optional[mx.array] = None,
        graph_features: Optional[mx.array] = None,
        node_labels: Optional[mx.array] = None,
        edge_labels: Optional[mx.array] = None,
        graph_labels: Optional[mx.array] = None,
        **kwargs,
    ):
        if edge_index:
            self.edge_index = edge_index
        if node_features:
            self.node_features = node_features
        if edge_features:
            self.edge_features = edge_features
        if graph_features:
            self.graph_features = graph_features
        if node_labels:
            self.node_labels = node_labels
        if edge_labels:
            self.edge_labels = edge_labels
        if graph_labels:
            self.graph_labels = graph_labels
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        """Converts the Data object to a dictionary.

        Returns:
            dict: A dictionary representation of the Data object.
        """
        return self.__dict__
