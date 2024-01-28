from typing import Optional

import mlx.core as mx


class GraphData:
    """
    Represents a graph data object with optional features and labels.

    Args:
        edge_index: edge index representing the topology of
            the graph, with shape [2, num_edges].
        node_features: Array of shape [num_nodes, num_node_features]
            containing the features of each node.
        edge_features: Array of shape [num_edges, num_edge_features]
            containing the features of each edge.
        graph_features: Array of shape [num_graph_features]
            containing the global features of the graph.
        node_labels: Array of shape [num_nodes, num_node_labels]
            containing the labels of each node.
        edge_labels: Array of shape [num_edges, num_edge_labels]
            containing the labels of each edge.
        graph_labels: Array of shape [num_graph_labels]
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
        self.edge_index = edge_index
        self.node_features = node_features
        self.edge_features = edge_features
        self.graph_features = graph_features
        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.graph_labels = graph_labels
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        strings = []
        for k, v in vars(self).items():
            if v is not None and not k.startswith("_"):
                strings.append(f"{k}={v}")

        return f"{type(self).__name__}({', '.join(strings)})"

    def to_dict(self) -> dict:
        """Converts the Data object to a dictionary.

        Returns:
            dict: A dictionary representation of the Data object.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    def __cat_dim__(self, key: str, *args, **kwargs) -> int:
        """This method can be overriden when batching is used with custom attributes.
        Given the name of a custom attribute `key`, returns the dimension where the
        concatenation happens during batching.

        By default, all attribute names containing "index" will be concatenated on axis 1,
        e.g. `edge_index`. Other attributes are concatenated on axis 0, e.g. node features.

        Args:
            key: Name of the attribute on which change the default concatenation dimension
                while using batching.

        Returns:
            int: The dimension where concatenation will happen when batching.
        """
        if "index" in key:
            return 1
        return 0

    def __inc__(self, key: str, *args, **kwargs) -> bool:
        """This method can be overriden when batching is used with custom attributes.
        Given the name of a custom attribute `key`, returns the whether the values
        of the attribute should be incremented by the cumulative sum of previous batches'
        number of nodes.

        By default, all attribute names containing "index" will be incremented to avoid
        duplicate nodes in the index, e.g. `edge_index`.
        Other attributes are cnot incremented and keep their original values, e.g. node features.

        Args:
            key: Name of the attribute on which change the default incrementation behavior
                while using batching.

        Returns:
            bool: Whether to use incrementation of values for a given attribute.
        """
        if "index" in key:
            return True
        return False
