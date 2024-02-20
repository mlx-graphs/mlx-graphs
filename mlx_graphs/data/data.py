from typing import Literal, Optional, Union

import mlx.core as mx
import numpy as np


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
        graph_labels: Array of shape [1, num_graph_labels]
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
                if isinstance(v, mx.array):
                    strings.append(
                        f"{k}(shape={v.shape}, {str(v.dtype).split('.')[-1]})"
                    )
                else:
                    strings.append(f"{k}={v}")

        prefix = "\n\t"
        return f"{type(self).__name__}({prefix + prefix.join(strings)})"

    def to_dict(self) -> dict:
        """Converts the Data object to a dictionary.

        Returns:
            A dictionary with all attributes of the `GraphData` object.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @property
    def num_nodes(self) -> Union[int, None]:
        """Number of nodes in the graph."""
        if self.node_features is not None:
            return self.node_features.shape[0]

        # NOTE: This may be slow for large graphs
        elif self.edge_index is not None:
            return np.unique(self.edge_index).size
        return None

    @property
    def num_edges(self) -> Union[int, None]:
        """Number of edges in the graph"""
        if self.edge_index is not None:
            return self.edge_index.shape[1]

        return None

    @property
    def num_node_classes(self) -> int:
        """Returns the number of node classes in the current graph."""
        return self._num_classes("node")

    @property
    def num_edge_classes(self) -> int:
        """Returns the number of edge classes in the current graph."""
        return self._num_classes("edge")

    @property
    def num_graph_classes(self) -> int:
        """Returns the number of graph classes in the current graph."""
        return self._num_classes("graph")

    @property
    def num_node_features(self) -> int:
        """Returns the number of node features."""
        if self.node_features is None:
            return 0
        return 1 if self.node_features.ndim == 1 else self.node_features.shape[-1]

    @property
    def num_edge_features(self) -> int:
        """Returns the number of edge features."""
        if self.edge_features is None:
            return 0
        return 1 if self.edge_features.ndim == 1 else self.edge_features.shape[-1]

    @property
    def num_graph_features(self) -> int:
        """Returns the number of graph features."""
        if self.graph_features is None:
            return 0
        return 1 if self.graph_features.ndim == 1 else self.graph_features.shape[-1]

    def _num_classes(self, task: Literal["node", "edge", "graph"]) -> int:
        labels = getattr(self, f"{task}_labels")
        if labels is None:
            return 0
        elif labels.size == labels.shape[0]:
            return np.unique(np.array(labels)).size
        return labels.shape[-1]

    def __cat_dim__(self, key: str, *args, **kwargs) -> int:
        """This method can be overriden when batching is used with custom attributes.
        Given the name of a custom attribute `key`, returns the dimension where the
        concatenation happens during batching.

        By default, all attribute names containing "index" will be concatenated on
        axis 1, e.g. `edge_index`. Other attributes are concatenated on axis 0,
        e.g. node features.

        Args:
            key: Name of the attribute on which change the default concatenation
                dimension while using batching.

        Returns:
            The dimension where concatenation will happen when batching.
        """
        if "index" in key:
            return 1
        return 0

    def __inc__(self, key: str, *args, **kwargs) -> Union[int, None]:
        """This method can be overriden when batching is used with custom attributes.
        Given the name of a custom attribute `key`, returns an integer indicating the
        incrementing value to apply to the elemnts of the attribute.

        By default, all attribute names containing "index" will be incremented based on
        the number of nodes in previous batches to avoid duplicate nodes in the index,
        e.g. `edge_index`. Other attributes are cnot incremented and keep their original
        values, e.g. node features.
        If incrementation is not used, the return value should be set to `None`.

        Args:
            key: Name of the attribute on which change the default incrementation
                behavior while using batching.

        Returns:
            Incrementing value for the given attribute or None.
        """
        if "index" in key:
            return self.num_nodes
        return None
