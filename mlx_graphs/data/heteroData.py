from typing import Dict, Literal, Optional, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.utils import has_isolated_nodes, has_self_loops
from mlx_graphs.utils.topology import is_undirected


class HeteroGraphData:
    def __init__(
        self,
        edge_index_dict: Dict[str, mx.array],
        node_features_dict: Optional[Dict[str, mx.array]] = None,
        edge_features_dict: Optional[Dict[str, mx.array]] = None,
        graph_features: Optional[mx.array] = None,
        node_labels_dict: Optional[Dict[str, mx.array]] = None,
        edge_labels_dict: Optional[Dict[str, mx.array]] = None,
        graph_labels: Optional[mx.array] = None,
    ) -> None:
        self.edge_index_dict = edge_index_dict
        self.node_features_dict = node_features_dict
        self.edge_features_dict = edge_features_dict
        self.graph_features = graph_features
        self.node_labels_dict = node_labels_dict
        self.edge_labels_dict = edge_labels_dict
        self.graph_labels = graph_labels

    def __repr__(self) -> str:
        strings = []
        for k, v in vars(self).items():
            if v is not None and not k.startswith("_"):
                if isinstance(v, dict):
                    for key, value in v.items():
                        if isinstance(value, mx.array):
                            strings.append(
                                f"{k}[{key}](shape={value.shape},{str(value.dtype).split('.')[-1]})"
                            )
                        else:
                            strings.append(f"{k}[{key}]={value}")
                elif isinstance(v, mx.array):
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
    def num_nodes(self) -> Dict[str, int]:
        num_nodes = {}
        if self.node_features_dict is not None:
            for node_type, node_features in self.node_features_dict.items():
                num_nodes[node_type] = node_features.shape[0]
        else:
            for edge_type, edge_index in self.edge_index_dict:
                src_node_type, _, dst_node_type = edge_type
                if src_node_type not in num_nodes:
                    num_nodes[src_node_type] = np.unique(
                        np.array(edge_index[0], copy=False)
                    ).size
                if dst_node_type not in num_nodes:
                    num_nodes[dst_node_type] = np.unique(
                        np.array(edge_index[1], copy=False)
                    ).size
        return num_nodes

    @property
    def num_edges_dict(self) -> Dict[str, int]:
        """Dictionary of number of edges for each edge type in the graph."""
        return {
            edge_type: edge_index.shape[1]
            for edge_type, edge_index in self.edge_index_dict.items()
        }

    @property
    def num_node_classes_dict(self) -> Dict[str, int]:
        """
        Returns a dictionary of the number of node classes
        for each node type in the current graph.
        """
        return {
            node_type: self._num_classes("node", node_type)
            for node_type in self.node_features_dict.keys()
        }

    @property
    def num_edge_classes_dict(self) -> Dict[str, int]:
        """
        Returns a dictionary of the number of edge classes
        for each edge type in the current graph.
        """
        return {
            edge_type: self._num_classes("edge", edge_type)
            for edge_type in self.edge_index_dict.keys()
        }

    @property
    def num_node_features_dict(self) -> Dict[str, int]:
        """Returns a dictionary of the number of node features for each node type."""
        num_node_features_dict = {}
        if self.node_features_dict is not None:
            for node_type, node_features in self.node_features_dict.items():
                num_node_features_dict[node_type] = (
                    1 if node_features.ndim == 1 else node_features.shape[-1]
                )
        return num_node_features_dict

    @property
    def num_edge_features_dict(self) -> Dict[str, int]:
        """Returns a dictionary of the number of edge features for each edge type."""
        num_edge_features_dict = {}
        if self.edge_features_dict is not None:
            for edge_type, edge_features in self.edge_features_dict.items():
                num_edge_features_dict[edge_type] = (
                    1 if edge_features.ndim == 1 else edge_features.shape[-1]
                )
        return num_edge_features_dict

    def _num_classes(self, task: Literal["node", "edge"], type_key: str) -> int:
        labels_dict = getattr(self, f"{task}_labels_dict")
        if labels_dict is None or type_key not in labels_dict:
            return 0
        labels = labels_dict[type_key]
        if labels.size == labels.shape[0]:
            return np.unique(np.array(labels)).size
        return labels.shape[-1]

    def __cat_dim__(self, key: str, *args, **kwargs) -> int:
        """This method can be overriden when batching is used with custom attributes.
        Given the name of a custom attribute `key`, returns the dimension where the
        concatenation happens during batching.

        By default, all attribute names containing "index" will be concatenated on
        axis 1, e.g. `edge_index_dict`. Other attributes are concatenated on axis 0,
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

    def __inc__(self, key: str, type_key: str, *args, **kwargs) -> Union[int, None]:
        """
        This method can be overriden when batching is used with custom attributes.
        Given the name of a custom attribute `key` and its type `type_key`, returns an
        integer indicating the incrementing value to apply to the elements
        of the attribute.

        By default, all attribute names containing "index" will be incremented based on
        the number of nodes of the corresponding node type in previous batches to avoid
        duplicate nodes in the index, e.g. `edge_index_dict`. Other attributes are not
        incremented and keep their original values, e.g. node features.
        If incrementation is not used, the return value should be set to `None`.

        Args:
            key: Name of the attribute on which change the default incrementation
                behavior while using batching.
            type_key: Type of the attribute (e.g., node type or edge type).

        Returns:
            Incrementing value for the given attribute or None.
        """
        if "index" in key:
            return self.num_nodes_dict[type_key]
        return None

    def has_isolated_nodes(self, node_type: str) -> bool:
        """Returns a boolean of whether the graph has isolated nodes of the given type
        (i.e., nodes that don't have a link to any other nodes)"""
        num_nodes = self.num_nodes_dict[node_type]
        for edge_type, edge_index in self.edge_index_dict.items():
            src_node_type, _, dst_node_type = edge_type
            if src_node_type == node_type or dst_node_type == node_type:
                if has_isolated_nodes(edge_index, num_nodes):
                    return True
        return False

    def has_self_loops(self, edge_type: str) -> bool:
        """
        Returns a boolean of whether the graph
        contains self loops for the given edge type.
        """
        return has_self_loops(self.edge_index_dict[edge_type])

    def is_undirected(self, edge_type: str) -> bool:
        """
        Returns a boolean of whether the
        given edge type is undirected or not.
        """
        edge_index = self.edge_index_dict[edge_type]
        edge_features = (
            self.edge_features_dict.get(edge_type, None)
            if self.edge_features_dict
            else None
        )
        return is_undirected(edge_index, edge_features)

    def is_directed(self, edge_type: str) -> bool:
        """
        Returns a boolean of whether the given
        edge type is directed or not.
        """
        return not self.is_undirected(edge_type)
