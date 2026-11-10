from typing import Any, Literal, Optional, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.utils import has_isolated_nodes, has_self_loops
from mlx_graphs.utils.topology import is_undirected


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
        edge_index: mx.array,
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
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        if self.node_features is not None:
            return self.node_features.shape[0]
        else:
            # NOTE: This may be slow for large graphs
            return np.unique(np.array(self.edge_index, copy=False)).size

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph"""
        return self.edge_index.shape[1]

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

    def has_isolated_nodes(self) -> bool:
        """Returns a boolean of whether the graph has isolated nodes or not
        (i.e., nodes that don't have a link to any other nodes)"""
        return has_isolated_nodes(self.edge_index, self.num_nodes)

    def has_self_loops(self) -> bool:
        """Returns a boolean of whether the graph contains self loops."""
        return has_self_loops(self.edge_index)

    def is_undirected(self) -> bool:
        """Returns a boolean of whether the graph is undirected or not."""
        return is_undirected(self.edge_index, self.edge_features)

    def is_directed(self) -> bool:
        """Returns a boolean of whether the graph is directed or not."""
        return not self.is_undirected()


class HeteroGraphData:
    """
    Represents a graph structure with multiple node and edge types

    Args:
        edge_index_dict: A dictionary mapping edge types
            to their corresponding edge indices. The edge indices are
            represented as a 2D array of shape `[2, num_edges]`, where the
            first row contains the source node indices and the second row
            contains the destination node indices.
        node_features_dict: A dictionary
            mapping node types to their corresponding node feature. Each node
            feature has shape `[num_nodes, num_features]`.
        edge_features_dict: A dictionary
            mapping edge types to their corresponding edge feature matrices.
            Each edge feature matrix has shape `[num_edges, num_features]`.
        graph_features: A 1D array containing graph-level
            features.
        node_labels_dict: A dictionary mapping
            node types to their corresponding node label arrays.
            Each node label array has shape `[num_nodes]`.
        edge_labels_dict: A dictionary mapping
            edge types to their corresponding edge label arrays.
            Each edge label array has shape `[num_edges]`.
        edge_labels_index_dict: A dictionary
            mapping edge types to their corresponding edge label index arrays.
            The edge label indices indicate the edges for which labels are
            available.
        graph_labels: A 1D array containing graph-level
            labels.
        **kwargs: Additional keyword arguments to store custom attributes.
            Pass ``num_nodes_dict={"node_type": count}`` to set the number of
            nodes of node types that have no features.

    Example:

    .. code-block:: python

        edge_index_dict = {
            ("user", "rates", "movie"): mx.array([[0, 1], [0, 1]]),
            ("movie", "rev_rates", "user"): mx.array([[0, 1], [0, 1]]),
        }
        node_features_dict = {
            "user": mx.array([[0.2], [0.8]]),
            "movie": mx.array([[0.5], [0.3]]),
        }
        data = HeteroGraphData(edge_index_dict, node_features_dict)
    """

    def __init__(
        self,
        edge_index_dict: dict[Any, mx.array],
        node_features_dict: Optional[dict[str, mx.array]] = None,
        edge_features_dict: Optional[dict[Any, mx.array]] = None,
        graph_features: Optional[mx.array] = None,
        node_labels_dict: Optional[dict[str, mx.array]] = None,
        edge_labels_dict: Optional[dict[Any, mx.array]] = None,
        edge_labels_index_dict: Optional[dict[Any, mx.array]] = None,
        graph_labels: Optional[mx.array] = None,
        **kwargs,
    ) -> None:
        self.edge_index_dict = edge_index_dict
        self.node_features_dict = node_features_dict
        self.edge_features_dict = edge_features_dict
        self.graph_features = graph_features
        self.node_labels_dict = node_labels_dict
        self.edge_labels_index_dict = edge_labels_index_dict
        self.edge_labels_dict = edge_labels_dict
        self.graph_labels = graph_labels
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]

        # Node features
        if self.node_features_dict is not None:
            node_features_lines = ["node_features("]
            for key, value in self.node_features_dict.items():
                node_features_lines.append(
                    f"  '{key}': "
                    + f"(shape={value.shape}, {str(value.dtype).split('.')[-1]}),"
                )
            node_features_lines.append(")\n edge_index(")
            lines.extend(node_features_lines)
        else:
            lines.append("\n edge_index(")
        # Edge index
        edge_index_lines = []
        for key, value in self.edge_index_dict.items():
            src_node_type, edge_type, dst_node_type = key
            edge_index_lines.append(
                f"  '{src_node_type} -> {edge_type} -> {dst_node_type}':"
                + f"(shape={value.shape}, {str(value.dtype).split('.')[-1]}),"
            )
        edge_index_lines.append(")\n edge_labels_index_dict(")
        lines.extend(edge_index_lines)

        # Edge labels index dict
        if self.edge_labels_index_dict is not None:
            edge_labels_index_lines = []
            for key, value in self.edge_labels_index_dict.items():
                src_node_type, edge_type, dst_node_type = key
                edge_labels_index_lines.append(
                    f"  '({src_node_type}', '{edge_type}', '{dst_node_type}')':"
                    + f"(shape={value.shape}, {str(value.dtype).split('.')[-1]}),"
                )
            edge_labels_index_lines.append(")\n edge_labels_dict(")
            lines.extend(edge_labels_index_lines)
        else:
            lines.append(")\n edge_labels_dict(")

        # Edge labels dict
        if self.edge_labels_dict is not None:
            edge_labels_lines = []
            for key, value in self.edge_labels_dict.items():
                src_node_type, edge_type, dst_node_type = key
                edge_labels_lines.append(
                    f"  '({src_node_type}', '{edge_type}', '{dst_node_type}')':"
                    + f"(shape={value.shape}, {str(value.dtype).split('.')[-1]}),"
                )
            edge_labels_lines.append(")")
            lines.extend(edge_labels_lines)
        else:
            lines.append(")")

        # Additional attributes
        for k, v in vars(self).items():
            if (
                v is not None
                and not k.startswith("_")
                and k
                not in [
                    "node_features_dict",
                    "edge_index_dict",
                    "edge_labels_index_dict",
                    "edge_labels_dict",
                ]
            ):
                if isinstance(v, dict):
                    attr_lines = [f"{k}("]
                    for key, value in v.items():
                        if isinstance(value, mx.array):
                            attr_lines.append(
                                f"  '{key}': "
                                + f"(shape={value.shape},"
                                + f"{str(value.dtype).split('.')[-1]}),"
                            )
                        else:
                            attr_lines.append(f"  '{key}': {value},")
                    attr_lines.append(")")
                    lines.extend(attr_lines)
                elif isinstance(v, mx.array):
                    lines.append(f"{k}(shape={v.shape}, {str(v.dtype).split('.')[-1]})")
                else:
                    lines.append(f"{k}={v}")

        lines.append(")")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Converts the Data object to a dictionary.

        Returns:
            A dictionary with all attributes of the `HeteroGraphData` object.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @property
    def node_types(self) -> list[str]:
        """All node types: those with features, those appearing in any edge
        type, and those listed in an optional ``num_nodes_dict`` attribute."""
        node_types: list[str] = []
        candidates = list((self.node_features_dict or {}).keys())
        for src_node_type, _, dst_node_type in self.edge_index_dict.keys():
            candidates += [src_node_type, dst_node_type]
        candidates += list((getattr(self, "num_nodes_dict", None) or {}).keys())
        for node_type in candidates:
            if node_type not in node_types:
                node_types.append(node_type)
        return node_types

    @property
    def num_nodes(self) -> dict[str, int]:
        """Number of nodes per node type.

        For each node type, the first available source is used:

        1. an explicit ``num_nodes_dict`` attribute, e.g.
           ``HeteroGraphData(..., num_nodes_dict={"conference": 20})``, for
           node types without features;
        2. the number of rows of its node features;
        3. the largest node id used by any edge type touching it, plus one.
        """
        explicit = getattr(self, "num_nodes_dict", None) or {}
        features = self.node_features_dict or {}
        num_nodes = {}
        for node_type in self.node_types:
            if node_type in explicit:
                num_nodes[node_type] = int(explicit[node_type])
            elif node_type in features:
                num_nodes[node_type] = features[node_type].shape[0]
            else:
                num_nodes[node_type] = self._max_node_id(node_type) + 1
        return num_nodes

    def _max_node_id(self, node_type: str) -> int:
        """Largest node id of ``node_type`` used in any edge index (-1 if none)."""
        max_id = -1
        for (
            src_node_type,
            _,
            dst_node_type,
        ), edge_index in self.edge_index_dict.items():
            if edge_index.shape[1] == 0:
                continue
            if src_node_type == node_type:
                max_id = max(max_id, int(edge_index[0].max().item()))
            if dst_node_type == node_type:
                max_id = max(max_id, int(edge_index[1].max().item()))
        return max_id

    def to_csc_dict(
        self,
    ) -> tuple[dict[Any, mx.array], dict[Any, mx.array], dict[Any, mx.array]]:
        """Converts every edge type to CSC format (edges grouped by destination).

        This is the input format of ``mlx_cluster.hetero_neighbor_sample``.

        Returns:
            A tuple ``(colptr_dict, row_dict, perm_dict)``, each keyed by edge
            type ``(src, rel, dst)``:

            - ``colptr_dict[et]``: ``int64`` array of length ``num_dst_nodes + 1``;
              the incoming edges of destination node ``v`` are at positions
              ``colptr[v]`` to ``colptr[v + 1]``.
            - ``row_dict[et]``: ``int64`` source node of each edge, in CSC order.
            - ``perm_dict[et]``: ``int64`` original position (column in
              ``edge_index_dict[et]``) of each edge in CSC order, used to look up
              edge features of sampled edges.
        """
        num_nodes = self.num_nodes
        colptr_dict, row_dict, perm_dict = {}, {}, {}
        for edge_type, edge_index in self.edge_index_dict.items():
            src_node_type, _, dst_node_type = edge_type
            src = np.asarray(edge_index[0]).astype(np.int64)
            dst = np.asarray(edge_index[1]).astype(np.int64)
            n_src = num_nodes[src_node_type]
            n_dst = num_nodes[dst_node_type]
            if src.size > 0 and (src.min() < 0 or src.max() >= n_src):
                raise ValueError(
                    f"Edge type {edge_type} has source ids outside [0, {n_src}) "
                    f"for node type '{src_node_type}'"
                )
            if dst.size > 0 and (dst.min() < 0 or dst.max() >= n_dst):
                raise ValueError(
                    f"Edge type {edge_type} has destination ids outside [0, {n_dst}) "
                    f"for node type '{dst_node_type}'"
                )

            order = np.argsort(dst, kind="stable")
            colptr = np.zeros(n_dst + 1, dtype=np.int64)
            np.cumsum(np.bincount(dst, minlength=n_dst), out=colptr[1:])

            colptr_dict[edge_type] = mx.array(colptr)
            row_dict[edge_type] = mx.array(src[order])
            perm_dict[edge_type] = mx.array(order.astype(np.int64))
        return colptr_dict, row_dict, perm_dict

    @property
    def num_graph_features(self) -> int:
        """Returns the number of graph features."""
        if self.graph_features is None:
            return 0
        return 1 if self.graph_features.ndim == 1 else self.graph_features.shape[-1]

    @property
    def num_edges(self) -> dict[Any, int]:
        """dictionary of number of edges for each edge type in the graph."""
        return {
            edge_type: edge_index.shape[1]
            for edge_type, edge_index in self.edge_index_dict.items()
        }

    @property
    def num_node_classes(self) -> Union[dict[str, int], None]:
        """
        Returns a dictionary of the number of node classes
        for each node type in the current graph.
        """
        if self.node_features_dict is not None:
            return {
                node_type: self._num_classes("node", node_type)
                for node_type in self.node_features_dict.keys()
            }
        return None

    @property
    def num_edge_classes(self) -> dict[Any, int]:
        """
        Returns a dictionary of the number of edge classes
        for each edge type in the current graph.
        """
        return {
            edge_type: self._num_classes("edge", edge_type)
            for edge_type in self.edge_index_dict.keys()
        }

    @property
    def num_node_features(self) -> dict[str, int]:
        """Returns a dictionary of the number of node features for each node type."""
        num_node_features_dict = {}
        if self.node_features_dict is not None:
            for node_type, node_features in self.node_features_dict.items():
                num_node_features_dict[node_type] = (
                    1 if node_features.ndim == 1 else node_features.shape[-1]
                )
        return num_node_features_dict

    @property
    def num_edge_features(self) -> dict[str, int]:
        """Returns a dictionary of the number of edge features for each edge type."""
        num_edge_features_dict = {}
        if self.edge_features_dict is not None:
            for edge_type, edge_features in self.edge_features_dict.items():
                num_edge_features_dict[edge_type] = (
                    1 if edge_features.ndim == 1 else edge_features.shape[-1]
                )
        return num_edge_features_dict

    def _num_classes(self, task: Literal["node", "edge"], type_key: Any) -> int:
        labels_dict = getattr(self, f"{task}_labels_dict")
        if labels_dict is None or type_key not in labels_dict:
            return 0
        labels = labels_dict[type_key]
        if labels.size == labels.shape[0]:
            return np.unique(np.array(labels)).size
        return labels.shape[-1]

    def has_isolated_nodes(self, node_type: str) -> bool:
        """Returns a boolean of whether the graph has isolated nodes of the
        given type
        (i.e., nodes that don't have a link to any other nodes)"""
        num_nodes = self.num_nodes[node_type]
        for edge_type, edge_index in self.edge_index_dict.items():
            src_node_type, _, dst_node_type = edge_type
            if src_node_type == node_type or dst_node_type == node_type:
                if has_isolated_nodes(edge_index, num_nodes):
                    return True
        return False

    def has_self_loops(self) -> bool:
        """
        Returns a boolean of whether the graph
        contains self loops for the given edge type.
        """

        return all(
            has_self_loops(self.edge_index_dict[edge_type])
            for edge_type in self.edge_index_dict
        )

    def is_undirected(self) -> bool:
        """
        Returns a boolean indicating whether all edge types in the graph are undirected.
        """
        return all(
            is_undirected(
                self.edge_index_dict[edge_type],
                (
                    self.edge_features_dict.get(edge_type, None)  # type: ignore
                    if self.edge_features_dict
                    else None
                ),
            )
            for edge_type in self.edge_index_dict
        )

    def is_directed(self) -> bool:
        """
        Returns a boolean of whether all
        edges are directed or not.
        """
        return all(
            not is_undirected(self.edge_index_dict[edge_type])
            for edge_type in self.edge_index_dict
        )
