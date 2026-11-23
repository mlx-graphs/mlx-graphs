import copy
import warnings
from typing import Any, Literal, Optional, Sequence, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.utils import has_isolated_nodes, has_self_loops
from mlx_graphs.utils.subgraph import subgraph as _node_subgraph
from mlx_graphs.utils.topology import is_undirected
from mlx_graphs.utils.transformations import coalesce as _coalesce_edges

# PyG attribute names that are aliases of mlx-graphs attributes. ``y`` is
# resolved dynamically (see ``GraphData.y``).
_ALIASES = {"x": "node_features", "edge_attr": "edge_features"}
# Attributes always present on a ``GraphData`` (possibly ``None``).
_DEFAULT_ATTRIBUTES = (
    "edge_index",
    "node_features",
    "edge_features",
    "graph_features",
    "node_labels",
    "edge_labels",
    "graph_labels",
)

IndexLike = Union[mx.array, np.ndarray, Sequence[int], Sequence[bool]]


def _to_host_index(index: IndexLike, size: int, name: str) -> np.ndarray:
    """Converts an index array or a boolean mask over ``size`` elements to host
    int64 indices."""
    index_np = np.asarray(np.array(index) if isinstance(index, mx.array) else index)
    index_np = index_np.reshape(-1)
    if index_np.dtype == np.bool_:
        if index_np.size != size:
            raise ValueError(
                f"The boolean mask `{name}` has {index_np.size} entries, "
                f"expected {size}"
            )
        return np.nonzero(index_np)[0]
    index_np = index_np.astype(np.int64)
    if index_np.size > 0 and (index_np.min() < -size or index_np.max() >= size):
        raise IndexError(f"`{name}` contains indices out of range [0, {size})")
    return np.where(index_np < 0, index_np + size, index_np)


def _take(array: mx.array, keep: np.ndarray, axis: int = 0) -> mx.array:
    """Entries ``keep`` (host indices) of ``array`` along ``axis``."""
    if keep.size == 0:
        shape = list(array.shape)
        shape[axis] = 0
        return mx.zeros(shape, dtype=array.dtype)
    return mx.take(array, mx.array(keep), axis=axis)


class GraphData:
    """
    Represents a graph data object with optional features and labels.

    PyG-style aliases are available for convenience: ``x`` for
    ``node_features``, ``edge_attr`` for ``edge_features`` and ``y`` for the
    labels (see :attr:`y`). They can be read, assigned and passed to the
    constructor (e.g. ``GraphData(edge_index, x=x, y=y)``), but they are not
    stored separately: only the mlx-graphs names appear in :meth:`to_dict`,
    :meth:`keys` and in batches.

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
        # `y` is resolved last: whether it holds node or graph labels depends
        # on the number of nodes, which other arguments (`x`, `num_nodes`) set.
        y_given = "y" in kwargs
        y = kwargs.pop("y", None)
        for key, value in kwargs.items():
            setattr(self, key, value)
        if y_given:
            self.y = y

    def __setstate__(self, state: dict):
        # Graphs pickled before `x`, `edge_attr` and `y` became aliases may
        # store them as plain attributes, which the alias properties would
        # hide: move them to the attributes they now stand for.
        state = dict(state)
        legacy = {k: state.pop(k) for k in ("x", "edge_attr", "y") if k in state}
        self.__dict__.update(state)
        for key, value in legacy.items():
            if value is None:
                continue
            if key == "y":
                taken = self.node_labels is not None or self.graph_labels is not None
            else:
                taken = self.__dict__.get(_ALIASES[key]) is not None
            if taken:
                # Both names were set: keep the old value under another name.
                warnings.warn(
                    f"Unpickled {type(self).__name__} has both '{key}' and the "
                    f"attribute it is now an alias of: '{key}' is kept as "
                    f"'{key}_'.",
                    stacklevel=2,
                )
                self.__dict__[f"{key}_"] = value
            else:
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
        """Number of nodes in the graph.

        The first available source is used:

        1. an explicit count, e.g. ``GraphData(edge_index, num_nodes=4)``;
        2. the number of rows of ``node_features``;
        3. the number of rows of ``pos`` (node positions, e.g. point clouds);
        4. the number of unique node ids in ``edge_index``.

        The last fallback cannot see isolated nodes (nodes without any edge), so
        graphs without node features that may contain isolated nodes should set
        ``num_nodes`` explicitly. Otherwise batching offsets will be wrong.
        """
        explicit = getattr(self, "_num_nodes", None)
        if explicit is not None:
            return explicit
        if self.node_features is not None:
            return self.node_features.shape[0]
        pos = self.__dict__.get("pos")
        if isinstance(pos, mx.array) and pos.ndim > 0:
            return pos.shape[0]
        else:
            # NOTE: This may be slow for large graphs
            return np.unique(np.array(self.edge_index, copy=False)).size

    @num_nodes.setter
    def num_nodes(self, value: Optional[int]):
        self._num_nodes = None if value is None else int(value)

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

    # ------------------------------------------------------------------
    # PyG-style aliases
    # ------------------------------------------------------------------

    @property
    def x(self) -> Optional[mx.array]:
        """PyG-style alias of ``node_features`` (readable and assignable)."""
        return self.node_features

    @x.setter
    def x(self, value: Optional[mx.array]):
        self.node_features = value

    @property
    def edge_attr(self) -> Optional[mx.array]:
        """PyG-style alias of ``edge_features`` (readable and assignable)."""
        return self.edge_features

    @edge_attr.setter
    def edge_attr(self, value: Optional[mx.array]):
        self.edge_features = value

    @property
    def y(self) -> Optional[mx.array]:
        """PyG-style alias of the labels of the graph.

        Reading ``y`` returns ``node_labels`` if they are set, and
        ``graph_labels`` otherwise (``None`` if neither is set). Edge labels
        are never returned.

        Assigning ``y`` replaces the labels that reading ``y`` returns. If the
        graph has no labels yet, the value becomes ``node_labels`` if it has
        one entry per node and ``graph_labels`` otherwise, which is the rule
        used by :func:`~mlx_graphs.utils.from_pyg`. On a single-node graph, a
        one-entry ``y`` is ambiguous and becomes ``graph_labels``, so that
        graph classification datasets with single-node graphs keep the same
        attributes in every graph (and can be batched): set ``node_labels``
        directly for the node labels of a single-node graph. Assigning
        ``None`` clears the labels that reading ``y`` returned.
        """
        if self.node_labels is not None:
            return self.node_labels
        return self.graph_labels

    @y.setter
    def y(self, value: Optional[mx.array]):
        if self.node_labels is not None:
            self.node_labels = value
        elif self.graph_labels is not None:
            self.graph_labels = value
        elif value is not None:
            num_nodes = self.num_nodes
            per_node = value.ndim > 0 and value.shape[0] == num_nodes > 1
            if per_node:
                self.node_labels = value
            else:
                self.graph_labels = value

    @property
    def num_features(self) -> int:
        """PyG-style alias of :attr:`num_node_features`."""
        return self.num_node_features

    # ------------------------------------------------------------------
    # Dictionary-like access
    # ------------------------------------------------------------------

    def keys(self) -> list[str]:
        """Names of the attributes that are set (not ``None``).

        Private attributes are omitted, and ``"num_nodes"`` is included when
        the number of nodes was set explicitly. PyG-style aliases (``x``,
        ``edge_attr``, ``y``) are not listed, only the attributes they refer
        to.

        Returns:
            The list of attribute names.
        """
        keys = [
            key
            for key, value in vars(self).items()
            if value is not None and not key.startswith("_")
        ]
        if getattr(self, "_num_nodes", None) is not None:
            keys.append("num_nodes")
        return keys

    def __contains__(self, key: str) -> bool:
        """Whether the attribute ``key`` (or a PyG-style alias) is set."""
        if key in ("x", "edge_attr", "y"):
            return getattr(self, key) is not None
        return key in self.keys()

    def __getitem__(self, key: str) -> Any:
        """Returns the attribute ``key``, e.g. ``data["node_features"]`` or
        ``data["x"]``. Raises a ``KeyError`` if it is not set."""
        if not isinstance(key, str):
            raise TypeError(
                f"{type(self).__name__} attributes are accessed by name "
                f"(got '{type(key).__name__}')"
            )
        if key not in self:
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        """Sets the attribute ``key``, e.g. ``data["train_mask"] = mask``."""
        setattr(self, key, value)

    def __delitem__(self, key: str):
        """Removes the attribute ``key``. Default attributes (e.g.
        ``node_features``) and aliases are set to ``None`` instead."""
        if key in _DEFAULT_ATTRIBUTES or key in ("x", "edge_attr", "y", "num_nodes"):
            setattr(self, key, None)
        elif key in vars(self):
            delattr(self, key)
        else:
            raise KeyError(key)

    def clone(self) -> "GraphData":
        """Returns a deep copy of the graph, with copies of all the arrays."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Node-level and edge-level attributes
    # ------------------------------------------------------------------

    def _full_num_nodes(self) -> int:
        """Number of nodes, covering every node id used in ``edge_index``."""
        num_nodes = self.num_nodes
        if self.edge_index is not None and self.edge_index.size > 0:
            num_nodes = max(num_nodes, int(mx.max(self.edge_index).item()) + 1)
        return num_nodes

    def _needs_explicit_num_nodes(self) -> bool:
        """Whether a transformed copy must store its number of nodes
        explicitly: when this graph does, or when it cannot be read from
        ``node_features`` (the fallback on ``edge_index`` misses isolated
        nodes)."""
        return getattr(self, "_num_nodes", None) is not None or (
            self.node_features is None
        )

    def _attribute_kind(
        self, key: str, value: Any, num_nodes: int, num_edges: int
    ) -> Optional[str]:
        """``"node"``, ``"edge"`` or ``None`` for the attribute ``key``.

        An array is per-node (per-edge) if its first dimension is the number of
        nodes (edges). If the graph has as many nodes as edges, only the arrays
        with "edge" in their name are per-edge. Index-like attributes (e.g.
        ``edge_index``), the ``edge_label`` that goes with ``edge_label_index``
        and graph-level attributes are neither.
        """
        if key.startswith(("_", "graph_")) or "index" in key:
            return None
        if key in ("edge_label", "face"):
            return None
        if not isinstance(value, mx.array) or value.ndim == 0:
            return None
        named_edge = "edge" in key
        if value.shape[0] == num_nodes and (num_nodes != num_edges or not named_edge):
            return "node"
        if value.shape[0] == num_edges and (num_nodes != num_edges or named_edge):
            return "edge"
        return None

    def _node_and_edge_keys(self) -> tuple[list[str], list[str]]:
        num_nodes = self._full_num_nodes()
        num_edges = self.num_edges if self.edge_index is not None else -1
        node_keys, edge_keys = [], []
        for key, value in vars(self).items():
            kind = self._attribute_kind(key, value, num_nodes, num_edges)
            if kind == "node":
                node_keys.append(key)
            elif kind == "edge":
                edge_keys.append(key)
        return node_keys, edge_keys

    def is_node_attr(self, key: str) -> bool:
        """Whether the attribute ``key`` holds one entry per node.

        An array attribute is per-node if its first dimension is the number of
        nodes. If the graph has as many nodes as edges, attributes with
        "edge" in their name are considered per-edge instead.

        Args:
            key: Name of the attribute (PyG-style aliases are accepted).

        Returns:
            ``True`` if the attribute is node-level.
        """
        key = _ALIASES.get(key, key)
        return key in self._node_and_edge_keys()[0]

    def is_edge_attr(self, key: str) -> bool:
        """Whether the attribute ``key`` holds one entry per edge (excluding
        ``edge_index`` itself).

        Args:
            key: Name of the attribute (PyG-style aliases are accepted).

        Returns:
            ``True`` if the attribute is edge-level.
        """
        key = _ALIASES.get(key, key)
        return key in self._node_and_edge_keys()[1]

    def subgraph(self, subset: IndexLike) -> "GraphData":
        """Returns the subgraph induced by the nodes ``subset``.

        The edges between nodes of ``subset`` are kept and relabeled, and every
        node-level and edge-level attribute (see :meth:`is_node_attr` and
        :meth:`is_edge_attr`) is sliced accordingly. Other attributes are
        shared with the original graph.

        Args:
            subset: The nodes to keep, as an array (or list) of node indices or
                as a boolean mask of shape ``[num_nodes]``. With indices, the
                nodes of the subgraph follow the order of ``subset``.

        Returns:
            A new graph with ``len(subset)`` nodes (or ``subset.sum()`` nodes
            for a mask).

        Example:

        .. code-block:: python

            graph = GraphData(
                edge_index=mx.array([[0, 1, 2], [1, 2, 3]]),
                node_features=mx.array([[0.0], [1.0], [2.0], [3.0]]),
            )
            sub = graph.subgraph(mx.array([1, 2, 3]))
            sub.edge_index
            # mx.array([[0, 1], [1, 2]])
            sub.node_features
            # mx.array([[1.0], [2.0], [3.0]])
        """
        num_nodes = self._full_num_nodes()
        node_keys, edge_keys = self._node_and_edge_keys()
        keep_nodes = _to_host_index(subset, num_nodes, "subset")

        data = copy.copy(self)
        if self.edge_index is not None:
            edge_index, _, edge_mask = _node_subgraph(
                keep_nodes,
                self.edge_index,
                relabel_nodes=True,
                num_nodes=num_nodes,
                return_edge_mask=True,
            )
            keep_edges = np.nonzero(np.asarray(edge_mask))[0]
            for key in edge_keys:
                setattr(data, key, _take(getattr(self, key), keep_edges))
            data.edge_index = edge_index
        for key in node_keys:
            setattr(data, key, _take(getattr(self, key), keep_nodes))
        if self._needs_explicit_num_nodes():
            data.num_nodes = keep_nodes.size
        return data

    def edge_subgraph(self, subset: IndexLike) -> "GraphData":
        """Returns the subgraph made of the edges ``subset``.

        Every node is kept, even if it becomes isolated, and every edge-level
        attribute (see :meth:`is_edge_attr`) is sliced accordingly.

        Args:
            subset: The edges to keep, as an array (or list) of edge indices
                or as a boolean mask of shape ``[num_edges]``.

        Returns:
            A new graph with the selected edges.

        Example:

        .. code-block:: python

            graph = GraphData(
                edge_index=mx.array([[0, 1, 2], [1, 2, 3]]),
                edge_features=mx.array([[0.0], [1.0], [2.0]]),
            )
            sub = graph.edge_subgraph(mx.array([0, 2]))
            sub.edge_index
            # mx.array([[0, 2], [1, 3]])
            sub.num_nodes
            # 4
        """
        num_nodes = self._full_num_nodes()
        _, edge_keys = self._node_and_edge_keys()
        keep = _to_host_index(subset, self.num_edges, "subset")
        data = copy.copy(self)
        for key in edge_keys:
            setattr(data, key, _take(getattr(self, key), keep))
        data.edge_index = _take(self.edge_index, keep, axis=1)
        if self._needs_explicit_num_nodes():
            data.num_nodes = num_nodes
        return data

    # ------------------------------------------------------------------
    # Validation and coalescing
    # ------------------------------------------------------------------

    def validate(self, raise_on_error: bool = True) -> bool:
        """Checks the consistency of the graph.

        The following properties are checked:

        - ``edge_index`` has shape ``[2, num_edges]`` and an integer dtype;
        - ``edge_index`` has no negative node id, and no node id greater than
          or equal to the number of nodes when it is known (set explicitly or
          given by ``node_features``);
        - node-level default attributes (``node_features``, ``node_labels``)
          have one entry per node, if the number of nodes is set explicitly;
        - ``edge_features`` and ``edge_labels`` have one entry per edge.

        Args:
            raise_on_error: If ``True``, raises a ``ValueError`` on the first
                failed check. Otherwise, emits a warning for every failed check.
                Default: ``True``

        Returns:
            ``True`` if every check passed, ``False`` otherwise.
        """
        name = type(self).__name__
        errors = []
        edge_index = self.edge_index
        known_num_nodes = getattr(self, "_num_nodes", None)
        if known_num_nodes is None and self.node_features is not None:
            known_num_nodes = self.node_features.shape[0]

        if edge_index is None:
            errors.append(f"'edge_index' is undefined in '{name}'")
        elif edge_index.ndim != 2 or edge_index.shape[0] != 2:
            errors.append(
                f"'edge_index' needs to be of shape [2, num_edges] in '{name}' "
                f"(found {tuple(edge_index.shape)})"
            )
        else:
            if not mx.issubdtype(edge_index.dtype, mx.integer):
                errors.append(
                    f"'edge_index' needs an integer dtype in '{name}' "
                    f"(found {edge_index.dtype})"
                )
            if edge_index.size > 0:
                min_id = int(mx.min(edge_index).item())
                max_id = int(mx.max(edge_index).item())
                if min_id < 0:
                    errors.append(
                        f"'edge_index' contains negative indices in '{name}' "
                        f"(found {min_id})"
                    )
                if known_num_nodes is not None and max_id >= known_num_nodes:
                    errors.append(
                        "'edge_index' contains larger indices than the number "
                        f"of nodes ({known_num_nodes}) in '{name}' "
                        f"(found {max_id})"
                    )
            for key in ("edge_features", "edge_labels"):
                value = getattr(self, key)
                if value is not None and (
                    value.ndim == 0 or value.shape[0] != edge_index.shape[1]
                ):
                    errors.append(
                        f"'{key}' needs one entry per edge ({edge_index.shape[1]}) "
                        f"in '{name}' (found shape {tuple(value.shape)})"
                    )

        explicit = getattr(self, "_num_nodes", None)
        if explicit is not None:
            for key in ("node_features", "node_labels"):
                value = getattr(self, key)
                if value is not None and (
                    value.ndim == 0 or value.shape[0] != explicit
                ):
                    errors.append(
                        f"'{key}' needs one entry per node ({explicit}) in "
                        f"'{name}' (found shape {tuple(value.shape)})"
                    )

        for error in errors:
            if raise_on_error:
                raise ValueError(error)
            warnings.warn(error, stacklevel=2)
        return len(errors) == 0

    def is_coalesced(self) -> bool:
        """Whether ``edge_index`` is sorted by source then destination node and
        contains no duplicate edge."""
        if self.edge_index.shape[1] < 2:
            return True
        edges = np.asarray(self.edge_index).astype(np.int64)
        src, dst = edges[0], edges[1]
        return bool(
            np.all(
                (src[1:] > src[:-1]) | ((src[1:] == src[:-1]) & (dst[1:] > dst[:-1]))
            )
        )

    def coalesce(self, reduce: str = "add") -> "GraphData":
        """Sorts ``edge_index`` by source then destination node and merges
        duplicate edges.

        Every edge-level attribute (see :meth:`is_edge_attr`, e.g.
        ``edge_features``, ``edge_weights``) follows the edges, and the values
        of merged edges are combined with ``reduce``. The original graph is
        left unchanged.

        Args:
            reduce: How the attributes of merged edges are combined: ``"add"``,
                ``"mean"``, ``"max"`` or ``"min"``. Default: ``"add"``

        Returns:
            A coalesced copy of the graph (the graph itself if it is already
            coalesced).

        Example:

        .. code-block:: python

            graph = GraphData(
                edge_index=mx.array([[1, 0, 1], [0, 1, 0]]),
                edge_features=mx.array([[1.0], [2.0], [3.0]]),
            )
            graph = graph.coalesce()
            graph.edge_index
            # mx.array([[0, 1], [1, 0]])
            graph.edge_features
            # mx.array([[2.0], [4.0]])
        """
        if self.is_coalesced():
            return self
        num_nodes = self._full_num_nodes()
        _, edge_keys = self._node_and_edge_keys()
        data = copy.copy(self)
        for key in edge_keys:
            _, values = _coalesce_edges(
                self.edge_index, getattr(self, key), num_nodes, reduce
            )
            setattr(data, key, values)
        data.edge_index = _coalesce_edges(self.edge_index, num_nodes=num_nodes)
        if self._needs_explicit_num_nodes():
            data.num_nodes = num_nodes
        return data

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

        By default, all attribute names containing "index" (and the mesh
        triangles ``face``) will be concatenated on axis 1, e.g. `edge_index`.
        Other attributes are concatenated on axis 0,
        e.g. node features.

        Args:
            key: Name of the attribute on which change the default concatenation
                dimension while using batching.

        Returns:
            The dimension where concatenation will happen when batching.
        """
        if "index" in key or key == "face":
            return 1
        return 0

    def __inc__(self, key: str, *args, **kwargs) -> Union[int, None]:
        """This method can be overriden when batching is used with custom attributes.
        Given the name of a custom attribute `key`, returns an integer indicating the
        incrementing value to apply to the elemnts of the attribute.

        By default, all attribute names containing "index" (and ``face``) will be
        incremented based on
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
        if "index" in key or key == "face":
            return self.num_nodes
        if key in ("batch", "batch_indices"):
            # A graph's own assignment of nodes to sub-batches (e.g. ShaDow
            # subgraphs): offset by its number of groups, as PyG does.
            value = self.to_dict().get(key)
            if isinstance(value, mx.array):
                return int(value.max().item()) + 1 if value.size > 0 else 0
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
    def edge_types(self) -> list[tuple]:
        """All edge types ``(src_node_type, relation, dst_node_type)``, in the
        order of ``edge_index_dict``."""
        return list(self.edge_index_dict.keys())

    def metadata(self) -> tuple[list[str], list[tuple]]:
        """Returns the heterogeneous metadata ``(node_types, edge_types)`` of
        the graph, as in PyG's ``HeteroData.metadata()``.

        Returns:
            A tuple with the list of node types and the list of edge types.
        """
        return self.node_types, self.edge_types

    def _typed_attributes(self) -> tuple[dict[str, dict], dict[str, dict]]:
        """Dict attributes ``<name>_dict`` keyed by node types and by edge
        types (excluding indices), as ``{<name>: dict}``."""
        node_types, edge_types = set(self.node_types), set(self.edge_types)
        skip = ("edge_index_dict", "edge_labels_index_dict", "num_nodes_dict")
        node_attrs, edge_attrs = {}, {}
        for key, value in vars(self).items():
            if key in skip or key.startswith("_") or not key.endswith("_dict"):
                continue
            if not isinstance(value, dict) or len(value) == 0:
                continue
            if not all(isinstance(v, mx.array) for v in value.values()):
                continue
            if set(value.keys()) <= node_types:
                node_attrs[key[: -len("_dict")]] = value
            elif set(value.keys()) <= edge_types:
                edge_attrs[key[: -len("_dict")]] = value
        return node_attrs, edge_attrs

    def to_homogeneous(
        self,
        node_attrs: Optional[list[str]] = None,
        edge_attrs: Optional[list[str]] = None,
        add_node_type: bool = True,
        add_edge_type: bool = True,
        dummy_values: bool = True,
    ) -> GraphData:
        """Converts the heterogeneous graph to a homogeneous :class:`GraphData`,
        as PyG's ``HeteroData.to_homogeneous()``.

        Nodes are stacked by node type in the order of :attr:`node_types`, and
        the edges of every edge type (in the order of :attr:`edge_types`) are
        offset accordingly and concatenated.

        Per-type attributes are the dict attributes named ``<name>_dict`` whose
        keys are node types (e.g. ``node_features_dict``,
        ``node_labels_dict``, ``train_mask_dict``) or edge types (e.g.
        ``edge_features_dict``). They are merged into a single ``<name>``
        attribute (e.g. ``node_features``) if every type has them with the
        same number of dimensions and the same shape beyond the first one
        (2D node attributes of different widths are zero-padded to the widest
        one). With ``dummy_values``, types missing an attribute are first
        filled with ``NaN`` (floats), ``False`` (booleans) or ``-1``
        (integers), provided the types that have it agree on its shape.

        If ``edge_labels_index_dict`` is set, ``edge_labels_dict`` is aligned
        with it rather than with the edges: both are then offset and
        concatenated into ``edge_labels_index`` and ``edge_labels``.
        Graph-level attributes (e.g. ``graph_labels``) are copied as they are,
        and the other dict attributes are dropped.

        Args:
            node_attrs: Names (without ``_dict``) of the node attributes to
                merge. All mergeable ones if ``None``. Default: ``None``
            edge_attrs: Names (without ``_dict``) of the edge attributes to
                merge. All mergeable ones if ``None``. Default: ``None``
            add_node_type: Whether to add a ``node_type`` array holding the
                index of the type of each node in :attr:`node_types`.
                Default: ``True``
            add_edge_type: Whether to add an ``edge_type`` array holding the
                index of the type of each edge in :attr:`edge_types`.
                Default: ``True``
            dummy_values: Whether to fill missing attributes with dummy
                values. Default: ``True``

        Returns:
            The homogeneous graph, with an explicit ``num_nodes``.

        Example:

        .. code-block:: python

            data = HeteroGraphData(
                edge_index_dict={
                    ("user", "rates", "movie"): mx.array([[0, 1], [0, 1]]),
                },
                node_features_dict={
                    "user": mx.array([[0.2], [0.8]]),
                    "movie": mx.array([[0.5], [0.3]]),
                },
            )
            graph = data.to_homogeneous()
            graph.edge_index
            # mx.array([[0, 1], [2, 3]])
            graph.node_type
            # mx.array([0, 0, 1, 1])
        """
        node_types, edge_types = self.metadata()
        num_nodes_dict = self.num_nodes
        offsets, total = {}, 0
        for node_type in node_types:
            offsets[node_type] = total
            total += num_nodes_dict[node_type]
        num_edges_dict = self.num_edges

        node_dicts, edge_dicts = self._typed_attributes()
        if self.edge_labels_index_dict:
            edge_dicts.pop("edge_labels", None)

        def merge(
            dicts: dict[str, dict],
            types: list,
            sizes: dict,
            keys: Optional[list[str]],
            pad_2d: bool,
        ) -> dict[str, mx.array]:
            merged = {}
            for name, values in dicts.items():
                if keys is not None and name not in keys:
                    continue
                trailing = {tuple(v.shape[1:]) for v in values.values()}
                ndims = {v.ndim for v in values.values()}
                if len(values) < len(types):
                    if not dummy_values or len(trailing) != 1:
                        continue
                    ref = next(iter(values.values()))
                    if mx.issubdtype(ref.dtype, mx.floating):
                        dummy: Any = float("nan")
                    elif ref.dtype == mx.bool_:
                        dummy = False
                    else:
                        dummy = -1
                    values = {
                        t: values.get(t)
                        if t in values
                        else mx.full((sizes[t], *ref.shape[1:]), dummy, ref.dtype)
                        for t in types
                    }
                if len(ndims) != 1:
                    continue
                arrays = [values[t] for t in types]
                if len(trailing) != 1:
                    if not (pad_2d and ndims == {2}):
                        continue
                    width = max(a.shape[1] for a in arrays)
                    arrays = [
                        mx.pad(a, [(0, 0), (0, width - a.shape[1])]) for a in arrays
                    ]
                merged[name] = mx.concatenate(arrays, axis=0)
            return merged

        attributes: dict[str, Any] = {}
        handled = {f"{name}_dict" for name in (*node_dicts, *edge_dicts)}
        handled |= {"edge_index_dict", "edge_labels_index_dict", "num_nodes_dict"}
        if self.edge_labels_index_dict:
            handled.add("edge_labels_dict")
        for key, value in vars(self).items():
            if key in handled or key.startswith("_"):
                continue
            # Other dicts cannot be merged nor batched as graph attributes.
            if value is not None and not isinstance(value, dict):
                attributes[key] = value

        attributes.update(
            merge(node_dicts, node_types, num_nodes_dict, node_attrs, pad_2d=True)
        )
        attributes.update(
            merge(edge_dicts, edge_types, num_edges_dict, edge_attrs, pad_2d=False)
        )

        def offset(edge_type: tuple, edge_index: mx.array) -> mx.array:
            src, _, dst = edge_type
            shift = mx.array([[offsets[src]], [offsets[dst]]], dtype=edge_index.dtype)
            return edge_index + shift

        if len(edge_types) > 0:
            edge_index = mx.concatenate(
                [offset(et, self.edge_index_dict[et]) for et in edge_types], axis=1
            )
        else:
            edge_index = mx.zeros((2, 0), dtype=mx.int32)

        if self.edge_labels_index_dict:
            label_types = list(self.edge_labels_index_dict.keys())
            attributes["edge_labels_index"] = mx.concatenate(
                [offset(et, self.edge_labels_index_dict[et]) for et in label_types],
                axis=1,
            )
            labels = self.edge_labels_dict or {}
            if all(et in labels for et in label_types):
                attributes["edge_labels"] = mx.concatenate(
                    [labels[et] for et in label_types], axis=0
                )

        if add_node_type:
            attributes["node_type"] = mx.array(
                np.repeat(
                    np.arange(len(node_types)),
                    [num_nodes_dict[t] for t in node_types],
                ).astype(np.int32)
            )
        if add_edge_type:
            attributes["edge_type"] = mx.array(
                np.repeat(
                    np.arange(len(edge_types)),
                    [num_edges_dict[t] for t in edge_types],
                ).astype(np.int32)
            )

        graph = GraphData(edge_index=edge_index, **attributes)
        graph.num_nodes = total
        return graph

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
