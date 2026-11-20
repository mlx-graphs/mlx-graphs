from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.base_transform import BaseTransform
from mlx_graphs.utils.subgraph import _take_rows, subgraph
from mlx_graphs.utils.transformations import (
    _take_edges,
    add_self_loops,
    to_undirected,
)


def _num_nodes(data: GraphData) -> int:
    """Number of nodes of ``data``, covering every node id of ``edge_index``.

    Without node features or an explicit count, ``data.num_nodes`` is the
    number of distinct node ids in ``edge_index``, which is too small when ids
    are skipped.
    """
    num_nodes = data.num_nodes
    if data.edge_index.size > 0:
        num_nodes = max(num_nodes, int(mx.max(data.edge_index).item()) + 1)
    return num_nodes


def _attribute_names(data: GraphData, num_nodes: int) -> tuple[list[str], list[str]]:
    """Names of the per-node and of the per-edge array attributes of ``data``.

    An array is per-node (per-edge) if its first dimension is the number of
    nodes (edges). If the graph has as many nodes as edges, only the arrays
    with "edge" in their name are per-edge. Index-like attributes (e.g.
    ``edge_index``, ``edge_label_index``), the ``edge_label`` that goes with
    ``edge_label_index`` and graph-level attributes are neither.
    """
    num_edges = data.num_edges
    node_keys, edge_keys = [], []
    for key, value in data.to_dict().items():
        if key.startswith(("_", "graph_")) or "index" in key or key == "edge_label":
            continue
        if not isinstance(value, mx.array) or value.ndim == 0:
            continue
        named_edge = "edge" in key
        if value.shape[0] == num_nodes and (num_nodes != num_edges or not named_edge):
            node_keys.append(key)
        elif value.shape[0] == num_edges and (num_nodes != num_edges or named_edge):
            edge_keys.append(key)
    return node_keys, edge_keys


def _select_edges(data: GraphData, keep: np.ndarray, edge_keys: list[str]) -> None:
    """Keeps the edges ``keep`` (host indices) of ``data`` and their attributes."""
    if keep.size == data.num_edges:
        return
    for key in edge_keys:
        setattr(data, key, _take_rows(getattr(data, key), keep))
    data.edge_index = _take_edges(data.edge_index, keep)


def _node_subgraph(data: GraphData, node_mask: np.ndarray, num_nodes: int) -> GraphData:
    """Restricts ``data`` to the nodes of ``node_mask``, relabeling the edges and
    slicing every per-node and per-edge attribute."""
    keep_nodes = np.nonzero(node_mask)[0]
    if keep_nodes.size < num_nodes:
        node_keys, edge_keys = _attribute_names(data, num_nodes)
        edge_index, _, edge_mask = subgraph(
            node_mask,
            data.edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
            return_edge_mask=True,
        )
        keep_edges = np.nonzero(np.asarray(edge_mask))[0]
        if keep_edges.size < data.num_edges:
            for key in edge_keys:
                setattr(data, key, _take_rows(getattr(data, key), keep_edges))
        for key in node_keys:
            setattr(data, key, _take_rows(getattr(data, key), keep_nodes))
        data.edge_index = edge_index
    data.num_nodes = keep_nodes.size
    return data


class ToUndirected(BaseTransform):
    """Makes a graph undirected by adding the reverse of every edge.

    Duplicate edges are merged, so that ``(i, j)`` and ``(j, i)`` are each
    present exactly once, and the edges are sorted by source then destination.
    Every per-edge attribute (e.g. ``edge_features``, ``edge_labels``) follows:
    the values of merged edges are combined with ``reduce``.

    Args:
        reduce: How the attributes of merged edges are combined: ``"add"``,
            ``"mean"``, ``"max"`` or ``"min"``. Default: ``"add"``

    Example:

    .. code-block:: python

        graph = GraphData(edge_index=mx.array([[0, 1, 1], [1, 0, 2]]))
        graph = ToUndirected()(graph)
        graph.edge_index
        # mx.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    """

    def __init__(self, reduce: str = "add"):
        self.reduce = reduce

    def process(self, data: GraphData) -> GraphData:
        """Adds the reverse edges to ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The undirected graph.
        """
        num_nodes = _num_nodes(data)
        _, edge_keys = _attribute_names(data, num_nodes)
        edge_index = data.edge_index
        for key in edge_keys:
            _, values = to_undirected(
                edge_index,
                getattr(data, key),
                coalesce=True,
                num_nodes=num_nodes,
                reduce=self.reduce,
            )
            setattr(data, key, values)
        data.edge_index = to_undirected(edge_index, coalesce=True, num_nodes=num_nodes)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reduce='{self.reduce}')"


class AddSelfLoops(BaseTransform):
    """Adds a self-loop to every node of a graph.

    A self-loop is appended for every node, including the nodes that already
    have one (use :class:`RemoveSelfLoops` first to avoid duplicates). Every
    per-edge attribute (e.g. ``edge_features``, ``edge_weights``) is extended
    with ``fill_value`` for the new edges.

    Args:
        fill_value: Value of the per-edge attributes of the added self-loops.
            Default: ``1.0``

    Example:

    .. code-block:: python

        graph = GraphData(edge_index=mx.array([[0, 1], [1, 2]]))
        graph = AddSelfLoops()(graph)
        graph.edge_index
        # mx.array([[0, 1, 0, 1, 2], [1, 2, 0, 1, 2]])
    """

    def __init__(self, fill_value: float = 1.0):
        self.fill_value = fill_value

    def process(self, data: GraphData) -> GraphData:
        """Adds the self-loops to ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with self-loops.
        """
        num_nodes = _num_nodes(data)
        _, edge_keys = _attribute_names(data, num_nodes)
        for key in edge_keys:
            values = getattr(data, key)
            # `mx.full` ignores an integer dtype for a float fill value.
            loop_values = mx.full(
                (num_nodes, *values.shape[1:]), self.fill_value, dtype=values.dtype
            ).astype(values.dtype)
            setattr(data, key, mx.concatenate([values, loop_values], axis=0))
        data.edge_index = add_self_loops(data.edge_index, num_nodes=num_nodes)
        data.num_nodes = num_nodes
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fill_value={self.fill_value})"


class RemoveSelfLoops(BaseTransform):
    """Removes the self-loops of a graph, together with their per-edge
    attributes (e.g. ``edge_features``, ``edge_labels``).

    Example:

    .. code-block:: python

        graph = GraphData(edge_index=mx.array([[0, 1, 1], [1, 1, 2]]))
        graph = RemoveSelfLoops()(graph)
        graph.edge_index
        # mx.array([[0, 1], [1, 2]])
    """

    def process(self, data: GraphData) -> GraphData:
        """Removes the self-loops of ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph without self-loops. Its number of nodes is unchanged.
        """
        num_nodes = _num_nodes(data)
        _, edge_keys = _attribute_names(data, num_nodes)
        edges = np.asarray(data.edge_index)
        _select_edges(data, np.nonzero(edges[0] != edges[1])[0], edge_keys)
        data.num_nodes = num_nodes
        return data


class RemoveIsolatedNodes(BaseTransform):
    """Removes the isolated nodes of a graph.

    A node is isolated if it has no edge to another node, as in
    :meth:`GraphData.has_isolated_nodes`: a node whose only edges are
    self-loops is isolated, and is removed together with its self-loops. The
    remaining nodes are relabeled to ``[0, num_remaining_nodes)`` in their
    original order, and every per-node attribute (e.g. ``node_features``,
    ``node_labels``, ``train_mask``) and per-edge attribute is sliced
    accordingly.

    Example:

    .. code-block:: python

        graph = GraphData(
            edge_index=mx.array([[0, 3], [3, 0]]),
            node_features=mx.array([[0.0], [1.0], [2.0], [3.0]]),
        )
        graph = RemoveIsolatedNodes()(graph)
        graph.edge_index
        # mx.array([[0, 1], [1, 0]])
        graph.node_features
        # mx.array([[0.0], [3.0]])
    """

    def process(self, data: GraphData) -> GraphData:
        """Removes the isolated nodes of ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph without isolated nodes.
        """
        num_nodes = _num_nodes(data)
        edges = np.asarray(data.edge_index)
        connected = np.zeros(num_nodes, dtype=bool)
        connected[edges[:, edges[0] != edges[1]].reshape(-1)] = True
        return _node_subgraph(data, connected, num_nodes)


class LargestConnectedComponents(BaseTransform):
    """Keeps only the largest connected components of a graph.

    The nodes are relabeled to ``[0, num_remaining_nodes)`` in their original
    order, and every per-node and per-edge attribute is sliced accordingly.
    Requires ``scipy``.

    Args:
        num_components: Number of largest components to keep. Default: ``1``
        connection: ``"weak"`` to ignore the direction of the edges when
            looking for components, ``"strong"`` to require directed paths in
            both directions. Default: ``"weak"``

    Example:

    .. code-block:: python

        graph = GraphData(edge_index=mx.array([[0, 1, 3], [1, 2, 4]]))
        graph = LargestConnectedComponents()(graph)
        graph.edge_index
        # mx.array([[0, 1], [1, 2]])
    """

    def __init__(
        self,
        num_components: int = 1,
        connection: Literal["weak", "strong"] = "weak",
    ):
        if connection not in ("weak", "strong"):
            raise ValueError(
                f"Unknown connection type '{connection}', expected 'weak' or 'strong'"
            )
        self.num_components = num_components
        self.connection = connection

    def process(self, data: GraphData) -> GraphData:
        """Restricts ``data`` to its largest connected components.

        Args:
            data: The graph to transform.

        Returns:
            The graph restricted to its largest components.
        """
        try:
            import scipy.sparse as sp
        except ImportError as error:
            raise ImportError(
                "scipy is required by LargestConnectedComponents, "
                "run `pip install scipy`"
            ) from error

        num_nodes = _num_nodes(data)
        edges = np.asarray(data.edge_index).astype(np.int64)
        adjacency = sp.coo_matrix(
            (np.ones(edges.shape[1], dtype=np.int8), (edges[0], edges[1])),
            shape=(num_nodes, num_nodes),
        )
        num_components, component = sp.csgraph.connected_components(
            adjacency, connection=self.connection
        )
        if num_components <= self.num_components:
            data.num_nodes = num_nodes
            return data

        _, count = np.unique(component, return_counts=True)
        largest = count.argsort()[-self.num_components :]
        return _node_subgraph(data, np.isin(component, largest), num_nodes)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.num_components})"
