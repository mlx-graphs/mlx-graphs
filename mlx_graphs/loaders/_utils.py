"""Host-side (numpy) helpers shared by the samplers and loaders."""

from typing import Any, Optional, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData

# Attributes of a `GraphData` that loaders handle explicitly.
_STANDARD_ATTRS = (
    "edge_index",
    "node_features",
    "edge_features",
    "graph_features",
    "node_labels",
    "edge_labels",
    "graph_labels",
)


def to_numpy(array: Union[mx.array, np.ndarray, list, tuple]) -> np.ndarray:
    """Host copy of ``array``; unsigned integers are widened to int64."""
    out = np.asarray(array)
    if out.dtype.kind == "u":
        out = out.astype(np.int64)
    return out


def to_index(
    nodes: Union[None, mx.array, np.ndarray, list, tuple], size: int
) -> np.ndarray:
    """Node indices (int64) from ``None`` (all), indices or a boolean mask."""
    if nodes is None:
        return np.arange(size, dtype=np.int64)
    nodes = to_numpy(nodes).reshape(-1)
    if nodes.dtype == np.bool_:
        if nodes.size != size:
            raise ValueError(f"Boolean mask has {nodes.size} entries, expected {size}.")
        return np.nonzero(nodes)[0].astype(np.int64)
    nodes = nodes.astype(np.int64)
    # The C++ samplers read their CSR/CSC buffers at these ids without bounds
    # checks.
    check_index_bounds(nodes, size, "Node")
    return nodes


def index_dtype(size: int) -> mx.Dtype:
    """dtype of ids in ``[0, size)`` returned by the loaders: int32, as
    everywhere in mlx-graphs, unless they cannot fit in it (graphs with more
    than 2**31 - 1 nodes or edges)."""
    return mx.int32 if size <= np.iinfo(np.int32).max else mx.int64


def index_array(values: Union[mx.array, np.ndarray, list], size: int) -> mx.array:
    """``values`` (ids in ``[0, size)``) as an ``mx.array`` of
    :func:`index_dtype`."""
    dtype = index_dtype(size)
    if isinstance(values, mx.array):
        return values.astype(dtype)
    return mx.array(np.asarray(values), dtype=dtype)


def check_index_bounds(index: np.ndarray, size: int, name: str):
    """Raises an ``IndexError`` if ``index`` has entries outside ``[0, size)``."""
    if index.size > 0 and (index.min() < 0 or index.max() >= size):
        raise IndexError(
            f"{name} indices must be in [0, {size}) (got values from "
            f"{index.min()} to {index.max()})"
        )


def segment_arange(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenation of ``arange(start, start + count)`` for every pair."""
    counts = counts.astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(starts.astype(np.int64), counts) + (
        np.arange(total, dtype=np.int64) - offsets
    )


class Adjacency:
    """CSR (by source) and CSC (by target) views of an ``edge_index``.

    Args:
        edge_index: ``[2, num_edges]`` array.
        num_nodes: Number of nodes of the graph.
    """

    def __init__(self, edge_index: Union[mx.array, np.ndarray], num_nodes: int):
        edges = to_numpy(edge_index).astype(np.int64).reshape(2, -1)
        self.num_nodes = int(num_nodes)
        self.num_edges = edges.shape[1]
        self.src, self.dst = edges[0], edges[1]

        # CSR: out-edges of each node, sorted by source.
        self.csr_perm = np.argsort(self.src, kind="stable")
        self.rowptr = self._ptr(self.src)
        self.csr_col = self.dst[self.csr_perm]

        # CSC: in-edges of each node, sorted by target.
        self.csc_perm = np.argsort(self.dst, kind="stable")
        self.colptr = self._ptr(self.dst)
        self.csc_row = self.src[self.csc_perm]

    def _ptr(self, index: np.ndarray) -> np.ndarray:
        ptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
        np.cumsum(np.bincount(index, minlength=self.num_nodes), out=ptr[1:])
        return ptr

    @property
    def out_degree(self) -> np.ndarray:
        return np.diff(self.rowptr)

    @property
    def in_degree(self) -> np.ndarray:
        return np.diff(self.colptr)

    def induced_subgraph(self, node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Edges whose two endpoints are in ``node_ids``.

        Args:
            node_ids: Distinct global node ids. Their positions become the
                local node ids.

        Returns:
            ``(edge_index, e_id)``: the local ``[2, num_sub_edges]`` edges
            (ordered by local source) and their original edge ids.
        """
        node_ids = np.asarray(node_ids, dtype=np.int64)
        local = np.full(self.num_nodes, -1, dtype=np.int64)
        local[node_ids] = np.arange(node_ids.size)

        starts = self.rowptr[node_ids]
        counts = self.rowptr[node_ids + 1] - starts
        pos = segment_arange(starts, counts)
        src_local = np.repeat(np.arange(node_ids.size), counts)
        dst_local = local[self.csr_col[pos]]
        keep = dst_local >= 0
        edge_index = np.stack([src_local[keep], dst_local[keep]])
        return edge_index, self.csr_perm[pos[keep]]


def _is_index_attr(name: str) -> bool:
    # Index-valued attributes (e.g. `edge_label_index`) cannot be sliced.
    return "index" in name


def slice_graph(
    data: GraphData,
    n_id: np.ndarray,
    e_id: np.ndarray,
    edge_index: Union[np.ndarray, mx.array],
    /,
    num_nodes: Optional[int] = None,
    num_edges: Optional[int] = None,
    **extra: Any,
) -> GraphData:
    """Builds the subgraph of ``data`` made of nodes ``n_id`` and edges ``e_id``.

    Node-level arrays (first dimension equal to the number of nodes) are
    indexed with ``n_id``, edge-level arrays (first dimension equal to the
    number of edges) with ``e_id``. Like PyG, an array matching both is treated
    as node-level, unless its name contains ``"edge"``. Other values are copied
    as they are, except index-valued custom attributes (with ``"index"`` in
    their name), which are dropped.

    Args:
        data: The full graph.
        n_id: Global id of every node of the subgraph.
        e_id: Original id of every edge of the subgraph.
        edge_index: Local ``[2, len(e_id)]`` edges of the subgraph.
        num_nodes: Number of nodes of ``data`` (computed if not given).
        num_edges: Number of edges of ``data`` (computed if not given).
        **extra: Additional attributes to set on the subgraph.

    Returns:
        The subgraph as a :class:`GraphData`.
    """
    N = data.num_nodes if num_nodes is None else num_nodes
    E = data.num_edges if num_edges is None else num_edges
    n_idx = mx.array(np.asarray(n_id, dtype=np.int64))
    e_idx = mx.array(np.asarray(e_id, dtype=np.int64))
    dtype = data.edge_index.dtype

    def take(name: str, value: Any) -> Any:
        if isinstance(value, mx.array) and value.ndim >= 1:
            # With as many nodes as edges, attributes named "edge..." (e.g.
            # `edge_features`) are edge-level, as in `GraphData.is_edge_attr`.
            if value.shape[0] == N and (N != E or "edge" not in name):
                return value[n_idx] if n_idx.size else value[:0]
            if value.shape[0] == E:
                return value[e_idx] if e_idx.size else value[:0]
        return value

    kwargs: dict[str, Any] = {}
    for name, value in vars(data).items():
        if name.startswith("_") or name in _STANDARD_ATTRS or name in extra:
            continue
        if _is_index_attr(name) or value is None:
            continue
        kwargs[name] = take(name, value)
    kwargs.update(extra)

    if isinstance(edge_index, np.ndarray):
        edge_index = mx.array(edge_index.reshape(2, -1).astype(np.int64), dtype=dtype)

    out = GraphData(
        edge_index=edge_index,
        node_features=take("node_features", data.node_features),
        edge_features=take("edge_features", data.edge_features),
        graph_features=data.graph_features,
        node_labels=take("node_labels", data.node_labels),
        edge_labels=take("edge_labels", data.edge_labels),
        graph_labels=data.graph_labels,
        **kwargs,
    )
    out.num_nodes = int(np.asarray(n_id).size)
    return out


def sample_neighbors_grouped(
    colptr: np.ndarray,
    row: np.ndarray,
    nodes: np.ndarray,
    num_neighbors: int,
    replace: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Samples up to ``num_neighbors`` in-neighbors of every node in ``nodes``.

    Args:
        colptr: CSC pointers.
        row: CSC sources.
        nodes: Nodes to sample for (may contain duplicates).
        num_neighbors: Fan-out; ``-1`` keeps every neighbor.
        replace: Sample with replacement (``np.random`` global state).

    Returns:
        ``(owner, neighbor)``: for each sampled neighbor, the position in
        ``nodes`` it was sampled for and its global id.
    """
    starts = colptr[nodes]
    degs = colptr[nodes + 1] - starts
    if num_neighbors < 0:
        pos = segment_arange(starts, degs)
        owner = np.repeat(np.arange(nodes.size), degs)
        return owner, row[pos]

    if replace:
        has = np.nonzero(degs > 0)[0]
        owner = np.repeat(has, num_neighbors)
        offs = np.floor(
            np.random.random(owner.size) * np.repeat(degs[has], num_neighbors)
        ).astype(np.int64)
        return owner, row[starts[owner] + offs]

    # Without replacement: random keys, keep the `num_neighbors` smallest per
    # node.
    pos = segment_arange(starts, degs)
    owner = np.repeat(np.arange(nodes.size), degs)
    order = np.lexsort((np.random.random(pos.size), owner))
    group_start = np.repeat(np.cumsum(degs) - degs, degs)
    rank = np.arange(pos.size) - group_start
    keep = order[rank < num_neighbors]
    return owner[keep], row[pos[keep]]
