from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.base_transform import BaseTransform
from mlx_graphs.transforms.features import _host_edges
from mlx_graphs.transforms.structure import _attribute_names, _num_nodes
from mlx_graphs.utils.transformations import coalesce


def _keeps_explicit_num_nodes(data: GraphData) -> bool:
    """Whether the number of nodes of a transformed ``data`` must be stored
    explicitly: when it already is, or when there are no node features to
    read it from."""
    return getattr(data, "_num_nodes", None) is not None or data.node_features is None


def _csr(src: np.ndarray, dst: np.ndarray, num_nodes: int):
    """``(ptr, order)``: the edges sorted by source are ``order``, and those of
    node ``v`` are ``order[ptr[v]:ptr[v + 1]]``."""
    order = np.argsort(src, kind="stable")
    ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=num_nodes), out=ptr[1:])
    return ptr, order


def _expand(ptr: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each entry ``e`` of ``groups``, all positions ``ptr[groups[e]]`` to
    ``ptr[groups[e] + 1] - 1``. Returns ``(owner, position)``: the entry each
    position belongs to and the position itself."""
    counts = ptr[groups + 1] - ptr[groups]
    owner = np.repeat(np.arange(groups.size), counts)
    starts = np.repeat(ptr[groups], counts)
    offsets = np.arange(owner.size) - np.repeat(np.cumsum(counts) - counts, counts)
    return owner, starts + offsets


def _unique_pairs(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> np.ndarray:
    """Sorted unique ``[2, num_pairs]`` int64 pairs."""
    keys = np.unique(src.astype(np.int64) * max(num_nodes, 1) + dst)
    return np.stack([keys // max(num_nodes, 1), keys % max(num_nodes, 1)])


class VirtualNode(BaseTransform):
    r"""Appends a virtual node connected to all the other nodes, as described
    in `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_, as PyG's ``VirtualNode``.

    The virtual node gets the id ``num_nodes``. An edge from every node to the
    virtual node and an edge from the virtual node to every node are
    appended, in this order. The type of each edge is stored in
    ``edge_type``: existing edges keep their ``edge_type`` (``0`` if there was
    none), and with :math:`t` the largest existing type, edges to the virtual
    node get type :math:`t + 1` and edges from it type :math:`t + 2`.

    Every per-node attribute (e.g. ``node_features``) gets a row of zeros for
    the virtual node, and every per-edge attribute (e.g. ``edge_features``)
    gets rows of zeros for the new edges, except ``edge_weights`` which is
    extended with ones.

    Example:

    .. code-block:: python

        graph = GraphData(
            edge_index=mx.array([[0, 1], [1, 2]]),
            node_features=mx.ones((3, 2)),
        )
        graph = VirtualNode()(graph)
        graph.edge_index
        # mx.array([[0, 1, 0, 1, 2, 3, 3, 3], [1, 2, 3, 3, 3, 0, 1, 2]])
        graph.edge_type
        # mx.array([0, 0, 1, 1, 1, 2, 2, 2])
        graph.node_features.shape
        # (4, 2)
    """

    def process(self, data: GraphData) -> GraphData:
        """Adds the virtual node to ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with the virtual node.
        """
        num_nodes = _num_nodes(data)
        node_keys, edge_keys = _attribute_names(data, num_nodes)
        dtype = data.edge_index.dtype
        edge_type = getattr(data, "edge_type", None)
        if edge_type is None:
            edge_type = mx.zeros((data.num_edges,), dtype=dtype)

        arange = mx.arange(num_nodes, dtype=dtype)
        full = mx.full((num_nodes,), num_nodes, dtype=dtype)
        edge_index = mx.concatenate(
            [
                data.edge_index,
                mx.stack([arange, full]),
                mx.stack([full, arange]),
            ],
            axis=1,
        )
        max_type = int(mx.max(edge_type).item()) if edge_type.size > 0 else 0
        new_type = mx.full((num_nodes,), max_type + 1, dtype=edge_type.dtype)
        edge_type = mx.concatenate([edge_type, new_type, new_type + 1])

        for key in edge_keys:
            if key == "edge_type":
                continue
            value = getattr(data, key)
            fill = 1 if key == "edge_weights" else 0
            new_value = mx.full((2 * num_nodes, *value.shape[1:]), fill, value.dtype)
            setattr(data, key, mx.concatenate([value, new_value.astype(value.dtype)]))
        for key in node_keys:
            value = getattr(data, key)
            new_value = mx.zeros((1, *value.shape[1:]), dtype=value.dtype)
            setattr(data, key, mx.concatenate([value, new_value]))

        data.edge_index = edge_index
        data.edge_type = edge_type
        if _keeps_explicit_num_nodes(data):
            data.num_nodes = num_nodes + 1
        return data


class TwoHop(BaseTransform):
    r"""Adds the two-hop edges to the graph, as PyG's ``TwoHop``.

    An edge :math:`(i, k)` is added for every path :math:`i \rightarrow j
    \rightarrow k` with :math:`i \neq k`. The edges are then coalesced: sorted
    by source then destination node, without duplicates. Every per-edge
    attribute (e.g. ``edge_features``) gets zeros for the new edges, and the
    values of merged duplicate edges are summed.

    Example:

    .. code-block:: python

        graph = GraphData(edge_index=mx.array([[0, 1], [1, 2]]))
        graph = TwoHop()(graph)
        graph.edge_index
        # mx.array([[0, 0, 1], [1, 2, 2]])
    """

    def process(self, data: GraphData) -> GraphData:
        """Adds the two-hop edges to ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with the two-hop edges.
        """
        num_nodes = _num_nodes(data)
        _, edge_keys = _attribute_names(data, num_nodes)
        src, dst = _host_edges(data)
        ptr, order = _csr(src, dst, num_nodes)
        owner, position = _expand(ptr, dst)
        hop_src, hop_dst = src[owner], dst[order[position]]
        not_loop = hop_src != hop_dst
        hops = _unique_pairs(hop_src[not_loop], hop_dst[not_loop], num_nodes)

        dtype = data.edge_index.dtype
        edge_index = mx.concatenate(
            [data.edge_index, mx.array(hops, dtype=dtype)], axis=1
        )
        for key in edge_keys:
            value = getattr(data, key)
            zeros = mx.zeros((hops.shape[1], *value.shape[1:]), dtype=value.dtype)
            _, value = coalesce(
                edge_index, mx.concatenate([value, zeros]), num_nodes, "add"
            )
            setattr(data, key, value)
        data.edge_index = coalesce(edge_index, num_nodes=num_nodes)
        if _keeps_explicit_num_nodes(data):
            data.num_nodes = num_nodes
        return data


class LineGraph(BaseTransform):
    r"""Converts a graph to its line graph, as PyG's ``LineGraph``.

    .. math::
        L(\mathcal{G}) &= (\mathcal{V}^{\prime}, \mathcal{E}^{\prime})

        \mathcal{V}^{\prime} &= \mathcal{E}

        \mathcal{E}^{\prime} &= \{ (e_1, e_2) : e_1 \cap e_2 \neq \emptyset \}

    The edges are first coalesced (sorted, duplicates merged by summing their
    ``edge_features``), and the nodes of the line graph follow that order.

    - For directed graphs (or with ``force_directed``), every edge is a node
      of the line graph, and there is a line edge from :math:`(u, v)` to
      every edge :math:`(v, w)`.
    - For undirected graphs, every pair of reciprocal edges :math:`(u, v)`,
      :math:`(v, u)` is a single node of the line graph (numbered in the
      order of the coalesced edges with :math:`u < v`), and two such nodes
      are connected in both directions if they share an endpoint.

    The new ``node_features`` are the (coalesced) old ``edge_features``, the
    features of reciprocal edges being summed for undirected graphs, and
    ``edge_features`` is removed. The number of nodes is stored explicitly.
    As in PyG, the other attributes are left unchanged.

    Args:
        force_directed: If ``True``, the graph is always treated as
            directed. Default: ``False``

    Example:

    .. code-block:: python

        # The path 0 - 1 - 2, undirected.
        graph = GraphData(edge_index=mx.array([[0, 1, 1, 2], [1, 0, 2, 1]]))
        graph = LineGraph()(graph)
        graph.edge_index
        # mx.array([[0, 1], [1, 0]])
        graph.num_nodes
        # 2
    """

    def __init__(self, force_directed: bool = False):
        self.force_directed = force_directed

    def process(self, data: GraphData) -> GraphData:
        """Converts ``data`` to its line graph.

        Args:
            data: The graph to transform.

        Returns:
            The line graph.
        """
        num_nodes = _num_nodes(data)
        dtype = data.edge_index.dtype
        edge_features = data.edge_features
        if edge_features is not None and edge_features.shape[0] == data.num_edges:
            edge_index, edge_features = coalesce(
                data.edge_index, edge_features, num_nodes, "add"
            )
        else:
            edge_index = coalesce(data.edge_index, num_nodes=num_nodes)
        src, dst = np.asarray(edge_index).astype(np.int64)
        num_edges = src.size

        if self.force_directed or data.is_directed():
            # Coalesced edges are sorted by source: those leaving `v` are
            # `ptr[v]` to `ptr[v + 1] - 1`.
            ptr = np.zeros(num_nodes + 1, dtype=np.int64)
            np.cumsum(np.bincount(src, minlength=num_nodes), out=ptr[1:])
            line_src, line_dst = _expand(ptr, dst)
            line_edges = np.stack([line_src, line_dst])
            node_features = edge_features
            num_line_nodes = num_edges
        else:
            # Id of each undirected edge {u, v}, in the order of the
            # coalesced edges with u < v.
            not_loop = src != dst
            low, high = np.minimum(src, dst), np.maximum(src, dst)
            pair_keys = low * max(num_nodes, 1) + high
            unique_keys = np.unique(pair_keys[not_loop])
            edge_ids = np.searchsorted(unique_keys, pair_keys)
            num_line_nodes = unique_keys.size

            # Line nodes incident to each node, grouped by node: every
            # undirected edge {u, v} (u < v) is incident to u and to v.
            forward = src < dst
            ends = np.concatenate([src[forward], dst[forward]])
            ids = np.concatenate([edge_ids[forward], edge_ids[forward]])
            order = np.lexsort((ids, ends))
            ends, ids = ends[order], ids[order]
            ptr = np.zeros(num_nodes + 1, dtype=np.int64)
            np.cumsum(np.bincount(ends, minlength=num_nodes), out=ptr[1:])
            owner, position = _expand(ptr, ends)
            line_src, line_dst = ids[owner], ids[position]
            not_self = line_src != line_dst
            line_edges = _unique_pairs(
                line_src[not_self], line_dst[not_self], num_line_nodes
            )

            node_features = None
            if edge_features is not None:
                selected = mx.array(np.nonzero(not_loop)[0])
                values = edge_features[selected] if selected.size else None
                node_features = mx.zeros(
                    (num_line_nodes, *edge_features.shape[1:]), edge_features.dtype
                )
                if values is not None:
                    node_features = node_features.at[mx.array(edge_ids[not_loop])].add(
                        values
                    )

        data.edge_index = mx.array(line_edges.astype(np.int64), dtype=dtype)
        data.node_features = node_features
        data.edge_features = None
        data.num_nodes = num_line_nodes
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(force_directed={self.force_directed})"


class ToDense(BaseTransform):
    r"""Converts the edges of a graph to a dense adjacency matrix, as PyG's
    ``ToDense``.

    The adjacency matrix is stored in ``adj``, with shape ``[num_nodes,
    num_nodes, *]``: entry ``adj[i, j]`` holds the ``edge_features`` of the
    edge :math:`(i, j)` (summed over duplicate edges), or 1 if the graph has
    no edge features. ``edge_index`` and ``edge_features`` are then set to
    ``None``.

    If ``num_nodes`` is given, the graph is padded to that many nodes:
    ``node_features``, ``pos`` and ``node_labels`` get rows of zeros, and a
    boolean ``mask`` attribute tells which nodes are real. The number of nodes
    is stored explicitly.

    Args:
        num_nodes: Number of nodes to pad the graph to. Defaults to the number
            of nodes of the graph.

    Example:

    .. code-block:: python

        graph = GraphData(edge_index=mx.array([[0, 1], [1, 2]]))
        graph = ToDense(num_nodes=4)(graph)
        graph.adj
        # mx.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        graph.mask
        # mx.array([True, True, True, False])
    """

    def __init__(self, num_nodes: Optional[int] = None):
        self.num_nodes = num_nodes

    def process(self, data: GraphData) -> GraphData:
        """Converts the edges of ``data`` to a dense adjacency matrix.

        Args:
            data: The graph to transform.

        Returns:
            The graph with ``adj`` and ``mask``.
        """
        orig_num_nodes = _num_nodes(data)
        num_nodes = orig_num_nodes if self.num_nodes is None else self.num_nodes
        if orig_num_nodes > num_nodes:
            raise ValueError(
                f"The graph has {orig_num_nodes} nodes, more than "
                f"num_nodes={num_nodes}"
            )
        edge_features = data.edge_features
        if edge_features is None:
            edge_features = mx.ones((data.num_edges,), dtype=mx.float32)
        src, dst = mx.array(_host_edges(data))
        adj = mx.zeros(
            (num_nodes, num_nodes, *edge_features.shape[1:]), dtype=edge_features.dtype
        )
        if data.num_edges > 0:
            adj = adj.at[src, dst].add(edge_features)

        data.adj = adj
        data.edge_index = None
        data.edge_features = None
        data.mask = mx.arange(num_nodes) < orig_num_nodes

        pad = num_nodes - orig_num_nodes
        for key in ("node_features", "pos", "node_labels"):
            value = getattr(data, key, None)
            if value is None or value.ndim == 0 or value.shape[0] != orig_num_nodes:
                continue
            zeros = mx.zeros((pad, *value.shape[1:]), dtype=value.dtype)
            setattr(data, key, mx.concatenate([value, zeros]))
        data.num_nodes = num_nodes
        return data

    def __repr__(self) -> str:
        if self.num_nodes is None:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}(num_nodes={self.num_nodes})"
