from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.negative_sampling import _host_rng
from mlx_graphs.utils.subgraph import _subgraph
from mlx_graphs.utils.transformations import (
    _infer_num_nodes,
    _take_edges,
    _to_numpy,
)
from mlx_graphs.utils.validators import validate_edge_index


def _check_probability(p: float) -> None:
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability has to be between 0 and 1 (got {p})")


@validate_edge_index
def dropout_edge(
    edge_index: mx.array,
    p: float = 0.5,
    force_undirected: bool = False,
    training: bool = True,
) -> tuple[mx.array, mx.array]:
    """
    Randomly drops edges of ``edge_index`` with probability ``p``.

    Randomness comes from ``mx.random.uniform``, so results are reproducible
    with ``mx.random.seed``.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        p: Probability of dropping each edge. Default: ``0.5``
        force_undirected: If ``True``, either both directions of an edge are
            dropped or both are kept: only the edges with ``src <= dst`` are
            sampled, and the kept ones are returned in both directions.
            Default: ``False``
        training: If ``False``, nothing is dropped. Default: ``True``

    Returns:
        A tuple ``(edge_index, edge_mask)`` with the kept edges and a boolean
        mask of shape ``[num_edges]`` of the input edges that were kept.

        When ``force_undirected=True`` (and edges are being dropped), the
        second element is instead, as in PyG, an array with the position in
        the input ``edge_index`` of each returned edge, e.g. to index edge
        features: an edge and its added reverse share the same position.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])

        edge_index, edge_mask = dropout_edge(edge_index, p=0.5)
        # edge_index: e.g. mx.array([[0, 1, 2, 2], [1, 2, 1, 3]])
        # edge_mask: e.g. mx.array([True, False, True, True, True, False])

        edge_index, edge_id = dropout_edge(edge_index, force_undirected=True)
        # edge_index: e.g. mx.array([[0, 1], [1, 0]])
        # edge_id: e.g. mx.array([0, 0])
    """
    _check_probability(p)
    num_edges = edge_index.shape[1]
    if not training or p == 0.0:
        return edge_index, mx.ones((num_edges,), dtype=mx.bool_)

    edge_mask = _to_numpy(mx.random.uniform(shape=(num_edges,))) >= p
    if force_undirected:
        edges = _to_numpy(edge_index)
        edge_mask = edge_mask & (edges[0] <= edges[1])

    keep = np.nonzero(edge_mask)[0]
    kept_edge_index = _take_edges(edge_index, keep)
    if not force_undirected:
        return kept_edge_index, mx.array(edge_mask)

    kept_edge_index = mx.concatenate([kept_edge_index, kept_edge_index[::-1]], axis=1)
    return kept_edge_index, mx.array(np.concatenate([keep, keep]))


@validate_edge_index
def dropout_node(
    edge_index: mx.array,
    p: float = 0.5,
    num_nodes: Optional[int] = None,
    training: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Randomly drops nodes with probability ``p``, together with all the edges
    connected to them. The kept edges are not relabeled.

    Randomness comes from ``mx.random.uniform``, so results are reproducible
    with ``mx.random.seed``.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        p: Probability of dropping each node. Default: ``0.5``
        num_nodes: Number of nodes in the graph. Inferred from ``edge_index``
            if not given.
        training: If ``False``, nothing is dropped. Default: ``True``

    Returns:
        A tuple ``(edge_index, edge_mask, node_mask)`` with the kept edges, a
        boolean mask of shape ``[num_edges]`` of the edges that were kept and
        a boolean mask of shape ``[num_nodes]`` of the nodes that were kept.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])

        edge_index, edge_mask, node_mask = dropout_node(edge_index, p=0.5)
        # edge_index: e.g. mx.array([[0, 1], [1, 0]])
        # edge_mask: e.g. mx.array([True, True, False, False, False, False])
        # node_mask: e.g. mx.array([True, True, False, False])
    """
    _check_probability(p)
    num_nodes = _infer_num_nodes(edge_index, num_nodes)
    if not training or p == 0.0:
        return (
            edge_index,
            mx.ones((edge_index.shape[1],), dtype=mx.bool_),
            mx.ones((num_nodes,), dtype=mx.bool_),
        )

    node_mask = _to_numpy(mx.random.uniform(shape=(num_nodes,))) > p
    kept_edge_index, _, edge_mask = _subgraph(
        edge_index, node_mask, None, False, num_nodes
    )
    return kept_edge_index, mx.array(edge_mask), mx.array(node_mask)


def _random_walk_edges(
    rowptr: np.ndarray,
    col: np.ndarray,
    start: np.ndarray,
    walk_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform random walks from ``start`` on a CSR graph, returning the
    ``(src, dst)`` pairs of all the steps. A walk that reaches a node without
    out-neighbors stays on it."""
    current = start
    srcs, dsts = [], []
    for _ in range(walk_length):
        begin = rowptr[current]
        deg = rowptr[current + 1] - begin
        offset = np.floor(rng.random(current.size) * np.maximum(deg, 1)).astype(
            np.int64
        )
        has_next = deg > 0
        nxt = current.copy()
        nxt[has_next] = col[begin[has_next] + offset[has_next]]
        srcs.append(current)
        dsts.append(nxt)
        current = nxt
    if not srcs:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(srcs), np.concatenate(dsts)


@validate_edge_index
def dropout_path(
    edge_index: mx.array,
    p: float = 0.2,
    walks_per_node: int = 1,
    walk_length: int = 3,
    num_nodes: Optional[int] = None,
    is_sorted: bool = False,
    training: bool = True,
) -> tuple[mx.array, mx.array]:
    """
    Drops edges along random walks (`"MaskGAE: Masked Graph Modeling Meets
    Graph Autoencoders" <https://arxiv.org/abs/2205.10053>`_).

    Each edge is sampled with probability ``p``. From the source of every
    sampled edge, ``walks_per_node`` uniform random walks of ``walk_length``
    steps are run, and every edge traversed by a walk is dropped (with all its
    duplicates).

    Randomness comes from ``mx.random``, so results are reproducible with
    ``mx.random.seed``.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        p: Probability of sampling each edge as the start of the walks.
            Default: ``0.2``
        walks_per_node: Number of walks started from each sampled edge.
            Default: ``1``
        walk_length: Number of steps of each walk. Default: ``3``
        num_nodes: Number of nodes in the graph. Inferred from ``edge_index``
            if not given.
        is_sorted: If ``True``, ``edge_index`` is assumed to be sorted by
            source node, which skips a sort. Default: ``False``
        training: If ``False``, nothing is dropped. Default: ``True``

    Returns:
        A tuple ``(edge_index, edge_mask)`` with the kept edges and a boolean
        mask of shape ``[num_edges]`` of the input edges that were kept.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])

        edge_index, edge_mask = dropout_path(edge_index)
        # edge_index: e.g. mx.array([[0, 1, 2, 2], [1, 2, 1, 3]])
        # edge_mask: e.g. mx.array([True, False, True, True, True, False])
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Sample probability has to be between 0 and 1 (got {p})")
    num_edges = edge_index.shape[1]
    if not training or p == 0.0 or num_edges == 0:
        return edge_index, mx.ones((num_edges,), dtype=mx.bool_)

    n = _infer_num_nodes(edge_index, num_nodes)
    edges = _to_numpy(edge_index).astype(np.int64)
    if is_sorted:
        row, col = edges[0], edges[1]
    else:
        order = np.argsort(edges[0], kind="stable")
        row, col = edges[0][order], edges[1][order]
    rowptr = np.concatenate([[0], np.cumsum(np.bincount(row, minlength=n))])

    sample_mask = _to_numpy(mx.random.uniform(shape=(num_edges,))) <= p
    start = np.tile(row[sample_mask], walks_per_node)
    rng = _host_rng()
    walk_src, walk_dst = _random_walk_edges(rowptr, col, start, walk_length, rng)

    dropped = np.isin(edges[0] * n + edges[1], walk_src * n + walk_dst)
    edge_mask = ~dropped
    return _take_edges(edge_index, np.nonzero(edge_mask)[0]), mx.array(edge_mask)
