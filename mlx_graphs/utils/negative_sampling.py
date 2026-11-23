from typing import Optional, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.transformations import (
    _empty_like,
    _infer_num_nodes,
    _to_numpy,
)
from mlx_graphs.utils.validators import validate_edge_index


def _host_rng() -> np.random.Generator:
    """NumPy generator seeded from MLX's global PRNG.

    Drawing the seed from ``mx.random`` makes everything sampled from the
    returned generator reproducible with ``mx.random.seed``.
    """
    seed = int(mx.random.randint(0, 2**31 - 1, (1,)).item())
    return np.random.default_rng(seed)


def _kth_missing(present: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Maps each rank ``r`` to the ``r``-th (0-based) integer not in ``present``.

    ``present`` must be sorted and without duplicates. ``present[k] - k`` is the
    number of missing integers below ``present[k]``, so the ``r``-th missing
    integer is ``r`` plus the number of entries with ``present[k] - k <= r``.
    """
    missing_before = present - np.arange(present.size, dtype=np.int64)
    return ranks + np.searchsorted(missing_before, ranks, side="right")


def _host_edges(edge_index: mx.array, num_nodes: Optional[int]) -> tuple:
    edges = _to_numpy(edge_index).astype(np.int64)
    n = _infer_num_nodes(edge_index, num_nodes)
    if edges.size > 0 and int(edges.max()) >= n:
        raise ValueError(
            f"num_nodes={n} but `edge_index` contains node {int(edges.max())}"
        )
    return edges[0], edges[1], n


def _num_negatives(num_neg_samples: Optional[Union[int, float]], num_edges: int) -> int:
    if num_neg_samples is None:
        return num_edges
    if isinstance(num_neg_samples, float):
        return int(num_neg_samples * num_edges)
    return int(num_neg_samples)


def _bipartite_negatives(
    edges: np.ndarray,
    size: tuple[int, int],
    num_neg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``[2, k]`` host negatives between ``size[0]`` source and ``size[1]``
    destination nodes, exact and without replacement."""
    num_src, num_dst = size
    population = num_src * num_dst
    present = np.unique(edges[0] * num_dst + edges[1])
    num_neg = min(num_neg, population - present.size)
    if num_neg <= 0:
        return np.zeros((2, 0), dtype=np.int64)
    ranks = rng.choice(population - present.size, size=num_neg, replace=False)
    ids = _kth_missing(present, ranks.astype(np.int64))
    return np.stack([ids // num_dst, ids % num_dst])


def _bipartite_negative_sampling(
    edge_index: mx.array,
    size: tuple[int, int],
    num_neg_samples: Optional[Union[int, float]],
) -> mx.array:
    """Negative sampling between ``size[0]`` source and ``size[1]``
    destination nodes, exact and without replacement."""
    num_src, num_dst = size
    edges = _to_numpy(edge_index).astype(np.int64).reshape(2, -1)
    if edges.size > 0 and (
        int(edges[0].max()) >= num_src or int(edges[1].max()) >= num_dst
    ):
        raise ValueError(f"`edge_index` contains nodes out of num_nodes={size}")
    num_neg = _num_negatives(num_neg_samples, edges.shape[1])
    if min(num_neg, num_src * num_dst) <= 0:
        return _empty_like(edge_index)
    negatives = _bipartite_negatives(edges, size, num_neg, _host_rng())
    if negatives.shape[1] == 0:
        return _empty_like(edge_index)
    return mx.array(negatives, dtype=edge_index.dtype)


def _negatives(
    src: np.ndarray,
    dst: np.ndarray,
    n: int,
    num_neg: int,
    force_undirected: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """``[2, k]`` host negatives of a graph of ``n`` nodes (see
    :func:`negative_sampling`), exact and without replacement."""
    not_loop = src != dst
    src, dst = src[not_loop], dst[not_loop]

    # Every candidate pair gets an id in [0, population). Self-loops have no id.
    if force_undirected:
        # Pairs (i, j) with i < j, row by row.
        num_neg = num_neg // 2
        population = n * (n - 1) // 2
        nodes = np.arange(n, dtype=np.int64)
        row_start = nodes * (2 * n - nodes - 1) // 2
        low, high = np.minimum(src, dst), np.maximum(src, dst)
        present = row_start[low] + high - low - 1
    else:
        # Pairs (i, j) with i != j: the diagonal is skipped in each row.
        population = n * (n - 1)
        present = src * (n - 1) + dst - (dst > src)
    present = np.unique(present)

    num_neg = min(num_neg, population - present.size)
    if num_neg <= 0:
        return np.zeros((2, 0), dtype=np.int64)

    ranks = rng.choice(population - present.size, size=num_neg, replace=False)
    ids = _kth_missing(present, ranks.astype(np.int64))

    if force_undirected:
        low = np.searchsorted(row_start, ids, side="right") - 1
        high = ids - row_start[low] + low + 1
        return np.stack([np.concatenate([low, high]), np.concatenate([high, low])])
    neg_src = ids // (n - 1)
    neg_dst = ids % (n - 1)
    neg_dst += neg_dst >= neg_src
    return np.stack([neg_src, neg_dst])


@validate_edge_index
def negative_sampling(
    edge_index: mx.array,
    num_nodes: Optional[Union[int, tuple[int, int]]] = None,
    num_neg_samples: Optional[Union[int, float]] = None,
    force_undirected: bool = False,
) -> mx.array:
    """
    Samples random negative edges, i.e. pairs of nodes ``(src, dst)`` with
    ``src != dst`` that are not connected by an edge of ``edge_index``.

    The negative edges are drawn uniformly without replacement among all the
    non-edges of the graph, so they are all different. The sampling is exact
    (no rejection loop): exactly ``num_neg_samples`` edges are returned, or
    all the non-edges of the graph if there are fewer than that (e.g. on
    dense graphs).

    Randomness comes from a NumPy generator seeded from ``mx.random``, so
    results are reproducible with ``mx.random.seed``.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        num_nodes: Number of nodes in the graph. Inferred from ``edge_index``
            if not given. For a bipartite graph, a tuple ``(num_src_nodes,
            num_dst_nodes)``: the negatives are then pairs of a source and a
            destination node (``src == dst`` is allowed) and
            ``force_undirected`` is ignored.
        num_neg_samples: Number of negative edges to sample. A float is
            interpreted as a ratio of the number of edges in ``edge_index``.
            Default: ``None`` (one negative edge per edge of ``edge_index``).
        force_undirected: If ``True``, the negative edges are undirected: both
            ``(src, dst)`` and ``(dst, src)`` are returned and neither
            direction is in ``edge_index``. ``num_neg_samples`` then counts
            both directions. Default: ``False``

    Returns:
        A ``[2, num_neg]`` array of negative edges, with the dtype of
        ``edge_index``.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 0, 1, 2], [0, 1, 2, 3]])

        negative_sampling(edge_index)
        # e.g. mx.array([[3, 1, 2, 0], [1, 0, 0, 3]])

        negative_sampling(edge_index, num_neg_samples=2, force_undirected=True)
        # e.g. mx.array([[0, 2], [2, 0]])
    """
    if isinstance(num_nodes, (tuple, list)):
        return _bipartite_negative_sampling(
            edge_index, (int(num_nodes[0]), int(num_nodes[1])), num_neg_samples
        )
    src, dst, n = _host_edges(edge_index, num_nodes)
    num_neg = _num_negatives(num_neg_samples, src.size)
    if num_neg <= 0 or n < 2:
        return _empty_like(edge_index)
    negatives = _negatives(src, dst, n, num_neg, force_undirected, _host_rng())
    if negatives.shape[1] == 0:
        return _empty_like(edge_index)
    return mx.array(negatives, dtype=edge_index.dtype)


@validate_edge_index
def structured_negative_sampling(
    edge_index: mx.array,
    num_nodes: Optional[int] = None,
    contains_neg_self_loops: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Samples one negative destination for every edge of ``edge_index``: for
    each edge ``(src, dst_pos)``, a node ``dst_neg`` such that
    ``(src, dst_neg)`` is not an edge.

    Each ``dst_neg`` is drawn uniformly among the nodes that are not
    neighbors of ``src``. The sampling is exact (no rejection loop), so it
    also terminates on dense graphs: an edge whose source is connected to
    every node has no valid negative and is left out of the result. The three
    returned arrays are therefore always aligned with each other, and they
    are aligned with ``edge_index`` whenever every source has at least one
    non-neighbor.

    Randomness comes from a NumPy generator seeded from ``mx.random``, so
    results are reproducible with ``mx.random.seed``.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        num_nodes: Number of nodes in the graph. Inferred from ``edge_index``
            if not given.
        contains_neg_self_loops: If ``False``, ``dst_neg`` is never equal to
            ``src``. Default: ``True`` (a self-loop that is not in
            ``edge_index`` is a valid negative, as in PyG).

    Returns:
        A tuple ``(src, dst_pos, dst_neg)`` of arrays of the same length, with
        the dtype of ``edge_index``.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 0, 1, 2], [0, 1, 2, 3]])

        src, dst_pos, dst_neg = structured_negative_sampling(edge_index)
        # src: mx.array([0, 0, 1, 2])
        # dst_pos: mx.array([0, 1, 2, 3])
        # dst_neg: e.g. mx.array([3, 2, 0, 1])
    """
    src, dst, n = _host_edges(edge_index, num_nodes)
    dtype = edge_index.dtype
    if src.size == 0:
        empty = mx.zeros((0,), dtype=dtype)
        return empty, empty, empty

    # Forbidden destinations of each source, as sorted unique keys.
    forbidden_src, forbidden_dst = src, dst
    if not contains_neg_self_loops:
        nodes = np.arange(n, dtype=np.int64)
        forbidden_src = np.concatenate([src, nodes])
        forbidden_dst = np.concatenate([dst, nodes])
    keys = np.unique(forbidden_src * n + forbidden_dst)
    key_src, key_dst = keys // n, keys % n

    counts = np.bincount(key_src, minlength=n)
    row_ptr = np.concatenate([[0], np.cumsum(counts)])
    # Number of allowed destinations of the row below each forbidden one (see
    # `_kth_missing`), made globally sorted by adding the row.
    rank_in_row = np.arange(keys.size, dtype=np.int64) - row_ptr[key_src]
    allowed_before = key_src * (n + 1) + key_dst - rank_in_row

    num_allowed = n - counts[src]
    feasible = np.nonzero(num_allowed > 0)[0]
    if feasible.size == 0:
        empty = mx.zeros((0,), dtype=dtype)
        return empty, empty, empty

    feasible_src = src[feasible]
    ranks = _host_rng().integers(0, num_allowed[feasible])
    forbidden_before = (
        np.searchsorted(allowed_before, feasible_src * (n + 1) + ranks, side="right")
        - row_ptr[feasible_src]
    )
    dst_neg = mx.array(ranks + forbidden_before, dtype=dtype)

    if feasible.size == src.size:
        return edge_index[0], edge_index[1], dst_neg
    feasible_mx = mx.array(feasible)
    return edge_index[0][feasible_mx], edge_index[1][feasible_mx], dst_neg


@validate_edge_index
def batched_negative_sampling(
    edge_index: mx.array,
    batch_indices: Union[mx.array, tuple[mx.array, mx.array]],
    num_neg_samples: Optional[Union[int, float]] = None,
    force_undirected: bool = False,
) -> mx.array:
    """
    Samples random negative edges within each graph of a batch of graphs (see
    :func:`negative_sampling`): no negative edge connects two different
    graphs.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge of the batch
        batch_indices: The graph of each node, of shape ``[num_nodes]``, with
            the nodes of each graph stored contiguously (as in
            :class:`~mlx_graphs.data.GraphDataBatch`). For bipartite graphs, a
            tuple with the graph of each source node and of each destination
            node.
        num_neg_samples: Number of negative edges to sample per graph. A float
            is interpreted as a ratio of the number of edges of each graph.
            Default: ``None`` (one negative edge per edge of each graph).
        force_undirected: If ``True``, the negative edges are undirected (see
            :func:`negative_sampling`). Ignored for bipartite graphs.
            Default: ``False``

    Returns:
        A ``[2, num_neg]`` array of negative edges, grouped by graph, with the
        dtype of ``edge_index``.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]])
        batch_indices = mx.array([0, 0, 0, 1, 1, 1])

        batched_negative_sampling(edge_index, batch_indices)
        # e.g. mx.array([[1, 2, 0, 5, 3, 4], [0, 1, 2, 4, 5, 3]])
    """
    bipartite = isinstance(batch_indices, (tuple, list))
    if bipartite:
        src_batch = _to_numpy(batch_indices[0]).astype(np.int64).reshape(-1)
        dst_batch = _to_numpy(batch_indices[1]).astype(np.int64).reshape(-1)
    else:
        src_batch = dst_batch = _to_numpy(batch_indices).astype(np.int64).reshape(-1)

    num_graphs = int(max(src_batch.max(initial=-1), dst_batch.max(initial=-1))) + 1
    num_src = np.bincount(src_batch, minlength=num_graphs)
    num_dst = np.bincount(dst_batch, minlength=num_graphs)
    src_ptr = np.concatenate([[0], np.cumsum(num_src)[:-1]])
    dst_ptr = np.concatenate([[0], np.cumsum(num_dst)[:-1]])

    edges = _to_numpy(edge_index).astype(np.int64).reshape(2, -1)
    edge_batch = src_batch[edges[0]]
    # Edges grouped by graph; one host generator for all the graphs (a
    # single draw from mx.random, instead of one GPU sync per graph).
    order = np.argsort(edge_batch, kind="stable")
    edges, edge_batch = edges[:, order], edge_batch[order]
    edge_ptr = np.searchsorted(edge_batch, np.arange(num_graphs + 1))
    rng = _host_rng() if num_graphs > 0 else None
    negatives = []
    for graph in range(num_graphs):
        local = edges[:, edge_ptr[graph] : edge_ptr[graph + 1]]
        local = np.stack([local[0] - src_ptr[graph], local[1] - dst_ptr[graph]])
        num_neg = _num_negatives(num_neg_samples, local.shape[1])
        if num_neg <= 0:
            continue
        if bipartite:
            size = (int(num_src[graph]), int(num_dst[graph]))
            neg = _bipartite_negatives(local, size, num_neg, rng)
        else:
            n = int(num_src[graph])
            if n < 2:
                continue
            neg = _negatives(local[0], local[1], n, num_neg, force_undirected, rng)
        negatives.append(np.stack([neg[0] + src_ptr[graph], neg[1] + dst_ptr[graph]]))
    if not negatives:
        return _empty_like(edge_index)
    return mx.array(np.concatenate(negatives, axis=1), dtype=edge_index.dtype)
