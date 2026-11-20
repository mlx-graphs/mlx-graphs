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


@validate_edge_index
def negative_sampling(
    edge_index: mx.array,
    num_nodes: Optional[int] = None,
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
            if not given.
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
    src, dst, n = _host_edges(edge_index, num_nodes)

    if num_neg_samples is None:
        num_neg = src.size
    elif isinstance(num_neg_samples, float):
        num_neg = int(num_neg_samples * src.size)
    else:
        num_neg = int(num_neg_samples)

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
        return _empty_like(edge_index)

    ranks = _host_rng().choice(population - present.size, size=num_neg, replace=False)
    ids = _kth_missing(present, ranks.astype(np.int64))

    if force_undirected:
        low = np.searchsorted(row_start, ids, side="right") - 1
        high = ids - row_start[low] + low + 1
        negatives = np.stack([np.concatenate([low, high]), np.concatenate([high, low])])
    else:
        neg_src = ids // (n - 1)
        neg_dst = ids % (n - 1)
        neg_dst += neg_dst >= neg_src
        negatives = np.stack([neg_src, neg_dst])
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
