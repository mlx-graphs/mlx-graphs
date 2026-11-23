from typing import Iterator, Literal, Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.transformations import _to_numpy

# Maximum number of entries of each block of the distance matrix.
_BLOCK_SIZE = 1 << 24


def _batch_vector(
    batch_indices: Optional[mx.array], num_points: int, name: str
) -> np.ndarray:
    if batch_indices is None:
        return np.zeros(num_points, dtype=np.int64)
    batch = _to_numpy(batch_indices).reshape(-1).astype(np.int64)
    if batch.size != num_points:
        raise ValueError(
            f"{name} must have one entry per point (got {batch.size} entries "
            f"for {num_points} points)"
        )
    return batch


def _dense_index(batch: np.ndarray, num_graphs: int) -> np.ndarray:
    """``[num_graphs, max_graph_size]`` array with the points of each graph
    (in increasing order), padded with ``-1``."""
    counts = np.bincount(batch, minlength=num_graphs)[:num_graphs]
    order = np.argsort(batch, kind="stable")
    order = order[batch[order] < num_graphs]
    starts = np.cumsum(counts) - counts
    slot = np.arange(order.size) - starts[batch[order]]
    dense = np.full((num_graphs, int(counts.max(initial=0))), -1, dtype=np.int64)
    dense[batch[order], slot] = order
    return dense


def _neighbor_blocks(
    x: mx.array,
    y: mx.array,
    batch_x: Optional[mx.array],
    batch_y: Optional[mx.array],
    num_graphs: Optional[int],
    num_neighbors: Optional[int],
    exclude_self: bool,
    cosine: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yields blocks ``(y_index, x_index, distance)`` of candidate neighbors.

    For each point of ``y``, the candidates are the points of ``x`` in the same
    graph, sorted by increasing distance (squared euclidean distance, or cosine
    distance), and restricted to the ``num_neighbors`` closest ones if given.
    Missing candidates (padding, or the point itself when ``exclude_self``) have
    an infinite distance. Distances are computed on the device, graph by graph
    and in blocks of bounded size.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x and y must have the same number of features ({x.shape[1]} != "
            f"{y.shape[1]})"
        )
    bx = _batch_vector(batch_x, x.shape[0], "batch_x")
    by = _batch_vector(batch_y, y.shape[0], "batch_y")
    if num_graphs is None:
        num_graphs = int(max(bx.max(initial=-1), by.max(initial=-1))) + 1
    if x.shape[0] == 0 or y.shape[0] == 0 or num_graphs == 0:
        return

    x = x.astype(mx.float32) if not mx.issubdtype(x.dtype, mx.floating) else x
    y = y.astype(mx.float32) if not mx.issubdtype(y.dtype, mx.floating) else y
    if cosine:
        x = x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
        y = y / mx.maximum(mx.linalg.norm(y, axis=-1, keepdims=True), 1e-12)

    x_dense = _dense_index(bx, num_graphs)
    y_dense = _dense_index(by, num_graphs)
    n_max, m_max = x_dense.shape[1], y_dense.shape[1]
    if n_max == 0 or m_max == 0:
        return
    keep = n_max if num_neighbors is None else min(num_neighbors, n_max)

    # Distances are computed from the differences of the points (not with a
    # matmul) to stay exact for points far from the origin: each block holds
    # at most `_BLOCK_SIZE` entries of the [B, M, N, F] differences.
    block_size = max(1, _BLOCK_SIZE // x.shape[1])
    graphs_per_block = max(1, block_size // (m_max * n_max))
    for g0 in range(0, num_graphs, graphs_per_block):
        x_ids = x_dense[g0 : g0 + graphs_per_block]  # [B, N]
        xb = x[mx.array(np.maximum(x_ids, 0))]  # [B, N, F]
        x_pad = mx.array(x_ids < 0)

        rows_per_block = max(1, block_size // (x_ids.shape[0] * n_max))
        for s0 in range(0, m_max, rows_per_block):
            y_ids = y_dense[g0 : g0 + graphs_per_block, s0 : s0 + rows_per_block]
            yb = y[mx.array(np.maximum(y_ids, 0))]  # [B, M, F]

            if cosine:
                dist = 1.0 - (yb[:, :, None, :] * xb[:, None, :, :]).sum(axis=-1)
            else:
                diff = yb[:, :, None, :] - xb[:, None, :, :]  # [B, M, N, F]
                dist = (diff * diff).sum(axis=-1)  # [B, M, N]
            invalid = x_pad[:, None, :]
            if exclude_self:
                invalid = invalid | mx.array(y_ids[:, :, None] == x_ids[:, None, :])
            dist = mx.where(invalid, float("inf"), dist)

            if keep < n_max:
                order = mx.argsort(dist, axis=-1)[..., :keep]
                dist = mx.take_along_axis(dist, order, axis=-1)
                cols = np.asarray(order).astype(np.int64)
            else:
                cols = np.broadcast_to(np.arange(n_max), dist.shape)
            dist_np = np.asarray(dist)

            neighbor_ids = np.take_along_axis(x_ids[:, None, :], cols, axis=-1)
            query_ids = np.broadcast_to(y_ids[:, :, None], cols.shape)
            valid = (query_ids >= 0) & np.isfinite(dist_np)
            yield query_ids[valid], neighbor_ids[valid], dist_np[valid]


def knn(
    x: mx.array,
    y: mx.array,
    k: int,
    batch_x: Optional[mx.array] = None,
    batch_y: Optional[mx.array] = None,
    cosine: bool = False,
    num_graphs: Optional[int] = None,
) -> mx.array:
    """Finds for each point of ``y`` the ``k`` nearest points of ``x``.

    Only points of the same graph are matched when batch vectors are given.
    Graphs with fewer than ``k`` points of ``x`` give fewer neighbors.

    Args:
        x: Points to search in, of shape ``[N, num_features]``.
        y: Query points, of shape ``[M, num_features]``.
        k: The number of neighbors.
        batch_x: Graph of each point of ``x``, of shape ``[N]``. Default:
            ``None`` (a single graph).
        batch_y: Graph of each point of ``y``, of shape ``[M]``. Default:
            ``None`` (a single graph).
        cosine: If ``True``, uses the cosine distance instead of the euclidean
            distance. Default: ``False``
        num_graphs: Number of graphs. Default: ``None`` (inferred from the
            batch vectors).

    Returns:
        An array of shape ``[2, num_pairs]`` whose first row holds the index of
        each query point of ``y`` and the second row the index of its neighbor
        in ``x``. Pairs are sorted by query, then by increasing distance.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.utils import knn

        x = mx.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        y = mx.array([[-1.0, 0.0], [1.0, 0.0]])
        assign_index = knn(x, y, 2)
        # [[0, 0, 1, 1], [0, 1, 2, 3]] (up to the order of equal distances)
    """
    rows = list(_neighbor_blocks(x, y, batch_x, batch_y, num_graphs, k, False, cosine))
    return _sorted_pairs(rows, by_distance=True)


def _sorted_pairs(rows: list, by_distance: bool) -> mx.array:
    """``[2, num_pairs]`` array of ``(query, neighbor)`` pairs, sorted by query
    then by distance (or by neighbor index)."""
    if not rows:
        return mx.zeros((2, 0), dtype=mx.int32)
    query = np.concatenate([r[0] for r in rows])
    neighbor = np.concatenate([r[1] for r in rows])
    if by_distance:
        distance = np.concatenate([r[2] for r in rows])
        order = np.lexsort((neighbor, distance, query))
    else:
        # one integer key sorts ~10x faster than a lexsort
        key = query.astype(np.int64) * (int(neighbor.max()) + 1) + neighbor
        order = np.argsort(key, kind="stable")
    return mx.array(np.stack([query[order], neighbor[order]]).astype(np.int32))


def knn_graph(
    x: mx.array,
    k: int,
    batch_indices: Optional[mx.array] = None,
    loop: bool = False,
    flow: Literal["source_to_target", "target_to_source"] = "source_to_target",
    cosine: bool = False,
    num_graphs: Optional[int] = None,
) -> mx.array:
    """Computes the graph connecting each point to its ``k`` nearest points.

    Args:
        x: Point coordinates (or features), of shape ``[N, num_features]``.
        k: The number of neighbors of each point.
        batch_indices: Graph of each point, of shape ``[N]``. Only points of the
            same graph are connected. Default: ``None`` (a single graph).
        loop: If ``True``, each point is one of its own neighbors (with a
            self-loop). Default: ``False``
        flow: Direction of the edges, for use with message passing. With
            ``"source_to_target"``, the edges go from the neighbors to the
            center point, so that each point aggregates its ``k`` neighbors.
            Default: ``"source_to_target"``
        cosine: If ``True``, uses the cosine distance instead of the euclidean
            distance. Default: ``False``
        num_graphs: Number of graphs. Default: ``None`` (inferred from
            ``batch_indices``).

    Returns:
        The ``edge_index`` of shape ``[2, num_edges]``. The edges are grouped
        by center point, closest neighbors first.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.utils import knn_graph

        x = mx.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        batch_indices = mx.array([0, 0, 0, 0])
        edge_index = knn_graph(x, k=2, batch_indices=batch_indices)
        >>> edge_index.shape
        (2, 8)
    """
    _check_flow(flow)
    rows = list(
        _neighbor_blocks(
            x, x, batch_indices, batch_indices, num_graphs, k, not loop, cosine
        )
    )
    edge_index = _sorted_pairs(rows, by_distance=True)
    return edge_index[::-1] if flow == "source_to_target" else edge_index


def _check_flow(flow: str) -> None:
    if flow not in ("source_to_target", "target_to_source"):
        raise ValueError(
            f"flow must be 'source_to_target' or 'target_to_source', got {flow!r}"
        )


def _within_radius(rows: list, r: float) -> list:
    """Keeps the candidates at a distance smaller than ``r`` (the distances of
    the candidates are squared euclidean distances)."""
    return [(q[d < r * r], n[d < r * r], d[d < r * r]) for q, n, d in rows]


def radius(
    x: mx.array,
    y: mx.array,
    r: float,
    batch_x: Optional[mx.array] = None,
    batch_y: Optional[mx.array] = None,
    max_num_neighbors: Optional[int] = 32,
    num_graphs: Optional[int] = None,
) -> mx.array:
    """Finds for each point of ``y`` the points of ``x`` within distance ``r``.

    Only points of the same graph are matched when batch vectors are given.

    Args:
        x: Points to search in, of shape ``[N, num_features]``.
        y: Query points, of shape ``[M, num_features]``.
        r: The radius. Points at a distance strictly smaller than ``r`` are
            matched.
        batch_x: Graph of each point of ``x``, of shape ``[N]``. Default:
            ``None`` (a single graph).
        batch_y: Graph of each point of ``y``, of shape ``[M]``. Default:
            ``None`` (a single graph).
        max_num_neighbors: The maximum number of neighbors returned for each
            query point; the closest ones are kept. ``None`` for no limit.
            Default: ``32``
        num_graphs: Number of graphs. Default: ``None`` (inferred from the
            batch vectors).

    Returns:
        An array of shape ``[2, num_pairs]`` whose first row holds the index of
        each query point of ``y`` and the second row the index of its neighbor
        in ``x``. Pairs are sorted by query, then by neighbor index.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.utils import radius

        x = mx.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        y = mx.array([[-1.0, 0.0], [1.0, 0.0]])
        assign_index = radius(x, y, 1.5)
        >>> assign_index
        array([[0, 0, 1, 1],
               [0, 1, 2, 3]], dtype=int32)
    """
    rows = _neighbor_blocks(
        x, y, batch_x, batch_y, num_graphs, max_num_neighbors, False
    )
    return _sorted_pairs(_within_radius(list(rows), r), by_distance=False)


def radius_graph(
    x: mx.array,
    r: float,
    batch_indices: Optional[mx.array] = None,
    loop: bool = False,
    max_num_neighbors: Optional[int] = 32,
    flow: Literal["source_to_target", "target_to_source"] = "source_to_target",
    num_graphs: Optional[int] = None,
) -> mx.array:
    """Computes the graph connecting each point to the points within distance
    ``r``.

    Args:
        x: Point coordinates (or features), of shape ``[N, num_features]``.
        r: The radius. Points at a distance strictly smaller than ``r`` are
            connected.
        batch_indices: Graph of each point, of shape ``[N]``. Only points of the
            same graph are connected. Default: ``None`` (a single graph).
        loop: If ``True``, the graph contains a self-loop for each point.
            Default: ``False``
        max_num_neighbors: The maximum number of neighbors of each point; the
            closest ones are kept. ``None`` for no limit. Default: ``32``
        flow: Direction of the edges, for use with message passing. With
            ``"source_to_target"``, the edges go from the neighbors to the
            center point. Default: ``"source_to_target"``
        num_graphs: Number of graphs. Default: ``None`` (inferred from
            ``batch_indices``).

    Returns:
        The ``edge_index`` of shape ``[2, num_edges]``. The edges are grouped by
        center point, by increasing neighbor index.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.utils import radius_graph

        x = mx.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        edge_index = radius_graph(x, r=2.5)
        >>> edge_index
        array([[1, 2, 0, 3, 0, 3, 1, 2],
               [0, 0, 1, 1, 2, 2, 3, 3]], dtype=int32)
    """
    _check_flow(flow)
    rows = _neighbor_blocks(
        x, x, batch_indices, batch_indices, num_graphs, max_num_neighbors, not loop
    )
    edge_index = _sorted_pairs(_within_radius(list(rows), r), by_distance=False)
    return edge_index[::-1] if flow == "source_to_target" else edge_index
