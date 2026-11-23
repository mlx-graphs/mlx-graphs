"""Sampling and clustering of point clouds and graphs: farthest point sampling,
voxel grids, nearest-point assignment and Graclus matching."""

from functools import lru_cache
from typing import List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.spatial import _batch_vector, _neighbor_blocks
from mlx_graphs.utils.transformations import _to_numpy

# Largest threadgroup used by the farthest point sampling kernel.
_FPS_MAX_THREADS = 1024

_FPS_SOURCE = """
    const uint graph = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint threads = threads_per_threadgroup.x;
    const int src_start = ptr[graph];
    const int num_points = ptr[graph + 1] - src_start;
    const int out_start = out_ptr[graph];
    const int out_end = out_ptr[graph + 1];

    threadgroup float best_values[32];
    threadgroup int best_indices[32];
    threadgroup int selected;

    if (num_points <= 0 || out_end <= out_start) {
        return;
    }
    int current = start[graph];
    if (tid == 0) {
        out[out_start] = src_start + current;
    }
    for (int k = out_start + 1; k < out_end; ++k) {
        // Distances to the last selected point, and the farthest point of
        // this thread (the first one on ties, as torch.argmax).
        const ulong center = ulong(src_start + current) * DIM;
        float best = -1.0f;
        int best_index = INT_MAX;
        for (int i = int(tid); i < num_points; i += int(threads)) {
            const ulong point = ulong(src_start + i) * DIM;
            float d = 0.0f;
            for (int c = 0; c < DIM; ++c) {
                const float diff = pos[point + c] - pos[center + c];
                d += diff * diff;
            }
            if (k > out_start + 1) {
                d = min(dist[src_start + i], d);
            }
            dist[src_start + i] = d;
            if (d > best) {
                best = d;
                best_index = i;
            }
        }
        for (ushort offset = 16; offset > 0; offset /= 2) {
            const float other = simd_shuffle_xor(best, offset);
            const int other_index = simd_shuffle_xor(best_index, offset);
            if (other > best || (other == best && other_index < best_index)) {
                best = other;
                best_index = other_index;
            }
        }
        if (lane == 0) {
            best_values[simd] = best;
            best_indices[simd] = best_index;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            const uint num_simds = threads / 32;
            best = lane < num_simds ? best_values[lane] : -1.0f;
            best_index = lane < num_simds ? best_indices[lane] : INT_MAX;
            for (ushort offset = 16; offset > 0; offset /= 2) {
                const float other = simd_shuffle_xor(best, offset);
                const int other_index = simd_shuffle_xor(best_index, offset);
                if (other > best || (other == best && other_index < best_index)) {
                    best = other;
                    best_index = other_index;
                }
            }
            if (lane == 0) {
                selected = best_index;
                out[k] = src_start + best_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        current = selected;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


@lru_cache(maxsize=1)
def _fps_kernel():
    return mx.fast.metal_kernel(
        name="farthest_point_sampling",
        input_names=["pos", "ptr", "out_ptr", "start"],
        output_names=["out", "dist"],
        source=_FPS_SOURCE,
        ensure_row_contiguous=True,
    )


def _use_metal() -> bool:
    return mx.metal.is_available() and mx.default_device() == mx.gpu


def _fps_reference(
    pos: mx.array, ptr: np.ndarray, counts: np.ndarray, start: np.ndarray
) -> mx.array:
    """Farthest point sampling with array operations, all graphs at once
    (padded to the size of the largest graph)."""
    num_graphs = ptr.size - 1
    sizes = np.diff(ptr)
    max_size = int(sizes.max())
    slots = np.arange(max_size)
    # [B, max_size] ids of the points of each graph (padding repeats the
    # first point and is excluded with an infinite negative distance)
    dense = ptr[:-1, None] + np.minimum(
        slots[None, :], np.maximum(sizes - 1, 0)[:, None]
    )
    dense = np.minimum(dense, pos.shape[0] - 1)  # empty graphs at the end
    valid = mx.array(slots[None, :] < sizes[:, None])
    points = pos[mx.array(dense)]  # [B, max_size, D]
    rows = mx.arange(num_graphs)
    current = mx.array(start.astype(np.int32))
    selected = [current]
    dist = None
    for step in range(1, int(counts.max(initial=0))):
        center = points[rows, current][:, None, :]
        d = ((points - center) ** 2).sum(axis=-1)
        dist = d if dist is None else mx.minimum(dist, d)
        dist = mx.where(valid, dist, -1.0)
        current = mx.argmax(dist, axis=1).astype(mx.int32)
        selected.append(current)
        if step % 64 == 0:
            mx.eval(dist, current)
    picked = np.stack([np.array(s) for s in selected], axis=1)  # [B, max_count]
    out = [ptr[b] + picked[b, : counts[b]] for b in range(num_graphs)]
    return mx.array(np.concatenate(out).astype(np.int32))


def fps(
    x: mx.array,
    batch: Optional[mx.array] = None,
    ratio: float = 0.5,
    random_start: bool = True,
    batch_size: Optional[int] = None,
) -> mx.array:
    r"""Farthest point sampling, from the `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper.

    Iteratively samples the point that is the most distant from the points
    already sampled. Each graph (point cloud) of the batch keeps
    :math:`\lceil \text{ratio} \cdot N_b \rceil` of its :math:`N_b` points.

    On the GPU, every graph is sampled by one threadgroup of a Metal kernel;
    otherwise the graphs are sampled together with array operations.

    Args:
        x: Point coordinates (or features), of shape ``[N, num_features]``.
        batch: Graph of each point, of shape ``[N]``, sorted in increasing
            order (as in a ``GraphDataBatch``). Default: ``None`` (a single
            graph).
        ratio: Sampling ratio, in :math:`(0, 1]`. Default: ``0.5``
        random_start: If ``False``, the first point of each graph is the
            starting point, as in PyG. Otherwise, a random point (from
            ``mx.random``). Default: ``True``
        batch_size: The number of graphs. Default: ``None`` (inferred from
            ``batch``).

    Returns:
        The indices of the sampled points, of shape ``[num_samples]``, grouped
        by graph and in sampling order.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.utils import fps

        x = mx.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        batch = mx.array([0, 0, 0, 0])
        index = fps(x, batch, ratio=0.5, random_start=False)
        >>> index
        array([0, 3], dtype=int32)
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    x = x.reshape(x.shape[0], -1) if x.ndim != 2 else x
    num_points = x.shape[0]
    batch_np = _batch_vector(batch, num_points, "batch")
    if batch_np.size > 1 and np.any(np.diff(batch_np) < 0):
        raise ValueError("fps expects `batch` to be sorted in increasing order")
    if batch_size is None:
        batch_size = int(batch_np.max(initial=-1)) + 1 if batch is not None else 1
    sizes = np.bincount(batch_np, minlength=batch_size)[:batch_size]
    ptr = np.zeros(batch_size + 1, dtype=np.int64)
    np.cumsum(sizes, out=ptr[1:])
    # same float32 arithmetic as PyG for the number of samples
    counts = np.ceil(sizes.astype(np.float32) * np.float32(ratio)).astype(np.int64)
    out_ptr = np.zeros(batch_size + 1, dtype=np.int64)
    np.cumsum(counts, out=out_ptr[1:])
    if out_ptr[-1] == 0:
        return mx.zeros((0,), dtype=mx.int32)

    if random_start:
        start = np.floor(np.array(mx.random.uniform(shape=(batch_size,))) * sizes)
        start = np.minimum(start.astype(np.int64), np.maximum(sizes - 1, 0))
    else:
        start = np.zeros(batch_size, dtype=np.int64)

    pos = x if mx.issubdtype(x.dtype, mx.floating) else x.astype(mx.float32)
    if _use_metal() and num_points < 2**31:
        pos = pos.astype(mx.float32)
        threads = min(_FPS_MAX_THREADS, max(32, 1 << int(sizes.max() - 1).bit_length()))
        out, _ = _fps_kernel()(
            inputs=[
                mx.stop_gradient(pos),
                mx.array(ptr.astype(np.int32)),
                mx.array(out_ptr.astype(np.int32)),
                mx.array(start.astype(np.int32)),
            ],
            template=[("DIM", pos.shape[1])],
            grid=(threads * batch_size, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(int(out_ptr[-1]),), (num_points,)],
            output_dtypes=[mx.int32, mx.float32],
        )
        return out
    return _fps_reference(mx.stop_gradient(pos), ptr, counts, start)


def nearest(
    x: mx.array,
    y: mx.array,
    batch_x: Optional[mx.array] = None,
    batch_y: Optional[mx.array] = None,
) -> mx.array:
    r"""Assigns each point of ``x`` to its nearest point of ``y``, in the same
    graph.

    Args:
        x: Points to assign, of shape ``[N, num_features]``.
        y: Cluster centers, of shape ``[M, num_features]``.
        batch_x: Graph of each point of ``x``, of shape ``[N]``. Default:
            ``None`` (a single graph).
        batch_y: Graph of each point of ``y``, of shape ``[M]``. Default:
            ``None`` (a single graph).

    Returns:
        The index in ``y`` of the nearest point of each point of ``x``, of
        shape ``[N]`` (the smallest index on ties).

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.utils import nearest

        x = mx.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        y = mx.array([[-1.0, 0.0], [1.0, 0.0]])
        >>> nearest(x, y)
        array([0, 0, 1, 1], dtype=int32)
    """
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    out = np.full(x.shape[0], -1, dtype=np.int64)
    best = np.full(x.shape[0], np.inf)
    for query, neighbor, dist in _neighbor_blocks(
        y, x, batch_y, batch_x, None, None, False
    ):
        # all the candidates of each query, in increasing order of distance
        # within the block: keep the smallest index among the closest ones
        order = np.lexsort((neighbor, dist, query))
        query, neighbor, dist = query[order], neighbor[order], dist[order]
        first = np.ones(query.size, dtype=bool)
        first[1:] = query[1:] != query[:-1]
        query, neighbor, dist = query[first], neighbor[first], dist[first]
        better = dist < best[query]
        out[query[better]] = neighbor[better]
        best[query[better]] = dist[better]
    if np.any(out < 0):
        raise ValueError(
            "Some points of `x` have no point of `y` in the same graph to be "
            "assigned to"
        )
    return mx.array(out.astype(np.int32))


def _repeat(value, dim: int, name: str) -> np.ndarray:
    value = np.array(_to_numpy(value) if isinstance(value, mx.array) else value)
    value = value.reshape(-1).astype(np.float64)
    if value.size == 1:
        value = np.repeat(value, dim)
    if value.size != dim:
        raise ValueError(f"{name} must have 1 or {dim} entries, got {value.size}")
    return value


def voxel_grid(
    pos: mx.array,
    size: Union[float, Sequence[float], mx.array],
    batch: Optional[mx.array] = None,
    start: Optional[Union[float, Sequence[float], mx.array]] = None,
    end: Optional[Union[float, Sequence[float], mx.array]] = None,
) -> mx.array:
    r"""Voxel grid pooling from the `"Dynamic Edge-Conditioned Filters in
    Convolutional Networks on Graphs" <https://arxiv.org/abs/1704.02901>`_
    paper: overlays a regular grid of user-defined size over a point cloud
    and clusters all the points within the same voxel.

    Args:
        pos: Node positions, of shape ``[N, D]``.
        size: Size of a voxel (in each dimension, or the same for all).
        batch: Graph of each node, of shape ``[N]``. Nodes of different graphs
            are never in the same voxel. Default: ``None`` (a single graph).
        start: Start coordinates of the grid (in each dimension, or the same
            for all). Default: ``None`` (the minimum of ``pos``).
        end: End coordinates of the grid. Default: ``None`` (the maximum of
            ``pos``).

    Returns:
        The voxel (cluster) id of each node, of shape ``[N]`` (``int64``). Ids
        are not consecutive, see :func:`consecutive_cluster`.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.utils import voxel_grid

        pos = mx.array([[0.0, 0.0], [0.4, 0.2], [1.5, 0.1], [0.2, 1.7]])
        >>> voxel_grid(pos, size=1.0)
        array([0, 0, 1, 2], dtype=int64)
    """
    pos = pos.reshape(-1, 1) if pos.ndim == 1 else pos
    if not mx.issubdtype(pos.dtype, mx.floating):
        pos = pos.astype(mx.float32)
    num_points, dim = pos.shape
    if batch is None:
        batch_column = mx.zeros((num_points, 1), dtype=pos.dtype)
    else:
        batch_column = batch.reshape(-1, 1).astype(pos.dtype)
    pos = mx.concatenate([pos, batch_column], axis=1)
    if num_points == 0:
        return mx.zeros((0,), dtype=mx.int64)

    size_ = np.concatenate([_repeat(size, dim, "size"), [1.0]])
    size_arr = mx.array(size_, dtype=pos.dtype)
    if start is None:
        start_arr = pos.min(axis=0)
    else:
        start_arr = mx.array(
            np.concatenate([_repeat(start, dim, "start"), [0.0]]), dtype=pos.dtype
        )
    if end is None:
        end_arr = pos.max(axis=0)
    else:
        end_arr = mx.concatenate(
            [
                mx.array(_repeat(end, dim, "end"), dtype=pos.dtype),
                batch_column.max(axis=0),
            ]
        )
    # Same operations as PyG's grid_cluster (truncation towards zero).
    pos = pos - start_arr[None, :]
    num_voxels = ((end_arr - start_arr) / size_arr).astype(mx.int64) + 1
    num_voxels = mx.cumprod(num_voxels)
    num_voxels = mx.concatenate([mx.ones((1,), dtype=mx.int64), num_voxels[:-1]])
    voxel = (pos / size_arr[None, :]).astype(mx.int64)
    return (voxel * num_voxels[None, :]).sum(axis=1)


def consecutive_cluster(cluster: mx.array) -> Tuple[mx.array, mx.array]:
    """Relabels cluster ids to consecutive integers.

    Args:
        cluster: Cluster id of each node, of shape ``[N]``.

    Returns:
        A tuple ``(cluster, perm)``: the new cluster ids in ``[0,
        num_clusters)`` (in increasing order of the original ids), and for
        each cluster the index of one of its nodes (the last one), as in PyG.
    """
    values = _to_numpy(cluster).reshape(-1)
    _, inverse = np.unique(values, return_inverse=True)
    inverse = inverse.reshape(-1)
    perm = np.empty(int(inverse.max(initial=-1)) + 1, dtype=np.int64)
    perm[inverse] = np.arange(inverse.size)
    return mx.array(inverse.astype(np.int64)), mx.array(perm)


def _graclus_round(
    row: mx.array, col: mx.array, weight: mx.array, cluster: mx.array, n: int
) -> Tuple[mx.array, bool]:
    """One round of the parallel matching: random blue nodes propose to their
    best unmatched red neighbor, which accepts its best proposal."""
    unmatched = cluster < 0
    active = unmatched[row] & unmatched[col]
    # unmatched nodes without unmatched neighbors stay alone
    has_active = mx.zeros((n,), dtype=mx.bool_).at[row].maximum(active)
    nodes = mx.arange(n, dtype=mx.int32)
    cluster = mx.where(unmatched & ~has_active, nodes, cluster)
    if not bool(mx.any(active).item()):
        return cluster, False

    blue = mx.random.uniform(shape=(n,)) < 0.5

    def best(src, dst, mask):
        # for each node `src`, its neighbor `dst` (among `mask`) with the
        # largest weight, the smallest index on ties; -1 if none
        w = mx.where(mask, weight, -np.inf)
        w_max = mx.full((n,), -np.inf).at[src].maximum(w)
        tie = mask & (w == w_max[src])
        choice = mx.full((n,), n, dtype=mx.int32).at[src].minimum(mx.where(tie, dst, n))
        return mx.where(choice < n, choice, -1)

    # proposals of blue nodes to red neighbors
    candidates = active & blue[row] & ~blue[col]
    proposal = best(row, col, candidates)
    # red nodes accept the best blue node that proposed to them
    accepted = best(col, row, candidates & (proposal[row] == col))
    matched = accepted >= 0  # red nodes that accepted a proposal
    pair = mx.minimum(nodes, accepted)
    cluster = mx.where(matched, pair, cluster)
    # the blue node accepted by each red node gets the same cluster (row `n`
    # is a dummy row for the red nodes without a partner)
    extended = mx.concatenate([cluster, mx.array([-1], dtype=cluster.dtype)])
    extended[mx.where(matched, accepted, n)] = pair
    return extended[:n], True


def graclus(
    edge_index: mx.array,
    weight: Optional[mx.array] = None,
    num_nodes: Optional[int] = None,
) -> mx.array:
    r"""A greedy clustering algorithm from the `"Weighted Graph Cuts without
    Eigenvectors: A Multilevel Approach" <http://www.cs.utexas.edu/users/
    inderjit/public_papers/multilevel_pami.pdf>`_ paper, which matches each
    node with one of its unmarked neighbors (the one maximizing the edge
    ``weight``, if given).

    The matching is computed in parallel rounds, as PyG's GPU implementation:
    unmatched nodes are randomly split into proposers and receivers, each
    proposer proposes to its best unmatched neighbor, and each receiver accepts
    its best proposal. The result is a maximal matching that depends on the
    random coloring (``mx.random``), as PyG's depends on the order in which
    the nodes are visited.

    Args:
        edge_index: The edges, of shape ``[2, num_edges]``. The graph is
            expected to be undirected.
        weight: Non-negative edge weights, of shape ``[num_edges]``. Edges with
            a negative weight are never matched, as in PyG. Default: ``None``
        num_nodes: The number of nodes. Default: ``None`` (inferred from
            ``edge_index``).

    Returns:
        The cluster of each node, of shape ``[N]`` (``int64``): a node matched
        with ``v`` has cluster ``min(u, v)``, an unmatched node ``u``.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.utils import graclus

        edge_index = mx.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        weight = mx.array([1.0, 1.0, 0.5, 0.5, 2.0, 2.0])
        >>> graclus(edge_index, weight)
        array([0, 0, 2, 2], dtype=int64)
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.size > 0 else 0
    if num_nodes == 0:
        return mx.zeros((0,), dtype=mx.int64)
    ei = _to_numpy(edge_index).astype(np.int64)
    keep = ei[0] != ei[1]
    if weight is not None:
        w = _to_numpy(weight).reshape(-1).astype(np.float32)
        keep &= w >= 0
        w = w[keep]
    else:
        w = np.ones(int(keep.sum()), dtype=np.float32)
    ei = ei[:, keep]
    row, col = mx.array(ei[0].astype(np.int32)), mx.array(ei[1].astype(np.int32))
    weight_arr = mx.array(w)
    cluster = mx.full((num_nodes,), -1, dtype=mx.int32)
    changed = True
    while changed:
        cluster, changed = _graclus_round(row, col, weight_arr, cluster, num_nodes)
        mx.eval(cluster)
    return cluster.astype(mx.int64)


__all__: List[str] = [
    "consecutive_cluster",
    "fps",
    "graclus",
    "nearest",
    "voxel_grid",
]
