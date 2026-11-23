"""Neighbor sampling and random walks, with or without ``mlx_cluster``.

Every function here has the signature and output of the ``mlx_cluster``
function of the same name. It runs that kernel when ``mlx_cluster`` is
installed (macOS) and an equivalent NumPy / MLX implementation otherwise (e.g.
on Linux, or with ``MLX_GRAPHS_DISABLE_MLX_CLUSTER=1``):

* the walks of :func:`random_walk` and :func:`metapath_random_walk` are
  identical given the same random values;
* neighbor sampling takes the same edges when it is deterministic (all
  neighbors, or a fan-out at least the degree) and otherwise samples them
  from the same distribution (uniformly, with or without replacement);
* :func:`rejection_sampling` (node2vec walks) follows the same transition
  probabilities.

Without ``mlx_cluster``, random numbers come from a NumPy generator seeded
from MLX's global RNG, so ``mx.random.seed`` makes sampling reproducible.
"""

from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.backend import HAS_MLX_CLUSTER
from mlx_graphs.backend import mlx_cluster as _mlxc

NodeType = str
EdgeType = Tuple[str, str, str]
NumNeighbors = Union[Sequence[int], Mapping[EdgeType, Sequence[int]]]

__all__ = [
    "HAS_MLX_CLUSTER",
    "HeteroSamplerOutput",
    "MetaPathCSR",
    "build_metapath_csr",
    "hetero_neighbor_sample",
    "metapath_node_types",
    "metapath_random_walk",
    "neighbor_sample",
    "random_walk",
    "rejection_sampling",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: Optional[int] = None) -> np.random.Generator:
    if seed is None:
        seed = int(mx.random.randint(0, 2**31 - 1, shape=()).item())
    return np.random.default_rng(seed)


def _index(a) -> np.ndarray:
    return np.asarray(a).astype(np.int64, copy=False).reshape(-1)


def _cat(parts: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def _ranges(starts: np.ndarray, counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Segment ids and values of the concatenated ``range(s, s + c)``."""
    seg = np.repeat(np.arange(counts.size), counts)
    first = np.cumsum(counts) - counts
    return seg, np.arange(seg.size) - first[seg] + starts[seg]


def _first_seen(values: np.ndarray) -> np.ndarray:
    """Distinct values in order of first appearance."""
    uniq, first = np.unique(values, return_index=True)
    return uniq[np.argsort(first, kind="stable")]


def _sample_edges(
    ptr: np.ndarray, nodes: np.ndarray, k: int, replace: bool, rng
) -> Tuple[np.ndarray, np.ndarray]:
    """Samples up to ``k`` edges in the CSR/CSC segment of each node.

    Same rule as ``mlx_cluster``: all edges if ``k < 0`` or (without
    replacement) ``k >= degree``, else ``k`` edges drawn uniformly with or
    without replacement.

    Returns:
        ``(pos, offsets)``: the position in ``nodes`` of each sampled edge's
        node, sorted, and the edge's offset in the CSR/CSC arrays.
    """
    empty = np.zeros(0, dtype=np.int64)
    if k == 0 or nodes.size == 0:
        return empty, empty
    start = ptr[nodes]
    deg = ptr[nodes + 1] - start
    if k < 0:
        take_all = deg > 0
    elif replace:
        take_all = np.zeros(deg.shape, dtype=bool)
    else:
        take_all = (deg > 0) & (deg <= k)

    pos_parts, off_parts = [], []
    idx = np.flatnonzero(take_all)
    if idx.size:
        seg, off = _ranges(start[idx], deg[idx])
        pos_parts.append(idx[seg])
        off_parts.append(off)

    rest = np.flatnonzero(~take_all & (deg > 0)) if k > 0 else empty
    if rest.size and replace:
        picks = rng.integers(0, deg[rest][:, None], size=(rest.size, k))
        pos_parts.append(np.repeat(rest, k))
        off_parts.append((start[rest][:, None] + picks).reshape(-1))
    elif rest.size:
        # Here k < degree. For small degrees, keep the k edges with the
        # smallest random keys in each segment.
        small = rest[deg[rest] <= 2 * k]
        if small.size:
            seg, off = _ranges(start[small], deg[small])
            order = np.lexsort((rng.random(seg.size), seg))
            rank = np.arange(seg.size) - (np.cumsum(deg[small]) - deg[small])[seg]
            keep = order[rank < k]
            pos_parts.append(small[seg[keep]])
            off_parts.append(off[keep])
        # For large degrees, draw k offsets and redraw duplicates, which are
        # rare (each draw collides with probability < 1/2). The result is a
        # uniform k-subset: the procedure is invariant under relabeling.
        large = rest[deg[rest] > 2 * k]
        if large.size:
            high = deg[large][:, None]
            picks = rng.integers(0, high, size=(large.size, k))
            while True:
                picks.sort(axis=1)
                dup = np.zeros(picks.shape, dtype=bool)
                dup[:, 1:] = picks[:, 1:] == picks[:, :-1]
                rows, cols = np.nonzero(dup)
                if rows.size == 0:
                    break
                picks[rows, cols] = rng.integers(0, high[rows, 0])
            pos_parts.append(np.repeat(large, k))
            off_parts.append((start[large][:, None] + picks).reshape(-1))

    pos, off = _cat(pos_parts), _cat(off_parts)
    order = np.argsort(pos, kind="stable")
    return pos[order], off[order]


# ---------------------------------------------------------------------------
# Homogeneous neighbor sampling
# ---------------------------------------------------------------------------


def neighbor_sample(
    colptr: mx.array,
    row: mx.array,
    input_node: mx.array,
    num_neighbors: Sequence[int],
    replace: bool = False,
    directed: bool = True,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    """Multi-hop neighbor sampling (GraphSAGE style) on a CSC graph.

    Args:
        colptr: CSC column pointers (int64, ``num_nodes + 1`` entries).
        row: CSC row indices, i.e. the source node of each edge (int64).
        input_node: Seed nodes (int64).
        num_neighbors: Fan-out per hop; ``-1`` takes all neighbors.
        replace: Sample with replacement.
        directed: If ``False``, return every edge between sampled nodes.

    Returns:
        ``(samples, rows, cols, edges)``: global ids of the sampled nodes
        (seeds first), local source and destination index of each sampled
        edge, and each edge's offset in ``row``. All int64.
    """
    if HAS_MLX_CLUSTER:
        return _mlxc.neighbor_sample(
            colptr, row, input_node, list(num_neighbors), replace, directed
        )
    return _neighbor_sample_numpy(
        colptr, row, input_node, num_neighbors, replace, directed
    )


def _neighbor_sample_numpy(
    colptr,
    row,
    input_node,
    num_neighbors: Sequence[int],
    replace: bool = False,
    directed: bool = True,
    seed: Optional[int] = None,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    colptr, row, seeds = _index(colptr), _index(row), _index(input_node)
    rng = _rng(seed)
    local = np.full(colptr.size - 1, -1, dtype=np.int64)
    uniq, first = np.unique(seeds, return_index=True)
    local[uniq] = first

    samples, num_samples = [seeds], seeds.size
    rows, cols, edges = [], [], []
    frontier, begin = seeds, 0
    for k in num_neighbors:
        pos, off = _sample_edges(colptr, frontier, int(k), replace, rng)
        src = row[off]
        new = _first_seen(src[local[src] < 0])
        local[new] = np.arange(num_samples, num_samples + new.size)
        if directed:
            rows.append(local[src])
            cols.append(pos + begin)
            edges.append(off)
        samples.append(new)
        frontier, begin = new, num_samples
        num_samples += new.size

    samples = _cat(samples)
    if not directed:
        start = colptr[samples]
        seg, off = _ranges(start, colptr[samples + 1] - start)
        src = local[row[off]]
        keep = src >= 0
        rows, cols, edges = [src[keep]], [seg[keep]], [off[keep]]
    return (
        mx.array(samples),
        mx.array(_cat(rows)),
        mx.array(_cat(cols)),
        mx.array(_cat(edges)),
    )


# ---------------------------------------------------------------------------
# Heterogeneous neighbor sampling
# ---------------------------------------------------------------------------


class HeteroSamplerOutput(NamedTuple):
    """Result of :func:`hetero_neighbor_sample` (same fields as in
    ``mlx_cluster``).

    Attributes:
        node_id: Global ids of the sampled nodes per node type, seeds first.
        row: Per edge type, local index (into ``node_id[src]``) of each
            sampled edge's source node.
        col: Per edge type, local index (into ``node_id[dst]``) of each
            sampled edge's destination node.
        edge_id: Per edge type, offset of each sampled edge in
            ``row_dict[edge_type]``.
        num_sampled_nodes: Per node type, ``[#seeds, #new at hop 1, ...]``.
        num_sampled_edges: Per edge type, ``[#edges at hop 1, ...]``.
    """

    node_id: Dict[NodeType, mx.array]
    row: Dict[EdgeType, mx.array]
    col: Dict[EdgeType, mx.array]
    edge_id: Dict[EdgeType, mx.array]
    num_sampled_nodes: Dict[NodeType, List[int]]
    num_sampled_edges: Dict[EdgeType, List[int]]


def hetero_neighbor_sample(
    colptr_dict: Mapping[EdgeType, mx.array],
    row_dict: Mapping[EdgeType, mx.array],
    seed_dict: Mapping[NodeType, mx.array],
    num_neighbors: NumNeighbors,
    num_nodes_dict: Optional[Mapping[NodeType, int]] = None,
    replace: bool = False,
    directed: bool = True,
    seed: Optional[int] = None,
) -> HeteroSamplerOutput:
    """Multi-hop neighbor sampling on a heterogeneous graph.

    For every edge type ``(src, rel, dst)`` and every hop, up to ``k``
    incoming ``src`` neighbors are sampled for each ``dst`` node discovered
    at the previous hop (the seeds at hop 0). See
    ``mlx_cluster.hetero_neighbor_sample`` for the arguments.
    """
    kwargs = dict(
        colptr_dict=colptr_dict,
        row_dict=row_dict,
        seed_dict=seed_dict,
        num_neighbors=num_neighbors,
        num_nodes_dict=num_nodes_dict,
        replace=replace,
        directed=directed,
        seed=seed,
    )
    if HAS_MLX_CLUSTER:
        return _mlxc.hetero_neighbor_sample(**kwargs)
    return _hetero_neighbor_sample_numpy(**kwargs)


def _resolve_num_neighbors(
    num_neighbors: NumNeighbors, edge_types: List[EdgeType]
) -> Dict[EdgeType, List[int]]:
    if isinstance(num_neighbors, Mapping):
        unknown = set(num_neighbors) - set(edge_types)
        if unknown:
            raise ValueError(
                f"num_neighbors has edge types not in colptr_dict: {sorted(unknown)}"
            )
        missing = [et for et in edge_types if et not in num_neighbors]
        if missing:
            raise ValueError(
                f"num_neighbors is missing edge types {missing}; "
                "pass [0] to skip an edge type"
            )
        resolved = {et: [int(k) for k in num_neighbors[et]] for et in edge_types}
    else:
        resolved = {et: [int(k) for k in num_neighbors] for et in edge_types}
    num_hops = max((len(v) for v in resolved.values()), default=0)
    if num_hops == 0:
        raise ValueError("num_neighbors must specify at least one hop")
    return {et: v + [0] * (num_hops - len(v)) for et, v in resolved.items()}


def _hetero_neighbor_sample_numpy(
    colptr_dict: Mapping[EdgeType, mx.array],
    row_dict: Mapping[EdgeType, mx.array],
    seed_dict: Mapping[NodeType, mx.array],
    num_neighbors: NumNeighbors,
    num_nodes_dict: Optional[Mapping[NodeType, int]] = None,
    replace: bool = False,
    directed: bool = True,
    seed: Optional[int] = None,
) -> HeteroSamplerOutput:
    if not directed:
        raise ValueError(
            "hetero_neighbor_sample: directed=False is not supported for "
            "heterogeneous graphs"
        )
    edge_types = [tuple(et) for et in colptr_dict]
    if not edge_types:
        raise ValueError("colptr_dict must contain at least one edge type")
    if set(row_dict) != set(colptr_dict):
        raise ValueError("colptr_dict and row_dict must have the same edge types")
    if not seed_dict:
        raise ValueError("seed_dict must contain at least one node type")

    node_types: List[NodeType] = []
    for src, _, dst in edge_types:
        for nt in (src, dst):
            if nt not in node_types:
                node_types.append(nt)
    for nt in list(seed_dict) + list(num_nodes_dict or {}):
        if nt not in node_types:
            node_types.append(nt)
    fanouts = _resolve_num_neighbors(num_neighbors, edge_types)
    num_hops = len(next(iter(fanouts.values())))

    colptr = {et: _index(colptr_dict[et]) for et in edge_types}
    row = {et: _index(row_dict[et]) for et in edge_types}
    num_nodes = {nt: int(n) for nt, n in (num_nodes_dict or {}).items()}
    for et in edge_types:
        n_dst = colptr[et].size - 1
        if n_dst < 0:
            raise ValueError(f"colptr_dict[{et}] is empty")
        if colptr[et][-1] != row[et].size:
            raise ValueError(
                f"colptr_dict[{et}][-1] = {colptr[et][-1]} does not match "
                f"len(row_dict[{et}]) = {row[et].size}"
            )
        if num_nodes.setdefault(et[2], n_dst) != n_dst:
            raise ValueError(
                f"colptr_dict[{et}] implies {n_dst} '{et[2]}' nodes, but "
                f"{num_nodes[et[2]]} were given/implied elsewhere"
            )
    seeds = {nt: _index(s) for nt, s in seed_dict.items()}
    # Size of the global -> local id map of each node type.
    size = {nt: num_nodes.get(nt, 0) for nt in node_types}
    for (src, _, _), r in row.items():
        if src not in num_nodes and r.size:
            size[src] = max(size[src], int(r.max()) + 1)
    for nt, s in seeds.items():
        size[nt] = max(size[nt], int(s.max()) + 1 if s.size else 0)

    rng = _rng(seed)
    local = {nt: np.full(size[nt], -1, dtype=np.int64) for nt in node_types}
    sampled: Dict[NodeType, List[np.ndarray]] = {nt: [] for nt in node_types}
    count = {nt: 0 for nt in node_types}
    for nt, s in seeds.items():
        if np.unique(s).size != s.size:
            raise ValueError(f"seed_dict['{nt}'] contains duplicate nodes")
        local[nt][s] = np.arange(s.size)
        sampled[nt].append(s)
        count[nt] = s.size
    frontier = {nt: _cat(sampled[nt]) for nt in node_types}
    begin = {nt: 0 for nt in node_types}
    num_sampled_nodes = {nt: [count[nt]] for nt in node_types}

    rows: Dict[EdgeType, List[np.ndarray]] = {et: [] for et in edge_types}
    cols: Dict[EdgeType, List[np.ndarray]] = {et: [] for et in edge_types}
    eids: Dict[EdgeType, List[np.ndarray]] = {et: [] for et in edge_types}
    num_sampled_edges: Dict[EdgeType, List[int]] = {et: [] for et in edge_types}

    for hop in range(num_hops):
        before = dict(count)
        for et in edge_types:
            src, _, dst = et
            dst_nodes = frontier[dst]
            n_dst = colptr[et].size - 1
            if dst_nodes.size and (dst_nodes.min() < 0 or dst_nodes.max() >= n_dst):
                raise IndexError(
                    f"a node of type '{dst}' is out of range for "
                    f"colptr_dict[{et}] ({n_dst} nodes)"
                )
            pos, off = _sample_edges(
                colptr[et], dst_nodes, fanouts[et][hop], replace, rng
            )
            src_nodes = row[et][off]
            new = _first_seen(src_nodes[local[src][src_nodes] < 0])
            local[src][new] = np.arange(count[src], count[src] + new.size)
            sampled[src].append(new)
            count[src] += new.size
            rows[et].append(local[src][src_nodes])
            cols[et].append(pos + begin[dst])
            eids[et].append(off)
            num_sampled_edges[et].append(int(off.size))
        for nt in node_types:
            # Nodes discovered during this hop are expanded at the next one.
            frontier[nt] = _cat(sampled[nt])[before[nt] :]
            begin[nt] = before[nt]
            num_sampled_nodes[nt].append(count[nt] - before[nt])

    return HeteroSamplerOutput(
        node_id={nt: mx.array(_cat(sampled[nt])) for nt in node_types},
        row={et: mx.array(_cat(rows[et])) for et in edge_types},
        col={et: mx.array(_cat(cols[et])) for et in edge_types},
        edge_id={et: mx.array(_cat(eids[et])) for et in edge_types},
        num_sampled_nodes=num_sampled_nodes,
        num_sampled_edges=num_sampled_edges,
    )


# ---------------------------------------------------------------------------
# Random walks
# ---------------------------------------------------------------------------


def _step_index(rand: mx.array, deg: mx.array) -> mx.array:
    # float32 product truncated toward zero, as in the mlx_cluster kernels.
    idx = (rand * deg.astype(mx.float32)).astype(mx.int64)
    return mx.minimum(idx, deg - 1)  # guards against rand == 1.0


def random_walk(
    rowptr: mx.array,
    col: mx.array,
    start: mx.array,
    rand: mx.array,
    walk_length: int,
    *,
    stream=None,
) -> Tuple[mx.array, mx.array]:
    """Uniform random walks on a CSR graph.

    Step ``l`` of walk ``n`` moves to neighbor ``floor(rand[n, l] * degree)``
    of the current node. A node without neighbors keeps the walk in place
    (edge id ``-1``).

    Args:
        rowptr: CSR row pointers (int64).
        col: CSR column indices (int64).
        start: Start nodes (int64).
        rand: Uniform ``[0, 1)`` values of shape ``[len(start), walk_length]``.
        walk_length: Number of steps per walk.
        stream: Where ``mlx_cluster`` runs (ignored by the fallback).

    Returns:
        ``(nodes, edges)`` of shapes ``[len(start), walk_length + 1]`` and
        ``[len(start), walk_length]``, int64.
    """
    if HAS_MLX_CLUSTER:
        return _mlxc.random_walk(rowptr, col, start, rand, walk_length, stream=stream)
    return _random_walk_mlx(rowptr, col, start, rand, walk_length)


def _random_walk_mlx(rowptr, col, start, rand, walk_length):
    rowptr = rowptr.astype(mx.int64)
    # A trailing entry makes the gather of a walk without neighbors safe.
    col = mx.concatenate([col.astype(mx.int64), mx.zeros((1,), mx.int64)])
    pad = col.size - 1
    rand = rand.astype(mx.float32)
    cur = start.astype(mx.int64).reshape(-1)
    nodes, edges = [cur], []
    for step in range(walk_length):
        row_start = rowptr[cur]
        deg = rowptr[cur + 1] - row_start
        has = deg > 0
        edge = mx.where(has, row_start + _step_index(rand[:, step], deg), -1)
        cur = mx.where(has, col[mx.where(has, edge, pad)], cur)
        nodes.append(cur)
        edges.append(edge)
    return _stack_walk(nodes, edges, start.size)


def _stack_walk(nodes, edges, n):
    edges = mx.stack(edges, axis=1) if edges else mx.zeros((n, 0), mx.int64)
    return mx.stack(nodes, axis=1), edges


def rejection_sampling(
    rowptr: mx.array,
    col: mx.array,
    start: mx.array,
    walk_length: int,
    p: float,
    q: float,
    *,
    key: Optional[mx.array] = None,
    max_attempts: int = 1000,
    stream=None,
) -> Tuple[mx.array, mx.array]:
    """node2vec biased random walks (rejection sampling).

    Having moved ``t -> v``, the next node ``x`` is a neighbor of ``v``
    chosen with weight ``1/p`` if ``x == t``, ``1`` if ``x`` is a neighbor of
    ``t``, and ``1/q`` otherwise. A node without neighbors keeps the walk in
    place (edge id ``-1``). After ``max_attempts`` rejections the last
    candidate is accepted.

    Returns:
        ``(nodes, edges)`` of shapes ``[len(start), walk_length + 1]`` and
        ``[len(start), walk_length]``, int64.
    """
    if HAS_MLX_CLUSTER:
        return _mlxc.rejection_sampling(
            rowptr,
            col,
            start,
            walk_length,
            p,
            q,
            key=key,
            max_attempts=max_attempts,
            stream=stream,
        )
    return _rejection_sampling_numpy(
        rowptr, col, start, walk_length, p, q, key=key, max_attempts=max_attempts
    )


def _rejection_sampling_numpy(
    rowptr, col, start, walk_length, p, q, *, key=None, max_attempts=1000
):
    rowptr, col, t = _index(rowptr), _index(col), _index(start)
    seed = None if key is None else [int(k) for k in np.asarray(key).reshape(-1)]
    rng = np.random.default_rng(seed) if seed is not None else _rng()
    max_prob = max(1.0 / p, 1.0, 1.0 / q)
    prob_return, prob_neighbor, prob_out = (
        1.0 / p / max_prob,
        1.0 / max_prob,
        1.0 / q / max_prob,
    )

    # Sorted (source, target) keys answer "is x a neighbor of t".
    n = rowptr.size - 1
    keys = np.repeat(np.arange(n, dtype=np.int64), np.diff(rowptr)) * n + col
    if keys.size > 1 and np.any(keys[1:] < keys[:-1]):
        keys = np.sort(keys)

    def is_neighbor(x, y):
        k = x * n + y
        i = np.minimum(np.searchsorted(keys, k), max(keys.size - 1, 0))
        return keys[i] == k if keys.size else np.zeros(k.shape, dtype=bool)

    nodes = np.empty((t.size, walk_length + 1), dtype=np.int64)
    edges = np.full((t.size, walk_length), -1, dtype=np.int64)
    nodes[:, 0] = t
    v = t.copy()
    for step in range(walk_length):
        row_start = rowptr[v]
        deg = rowptr[v + 1] - row_start
        x, edge = v.copy(), np.full(v.size, -1, dtype=np.int64)
        if step == 0:
            # First-order step: uniform among neighbors.
            has = np.flatnonzero(deg > 0)
            edge[has] = row_start[has] + rng.integers(0, deg[has])
            x[has] = col[edge[has]]
        else:
            one = np.flatnonzero(deg == 1)
            edge[one] = row_start[one]
            x[one] = col[edge[one]]
            pending = np.flatnonzero(deg > 1)
            for _ in range(max_attempts):
                if pending.size == 0:
                    break
                cand = row_start[pending] + rng.integers(0, deg[pending])
                cx, prev = col[cand], t[pending]
                prob = np.where(
                    cx == prev,
                    prob_return,
                    np.where(is_neighbor(cx, prev), prob_neighbor, prob_out),
                )
                edge[pending], x[pending] = cand, cx
                pending = pending[rng.random(pending.size) >= prob]
        nodes[:, step + 1] = x
        edges[:, step] = edge
        t, v = v, x
    return mx.array(nodes), mx.array(edges)


# ---------------------------------------------------------------------------
# Meta-path random walks
# ---------------------------------------------------------------------------


class MetaPathCSR(NamedTuple):
    """Per-relation CSR adjacency for the relations of a meta-path (same
    fields as ``mlx_cluster.MetaPathCSR``)."""

    metapath: List[EdgeType]
    rowptr: mx.array
    col: mx.array
    rowptr_offset: mx.array
    col_offset: mx.array
    rel_ids: List[int]
    num_nodes_dict: Dict[NodeType, int]


def metapath_node_types(metapath: Sequence[EdgeType], walk_length: int) -> List[str]:
    """Node type of each of the ``walk_length + 1`` positions of a walk."""
    m = len(metapath)
    return [metapath[0][0]] + [metapath[step % m][2] for step in range(walk_length)]


def build_metapath_csr(
    edge_index_dict: Mapping[EdgeType, mx.array],
    metapath: Sequence[EdgeType],
    num_nodes_dict: Optional[Mapping[NodeType, int]] = None,
):
    """Builds the CSR adjacency used by :func:`metapath_random_walk`.

    Args:
        edge_index_dict: ``(src, rel, dst) -> [2, num_edges]`` edge indices.
        metapath: List of ``(src, rel, dst)`` edge types, each ending at the
            node type the next one starts from.
        num_nodes_dict: Optional number of nodes per node type. Missing types
            are inferred from the largest node id in ``edge_index_dict``.
    """
    if HAS_MLX_CLUSTER:
        return _mlxc.build_metapath_csr(edge_index_dict, metapath, num_nodes_dict)
    return _build_metapath_csr_numpy(edge_index_dict, metapath, num_nodes_dict)


def _build_metapath_csr_numpy(edge_index_dict, metapath, num_nodes_dict=None):
    metapath = [tuple(et) for et in metapath]
    if not metapath:
        raise ValueError("metapath must contain at least one edge type")
    for et in metapath:
        if len(et) != 3 or not all(isinstance(part, str) for part in et):
            raise ValueError(f"Edge types must be (src, rel, dst) str tuples: {et!r}")
    for a, b in zip(metapath[:-1], metapath[1:]):
        if a[2] != b[0]:
            raise ValueError(
                f"Invalid metapath: {a!r} ends at '{a[2]}' but {b!r} "
                f"starts at '{b[0]}'"
            )
    missing = [et for et in metapath if et not in edge_index_dict]
    if missing:
        raise ValueError(f"Edge types {missing} are not in edge_index_dict")

    num_nodes = dict(num_nodes_dict or {})
    inferred: Dict[NodeType, int] = {}
    for (src, _, dst), ei in edge_index_dict.items():
        if ei.size == 0:
            continue
        ei = np.asarray(ei)
        inferred[src] = max(inferred.get(src, 0), int(ei[0].max()) + 1)
        inferred[dst] = max(inferred.get(dst, 0), int(ei[1].max()) + 1)
    for nt, n in inferred.items():
        if nt not in num_nodes:
            num_nodes[nt] = n
        elif num_nodes[nt] < n:
            raise ValueError(
                f"num_nodes_dict['{nt}'] = {num_nodes[nt]} but edges reference "
                f"node id {n - 1}"
            )

    relations = list(dict.fromkeys(metapath))
    rowptrs, cols, rowptr_offset, col_offset = [], [], [], []
    for et in relations:
        ei = np.asarray(edge_index_dict[et]).astype(np.int64).reshape(2, -1)
        src, dst = ei
        counts = np.bincount(src, minlength=num_nodes.get(et[0], 0))
        rowptr_offset.append(sum(r.size for r in rowptrs))
        col_offset.append(sum(c.size for c in cols))
        rowptrs.append(np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
        cols.append(dst[np.argsort(src, kind="stable")])
    return MetaPathCSR(
        metapath=metapath,
        rowptr=mx.array(np.concatenate(rowptrs)),
        col=mx.array(np.concatenate(cols)),
        rowptr_offset=mx.array(np.array(rowptr_offset, dtype=np.int64)),
        col_offset=mx.array(np.array(col_offset, dtype=np.int64)),
        rel_ids=[relations.index(et) for et in metapath],
        num_nodes_dict=num_nodes,
    )


def metapath_random_walk(
    csr,
    start: mx.array,
    walk_length: int,
    rand: Optional[mx.array] = None,
    stream=None,
) -> Tuple[mx.array, mx.array]:
    """Meta-path guided uniform random walks.

    Step ``l`` follows edge type ``metapath[l % len(metapath)]``, moving to
    neighbor ``floor(rand[n, l] * degree)``. If a node has no such neighbor,
    the walk stops and the remaining entries are ``-1``.

    Args:
        csr: Output of :func:`build_metapath_csr`.
        start: Start node ids, local to ``metapath[0][0]``'s node type.
        walk_length: Number of steps per walk.
        rand: Optional uniform ``[0, 1)`` values of shape
            ``[len(start), walk_length]``, drawn from MLX's RNG if omitted.
        stream: Where ``mlx_cluster`` runs (ignored by the fallback).

    Returns:
        ``(nodes, edges)``: ``[len(start), walk_length + 1]`` local node ids
        and ``[len(start), walk_length]`` edge offsets within each relation.
    """
    if HAS_MLX_CLUSTER:
        return _mlxc.metapath_random_walk(csr, start, walk_length, rand, stream)
    return _metapath_random_walk_mlx(csr, start, walk_length, rand)


def _metapath_random_walk_mlx(csr, start, walk_length, rand=None):
    metapath = csr.metapath
    m = len(metapath)
    if walk_length < 1:
        raise ValueError("walk_length must be >= 1")
    if walk_length > m and metapath[-1][2] != metapath[0][0]:
        raise ValueError(
            f"walk_length ({walk_length}) is longer than the metapath ({m}), "
            "so the metapath must be cyclic: it ends at "
            f"'{metapath[-1][2]}' but starts at '{metapath[0][0]}'"
        )
    start = start if isinstance(start, mx.array) else mx.array(start)
    cur = start.astype(mx.int64).reshape(-1)
    if rand is None:
        rand = mx.random.uniform(shape=(cur.size, walk_length))
    rand = rand.astype(mx.float32)
    rowptr = csr.rowptr.astype(mx.int64)
    col = mx.concatenate([csr.col.astype(mx.int64), mx.zeros((1,), mx.int64)])
    pad = col.size - 1
    rowptr_offset = np.asarray(csr.rowptr_offset).tolist()
    col_offset = np.asarray(csr.col_offset).tolist()

    nodes, edges = [cur], []
    for step in range(walk_length):
        rel = csr.rel_ids[step % m]
        alive = cur >= 0
        ptr = rowptr_offset[rel] + mx.where(alive, cur, 0)
        row_start = rowptr[ptr]
        deg = mx.where(alive, rowptr[ptr + 1] - row_start, 0)
        has = deg > 0
        edge = mx.where(has, row_start + _step_index(rand[:, step], deg), -1)
        cur = mx.where(has, col[mx.where(has, col_offset[rel] + edge, pad)], -1)
        nodes.append(cur)
        edges.append(edge)
    return _stack_walk(nodes, edges, cur.size)
