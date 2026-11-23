"""Vectorized (numpy) disjoint and temporal neighbor sampling over CSC.

``mlx_cluster.neighbor_sample`` samples one merged subgraph for all the seeds
of a mini-batch and has no notion of time. Temporal sampling (PyG's
``time_attr`` / ``temporal_strategy``) needs one subgraph per seed (a
neighbor is only valid for a seed if its timestamp is not later than the
seed's), which is what this module implements, following pyg-lib's
semantics:

- the in-neighbors of every node are sorted by time in the CSC structure, so
  the neighbors valid for a seed time ``t`` form a prefix of each column,
  found by a (vectorized) binary search;
- ``temporal_strategy="uniform"`` samples uniformly among them,
  ``"last"`` keeps the ``k`` most recent ones;
- each seed gets its own subgraph (``disjoint``): nodes are deduplicated per
  seed, and every node of the output carries the position of its seed
  (``batch``).

Every hop is a handful of numpy operations over the whole frontier: there is
no per-node Python loop.
"""

from typing import Any, Optional

import numpy as np

from ._utils import segment_arange


def build_csc(
    src: np.ndarray,
    dst: np.ndarray,
    num_nodes: int,
    node_time: Optional[np.ndarray] = None,
    edge_time: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """CSC (in-edges by target) of ``(src, dst)``, sorted by time per column.

    As in PyG's ``sort_csc``, the in-edges of every node are sorted by
    ``edge_time`` (edge-level time) or by the time of their source node
    (node-level time), ties keeping the original edge order.

    Args:
        src: Source node of every edge.
        dst: Target node of every edge.
        num_nodes: Number of nodes.
        node_time: Optional timestamp of every node.
        edge_time: Optional timestamp of every edge.

    Returns:
        ``(colptr, row, perm, csc_time)``: column pointers, the source of
        every CSC entry, the original edge id of every CSC entry and the time
        of every CSC entry (``None`` without time).
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    if edge_time is not None:
        perm = np.lexsort((np.asarray(edge_time), dst))
    elif node_time is not None:
        perm = np.lexsort((np.asarray(node_time)[src], dst))
    else:
        perm = np.argsort(dst, kind="stable")
    row = src[perm]
    colptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(dst, minlength=num_nodes), out=colptr[1:])
    csc_time = None
    if edge_time is not None:
        csc_time = np.asarray(edge_time)[perm]
    elif node_time is not None:
        csc_time = np.asarray(node_time)[row]
    return colptr, row, perm.astype(np.int64), csc_time


def segment_upper_bound(
    values: np.ndarray, starts: np.ndarray, ends: np.ndarray, bound: np.ndarray
) -> np.ndarray:
    """For every segment ``values[starts[i]:ends[i]]`` (sorted), the first
    position whose value is greater than ``bound[i]`` (``ends[i]`` if none).

    A binary search run on all the segments at once, in
    ``ceil(log2(max segment length)) + 1`` vectorized steps.
    """
    lo = starts.astype(np.int64, copy=True)
    hi = ends.astype(np.int64, copy=True)
    active = np.nonzero(lo < hi)[0]
    while active.size:
        lo_a, hi_a = lo[active], hi[active]
        mid = (lo_a + hi_a) // 2
        ok = values[mid] <= bound[active]
        lo[active] = np.where(ok, mid + 1, lo_a)
        hi[active] = np.where(ok, hi_a, mid)
        active = active[lo[active] < hi[active]]
    return lo


def _floyd_offsets(population: np.ndarray, k: int) -> np.ndarray:
    """``k`` distinct offsets in ``[0, population[g])`` for every group ``g``
    (all populations larger than ``k``), drawn uniformly with Robert Floyd's
    algorithm, vectorized across groups."""
    num_groups = population.size
    chosen = np.empty((num_groups, k), dtype=np.int64)
    for i in range(k):
        j = population - k + i  # candidate range is [0, j]
        t = np.floor(np.random.random(num_groups) * (j + 1)).astype(np.int64)
        np.minimum(t, j, out=t)
        if i:
            dup = (chosen[:, :i] == t[:, None]).any(axis=1)
            t = np.where(dup, j, t)
        chosen[:, i] = t
    return chosen.reshape(-1)


def sample_positions(
    starts: np.ndarray, counts: np.ndarray, k: int, replace: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Samples up to ``k`` positions in every segment
    ``[starts[i], starts[i] + counts[i])``.

    Returns:
        ``(owner, pos)``, grouped by owner (in increasing order): the segment
        each sampled position belongs to, and the position. When a segment
        has at most ``k`` entries (or ``k < 0``) and ``replace`` is ``False``,
        all its entries are kept in order.
    """
    num = starts.size
    if k < 0:
        return np.repeat(np.arange(num), counts), segment_arange(starts, counts)
    if replace:
        has = np.nonzero(counts > 0)[0]
        owner = np.repeat(has, k)
        offs = np.floor(np.random.random(owner.size) * counts[owner]).astype(np.int64)
        return owner, starts[owner] + np.minimum(offs, counts[owner] - 1)

    small = np.nonzero(counts <= k)[0]
    big = np.nonzero(counts > k)[0]
    owner_small = np.repeat(small, counts[small])
    pos_small = segment_arange(starts[small], counts[small])
    if big.size == 0:
        return owner_small, pos_small
    owner_big = np.repeat(big, k)
    pos_big = np.repeat(starts[big], k) + _floyd_offsets(counts[big], k)
    owner = np.concatenate([owner_small, owner_big])
    pos = np.concatenate([pos_small, pos_big])
    order = np.argsort(owner, kind="stable")
    return owner[order], pos[order]


class DisjointSample:
    """Output of :func:`disjoint_neighbor_sample`.

    Attributes:
        node: Global id of every sampled node; the seeds come first, in
            order (a node sampled for several seeds appears once per seed).
        batch: Position of the seed every node was sampled for.
        row: Local source (sampled neighbor) of every sampled edge.
        col: Local target of every sampled edge.
        edge: CSC position of every sampled edge.
        num_sampled_nodes: Number of new nodes per hop (seeds first).
        num_sampled_edges: Number of sampled edges per hop.
    """

    def __init__(
        self, node, batch, row, col, edge, num_sampled_nodes, num_sampled_edges
    ):
        self.node = node
        self.batch = batch
        self.row = row
        self.col = col
        self.edge = edge
        self.num_sampled_nodes = num_sampled_nodes
        self.num_sampled_edges = num_sampled_edges


def disjoint_neighbor_sample(
    colptr: np.ndarray,
    row: np.ndarray,
    seeds: np.ndarray,
    num_neighbors: list[int],
    seed_time: Optional[np.ndarray] = None,
    csc_time: Optional[np.ndarray] = None,
    replace: bool = False,
    temporal_strategy: str = "uniform",
) -> DisjointSample:
    """Samples one K-hop subgraph per seed (pyg-lib's ``disjoint=True``).

    Args:
        colptr: CSC column pointers.
        row: CSC sources. With time, every column must be sorted by
            ``csc_time`` (see :func:`build_csc`).
        seeds: Seed nodes (duplicates are separate seeds).
        num_neighbors: Fan-out per hop (``-1`` keeps all valid neighbors).
        seed_time: Timestamp of every seed. With ``csc_time``, only the
            neighbors whose time is ``<=`` the time of their seed are
            sampled.
        csc_time: Timestamp of every CSC entry (time of the edge or of its
            source node).
        replace: Sample with replacement.
        temporal_strategy: ``"uniform"`` or ``"last"`` (the ``k`` most
            recent valid neighbors).

    Returns:
        A :class:`DisjointSample`.
    """
    if temporal_strategy not in ("uniform", "last"):
        raise ValueError(
            "`temporal_strategy` must be 'uniform' or 'last' "
            f"(got '{temporal_strategy}')"
        )
    temporal = csc_time is not None and seed_time is not None
    num_nodes = colptr.size - 1
    seeds = np.asarray(seeds, dtype=np.int64)
    num_seeds = seeds.size

    nodes = [seeds]
    batches = [np.arange(num_seeds, dtype=np.int64)]
    rows, cols, edges = [], [], []
    num_sampled_nodes, num_sampled_edges = [num_seeds], []

    # (seed, node) pairs already in the output, as sorted int64 keys
    # `seed * num_nodes + node`, with their local ids.
    keys = batches[0] * num_nodes + seeds
    key_order = np.argsort(keys, kind="stable")
    sorted_keys, sorted_local = keys[key_order], key_order.astype(np.int64)

    f_local = np.arange(num_seeds, dtype=np.int64)
    f_node, f_batch = seeds, batches[0]
    total = num_seeds

    for k in num_neighbors:
        starts = colptr[f_node]
        ends = colptr[f_node + 1]
        if temporal:
            ends = segment_upper_bound(csc_time, starts, ends, seed_time[f_batch])
        if temporal_strategy == "last" and k >= 0:
            starts = np.maximum(starts, ends - k)
        owner, pos = sample_positions(starts, ends - starts, k, replace)

        nbr = row[pos]
        b = f_batch[owner]
        key = b * num_nodes + nbr
        local = np.empty(key.size, dtype=np.int64)
        if sorted_keys.size:
            idx = np.searchsorted(sorted_keys, key)
            hit = idx < sorted_keys.size
            hit[hit] = sorted_keys[idx[hit]] == key[hit]
            local[hit] = sorted_local[idx[hit]]
        else:
            hit = np.zeros(key.size, dtype=bool)

        miss = np.nonzero(~hit)[0]
        uniq, first, inv = np.unique(key[miss], return_index=True, return_inverse=True)
        # New nodes get local ids in order of first appearance.
        appearance = np.argsort(first, kind="stable")
        rank = np.empty(uniq.size, dtype=np.int64)
        rank[appearance] = np.arange(uniq.size)
        local[miss] = total + rank[inv.reshape(-1)]
        new_at = miss[first[appearance]]
        new_nodes, new_batch = nbr[new_at], b[new_at]

        rows.append(local)
        cols.append(f_local[owner])
        edges.append(pos)
        nodes.append(new_nodes)
        batches.append(new_batch)
        num_sampled_nodes.append(int(new_nodes.size))
        num_sampled_edges.append(int(pos.size))

        new_keys = key[new_at]
        new_local = np.arange(total, total + new_nodes.size, dtype=np.int64)
        all_keys = np.concatenate([sorted_keys, new_keys])
        all_local = np.concatenate([sorted_local, new_local])
        order = np.argsort(all_keys, kind="stable")
        sorted_keys, sorted_local = all_keys[order], all_local[order]

        f_local, f_node, f_batch = new_local, new_nodes, new_batch
        total += new_nodes.size

    def cat(arrays):
        return np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.int64)

    return DisjointSample(
        node=cat(nodes),
        batch=cat(batches),
        row=cat(rows),
        col=cat(cols),
        edge=cat(edges),
        num_sampled_nodes=num_sampled_nodes,
        num_sampled_edges=num_sampled_edges,
    )


def graph_time(
    data: Any, time_attr: Optional[str]
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """``(node_time, edge_time)``: the ``time_attr`` attribute of a
    :class:`~mlx_graphs.data.GraphData`, as node-level or edge-level time
    (node-level first, as in PyG)."""
    if time_attr is None:
        return None, None
    value = getattr(data, time_attr, None)
    if value is None or not hasattr(value, "shape") or len(value.shape) == 0:
        raise ValueError(f"The graph has no time attribute '{time_attr}'.")
    time = np.asarray(value)
    if time.ndim > 1:
        if int(np.prod(time.shape[1:])) != 1:
            raise ValueError(f"The time attribute '{time_attr}' must be 1-D.")
        time = time.reshape(-1)
    if data.is_node_attr(time_attr):
        return time, None
    if data.is_edge_attr(time_attr):
        return None, time
    raise ValueError(
        f"The time attribute '{time_attr}' is neither a node-level nor an "
        "edge-level attribute."
    )
