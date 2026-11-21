import importlib.util
import math
from typing import Literal, Optional, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData

from ._utils import Adjacency, index_array, segment_arange, slice_graph, to_numpy


def _symmetric_csr(adj: Adjacency) -> tuple[np.ndarray, np.ndarray]:
    """CSR of the undirected version of the graph, without self-loops or
    duplicate edges."""
    N = adj.num_nodes
    src = np.concatenate([adj.src, adj.dst])
    dst = np.concatenate([adj.dst, adj.src])
    keys = np.unique(src[src != dst] * N + dst[src != dst])
    src, dst = keys // N, keys % N
    rowptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=N), out=rowptr[1:])
    return rowptr, dst


def _bfs_order(rowptr: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Breadth-first ordering of all nodes, component by component, each
    started from a node of minimum degree. Isolated nodes come last."""
    N = rowptr.size - 1
    deg = np.diff(rowptr)
    visited = deg == 0
    order = []
    # Start nodes, tried by increasing degree.
    candidates = np.argsort(deg, kind="stable")
    candidates = candidates[deg[candidates] > 0]
    next_candidate = 0
    while next_candidate < candidates.size:
        start = candidates[next_candidate]
        next_candidate += 1
        if visited[start]:
            continue
        visited[start] = True
        frontier = np.array([start], dtype=np.int64)
        while frontier.size:
            order.append(frontier)
            starts = rowptr[frontier]
            nbrs = col[segment_arange(starts, rowptr[frontier + 1] - starts)]
            nbrs = nbrs[~visited[nbrs]]
            # Unique, in order of first appearance.
            _, first = np.unique(nbrs, return_index=True)
            frontier = nbrs[np.sort(first)]
            visited[frontier] = True
    order.append(np.nonzero(deg == 0)[0])
    return np.concatenate(order).astype(np.int64) if N else np.zeros(0, np.int64)


def _refine(
    part: np.ndarray,
    rowptr: np.ndarray,
    col: np.ndarray,
    num_parts: int,
    num_iters: int,
    imbalance: float,
) -> np.ndarray:
    """Balanced label propagation: nodes move to the partition holding most of
    their neighbors, as long as partition sizes stay within ``imbalance`` of
    the average."""
    N = part.size
    avg = N / num_parts
    max_size = max(math.ceil(avg * (1 + imbalance)), 1)
    min_size = max(math.floor(avg * (1 - imbalance)), 1)
    src = np.repeat(np.arange(N), np.diff(rowptr))

    for _ in range(num_iters):
        # Only half of the nodes may move at once, to avoid oscillations.
        movable = np.random.random(N) < 0.5
        keys, counts = np.unique(src * num_parts + part[col], return_counts=True)
        node, target = keys // num_parts, keys % num_parts

        # Number of neighbors in the current partition of each node.
        current = np.zeros(N, dtype=np.int64)
        in_current = target == part[node]
        current[node[in_current]] = counts[in_current]

        # Best partition of each node (ties broken by partition id).
        order = np.lexsort((target, -counts, node))
        first = np.ones(order.size, dtype=bool)
        first[1:] = node[order][1:] != node[order][:-1]
        best = order[first]
        cand_node, cand_target = node[best], target[best]
        gain = counts[best] - current[cand_node]
        ok = (gain > 0) & (cand_target != part[cand_node]) & movable[cand_node]
        cand_node, cand_target, gain = cand_node[ok], cand_target[ok], gain[ok]
        if cand_node.size == 0:
            break

        # Accept the moves with the highest gains within the size limits.
        sizes = np.bincount(part, minlength=num_parts)
        by_gain = np.argsort(-gain, kind="stable")
        cand_node, cand_target = cand_node[by_gain], cand_target[by_gain]
        cand_source = part[cand_node]

        def rank_in_group(group: np.ndarray) -> np.ndarray:
            grouped = np.argsort(group, kind="stable")
            counts_ = np.bincount(group, minlength=num_parts)
            starts = np.cumsum(counts_) - counts_
            rank = np.empty(group.size, dtype=np.int64)
            rank[grouped] = np.arange(group.size) - np.repeat(starts, counts_)
            return rank

        accept = (rank_in_group(cand_target) < (max_size - sizes)[cand_target]) & (
            rank_in_group(cand_source) < (sizes - min_size)[cand_source]
        )
        if not accept.any():
            break
        part = part.copy()
        part[cand_node[accept]] = cand_target[accept]
    return part


def balanced_partition(
    edge_index: Union[mx.array, np.ndarray],
    num_nodes: int,
    num_parts: int,
    num_iters: int = 10,
    imbalance: float = 0.05,
) -> np.ndarray:
    """Dependency-free balanced graph partitioning.

    Nodes are ordered by a breadth-first traversal of the (undirected) graph,
    the order is cut into ``num_parts`` chunks of equal size, and the
    partitions are refined with balanced label propagation (a node moves to the
    partition holding most of its neighbors while sizes stay within
    ``imbalance`` of the average). Partitions are therefore connected-ish and
    balanced, but their edge cut is usually worse than METIS'.

    Randomness comes from ``np.random`` (seed it for reproducible partitions).

    Args:
        edge_index: ``[2, num_edges]`` edges.
        num_nodes: Number of nodes.
        num_parts: Number of partitions.
        num_iters: Number of label propagation rounds. Defaults to 10
        imbalance: Allowed relative deviation of the partition sizes from the
            average during refinement. Defaults to 0.05

    Returns:
        The partition id of every node, as a ``[num_nodes]`` numpy array.
    """
    if num_parts < 1:
        raise ValueError(f"`num_parts` must be positive (got {num_parts})")
    adj = Adjacency(edge_index, num_nodes)
    rowptr, col = _symmetric_csr(adj)
    order = _bfs_order(rowptr, col)
    part = np.empty(num_nodes, dtype=np.int64)
    part[order] = (np.arange(num_nodes) * num_parts) // max(num_nodes, 1)
    if num_parts > 1 and num_iters > 0:
        part = _refine(part, rowptr, col, num_parts, num_iters, imbalance)
    return part


def _metis_partition(
    edge_index: Union[mx.array, np.ndarray], num_nodes: int, num_parts: int
) -> np.ndarray:
    import pymetis  # type: ignore

    rowptr, col = _symmetric_csr(Adjacency(edge_index, num_nodes))
    _, membership = pymetis.part_graph(num_parts, xadj=rowptr, adjncy=col)
    return np.asarray(membership, dtype=np.int64)


class ClusterData:
    """Partitions a graph into ``num_parts`` clusters, for Cluster-GCN
    training (`"Cluster-GCN: An Efficient Algorithm for Training Deep and Large
    Graph Convolutional Networks" <https://arxiv.org/abs/1905.07953>`_),
    following PyG's ``ClusterData``. Use it with :class:`ClusterLoader`.

    PyG partitions with METIS. Here, METIS (through the optional ``pymetis``
    package) is used only if it is installed or ``method="metis"`` is set;
    otherwise a dependency-free balanced partitioner is used (see
    :func:`balanced_partition`): partitions are balanced but differ from
    METIS', with a usually larger edge cut. Any custom partition can also be
    given with ``partition``.

    ``cluster_data[i]`` is the subgraph induced by the nodes of cluster ``i``
    (edges between clusters are dropped), with ``n_id`` and ``e_id``.

    Args:
        data: The graph to partition.
        num_parts: Number of clusters. Ignored if ``partition`` is given
            (``max(partition) + 1`` is used).
        partition: Optional cluster id of every node, ``[num_nodes]``, to use
            instead of computing one.
        method: ``"auto"`` (METIS if ``pymetis`` is installed, else
            ``"balanced"``), ``"metis"`` or ``"balanced"``.
            Defaults to ``"auto"``
        num_iters: Refinement rounds of the ``"balanced"`` partitioner.
            Defaults to 10

    Attributes:
        cluster: Cluster id of every node (numpy array).
        node_perm: Node ids sorted by cluster (numpy array).
        partptr: Start of every cluster in ``node_perm``, ``[num_parts + 1]``.

    Example:

    .. code-block:: python

        from mlx_graphs.loaders import ClusterData, ClusterLoader

        cluster_data = ClusterData(data, num_parts=100)
        loader = ClusterLoader(cluster_data, batch_size=10, shuffle=True)
        for batch in loader:
            out = model(batch.node_features, batch.edge_index)
    """

    def __init__(
        self,
        data: GraphData,
        num_parts: int,
        partition: Optional[Union[mx.array, np.ndarray]] = None,
        method: Literal["auto", "metis", "balanced"] = "auto",
        num_iters: int = 10,
    ):
        if method not in ("auto", "metis", "balanced"):
            raise ValueError(
                f"`method` must be 'auto', 'metis' or 'balanced' (got '{method}')"
            )
        self.data = data
        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges

        if partition is not None:
            cluster = to_numpy(partition).reshape(-1).astype(np.int64)
            if cluster.size != self.num_nodes:
                raise ValueError(
                    f"`partition` has {cluster.size} entries, expected "
                    f"{self.num_nodes} (one per node)."
                )
            if cluster.size and cluster.min() < 0:
                raise ValueError("`partition` must contain non-negative ids.")
            num_parts = int(cluster.max()) + 1 if cluster.size else 0
            self.method = "custom"
        else:
            has_metis = importlib.util.find_spec("pymetis") is not None
            if method == "metis" or (method == "auto" and has_metis):
                cluster = _metis_partition(data.edge_index, self.num_nodes, num_parts)
                self.method = "metis"
            else:
                cluster = balanced_partition(
                    data.edge_index, self.num_nodes, num_parts, num_iters=num_iters
                )
                self.method = "balanced"

        self.num_parts = num_parts
        self.cluster = cluster
        self.node_perm = np.argsort(cluster, kind="stable")
        self.partptr = np.zeros(num_parts + 1, dtype=np.int64)
        np.cumsum(np.bincount(cluster, minlength=num_parts), out=self.partptr[1:])
        self._adj = Adjacency(data.edge_index, self.num_nodes)

    def __len__(self) -> int:
        return self.num_parts

    def nodes_of(self, parts: Union[int, np.ndarray, list]) -> np.ndarray:
        """Global ids of the nodes of the given cluster(s), cluster by cluster."""
        parts = np.atleast_1d(np.asarray(parts, dtype=np.int64))
        starts = self.partptr[parts]
        return self.node_perm[segment_arange(starts, self.partptr[parts + 1] - starts)]

    def subgraph(self, parts: Union[int, np.ndarray, list]) -> GraphData:
        """Subgraph induced by the union of the given clusters."""
        n_id = self.nodes_of(parts)
        edge_index, e_id = self._adj.induced_subgraph(n_id)
        return slice_graph(
            self.data,
            n_id,
            e_id,
            edge_index,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            n_id=index_array(n_id, self.num_nodes),
            e_id=index_array(e_id, self.num_edges),
        )

    def __getitem__(self, idx: int) -> GraphData:
        if not -self.num_parts <= idx < self.num_parts:
            raise IndexError(f"Cluster index {idx} out of range.")
        return self.subgraph(idx % self.num_parts)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_parts={self.num_parts}, method={self.method})"
        )


class ClusterLoader:
    """Merges ``batch_size`` clusters of a :class:`ClusterData` per mini-batch
    and yields the subgraph they induce, *including* the edges between the
    merged clusters (Cluster-GCN's stochastic multiple partitions), following
    PyG's ``ClusterLoader``.

    Each batch is a :class:`GraphData` with node- and edge-level attributes
    sliced to the subgraph, plus ``n_id`` and ``e_id``.

    Args:
        cluster_data: The partitioned graph.
        batch_size: Number of clusters per mini-batch. Defaults to 1
        shuffle: Whether to shuffle the clusters at each epoch.
            Defaults to False

    Example:

    .. code-block:: python

        cluster_data = ClusterData(data, num_parts=1500)
        loader = ClusterLoader(cluster_data, batch_size=20, shuffle=True)
    """

    def __init__(
        self, cluster_data: ClusterData, batch_size: int = 1, shuffle: bool = False
    ):
        if batch_size < 1:
            raise ValueError(f"`batch_size` must be positive (got {batch_size})")
        self.cluster_data = cluster_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = np.arange(len(cluster_data))
        self._current_index = 0

    def __len__(self) -> int:
        """Number of mini-batches per epoch."""
        return (len(self.cluster_data) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> "ClusterLoader":
        self._current_index = 0
        self._indices = np.arange(len(self.cluster_data))
        if self.shuffle:
            np.random.shuffle(self._indices)
        return self

    def __next__(self) -> GraphData:
        if self._current_index >= self._indices.size:
            raise StopIteration
        end = self._current_index + self.batch_size
        parts = self._indices[self._current_index : end]
        self._current_index = end
        return self.cluster_data.subgraph(parts)
