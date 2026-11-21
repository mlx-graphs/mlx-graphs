import os
from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData

from ._utils import Adjacency, index_array, slice_graph


class GraphSAINTSampler:
    """Base class of the GraphSAINT samplers, from the `"GraphSAINT: Graph
    Sampling Based Inductive Learning Method"
    <https://arxiv.org/abs/1907.04931>`_ paper, following PyG's
    ``GraphSAINTSampler``.

    Each iteration samples a set of nodes (how depends on the subclass) and
    yields the subgraph they induce, as a :class:`GraphData` with node- and
    edge-level attributes sliced to the subgraph, plus ``n_id`` (global id of
    every node, sorted) and ``e_id`` (original id of every edge).

    If ``sample_coverage > 0``, normalization statistics are estimated
    beforehand by sampling subgraphs until every node has been sampled
    ``sample_coverage`` times on average, and every batch also holds:

    - ``node_norm``: per-node loss weight, ``num_samples / (count(v) * N)``,
      to use as ``(node_norm * loss).sum()``;
    - ``edge_norm``: per-edge aggregation weight,
      ``count(target(e)) / count(e)``, to scale messages with (with the
      default ``source -> target`` message passing, so that, for a sampled
      node, the expected normalized sum of the messages it receives equals
      the full-graph sum).

    Note:
        PyG divides the count of the edge's *source* node instead
        (``count(source(e)) / count(e)``), which is the paper's normalization
        for the reverse edge: aggregations are then biased by
        ``count(source) / count(target)`` (several-fold for nodes of very
        different degrees). This class follows the paper,
        :math:`\\alpha_{u,v} = p_{u,v} / p_v` with :math:`v` the aggregating
        node.

    Args:
        data: The full graph.
        batch_size: Sampling budget of each subgraph (meaning depends on the
            subclass).
        num_steps: Number of subgraphs (iterations) per epoch. Defaults to 1
        sample_coverage: Average number of samples per node used to compute
            the normalization statistics. ``0`` disables them. Defaults to 0
        save_dir: Optional directory where the normalization statistics are
            cached (as ``.npz``) and loaded from. Defaults to None
    """

    def __init__(
        self,
        data: GraphData,
        batch_size: int,
        num_steps: int = 1,
        sample_coverage: int = 0,
        save_dir: Optional[str] = None,
    ):
        if batch_size < 1:
            raise ValueError(f"`batch_size` must be positive (got {batch_size})")
        for name in ("node_norm", "edge_norm"):
            if getattr(data, name, None) is not None:
                raise ValueError(f"`data` already has a '{name}' attribute.")

        self.data = data
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.sample_coverage = sample_coverage
        self.save_dir = save_dir

        self.N = data.num_nodes
        self.E = data.num_edges
        self._adj = Adjacency(data.edge_index, self.N)
        self._step = 0

        self.node_norm: Optional[np.ndarray] = None
        self.edge_norm: Optional[np.ndarray] = None
        if sample_coverage > 0:
            path = (
                os.path.join(save_dir, self._filename) if save_dir is not None else None
            )
            if path is not None and os.path.exists(path):
                saved = np.load(path)
                self.node_norm, self.edge_norm = saved["node_norm"], saved["edge_norm"]
            else:
                self.node_norm, self.edge_norm = self._compute_norm()
                if path is not None:
                    os.makedirs(save_dir, exist_ok=True)  # type: ignore
                    np.savez(path, node_norm=self.node_norm, edge_norm=self.edge_norm)

    @property
    def _filename(self) -> str:
        return f"{type(self).__name__.lower()}_{self.sample_coverage}.npz"

    def _sample_nodes(self) -> np.ndarray:
        """Returns the (possibly repeated) sampled node ids of one subgraph."""
        raise NotImplementedError

    def _sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        node_idx = np.unique(self._sample_nodes())
        edge_index, e_id = self._adj.induced_subgraph(node_idx)
        return node_idx, edge_index, e_id

    def _compute_norm(self) -> tuple[np.ndarray, np.ndarray]:
        node_count = np.zeros(self.N, dtype=np.float64)
        edge_count = np.zeros(self.E, dtype=np.float64)

        num_samples = total_sampled_nodes = 0
        while total_sampled_nodes < self.N * self.sample_coverage:
            for _ in range(self.num_steps):
                node_idx, _, e_id = self._sample()
                node_count[node_idx] += 1
                edge_count[e_id] += 1
                total_sampled_nodes += node_idx.size
            num_samples += self.num_steps

        # Count of the target (aggregating) node of each edge, divided by the
        # edge count: GraphSAINT's alpha_{u,v} = C_v / C_{u,v}. PyG uses the
        # source count, which biases the aggregation (see the class docstring).
        t = node_count[self._adj.dst]
        with np.errstate(divide="ignore", invalid="ignore"):
            edge_norm = np.clip(t / edge_count, 0, 1e4)
        edge_norm[np.isnan(edge_norm)] = 0.1

        node_count[node_count == 0] = 0.1
        node_norm = num_samples / node_count / self.N
        return node_norm.astype(np.float32), edge_norm.astype(np.float32)

    def __len__(self) -> int:
        """Number of subgraphs per epoch."""
        return self.num_steps

    def __iter__(self) -> "GraphSAINTSampler":
        self._step = 0
        return self

    def __next__(self) -> GraphData:
        if self._step >= self.num_steps:
            raise StopIteration
        self._step += 1

        n_id, edge_index, e_id = self._sample()
        extra = {"n_id": index_array(n_id, self.N), "e_id": index_array(e_id, self.E)}
        if self.node_norm is not None and self.edge_norm is not None:
            extra["node_norm"] = mx.array(self.node_norm[n_id])
            extra["edge_norm"] = mx.array(self.edge_norm[e_id])
        return slice_graph(
            self.data,
            n_id,
            e_id,
            edge_index,
            num_nodes=self.N,
            num_edges=self.E,
            **extra,
        )


class GraphSAINTNodeSampler(GraphSAINTSampler):
    """GraphSAINT node sampler: samples ``batch_size`` edges uniformly at
    random and keeps their source nodes (so nodes are drawn proportionally to
    their out-degree), then takes the induced subgraph.

    See :class:`GraphSAINTSampler` for the arguments and the batch content.

    Example:

    .. code-block:: python

        from mlx_graphs.loaders import GraphSAINTNodeSampler

        loader = GraphSAINTNodeSampler(
            data, batch_size=6000, num_steps=5, sample_coverage=100
        )
        for batch in loader:
            out = model(batch.node_features, batch.edge_index)
            loss = (batch.node_norm * loss_fn(out, batch.node_labels)).sum()
    """

    def _sample_nodes(self) -> np.ndarray:
        if self.E == 0:
            return np.random.randint(0, self.N, size=self.batch_size)
        edge_sample = np.random.randint(0, self.E, size=self.batch_size)
        return self._adj.src[edge_sample]


class GraphSAINTEdgeSampler(GraphSAINTSampler):
    """GraphSAINT edge sampler: samples ``batch_size`` edges without
    replacement with probability proportional to
    ``1 / out_degree(source) + 1 / in_degree(target)``, and takes the subgraph
    induced by their endpoints.

    Note:
        This follows the edge probabilities of the GraphSAINT paper (and of
        its reference implementation), which minimize the variance of the
        normalized aggregation. PyG's implementation instead uses
        ``in_degree(source) + out_degree(target)``, which favors the edges
        between high-degree nodes, and draws the ``batch_size`` edges with
        replacement.

    See :class:`GraphSAINTSampler` for the arguments and the batch content.

    Example:

    .. code-block:: python

        from mlx_graphs.loaders import GraphSAINTEdgeSampler

        loader = GraphSAINTEdgeSampler(data, batch_size=4000, num_steps=5)
    """

    _edge_prob: Optional[np.ndarray] = None

    def _sample_nodes(self) -> np.ndarray:
        if self.E == 0:
            return np.random.randint(0, self.N, size=self.batch_size)
        if self._edge_prob is None:
            adj = self._adj
            out_deg = np.maximum(adj.out_degree, 1).astype(np.float64)
            in_deg = np.maximum(adj.in_degree, 1).astype(np.float64)
            self._edge_prob = 1.0 / out_deg[adj.src] + 1.0 / in_deg[adj.dst]
        # Weighted sampling without replacement (Efraimidis-Spirakis keys).
        keys = np.log(np.random.random(self.E)) / (self._edge_prob + 1e-10)
        k = min(self.batch_size, self.E)
        edge_sample = np.argpartition(-keys, k - 1)[:k]
        return np.concatenate([self._adj.src[edge_sample], self._adj.dst[edge_sample]])


class GraphSAINTRandomWalkSampler(GraphSAINTSampler):
    """GraphSAINT random walk sampler: starts ``batch_size`` random walks of
    length ``walk_length`` (following out-edges) from uniformly sampled root
    nodes, and takes the subgraph induced by the visited nodes. A walk
    reaching a node without out-edges stays on it.

    See :class:`GraphSAINTSampler` for the other arguments and the batch
    content.

    Args:
        walk_length: Length of each random walk.

    Example:

    .. code-block:: python

        from mlx_graphs.loaders import GraphSAINTRandomWalkSampler

        loader = GraphSAINTRandomWalkSampler(
            data, batch_size=3000, walk_length=2, num_steps=5, sample_coverage=100
        )
    """

    def __init__(
        self,
        data: GraphData,
        batch_size: int,
        walk_length: int,
        num_steps: int = 1,
        sample_coverage: int = 0,
        save_dir: Optional[str] = None,
    ):
        self.walk_length = walk_length
        super().__init__(data, batch_size, num_steps, sample_coverage, save_dir)

    @property
    def _filename(self) -> str:
        return (
            f"{type(self).__name__.lower()}_{self.walk_length}_"
            f"{self.sample_coverage}.npz"
        )

    def _sample_nodes(self) -> np.ndarray:
        adj = self._adj
        current = np.random.randint(0, self.N, size=self.batch_size)
        visited = [current]
        out_deg = adj.out_degree
        for _ in range(self.walk_length):
            deg = out_deg[current]
            offset = np.floor(np.random.random(current.size) * deg).astype(np.int64)
            has_next = deg > 0
            nxt = current.copy()
            nxt[has_next] = adj.csr_col[
                adj.rowptr[current[has_next]] + offset[has_next]
            ]
            current = nxt
            visited.append(current)
        return np.concatenate(visited)
