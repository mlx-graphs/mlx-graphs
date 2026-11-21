from typing import Any, Iterable, Optional, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import HeteroGraphData

from ._utils import index_array, segment_arange
from .hetero_neighbor_loader import _HeteroNeighborLoader

# Nodes with more neighbors than this contribute only a random subset of
# them, as in PyG's (torch-sparse) implementation.
MAX_NEIGHBORS = 50


def _capped_neighbors(
    colptr: np.ndarray, nodes: np.ndarray, cap: int = MAX_NEIGHBORS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """In-neighbors of ``nodes``, at most ``cap`` random ones per node.

    Returns:
        ``(owner, pos, weight)``: for every kept neighbor, the position in
        ``nodes`` it belongs to, its position in the CSC arrays, and
        ``1 / degree`` (``1 / cap`` for capped nodes).
    """
    starts = colptr[nodes]
    degs = colptr[nodes + 1] - starts
    index = np.arange(nodes.size)
    owners, offsets = [], []

    # All neighbors of nodes with at most `cap` of them, and of the nodes with
    # a few more (random keys, keep the `cap` smallest per node).
    few = degs <= 4 * cap
    owner = np.repeat(index[few], degs[few])
    offset = segment_arange(np.zeros(few.sum(), np.int64), degs[few])
    if owner.size > 0 and degs[few].max() > cap:
        order = np.lexsort((np.random.random(owner.size), owner))
        group = np.repeat(np.cumsum(degs[few]) - degs[few], degs[few])
        keep = np.sort(order[(np.arange(owner.size) - group) < cap])
        owner, offset = owner[keep], offset[keep]
    owners.append(owner)
    offsets.append(offset)

    # High-degree nodes, without listing all their neighbors: draw 2 * cap
    # offsets with replacement, keep a random `cap` of the distinct ones (a
    # uniform random subset, by symmetry); redraw for the rare nodes with
    # fewer than `cap` distinct offsets.
    pending = index[~few]
    while pending.size > 0:
        owner = np.repeat(pending, 2 * cap)
        offset = np.floor(np.random.random(owner.size) * degs[owner]).astype(np.int64)
        pairs = np.unique(np.stack([owner, offset], 1), axis=0)
        order = np.lexsort((np.random.random(len(pairs)), pairs[:, 0]))
        pairs = pairs[order]
        counts = np.bincount(pairs[:, 0], minlength=nodes.size)
        rank = np.arange(len(pairs)) - np.repeat(np.cumsum(counts) - counts, counts)
        done = counts[pending] >= cap
        keep = (rank < cap) & np.isin(pairs[:, 0], pending[done])
        owners.append(pairs[keep, 0])
        offsets.append(pairs[keep, 1])
        pending = pending[~done]

    owner = np.concatenate(owners)
    offset = np.concatenate(offsets)
    order = np.lexsort((offset, owner))
    owner, pos = owner[order], starts[owner[order]] + offset[order]
    weight = 1.0 / np.maximum(np.minimum(degs[owner], cap), 1)
    return owner, pos, weight


def _weighted_sample(
    candidates: np.ndarray, weights: np.ndarray, num_samples: int
) -> np.ndarray:
    """``num_samples`` of ``candidates`` without replacement, with
    probabilities proportional to ``weights`` (successive sampling, like
    ``torch.multinomial(weights, num_samples, replacement=False)``). Takes all
    candidates if there are not more than ``num_samples``."""
    if num_samples <= 0 or candidates.size == 0:
        return np.zeros(0, dtype=np.int64)
    if candidates.size <= num_samples:
        return candidates
    # Efraimidis-Spirakis: the k smallest Exp(1) / w keys.
    keys = np.random.exponential(size=candidates.size) / weights
    chosen = np.argpartition(keys, num_samples - 1)[:num_samples]
    return candidates[chosen[np.argsort(keys[chosen])]]


class HGTLoader(_HeteroNeighborLoader):
    r"""Heterogeneous mini-batch loader using the layer-wise sampling of
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
    (HGSampling), as PyG's ``HGTLoader``.

    Instead of a number of neighbors per node (:class:`NeighborLoader`), each
    hop samples a *node budget per node type*: every node type keeps a budget
    of candidate nodes, adjacent to the nodes sampled so far. Each sampled
    node adds ``1 / degree`` to the budget of each of its (not yet sampled)
    neighbors, for every edge type pointing to it (nodes with more than 50
    neighbors only add ``1 / 50`` to 50 random ones). At hop ``l``,
    ``num_samples[node_type][l]`` nodes of each type are drawn from its budget
    without replacement, with probability proportional to the squared budget,
    which favours nodes connected to many sampled nodes. The mini-batch is
    the subgraph induced by the sampled nodes (for destination nodes with more
    than 50 neighbors, edges to 50 random ones, as in PyG).

    Sampling runs on the host with NumPy (seed it with ``np.random.seed``).

    The first ``batch_size`` nodes of the seed node type are the seed nodes,
    so the loss is computed on them::

        for batch in loader:
            out = model(batch.edge_index_dict, batch.node_features_dict)
            n = batch.batch_size
            loss = loss_fn(out["paper"][:n], batch.node_labels_dict["paper"][:n])

    Each batch is a :class:`~mlx_graphs.data.HeteroGraphData` with the same
    attributes as the batches of :class:`NeighborLoader`: ``edge_index_dict``
    (local ids per node type), ``node_features_dict``, ``node_labels_dict``,
    ``edge_features_dict``, ``n_id_dict`` (global id of every sampled node),
    ``e_id_dict`` (original position of every sampled edge), ``batch_size``,
    ``input_type``, ``num_nodes_dict``, ``num_sampled_nodes_dict`` (number of
    nodes added at each hop, starting with the seeds), and the per-node
    attributes of ``data`` (e.g. ``paper_train_mask``) sliced to the sampled
    nodes. Every node type and edge type is present, possibly empty.

    Args:
        data: The full graph.
        num_samples: Number of nodes to sample per hop, either one list for
            every node type (e.g. ``[512] * 4``) or a dict mapping each node
            type to its own list. Node types missing from the dict, or with a
            shorter list, sample no nodes at the missing hops.
        input_nodes: The seed node type and which of its nodes to iterate
            over: ``"paper"`` (all papers), ``("paper", indices)`` or
            ``("paper", boolean_mask)``.
        batch_size: Number of seed nodes per mini-batch.
        shuffle: Whether to shuffle the seed nodes at the start of each epoch.
        sampler: Iterable yielding positions in ``input_nodes`` for each epoch
            (e.g. an :class:`ImbalancedSampler`). Overrides ``shuffle``.
            Duplicate seeds within a mini-batch are merged, so ``batch_size``
            is then the number of distinct seeds.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import DBLP
        from mlx_graphs.loaders import HGTLoader

        data = DBLP()[0]
        loader = HGTLoader(
            data,
            num_samples={
                "author": [512] * 2,
                "paper": [512] * 2,
                "term": [256] * 2,
                "conference": [20] * 2,
            },
            input_nodes=("author", data.author_train_mask),
            batch_size=128,
            shuffle=True,
        )
        batch = next(iter(loader))
    """

    def __init__(
        self,
        data: HeteroGraphData,
        num_samples: Union[list[int], dict[str, list[int]]],
        input_nodes: Union[str, tuple[str, Optional[mx.array]]],
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Iterable[int]] = None,
    ):
        if not isinstance(data, HeteroGraphData):
            raise TypeError(
                f"HGTLoader needs a HeteroGraphData, got {type(data).__name__}"
            )
        if isinstance(input_nodes, mx.array) or input_nodes is None:
            raise ValueError(
                "`input_nodes` must name the seed node type: 'paper', "
                "('paper', indices) or ('paper', boolean_mask)."
            )
        super().__init__(
            data,
            num_neighbors=[],
            input_nodes=input_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
        )
        node_types = list(self.num_nodes)
        if isinstance(num_samples, dict):
            unknown = [t for t in num_samples if t not in self.num_nodes]
            if unknown:
                raise ValueError(f"Unknown node types in num_samples: {unknown}")
            num_samples = {t: list(num_samples.get(t, [])) for t in node_types}
        else:
            num_samples = {t: list(num_samples) for t in node_types}
        if any(n < 0 for v in num_samples.values() for n in v):
            raise ValueError("num_samples must be non-negative")
        self.num_samples = num_samples
        self.num_hops = max((len(v) for v in num_samples.values()), default=0)

        self._colptr = {et: np.asarray(v) for et, v in self._colptr_dict.items()}
        self._row = {et: np.asarray(v) for et, v in self._row_dict.items()}
        self._perm = {et: np.asarray(v) for et, v in self._perm_dict.items()}
        self._edge_types = list(data.edge_index_dict)

    def _update_budget(
        self,
        budget: dict[str, np.ndarray],
        local: dict[str, np.ndarray],
        node_type: str,
        nodes: np.ndarray,
    ):
        """Adds the neighbors of newly sampled ``nodes`` (of ``node_type``)
        to the budgets of their node types."""
        if nodes.size == 0:
            return
        for edge_type in self._edge_types:
            src, _, dst = edge_type
            if dst != node_type:
                continue
            _, pos, weight = _capped_neighbors(self._colptr[edge_type], nodes)
            neighbors = self._row[edge_type][pos]
            new = local[src][neighbors] < 0  # only nodes not sampled yet
            budget[src] += np.bincount(
                neighbors[new], weights=weight[new], minlength=budget[src].size
            )

    def sample(self, seed_nodes: np.ndarray) -> tuple[dict, dict, dict, dict]:
        """Samples the subgraph of a mini-batch.

        Args:
            seed_nodes: Global ids of the (distinct) seed nodes of
                ``self.input_type``.

        Returns:
            ``(n_id_dict, edge_index_dict, e_id_dict, num_sampled_nodes_dict)``
            as NumPy arrays: the sampled nodes per node type (seeds first),
            the induced edges in local ids, their original edge positions, and
            the number of nodes added per hop.
        """
        node_types = list(self.num_nodes)
        nodes = {t: [] for t in node_types}
        counts = {t: [0] for t in node_types}
        local = {t: np.full(self.num_nodes[t], -1, dtype=np.int64) for t in node_types}
        budget = {t: np.zeros(self.num_nodes[t]) for t in node_types}

        def add(node_type, new):
            local[node_type][new] = sum(len(n) for n in nodes[node_type]) + np.arange(
                new.size
            )
            nodes[node_type].append(new)
            budget[node_type][new] = 0.0

        add(self.input_type, seed_nodes)
        counts[self.input_type][0] = seed_nodes.size
        self._update_budget(budget, local, self.input_type, seed_nodes)

        for hop in range(self.num_hops):
            samples = {}
            for node_type in node_types:
                plan = self.num_samples[node_type]
                k = plan[hop] if hop < len(plan) else 0
                candidates = np.nonzero(budget[node_type] > 0)[0]
                samples[node_type] = _weighted_sample(
                    candidates, budget[node_type][candidates] ** 2, k
                )
                add(node_type, samples[node_type])
                counts[node_type].append(samples[node_type].size)
            if hop < self.num_hops - 1:
                for node_type, new in samples.items():
                    self._update_budget(budget, local, node_type, new)

        n_id = {
            t: np.concatenate(v) if v else np.zeros(0, dtype=np.int64)
            for t, v in nodes.items()
        }

        # Induced subgraph (at most 50 random in-neighbors per destination).
        edge_index, e_id = {}, {}
        for edge_type in self._edge_types:
            src, _, dst = edge_type
            owner, pos, _ = _capped_neighbors(self._colptr[edge_type], n_id[dst])
            src_local = local[src][self._row[edge_type][pos]]
            keep = src_local >= 0
            edge_index[edge_type] = np.stack([src_local[keep], owner[keep]])
            e_id[edge_type] = self._perm[edge_type][pos[keep]]
        return n_id, edge_index, e_id, counts

    def __next__(self) -> HeteroGraphData:
        if self._current_index >= len(self._indices):
            raise StopIteration

        end = min(self._current_index + self.batch_size, len(self._indices))
        positions = self._indices[self._current_index : end]
        self._current_index = end
        seeds = np.asarray(self.input_nodes)[positions].astype(np.int64)
        # Duplicate seeds (e.g. from a `sampler`) are merged.
        _, first = np.unique(seeds, return_index=True)
        seeds = seeds[np.sort(first)]

        n_id, edge_index, e_id, counts = self.sample(seeds)
        edge_index_full = self.data.edge_index_dict
        n_id_dict = {t: index_array(v, self.num_nodes[t]) for t, v in n_id.items()}
        e_id_dict = {
            et: index_array(v, edge_index_full[et].shape[1]) for et, v in e_id.items()
        }
        # local edges in the dtype of the graph's edges
        edge_index_dict = {
            et: mx.array(v, dtype=edge_index_full[et].dtype)
            for et, v in edge_index.items()
        }

        kwargs: dict[str, Any] = {
            "batch_size": int(seeds.size),
            "input_type": self.input_type,
            "n_id_dict": n_id_dict,
            "e_id_dict": e_id_dict,
            "num_sampled_nodes_dict": counts,
            "num_nodes_dict": {t: int(v.size) for t, v in n_id.items()},
        }
        kwargs.update(self._slice_custom_attributes(n_id_dict))
        return HeteroGraphData(
            edge_index_dict=edge_index_dict,
            node_features_dict=self._slice_node_dict(
                self.data.node_features_dict, n_id_dict
            ),
            edge_features_dict=self._slice_edge_dict(
                self.data.edge_features_dict, e_id_dict
            ),
            graph_features=self.data.graph_features,
            node_labels_dict=self._slice_node_dict(
                self.data.node_labels_dict, n_id_dict
            ),
            graph_labels=self.data.graph_labels,
            **kwargs,
        )
