"""Synthetic benchmarks with ground-truth explanations."""

from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData

# House motif as in PyG's HouseMotif: a square (0-1-2-3) with a roof (4).
_HOUSE_EDGES = np.array([[0, 0, 0, 1, 1, 2], [1, 3, 4, 4, 2, 3]])
_HOUSE_LABELS = np.array([1, 1, 2, 2, 3])


def _barabasi_albert(num_nodes: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Undirected BA edges as a ``[2, E]`` array of unique pairs (i < j)."""
    targets = list(range(m))
    repeated: list[int] = []
    edges = set()
    for source in range(m, num_nodes):
        for t in set(targets):
            edges.add((min(source, t), max(source, t)))
        repeated.extend(targets)
        repeated.extend([source] * m)
        targets = []
        while len(set(targets)) < m:
            targets.append(int(rng.choice(repeated)))
        targets = list(set(targets))
    return np.array(sorted(edges)).T


def ba_shapes(
    num_base_nodes: int = 300,
    num_motifs: int = 80,
    edges_per_node: int = 5,
    num_node_features: int = 10,
    seed: Optional[int] = 0,
) -> GraphData:
    """BA-Shapes from the GNNExplainer paper.

    A Barabasi-Albert graph with ``num_motifs`` "house" motifs attached to
    random nodes. Node labels: 0 for base nodes, and 1/2/3 for the middle,
    bottom and top nodes of a house. Node features are constant, so a model
    can only classify nodes from the graph structure.

    Returns:
        A :class:`~mlx_graphs.data.GraphData` with ``edge_index`` (undirected,
        both directions), ``node_features``, ``node_labels`` and
        ``edge_groundtruth`` (1 for edges inside a house motif, else 0).
    """
    rng = np.random.default_rng(seed)
    base = _barabasi_albert(num_base_nodes, edges_per_node, rng)

    pairs = [base]
    truth = [np.zeros(base.shape[1], dtype=np.int32)]
    labels = [np.zeros(num_base_nodes, dtype=np.int32)]
    offset = num_base_nodes
    for _ in range(num_motifs):
        pairs.append(_HOUSE_EDGES + offset)
        truth.append(np.ones(_HOUSE_EDGES.shape[1], dtype=np.int32))
        labels.append(_HOUSE_LABELS)
        anchor = int(rng.integers(num_base_nodes))
        pairs.append(np.array([[anchor], [offset]]))  # attach the house
        truth.append(np.zeros(1, dtype=np.int32))
        offset += 5

    one_way = np.concatenate(pairs, axis=1)
    one_way_truth = np.concatenate(truth)
    edge_index = np.concatenate([one_way, one_way[::-1]], axis=1)
    edge_truth = np.concatenate([one_way_truth, one_way_truth])

    return GraphData(
        edge_index=mx.array(edge_index.astype(np.int32)),
        node_features=mx.ones((offset, num_node_features)),
        node_labels=mx.array(np.concatenate(labels)),
        edge_groundtruth=mx.array(edge_truth),
    )
