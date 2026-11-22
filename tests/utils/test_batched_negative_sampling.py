import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils.negative_sampling import (
    batched_negative_sampling,
    negative_sampling,
)


def batch_of_graphs(seed: int, sizes=(5, 7, 4), edges_per_graph=8):
    rng = np.random.default_rng(seed)
    edges, batch, offset = [], [], 0
    for graph, n in enumerate(sizes):
        edges.append(rng.integers(0, n, size=(2, edges_per_graph)) + offset)
        batch.append(np.full(n, graph))
        offset += n
    return np.concatenate(edges, axis=1).astype(np.int32), np.concatenate(batch)


def edge_set(edges: np.ndarray) -> set:
    return set(map(tuple, edges.T.tolist()))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("force_undirected", [False, True])
def test_batched_negative_sampling(seed, force_undirected):
    edges, batch = batch_of_graphs(seed)
    mx.random.seed(seed)
    neg = batched_negative_sampling(
        mx.array(edges), mx.array(batch), force_undirected=force_undirected
    )
    neg_np = np.asarray(neg)
    assert neg.dtype == mx.int32

    # Negatives stay within one graph, are not edges, nor self-loops.
    assert np.array_equal(batch[neg_np[0]], batch[neg_np[1]])
    assert not (edge_set(neg_np) & edge_set(edges))
    assert np.all(neg_np[0] != neg_np[1])
    assert len(edge_set(neg_np)) == neg_np.shape[1]
    # Grouped by graph.
    assert np.all(np.diff(batch[neg_np[0]]) >= 0)
    if force_undirected:
        assert edge_set(neg_np) == edge_set(neg_np[::-1])
    else:
        # One negative per edge when the graph has enough non-edges.
        for graph in range(3):
            num_edges = int((batch[edges[0]] == graph).sum())
            n = int((batch == graph).sum())
            num_non_edges = n * (n - 1) - len(
                {(s, d) for s, d in edge_set(edges) if s != d and batch[s] == graph}
            )
            assert (batch[neg_np[0]] == graph).sum() == min(num_edges, num_non_edges)


def test_batched_negative_sampling_num_samples_and_reproducibility():
    edges, batch = batch_of_graphs(0, sizes=(10, 10), edges_per_graph=5)
    mx.random.seed(3)
    a = batched_negative_sampling(mx.array(edges), mx.array(batch), 3)
    mx.random.seed(3)
    b = batched_negative_sampling(mx.array(edges), mx.array(batch), 3)
    assert mx.array_equal(a, b)
    assert a.shape == (2, 6)

    c = batched_negative_sampling(mx.array(edges), mx.array(batch), 0.4)
    assert c.shape == (2, 4)


def test_batched_negative_sampling_bipartite():
    rng = np.random.default_rng(0)
    src_batch = np.array([0, 0, 0, 1, 1])
    dst_batch = np.array([0, 0, 1, 1, 1, 1])
    edges = np.array([[0, 1, 2, 3, 4, 4], [0, 1, 1, 2, 3, 5]])
    neg = np.asarray(
        batched_negative_sampling(
            mx.array(edges), (mx.array(src_batch), mx.array(dst_batch))
        )
    )
    assert neg.shape == (2, 6)
    assert np.array_equal(src_batch[neg[0]], dst_batch[neg[1]])
    assert not (edge_set(neg) & edge_set(edges))
    del rng


def test_negative_sampling_bipartite():
    edges = np.array([[0, 0, 1, 2], [0, 3, 1, 2]])
    mx.random.seed(0)
    neg = np.asarray(negative_sampling(mx.array(edges), (3, 4), 100))
    # Every non-edge of the 3 x 4 bipartite graph, including (i, i) pairs.
    assert neg.shape == (2, 8)
    assert np.all(neg[0] < 3) and np.all(neg[1] < 4)
    assert not (edge_set(neg) & edge_set(edges))
    assert len(edge_set(neg)) == 8

    with pytest.raises(ValueError):
        negative_sampling(mx.array(edges), (2, 4))
    empty = negative_sampling(mx.array([[0], [0]]), (1, 1))
    assert empty.shape == (2, 0)


def test_batched_negative_sampling_one_host_generator(monkeypatch):
    """A single generator for all the graphs (each seeding from mx.random
    syncs with the GPU), and the negatives of every graph stay uniform over
    its non-edges."""
    import sys

    module = sys.modules["mlx_graphs.utils.negative_sampling"]

    calls = []
    host_rng = module._host_rng
    monkeypatch.setattr(module, "_host_rng", lambda: calls.append(1) or host_rng())
    edges, batch = batch_of_graphs(0, sizes=(5, 7, 4, 6))
    batched_negative_sampling(mx.array(edges), mx.array(batch))
    assert len(calls) == 1

    # graph 0: non-edges (0, 2), (1, 0), (2, 0), (2, 1); 2 negatives per draw
    edges = np.array([[0, 1, 3, 4, 5], [1, 2, 4, 5, 3]])
    batch = np.array([0, 0, 0, 1, 1, 1])
    counts = {}
    for seed in range(400):
        mx.random.seed(seed)
        neg = np.asarray(batched_negative_sampling(mx.array(edges), mx.array(batch)))
        for pair in map(tuple, neg.T.tolist()):
            counts[pair] = counts.get(pair, 0) + 1
    assert {pair: counts[pair] for pair in [(3, 5), (4, 3), (5, 4)]} == {
        (3, 5): 400, (4, 3): 400, (5, 4): 400  # graph 1: its only non-edges
    }  # fmt: skip
    for pair in [(0, 2), (1, 0), (2, 0), (2, 1)]:
        assert 160 < counts[pair] < 240  # expected 200, std 10
