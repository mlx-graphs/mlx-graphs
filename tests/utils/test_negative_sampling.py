import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils.negative_sampling import (
    _kth_missing,
    negative_sampling,
    structured_negative_sampling,
)


def random_edges(seed: int, num_nodes: int, num_edges: int) -> np.ndarray:
    """Random edges with duplicates and self-loops."""
    return np.random.default_rng(seed).integers(0, num_nodes, size=(2, num_edges))


def edge_set(edges) -> set:
    edges = np.asarray(edges)
    return set(zip(edges[0].tolist(), edges[1].tolist()))


def all_non_edges(edges: np.ndarray, num_nodes: int) -> set:
    existing = edge_set(edges)
    return {
        (i, j)
        for i in range(num_nodes)
        for j in range(num_nodes)
        if i != j and (i, j) not in existing
    }


def test_kth_missing_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        present = np.unique(rng.integers(0, 60, size=rng.integers(0, 40)))
        missing = np.setdiff1d(np.arange(200), present)
        ranks = rng.permutation(100)
        assert np.array_equal(_kth_missing(present, ranks), missing[ranks])


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("num_neg_samples", [None, 7, 0.5, 2.0])
def test_negative_sampling_invariants(seed, num_neg_samples):
    num_nodes, num_edges = 20, 60
    edges = random_edges(seed, num_nodes, num_edges)
    mx.random.seed(seed)

    negatives = negative_sampling(
        mx.array(edges), num_nodes=num_nodes, num_neg_samples=num_neg_samples
    )

    if num_neg_samples is None:
        expected = num_edges
    elif isinstance(num_neg_samples, float):
        expected = int(num_neg_samples * num_edges)
    else:
        expected = num_neg_samples
    assert negatives.shape == (2, expected)

    negatives_np = np.asarray(negatives)
    assert negatives_np.min() >= 0 and negatives_np.max() < num_nodes
    assert not np.any(negatives_np[0] == negatives_np[1]), "self-loop sampled"
    assert edge_set(negatives).isdisjoint(edge_set(edges)), "existing edge sampled"
    assert len(edge_set(negatives)) == expected, "duplicate negative edges"


@pytest.mark.parametrize("dtype", [mx.uint32, mx.int32, mx.int64, mx.uint64])
def test_negative_sampling_preserves_dtype(dtype):
    edge_index = mx.array([[0, 1, 2], [1, 2, 3]], dtype=dtype)
    assert negative_sampling(edge_index).dtype == dtype
    assert negative_sampling(edge_index, force_undirected=True).dtype == dtype
    for array in structured_negative_sampling(edge_index):
        assert array.dtype == dtype

    complete = mx.array([[0, 1], [1, 0]], dtype=dtype)
    empty = negative_sampling(complete)
    assert empty.shape == (2, 0)
    assert empty.dtype == dtype


def test_negative_sampling_reproducible_with_seed():
    edge_index = mx.array(random_edges(0, 50, 100))

    mx.random.seed(42)
    first = negative_sampling(edge_index)
    mx.random.seed(42)
    second = negative_sampling(edge_index)
    mx.random.seed(43)
    third = negative_sampling(edge_index)

    assert mx.array_equal(first, second)
    assert not mx.array_equal(first, third)


@pytest.mark.parametrize("seed", range(3))
def test_negative_sampling_dense_graph_returns_all_non_edges(seed):
    # Asking for more negatives than there are non-edges returns every
    # non-edge exactly once.
    num_nodes = 9
    edges = random_edges(seed, num_nodes, 60)
    expected = all_non_edges(edges, num_nodes)
    assert 0 < len(expected) < 60

    mx.random.seed(seed)
    negatives = negative_sampling(mx.array(edges), num_nodes, num_neg_samples=1000)
    assert negatives.shape == (2, len(expected))
    assert edge_set(negatives) == expected


def test_negative_sampling_complete_graph_is_empty():
    num_nodes = 6
    src, dst = np.nonzero(~np.eye(num_nodes, dtype=bool))
    complete = mx.array(np.stack([src, dst]))

    assert negative_sampling(complete).shape == (2, 0)
    assert negative_sampling(complete, force_undirected=True).shape == (2, 0)
    # Self-loops are never negatives, even when they are the only non-edges.
    assert negative_sampling(complete, num_neg_samples=10).shape == (2, 0)


def test_negative_sampling_covers_all_non_edges():
    # Every non-edge can be sampled, and nothing else.
    num_nodes = 5
    edges = np.array([[0, 1, 2, 3, 0], [1, 2, 3, 4, 0]])
    expected = all_non_edges(edges, num_nodes)

    mx.random.seed(0)
    seen = set()
    for _ in range(300):
        seen |= edge_set(negative_sampling(mx.array(edges), num_neg_samples=1))
    assert seen == expected


def test_negative_sampling_isolated_nodes_and_empty_edges():
    # Nodes 3 and 4 are isolated: they are only known through `num_nodes`.
    edge_index = mx.array([[0, 1], [1, 2]])
    mx.random.seed(0)
    negatives = negative_sampling(edge_index, num_nodes=5, num_neg_samples=18)
    assert negatives.shape == (2, 18)  # 5 * 4 - 2 non-edges
    assert int(negatives.max().item()) == 4

    no_edges = mx.zeros((2, 0), dtype=mx.int32)
    assert negative_sampling(no_edges).shape == (2, 0)
    assert negative_sampling(no_edges, num_nodes=4).shape == (2, 0)
    negatives = negative_sampling(no_edges, num_nodes=4, num_neg_samples=5)
    assert negatives.shape == (2, 5)
    assert len(edge_set(negatives)) == 5

    with pytest.raises(ValueError):
        negative_sampling(edge_index, num_nodes=2)


@pytest.mark.parametrize("seed", range(5))
def test_negative_sampling_force_undirected(seed):
    num_nodes = 15
    edges = random_edges(seed, num_nodes, 40)
    mx.random.seed(seed)

    negatives = negative_sampling(
        mx.array(edges), num_nodes, num_neg_samples=21, force_undirected=True
    )
    assert negatives.shape == (2, 20)  # 2 * (21 // 2)

    sampled = edge_set(negatives)
    assert len(sampled) == 20
    assert {(j, i) for i, j in sampled} == sampled, "not symmetric"
    assert all(i != j for i, j in sampled)
    existing = edge_set(edges)
    existing |= {(j, i) for i, j in existing}
    assert sampled.isdisjoint(existing), "an existing edge (or its reverse) sampled"


def test_negative_sampling_force_undirected_dense_graph():
    # 4 nodes, 6 undirected pairs, 4 of them taken: 2 pairs left.
    edge_index = mx.array([[0, 2, 1, 3], [1, 0, 2, 2]])
    mx.random.seed(0)
    negatives = negative_sampling(edge_index, num_neg_samples=50, force_undirected=True)
    assert edge_set(negatives) == {(0, 3), (3, 0), (1, 3), (3, 1)}
    assert negatives.shape == (2, 4)


def test_negative_sampling_large_sparse_graph():
    # The population (num_nodes ** 2) is far too large to enumerate.
    num_nodes, num_edges = 1_000_000, 20_000
    edges = random_edges(0, num_nodes, num_edges)
    mx.random.seed(0)

    negatives = negative_sampling(mx.array(edges), num_nodes)
    assert negatives.shape == (2, num_edges)
    negatives_np = np.asarray(negatives).astype(np.int64)
    assert not np.any(negatives_np[0] == negatives_np[1])
    keys = negatives_np[0] * num_nodes + negatives_np[1]
    assert np.unique(keys).size == num_edges
    assert not np.isin(keys, edges[0] * num_nodes + edges[1]).any()
    # Sources are spread over the whole range of nodes.
    assert negatives_np[0].min() < 0.01 * num_nodes
    assert negatives_np[0].max() > 0.99 * num_nodes


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("contains_neg_self_loops", [True, False])
def test_structured_negative_sampling_invariants(seed, contains_neg_self_loops):
    num_nodes, num_edges = 20, 80
    edges = random_edges(seed, num_nodes, num_edges)
    mx.random.seed(seed)

    src, dst_pos, dst_neg = structured_negative_sampling(
        mx.array(edges), num_nodes, contains_neg_self_loops=contains_neg_self_loops
    )

    # No source is connected to every node here: the output is aligned with
    # the input edges.
    assert np.array_equal(np.asarray(src), edges[0])
    assert np.array_equal(np.asarray(dst_pos), edges[1])
    assert dst_neg.shape == (num_edges,)

    dst_neg_np = np.asarray(dst_neg)
    assert dst_neg_np.min() >= 0 and dst_neg_np.max() < num_nodes
    negatives = set(zip(edges[0].tolist(), dst_neg_np.tolist()))
    assert negatives.isdisjoint(edge_set(edges)), "existing edge sampled"
    if not contains_neg_self_loops:
        assert not np.any(dst_neg_np == edges[0])


def test_structured_negative_sampling_covers_all_non_neighbors():
    # Node 0 is connected to 1 and 2 out of 5 nodes.
    edge_index = mx.array([[0, 0], [1, 2]])

    mx.random.seed(0)
    with_loops, without_loops = set(), set()
    for _ in range(100):
        with_loops |= set(structured_negative_sampling(edge_index, 5)[2].tolist())
        without_loops |= set(
            structured_negative_sampling(edge_index, 5, contains_neg_self_loops=False)[
                2
            ].tolist()
        )
    assert with_loops == {0, 3, 4}
    assert without_loops == {3, 4}


def test_structured_negative_sampling_dense_graph_terminates():
    # Node 0 is connected to every node (itself included): its edges have no
    # valid negative and are dropped. Node 1 can only get node 2.
    edge_index = mx.array([[0, 0, 0, 1, 1], [0, 1, 2, 0, 1]])
    src, dst_pos, dst_neg = structured_negative_sampling(edge_index, num_nodes=3)
    assert src.tolist() == [1, 1]
    assert dst_pos.tolist() == [0, 1]
    assert dst_neg.tolist() == [2, 2]

    # Complete graph with self-loops: nothing can be sampled.
    src_np, dst_np = np.nonzero(np.ones((4, 4), dtype=bool))
    complete = mx.array(np.stack([src_np, dst_np]))
    for array in structured_negative_sampling(complete):
        assert array.shape == (0,)

    # Without self-loops as negatives, a node connected to all the others is
    # infeasible even if it has no self-loop.
    star = mx.array([[0, 0, 1], [1, 2, 0]])
    src, dst_pos, dst_neg = structured_negative_sampling(
        star, contains_neg_self_loops=False
    )
    assert src.tolist() == [1]
    assert dst_pos.tolist() == [0]
    assert dst_neg.tolist() == [2]


def test_structured_negative_sampling_empty_and_seed():
    no_edges = mx.zeros((2, 0), dtype=mx.int32)
    for array in structured_negative_sampling(no_edges, num_nodes=3):
        assert array.shape == (0,)

    edge_index = mx.array(random_edges(0, 30, 60))
    mx.random.seed(7)
    first = structured_negative_sampling(edge_index)[2]
    mx.random.seed(7)
    second = structured_negative_sampling(edge_index)[2]
    assert mx.array_equal(first, second)
