"""The NumPy / MLX fallbacks of mlx_graphs.sampling (used without mlx_cluster,
e.g. on Linux) against mlx_cluster and against the sampling distributions."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs import sampling


def _mlx_cluster():
    # Imported directly so the comparison also runs when mlx-graphs itself is
    # told to ignore mlx_cluster (MLX_GRAPHS_DISABLE_MLX_CLUSTER=1).
    return pytest.importorskip("mlx_cluster")


def _graph(num_nodes=60, num_edges=500, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, num_nodes, num_edges)
    dst = rng.integers(0, num_nodes, num_edges)
    return src.astype(np.int64), dst.astype(np.int64)


def _compressed(index, other, num_nodes):
    """(ptr, values) of the CSR (index = src) or CSC (index = dst) layout."""
    order = np.argsort(index, kind="stable")
    ptr = np.concatenate([[0], np.cumsum(np.bincount(index, minlength=num_nodes))])
    return mx.array(ptr.astype(np.int64)), mx.array(other[order])


def _assert_equal(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# ---------------------------------------------------------------------------
# Homogeneous neighbor sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("num_neighbors", [[-1], [-1, -1], [1000, 1000]])
def test_neighbor_sample_matches_mlx_cluster(directed, num_neighbors):
    """Taking all neighbors is deterministic, so both give the same output."""
    mlx_cluster = _mlx_cluster()
    src, dst = _graph()
    colptr, row = _compressed(dst, src, 60)
    seeds = mx.array([3, 7, 7, 11, 59], dtype=mx.int64)  # with a duplicate
    expected = mlx_cluster.neighbor_sample(
        colptr, row, seeds, num_neighbors, False, directed
    )
    actual = sampling._neighbor_sample_numpy(
        colptr, row, seeds, num_neighbors, False, directed
    )
    _assert_equal(actual, expected)


@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("num_neighbors", [[3, 2], [5], [12, 0, 4]])
def test_neighbor_sample_fallback_is_valid(replace, num_neighbors):
    src, dst = _graph(num_nodes=80, num_edges=900, seed=1)
    colptr, row = _compressed(dst, src, 80)
    ptr, rows_csc = np.asarray(colptr), np.asarray(row)
    seeds = np.array([0, 5, 9, 33, 70])
    samples, rows, cols, edges = (
        np.asarray(a)
        for a in sampling._neighbor_sample_numpy(
            colptr, row, mx.array(seeds), num_neighbors, replace
        )
    )
    np.testing.assert_array_equal(samples[: seeds.size], seeds)
    assert np.unique(samples).size == samples.size
    # Every sampled edge is an edge of the graph, between the right nodes.
    np.testing.assert_array_equal(samples[rows], rows_csc[edges])
    targets = samples[cols]
    assert np.all((ptr[targets] <= edges) & (edges < ptr[targets + 1]))

    # Per hop, each frontier node gets min(k, degree) distinct edges (or k
    # edges with replacement).
    begin, end = 0, seeds.size
    for k in num_neighbors:
        for i in range(begin, end):
            mine = edges[cols == i]
            deg = ptr[samples[i] + 1] - ptr[samples[i]]
            want = 0 if deg == 0 or k == 0 else (k if replace else min(k, deg))
            assert mine.size == want
            if not replace:
                assert np.unique(mine).size == mine.size
        # Nodes first reached at this hop get the next local ids, in order.
        hop_rows = rows[(cols >= begin) & (cols < end)]
        new = np.unique(hop_rows[hop_rows >= end])
        np.testing.assert_array_equal(new, np.arange(end, end + new.size))
        begin, end = end, end + new.size
    assert end == samples.size


@pytest.mark.parametrize("degree, k", [(5, 3), (12, 3), (40, 7)])
def test_sampling_without_replacement_is_uniform(degree, k):
    """Each neighbor is picked with probability k / degree, for the small-
    and large-degree branches of the sampler."""
    # Node 0 has incoming edges from every node, including itself.
    colptr = mx.array([0] + [degree] * degree, dtype=mx.int64)
    row = mx.arange(degree, dtype=mx.int64)
    trials = 2000
    counts = np.zeros(degree)
    mx.random.seed(0)
    for _ in range(trials):
        _, _, _, edges = sampling._neighbor_sample_numpy(
            colptr, row, mx.array([0], dtype=mx.int64), [k]
        )
        edges = np.asarray(edges)
        assert np.unique(edges).size == k
        counts[edges] += 1
    np.testing.assert_allclose(counts / trials, k / degree, atol=0.05)


def test_neighbor_sample_fallback_is_reproducible():
    src, dst = _graph()
    colptr, row = _compressed(dst, src, 60)
    seeds = mx.array([1, 2, 3], dtype=mx.int64)
    mx.random.seed(3)
    a = sampling._neighbor_sample_numpy(colptr, row, seeds, [4, 4])
    mx.random.seed(3)
    b = sampling._neighbor_sample_numpy(colptr, row, seeds, [4, 4])
    _assert_equal(a, b)


# ---------------------------------------------------------------------------
# Heterogeneous neighbor sampling
# ---------------------------------------------------------------------------

NUM_NODES = {"author": 30, "paper": 50, "venue": 4}


def _hetero_csc():
    rng = np.random.default_rng(5)
    edges = {
        ("author", "writes", "paper"): (
            rng.integers(0, 30, 120),
            rng.integers(0, 50, 120),
        ),
        ("paper", "cites", "paper"): (
            rng.integers(0, 50, 200),
            rng.integers(0, 50, 200),
        ),
        ("paper", "in", "venue"): (np.arange(50), rng.integers(0, 4, 50)),
        ("paper", "written_by", "author"): None,
    }
    w = edges[("author", "writes", "paper")]
    edges[("paper", "written_by", "author")] = (w[1], w[0])
    colptr, row = {}, {}
    for et, (s, d) in edges.items():
        colptr[et], row[et] = _compressed(d, s.astype(np.int64), NUM_NODES[et[2]])
    return colptr, row


@pytest.mark.parametrize(
    "num_neighbors",
    [
        [-1, -1],
        [100],
        {
            ("author", "writes", "paper"): [-1, 0],
            ("paper", "cites", "paper"): [-1, -1],
            ("paper", "in", "venue"): [0, -1],
            ("paper", "written_by", "author"): [-1],
        },
    ],
)
def test_hetero_neighbor_sample_matches_mlx_cluster(num_neighbors):
    mlx_cluster = _mlx_cluster()
    colptr, row = _hetero_csc()
    seeds = {"paper": mx.array([4, 1, 17], dtype=mx.int64)}
    kwargs = dict(num_nodes_dict=NUM_NODES)
    expected = mlx_cluster.hetero_neighbor_sample(
        colptr, row, seeds, num_neighbors, **kwargs
    )
    actual = sampling._hetero_neighbor_sample_numpy(
        colptr, row, seeds, num_neighbors, **kwargs
    )
    for field in expected._fields:
        want, got = getattr(expected, field), getattr(actual, field)
        assert set(want) == set(got), field
        for key in want:
            np.testing.assert_array_equal(np.asarray(got[key]), np.asarray(want[key]))


@pytest.mark.parametrize("replace", [False, True])
def test_hetero_neighbor_sample_fallback_is_valid(replace):
    colptr, row = _hetero_csc()
    seeds = {"venue": mx.array([0, 2], dtype=mx.int64)}
    k = 3
    out = sampling._hetero_neighbor_sample_numpy(
        colptr, row, seeds, [k, k], num_nodes_dict=NUM_NODES, replace=replace
    )
    node_id = {t: np.asarray(v) for t, v in out.node_id.items()}
    for t, ids in node_id.items():
        assert np.unique(ids).size == ids.size
        assert sum(out.num_sampled_nodes[t]) == ids.size
    np.testing.assert_array_equal(node_id["venue"][:2], [0, 2])
    for et in colptr:
        src_t, _, dst_t = et
        ptr, rows_csc = np.asarray(colptr[et]), np.asarray(row[et])
        r, c, e = (np.asarray(a) for a in (out.row[et], out.col[et], out.edge_id[et]))
        assert sum(out.num_sampled_edges[et]) == e.size
        np.testing.assert_array_equal(node_id[src_t][r], rows_csc[e])
        targets = node_id[dst_t][c]
        assert np.all((ptr[targets] <= e) & (e < ptr[targets + 1]))
        for target in np.unique(c):
            mine = e[c == target]
            deg = ptr[node_id[dst_t][target] + 1] - ptr[node_id[dst_t][target]]
            assert mine.size == (k if replace else min(k, deg))


def test_hetero_neighbor_sample_fallback_errors():
    colptr, row = _hetero_csc()
    with pytest.raises(ValueError, match="duplicate"):
        sampling._hetero_neighbor_sample_numpy(
            colptr, row, {"paper": mx.array([1, 1])}, [2]
        )
    with pytest.raises(ValueError, match="directed"):
        sampling._hetero_neighbor_sample_numpy(
            colptr, row, {"paper": mx.array([1])}, [2], directed=False
        )
    with pytest.raises(IndexError):
        sampling._hetero_neighbor_sample_numpy(
            colptr, row, {"paper": mx.array([50])}, [2]
        )


# ---------------------------------------------------------------------------
# Random walks
# ---------------------------------------------------------------------------


def _devices():
    return [mx.cpu] + ([mx.gpu] if mx.metal.is_available() else [])


@pytest.mark.parametrize("device", _devices())
def test_random_walk_matches_mlx_cluster(device):
    mlx_cluster = _mlx_cluster()
    src, dst = _graph(num_nodes=40, num_edges=90)  # some nodes have no edges
    rowptr, col = _compressed(src, dst, 40)
    start = mx.arange(40, dtype=mx.int64)
    rand = mx.random.uniform(shape=(40, 12), key=mx.random.key(1))
    rand = rand.at[0].add(1.0 - rand[0])  # rand == 1.0 is clamped
    expected = mlx_cluster.random_walk(rowptr, col, start, rand, 12, stream=device)
    _assert_equal(sampling._random_walk_mlx(rowptr, col, start, rand, 12), expected)


def test_random_walk_fallback_on_edgeless_graph():
    nodes, edges = sampling._random_walk_mlx(
        mx.zeros((4,), mx.int64),
        mx.zeros((0,), mx.int64),
        mx.arange(3),
        mx.zeros((3, 2)),
        2,
    )
    np.testing.assert_array_equal(nodes, [[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    np.testing.assert_array_equal(edges, -1)


METAPATH = [
    ("author", "writes", "paper"),
    ("paper", "in", "venue"),
    ("venue", "has", "paper"),
    ("paper", "written_by", "author"),
]


def _metapath_edges():
    rng = np.random.default_rng(2)
    writes = mx.array(np.stack([rng.integers(0, 9, 25), rng.integers(0, 15, 25)]))
    paper_in = mx.array(np.stack([np.arange(14), rng.integers(0, 3, 14)]))
    return {
        ("author", "writes", "paper"): writes,
        ("paper", "written_by", "author"): writes[::-1],
        ("paper", "in", "venue"): paper_in,
        ("venue", "has", "paper"): paper_in[::-1],
    }


@pytest.mark.parametrize("device", _devices())
def test_metapath_random_walk_matches_mlx_cluster(device):
    mlx_cluster = _mlx_cluster()
    edge_index_dict = _metapath_edges()
    num_nodes = {"author": 10, "paper": 15}  # author 9 and paper 14+ are isolated
    expected_csr = mlx_cluster.build_metapath_csr(edge_index_dict, METAPATH, num_nodes)
    csr = sampling._build_metapath_csr_numpy(edge_index_dict, METAPATH, num_nodes)
    for field in expected_csr._fields:
        want, got = getattr(expected_csr, field), getattr(csr, field)
        if isinstance(want, mx.array):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        else:
            assert got == want, field
    assert sampling.metapath_node_types(METAPATH, 9) == mlx_cluster.metapath_node_types(
        METAPATH, 9
    )

    start = mx.arange(10, dtype=mx.int64)
    rand = mx.random.uniform(shape=(10, 9), key=mx.random.key(4))
    expected = mlx_cluster.metapath_random_walk(
        expected_csr, start, 9, rand, stream=device
    )
    actual = sampling._metapath_random_walk_mlx(csr, start, 9, rand)
    _assert_equal(actual, expected)
    assert (np.asarray(actual[0]) == -1).any()  # some walks hit a dead end


def _node2vec_transition(rowptr, col, t, v, p, q):
    """Exact probability of each next node x after the step t -> v."""
    neighbors = lambda u: col[rowptr[u] : rowptr[u + 1]]  # noqa: E731
    weights = {}
    for x in neighbors(v):
        w = 1 / p if x == t else (1.0 if x in neighbors(t) else 1 / q)
        weights[int(x)] = weights.get(int(x), 0) + w
    total = sum(weights.values())
    return {x: w / total for x, w in weights.items()}


@pytest.mark.parametrize("p, q", [(0.25, 4.0), (4.0, 0.25), (1.0, 1.0)])
def test_rejection_sampling_fallback_follows_node2vec(p, q):
    # Node 0 - 1 - {2, 3, 4}, with 2 - 0 and 3 - 4 also connected.
    edges = [(0, 1), (1, 2), (1, 3), (1, 4), (2, 0), (3, 4), (0, 5)]
    edges += [(b, a) for a, b in edges]
    src, dst = (np.array(x, dtype=np.int64) for x in zip(*edges))
    rowptr, col = _compressed(src, dst, 6)
    rp, cl = np.asarray(rowptr), np.asarray(col)
    walks = 20000
    start = mx.zeros((walks,), dtype=mx.int64)

    nodes, edge_ids = sampling._rejection_sampling_numpy(
        rowptr, col, start, 3, p, q, key=mx.random.key(0)
    )
    nodes, edge_ids = np.asarray(nodes), np.asarray(edge_ids)
    # Every step follows an edge.
    np.testing.assert_array_equal(cl[edge_ids], nodes[:, 1:])
    assert np.all(rp[nodes[:, :-1]] <= edge_ids)
    assert np.all(edge_ids < rp[nodes[:, :-1] + 1])

    # Second step after 0 -> 1 follows the node2vec transition probabilities.
    after = nodes[nodes[:, 1] == 1][:, 2]
    expected = _node2vec_transition(rp, cl, 0, 1, p, q)
    for x, prob in expected.items():
        assert abs(np.mean(after == x) - prob) < 0.02, (x, prob)


def test_rejection_sampling_fallback_key_is_reproducible():
    src, dst = _graph(num_nodes=20, num_edges=80)
    rowptr, col = _compressed(src, dst, 20)
    start = mx.arange(20, dtype=mx.int64)
    a = sampling._rejection_sampling_numpy(
        rowptr, col, start, 5, 2.0, 0.5, key=mx.random.key(7)
    )
    b = sampling._rejection_sampling_numpy(
        rowptr, col, start, 5, 2.0, 0.5, key=mx.random.key(7)
    )
    _assert_equal(a, b)
