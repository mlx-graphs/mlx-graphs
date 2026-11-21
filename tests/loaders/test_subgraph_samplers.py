import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.loaders import (
    ClusterData,
    ClusterLoader,
    GraphSAINTEdgeSampler,
    GraphSAINTNodeSampler,
    GraphSAINTRandomWalkSampler,
    RandomNodeLoader,
    ShaDowKHopSampler,
    balanced_partition,
)
from mlx_graphs.utils import k_hop_subgraph

from .helpers import check_subgraph, cliques_ring, edge_set, random_graph

# ---------------------------------------------------------------------------
# RandomNodeLoader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shuffle", [False, True])
def test_random_node_loader(shuffle):
    data = random_graph()
    loader = RandomNodeLoader(data, num_parts=4, shuffle=shuffle)
    assert len(loader) == 4
    batches = list(loader)
    assert len(batches) == 4
    all_nodes = np.concatenate([np.asarray(b.n_id) for b in batches])
    assert sorted(all_nodes.tolist()) == list(range(30))
    for b in batches:
        check_subgraph(b, data, induced=True)
        np.testing.assert_array_equal(
            np.asarray(b.node_labels), np.asarray(data.node_labels)[np.asarray(b.n_id)]
        )
    if not shuffle:
        np.testing.assert_array_equal(np.asarray(batches[0].n_id), np.arange(8))


def test_random_node_loader_reshuffles():
    data = random_graph()
    loader = RandomNodeLoader(data, num_parts=3, shuffle=True)
    np.random.seed(0)
    first = [np.asarray(b.n_id).tolist() for b in loader]
    second = [np.asarray(b.n_id).tolist() for b in loader]
    assert first != second
    np.random.seed(0)
    assert first == [np.asarray(b.n_id).tolist() for b in loader]


# ---------------------------------------------------------------------------
# GraphSAINT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda d, **kw: GraphSAINTNodeSampler(d, batch_size=15, **kw),
        lambda d, **kw: GraphSAINTEdgeSampler(d, batch_size=8, **kw),
        lambda d, **kw: GraphSAINTRandomWalkSampler(
            d, batch_size=4, walk_length=3, **kw
        ),
    ],
)
def test_graph_saint_samplers(make):
    data = random_graph()
    np.random.seed(0)
    loader = make(data, num_steps=3, sample_coverage=10)
    assert len(loader) == 3
    batches = list(loader)
    assert len(batches) == 3
    for b in batches:
        n_id = np.asarray(b.n_id)
        assert np.all(np.diff(n_id) > 0)  # unique and sorted
        check_subgraph(b, data, induced=True)
        assert b.node_norm.shape == (n_id.size,)
        assert b.edge_norm.shape == (b.num_edges,)
        np.testing.assert_allclose(np.asarray(b.node_norm), loader.node_norm[n_id])
        np.testing.assert_allclose(
            np.asarray(b.edge_norm), loader.edge_norm[np.asarray(b.e_id)]
        )
    assert np.all(loader.node_norm > 0)
    assert np.all((loader.edge_norm >= 0) & (loader.edge_norm <= 1e4))

    # Without sample coverage, there is no normalization
    b = next(iter(make(data, num_steps=1)))
    assert getattr(b, "node_norm", None) is None


def test_graph_saint_norm_values():
    # Every sample holds every node of this complete graph (with overwhelming
    # probability): each node is counted in all samples, each edge too.
    n = 5
    ei = np.array([(u, v) for u in range(n) for v in range(n) if u != v]).T
    data = GraphData(edge_index=mx.array(ei), node_features=mx.ones((n, 1)))
    np.random.seed(0)
    loader = GraphSAINTNodeSampler(data, batch_size=500, num_steps=2, sample_coverage=4)
    np.testing.assert_allclose(loader.node_norm, 1.0 / n, rtol=1e-6)
    np.testing.assert_allclose(loader.edge_norm, 1.0, rtol=1e-6)


def test_graph_saint_edge_norm_uses_target_count():
    # Samples alternate between {0, 1} and {0}: count(0) = 2 count(1) and
    # count(edge) = count(1). The aggregation weight of an edge is the count
    # of the node aggregating it (its target) over the edge count, so that
    # node 0 receives, in expectation over the samples holding it, exactly
    # its full-graph sum: P(1 sampled | 0 sampled) * 2 = 1.
    class Alternating(GraphSAINTNodeSampler):
        calls = 0

        def _sample_nodes(self):
            self.calls += 1
            return np.array([0, 1]) if self.calls % 2 else np.array([0])

    data = GraphData(
        edge_index=mx.array([[0, 1], [1, 0]]), node_features=mx.ones((2, 1))
    )
    loader = Alternating(data, batch_size=1, num_steps=2, sample_coverage=30)
    # edge 0 -> 1: count(1) / count(edge) = 1; edge 1 -> 0: count(0) / ... = 2
    np.testing.assert_allclose(loader.edge_norm, [1.0, 2.0])
    # node_norm = num_samples / (count * N)
    np.testing.assert_allclose(loader.node_norm * loader.N, [1.0, 2.0])


def test_graph_saint_aggregation_is_unbiased():
    # Star graph: the center is in every sample, a leaf in a few. With the
    # target-count normalization, the center's normalized sum over a sample
    # is an unbiased estimate of the sum over all leaves.
    num_leaves = 20
    leaves = np.arange(1, num_leaves + 1)
    edge_index = np.concatenate(
        [
            np.stack([leaves, np.zeros_like(leaves)]),
            np.stack([np.zeros_like(leaves), leaves]),
        ],
        axis=1,
    )
    data = GraphData(
        edge_index=mx.array(edge_index), node_features=mx.ones((num_leaves + 1, 1))
    )
    np.random.seed(0)
    loader = GraphSAINTEdgeSampler(
        data, batch_size=3, num_steps=10, sample_coverage=200
    )
    x = np.arange(num_leaves + 1, dtype=np.float64)
    total, num_samples = 0.0, 2000
    for _ in range(num_samples):
        _, _, e_id = loader._sample()
        to_center = e_id[edge_index[1, e_id] == 0]
        total += (loader.edge_norm[to_center] * x[edge_index[0, to_center]]).sum()
    np.testing.assert_allclose(total / num_samples, x[leaves].sum(), rtol=0.05)


def test_graph_saint_node_sampler_uses_sources():
    # Only nodes 0 and 1 have out-edges: the node sampler draws sources.
    data = GraphData(
        edge_index=mx.array([[0, 0, 1], [2, 3, 4]]),
        node_features=mx.arange(5).reshape(-1, 1).astype(mx.float32),
    )
    np.random.seed(0)
    for b in GraphSAINTNodeSampler(data, batch_size=50, num_steps=3):
        assert set(np.asarray(b.n_id).tolist()) <= {0, 1}
        assert b.num_edges == 0


def test_graph_saint_random_walk_follows_edges():
    # A directed path: walks only move forward and stop at the end.
    data = GraphData(
        edge_index=mx.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
        node_features=mx.arange(5).reshape(-1, 1).astype(mx.float32),
    )
    np.random.seed(0)
    loader = GraphSAINTRandomWalkSampler(data, batch_size=1, walk_length=2, num_steps=5)
    for b in loader:
        n_id = np.asarray(b.n_id)
        # Contiguous nodes, at most walk_length + 1 of them
        assert n_id.size <= 3
        assert np.all(np.diff(n_id) == 1)
        assert b.num_edges == n_id.size - 1


def test_graph_saint_save_dir(tmp_path):
    data = random_graph()
    np.random.seed(0)
    first = GraphSAINTNodeSampler(
        data, batch_size=10, sample_coverage=5, save_dir=str(tmp_path)
    )
    assert (tmp_path / "graphsaintnodesampler_5.npz").exists()
    np.random.seed(1)
    second = GraphSAINTNodeSampler(
        data, batch_size=10, sample_coverage=5, save_dir=str(tmp_path)
    )
    np.testing.assert_array_equal(first.node_norm, second.node_norm)
    np.testing.assert_array_equal(first.edge_norm, second.edge_norm)

    data.node_norm = mx.ones(30)
    with pytest.raises(ValueError):
        GraphSAINTNodeSampler(data, batch_size=10)


# ---------------------------------------------------------------------------
# ShaDowKHopSampler
# ---------------------------------------------------------------------------


def test_shadow_k_hop_sampler():
    data = random_graph()
    roots_all = np.nonzero(np.asarray(data.train_mask))[0]
    loader = ShaDowKHopSampler(
        data, depth=2, num_neighbors=3, node_idx=data.train_mask, batch_size=4
    )
    assert len(loader) == (roots_all.size + 3) // 4
    seen = []
    original = edge_set(data.edge_index)
    for b in loader:
        n_id = np.asarray(b.n_id)
        ptr = np.asarray(b.ptr)
        batch = np.asarray(b.batch_indices)
        root_n_id = np.asarray(b.root_n_id)
        num_roots = root_n_id.size
        roots = n_id[root_n_id]
        seen.extend(roots.tolist())

        assert ptr[0] == 0 and ptr[-1] == n_id.size and ptr.size == num_roots + 1
        np.testing.assert_array_equal(
            batch, np.repeat(np.arange(num_roots), np.diff(ptr))
        )
        # Each root lies in its own subgraph
        assert np.all((root_n_id >= ptr[:-1]) & (root_n_id < ptr[1:]))
        # Labels are the roots' labels (PyG's `y` semantics)
        np.testing.assert_array_equal(
            np.asarray(b.node_labels), np.asarray(data.node_labels)[roots]
        )
        # Node features / masks follow n_id, edges map back to original edges
        np.testing.assert_array_equal(np.asarray(b.node_features).reshape(-1), n_id)
        ei = np.asarray(b.edge_index)
        e_id = np.asarray(b.e_id)
        np.testing.assert_array_equal(n_id[ei], np.asarray(data.edge_index)[:, e_id])
        # Edges never cross subgraphs
        np.testing.assert_array_equal(batch[ei[0]], batch[ei[1]])

        for g in range(num_roots):
            nodes = n_id[ptr[g] : ptr[g + 1]]
            assert np.all(np.diff(nodes) > 0)
            # The subgraph is induced: every original edge between its nodes
            sub_edges = {(int(n_id[u]), int(n_id[v])) for u, v in ei.T if batch[u] == g}
            node_set = set(nodes.tolist())
            expected = {(u, v) for u, v in original if u in node_set and v in node_set}
            assert sub_edges == expected
            # Nodes lie within `depth` hops (following incoming edges)
            subset, *_ = k_hop_subgraph(
                int(roots[g]), 2, data.edge_index, num_nodes=data.num_nodes
            )
            assert node_set <= set(np.asarray(subset).tolist())
    assert seen == roots_all.tolist()


def test_shadow_full_neighborhood_matches_k_hop_subgraph():
    data = random_graph()
    loader = ShaDowKHopSampler(data, depth=2, num_neighbors=-1, batch_size=3)
    b = next(iter(loader))
    n_id, ptr = np.asarray(b.n_id), np.asarray(b.ptr)
    for g, root in enumerate([0, 1, 2]):
        subset, *_ = k_hop_subgraph(root, 2, data.edge_index, num_nodes=data.num_nodes)
        assert set(n_id[ptr[g] : ptr[g + 1]].tolist()) == set(
            np.asarray(subset).tolist()
        )


def test_shadow_with_replacement_and_shuffle():
    data = random_graph()
    np.random.seed(0)
    loader = ShaDowKHopSampler(
        data, depth=1, num_neighbors=2, replace=True, batch_size=7, shuffle=True
    )
    roots = []
    for b in loader:
        n_id, ptr = np.asarray(b.n_id), np.asarray(b.ptr)
        roots.extend(n_id[np.asarray(b.root_n_id)].tolist())
        # Depth 1 with 2 draws: at most 3 nodes per subgraph
        assert np.all(np.diff(ptr) <= 3)
    assert sorted(roots) == list(range(30))
    assert roots != list(range(30))


# ---------------------------------------------------------------------------
# ClusterData / ClusterLoader
# ---------------------------------------------------------------------------


def _cut(part, edge_index):
    e = np.asarray(edge_index)
    return int((part[e[0]] != part[e[1]]).sum())


def test_balanced_partition_recovers_cliques():
    data = cliques_ring(4, 8)
    np.random.seed(0)
    part = balanced_partition(data.edge_index, data.num_nodes, 4)
    assert np.bincount(part, minlength=4).tolist() == [8, 8, 8, 8]
    # Only the 4 ring edges (8 directed edges) are cut
    assert _cut(part, data.edge_index) == 8


def _grid(k):
    edges = []
    for i in range(k):
        for j in range(k):
            u = i * k + j
            if j + 1 < k:
                edges += [(u, u + 1), (u + 1, u)]
            if i + 1 < k:
                edges += [(u, u + k), (u + k, u)]
    return mx.array(np.array(edges).T)


def test_balanced_partition_is_balanced():
    edge_index = _grid(12)
    np.random.seed(0)
    part = balanced_partition(edge_index, 144, 6)
    sizes = np.bincount(part, minlength=6)
    # Sizes stay within 5% of the average (24)
    assert sizes.sum() == 144 and sizes.min() >= 22 and sizes.max() <= 26
    random_part = np.random.randint(0, 6, 144)
    assert 2 * _cut(part, edge_index) < _cut(random_part, edge_index)
    # Refinement does not increase the cut of the BFS split
    unrefined = balanced_partition(edge_index, 144, 6, num_iters=0)
    assert np.bincount(unrefined).tolist() == [24] * 6
    assert _cut(part, edge_index) <= _cut(unrefined, edge_index)

    # Isolated nodes and empty graphs are handled
    part = balanced_partition(mx.array([[0], [1]]), 5, 2)
    assert sorted(np.bincount(part).tolist()) == [2, 3]


def test_cluster_data():
    data = cliques_ring(4, 8)
    data.train_mask = mx.array(np.arange(32) % 2 == 0)
    np.random.seed(0)
    cluster_data = ClusterData(data, num_parts=4, method="balanced")
    assert len(cluster_data) == 4
    assert cluster_data.method == "balanced"
    assert cluster_data.partptr.tolist() == [0, 8, 16, 24, 32]
    for i in range(4):
        sub = cluster_data[i]
        check_subgraph(sub, data, induced=True)
        assert sub.num_nodes == 8 and sub.num_edges == 56  # one full clique
    assert cluster_data[-1].num_nodes == 8
    with pytest.raises(IndexError):
        cluster_data[4]


def test_cluster_data_custom_partition():
    data = random_graph()
    partition = np.arange(30) % 3
    cluster_data = ClusterData(data, num_parts=10, partition=mx.array(partition))
    assert cluster_data.method == "custom" and len(cluster_data) == 3
    np.testing.assert_array_equal(cluster_data[1].n_id, np.arange(1, 30, 3))
    check_subgraph(cluster_data[1], data, induced=True)

    with pytest.raises(ValueError):
        ClusterData(data, 2, partition=np.zeros(5))
    with pytest.raises(ValueError):
        ClusterData(data, 2, method="spectral")


def test_cluster_loader():
    data = random_graph()
    cluster_data = ClusterData(data, num_parts=5, partition=np.arange(30) % 5)
    loader = ClusterLoader(cluster_data, batch_size=2)
    assert len(loader) == 3
    batches = list(loader)
    assert [b.num_nodes for b in batches] == [12, 12, 6]
    for b in batches:
        # Inter-cluster edges between merged clusters are kept
        check_subgraph(b, data, induced=True)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([np.asarray(b.n_id) for b in batches])), np.arange(30)
    )

    np.random.seed(0)
    shuffled = ClusterLoader(cluster_data, batch_size=1, shuffle=True)
    orders = {tuple(int(b.n_id[0].item()) for b in shuffled) for _ in range(5)}
    assert len(orders) > 1


# ---------------------------------------------------------------------------
# Graphs with as many nodes as edges
# ---------------------------------------------------------------------------


def _ring(num_nodes=6):
    # A directed cycle: as many edges as nodes, so array lengths alone cannot
    # tell node-level from edge-level attributes.
    src = np.arange(num_nodes)
    return GraphData(
        edge_index=mx.array(np.stack([src, (src + 1) % num_nodes]).astype(np.int32)),
        node_features=mx.arange(num_nodes).reshape(-1, 1).astype(mx.float32),
        edge_features=(100 + mx.arange(num_nodes)).reshape(-1, 1).astype(mx.float32),
        edge_labels=200 + mx.arange(num_nodes),
        edge_weight=300 + mx.arange(num_nodes),
        node_labels=mx.arange(num_nodes),
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda d: RandomNodeLoader(d, num_parts=2, shuffle=True),
        lambda d: GraphSAINTNodeSampler(d, batch_size=4, num_steps=3),
        lambda d: GraphSAINTRandomWalkSampler(d, batch_size=2, walk_length=2),
        lambda d: ShaDowKHopSampler(d, depth=2, num_neighbors=2, batch_size=2),
        lambda d: ClusterLoader(
            ClusterData(d, 3, partition=np.array([2, 0, 1, 1, 0, 2])), batch_size=2
        ),
    ],
)
def test_edge_attributes_follow_edges_when_num_nodes_equals_num_edges(make):
    data = _ring()
    np.random.seed(0)
    for b in make(data):
        e_id = np.asarray(b.e_id)
        np.testing.assert_array_equal(np.asarray(b.edge_features)[:, 0], 100 + e_id)
        np.testing.assert_array_equal(np.asarray(b.edge_labels), 200 + e_id)
        np.testing.assert_array_equal(np.asarray(b.edge_weight), 300 + e_id)
        np.testing.assert_array_equal(
            np.asarray(b.node_features)[:, 0], np.asarray(b.n_id)
        )


@pytest.mark.parametrize(
    "make",
    [
        lambda d: RandomNodeLoader(d, num_parts=2),
        lambda d: GraphSAINTNodeSampler(d, batch_size=4, num_steps=1),
        lambda d: ShaDowKHopSampler(d, depth=2, num_neighbors=2, batch_size=2),
        lambda d: ClusterLoader(ClusterData(d, 2, partition=np.arange(6) % 2)),
    ],
)
def test_index_outputs_are_int32(make):
    data = _ring()
    for b in make(data):
        assert b.edge_index.dtype == data.edge_index.dtype
        for key in ("n_id", "e_id", "batch_indices", "ptr", "root_n_id"):
            if hasattr(b, key):
                assert getattr(b, key).dtype == mx.int32, key
