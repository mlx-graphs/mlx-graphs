"""Temporal (and disjoint) neighbor sampling of NeighborLoader and
LinkNeighborLoader, against brute-force references and, when pyg-lib is
installed, against PyG."""

import mlx.core as mx
import numpy as np
import pytest
from scipy.stats import chisquare

from mlx_graphs.data import GraphData, HeteroGraphData
from mlx_graphs.loaders import LinkNeighborLoader, NeighborLoader
from mlx_graphs.loaders._disjoint_sampler import (
    _floyd_offsets,
    disjoint_neighbor_sample,
    segment_upper_bound,
)


def _temporal_graph(num_nodes=120, num_edges=1500, level="node", seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, num_nodes, num_edges)
    dst = rng.integers(0, num_nodes, num_edges)
    size = num_nodes if level == "node" else num_edges
    time = rng.integers(0, 40, size)
    data = GraphData(
        edge_index=mx.array(np.stack([src, dst]).astype(np.int32)),
        node_features=mx.array(rng.standard_normal((num_nodes, 3)).astype(np.float32)),
        time=mx.array(time),
    )
    return data, src, dst, time


def _edge_times(batch, level, node_time_or_edge_time):
    """Time of every sampled edge, and the time of its seed."""
    e_id = np.asarray(batch.e_id)
    n_id = np.asarray(batch.n_id)
    src_local = np.asarray(batch.edge_index[0])
    if level == "node":
        t = node_time_or_edge_time[n_id[src_local]]
    else:
        t = node_time_or_edge_time[e_id]
    seed_of = np.asarray(batch.batch)[src_local]
    return t, seed_of


@pytest.mark.parametrize("level", ["node", "edge"])
@pytest.mark.parametrize("strategy", ["uniform", "last"])
@pytest.mark.parametrize("fanout", [[4, 3], [-1, 2], [2, 2, 2]])
def test_no_sampled_edge_violates_time(level, strategy, fanout):
    data, src, dst, time = _temporal_graph(level=level)
    seeds = np.arange(0, 120, 3)
    input_time = None
    if level == "edge":
        input_time = np.random.default_rng(1).integers(5, 40, seeds.size)
    np.random.seed(0)
    loader = NeighborLoader(
        data,
        num_neighbors=fanout,
        input_nodes=mx.array(seeds),
        batch_size=7,
        time_attr="time",
        input_time=None if input_time is None else mx.array(input_time),
        temporal_strategy=strategy,
    )
    seed_times = time[seeds] if level == "node" else input_time
    start = 0
    total_edges = 0
    for batch in loader:
        b = batch.batch_size
        t, seed_of = _edge_times(batch, level, time)
        assert np.all(t <= seed_times[start : start + b][seed_of])
        # The seeds come first, one per input node, with their own batch id
        assert np.array_equal(np.asarray(batch.n_id[:b]), seeds[start : start + b])
        assert np.array_equal(np.asarray(batch.batch[:b]), np.arange(b))
        # Edges stay inside the subgraph of one seed
        ei = np.asarray(batch.edge_index)
        bv = np.asarray(batch.batch)
        assert np.array_equal(bv[ei[0]], bv[ei[1]])
        # Edges are real edges of the graph
        e_id = np.asarray(batch.e_id)
        n_id = np.asarray(batch.n_id)
        assert np.array_equal(src[e_id], n_id[ei[0]])
        assert np.array_equal(dst[e_id], n_id[ei[1]])
        assert sum(batch.num_sampled_nodes) == batch.num_nodes
        assert sum(batch.num_sampled_edges) == ei.shape[1]
        assert ("seed_time" in batch) == (input_time is not None)
        total_edges += ei.shape[1]
        start += b
    assert total_edges > 0


def _valid_in_neighbors(src, dst, ref_time, node, t, level):
    """In-edges of ``node`` valid at time ``t``, sorted by time (stable)."""
    edges = np.nonzero(dst == node)[0]
    times = ref_time[src[edges]] if level == "node" else ref_time[edges]
    order = np.argsort(times, kind="stable")
    edges, times = edges[order], times[order]
    return edges[times <= t]


@pytest.mark.parametrize("level", ["node", "edge"])
def test_last_strategy_takes_the_latest_neighbors(level):
    data, src, dst, time = _temporal_graph(level=level, seed=3)
    seeds = np.arange(120)
    input_time = np.full(seeds.size, 25) if level == "edge" else None
    k = 3
    loader = NeighborLoader(
        data,
        num_neighbors=[k],
        input_nodes=mx.array(seeds),
        batch_size=120,
        time_attr="time",
        input_time=None if input_time is None else mx.array(input_time),
        temporal_strategy="last",
    )
    batch = next(iter(loader))
    e_id = np.asarray(batch.e_id)
    bv = np.asarray(batch.batch)
    col = np.asarray(batch.edge_index[1])
    for seed in seeds:
        t = time[seed] if level == "node" else 25
        expected = _valid_in_neighbors(src, dst, time, seed, t, level)[-k:]
        got = e_id[(bv[col] == seed)]
        assert np.array_equal(got, expected), seed


def test_uniform_strategy_is_uniform_over_valid_neighbors():
    # A star: node 0 has 30 in-neighbors with times 0..29; seed time 19, so
    # 20 valid neighbors, each sampled with probability 5 / 20.
    num = 30
    src = np.arange(1, num + 1)
    dst = np.zeros(num, dtype=np.int64)
    time = np.concatenate([[19], np.arange(num)])
    data = GraphData(
        edge_index=mx.array(np.stack([src, dst]).astype(np.int32)),
        time=mx.array(time),
        num_nodes=num + 1,
    )
    np.random.seed(0)
    loader = NeighborLoader(
        data,
        num_neighbors=[5],
        input_nodes=mx.zeros(2000, dtype=mx.int32),
        batch_size=500,
        time_attr="time",
    )
    counts = np.zeros(num + 1)
    for batch in loader:
        n_id = np.asarray(batch.n_id)
        ei = np.asarray(batch.edge_index)
        # 5 distinct neighbors per seed (without replacement)
        assert ei.shape[1] == 5 * batch.batch_size
        np.add.at(counts, n_id[ei[0]], 1)
    assert counts[21:].sum() == 0  # times 20..29 are in the future
    assert chisquare(counts[1:21]).pvalue > 1e-3


def test_floyd_offsets_are_distinct_and_uniform():
    np.random.seed(1)
    pop = np.full(20000, 12)
    out = _floyd_offsets(pop, 4).reshape(-1, 4)
    assert np.all(out >= 0) and np.all(out < 12)
    assert all(len(set(row)) == 4 for row in out[:500])
    assert chisquare(np.bincount(out.reshape(-1), minlength=12)).pvalue > 1e-3


def test_segment_upper_bound():
    values = np.array([1, 3, 3, 7, 0, 2, 5, 5, 9], dtype=np.int64)
    starts = np.array([0, 4, 4, 9])
    ends = np.array([4, 9, 4, 9])
    bound = np.array([3, 5, -1, 4])
    assert np.array_equal(
        segment_upper_bound(values, starts, ends, bound), np.array([3, 8, 4, 9])
    )


def test_disjoint_without_time_matches_full_k_hop():
    data, src, dst, _ = _temporal_graph(num_nodes=40, num_edges=120, seed=5)
    loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        input_nodes=mx.array([3, 3, 7]),
        batch_size=3,
        disjoint=True,
    )
    batch = next(iter(loader))
    n_id = np.asarray(batch.n_id)
    bv = np.asarray(batch.batch)
    for b, seed in enumerate([3, 3, 7]):
        hop1 = set(src[dst == seed].tolist())
        hop2 = set(src[np.isin(dst, list(hop1))].tolist())
        assert set(n_id[bv == b].tolist()) == {seed} | hop1 | hop2
        # nodes are unique within each subgraph
        assert len(n_id[bv == b]) == len(set(n_id[bv == b].tolist()))


def test_temporal_errors():
    data, *_ = _temporal_graph(level="edge")
    with pytest.raises(ValueError, match="input_time"):
        NeighborLoader(data, [2], time_attr="time")  # edge-level needs input_time
    node_data, *_ = _temporal_graph(level="node")
    with pytest.raises(ValueError, match="time_attr"):
        NeighborLoader(node_data, [2], input_time=mx.zeros(120))
    with pytest.raises(ValueError, match="temporal_strategy"):
        NeighborLoader(node_data, [2], time_attr="time", temporal_strategy="first")
    with pytest.raises(ValueError, match="no time attribute"):
        NeighborLoader(node_data, [2], time_attr="missing")
    with pytest.raises(ValueError, match="entries"):
        NeighborLoader(node_data, [2], time_attr="time", input_time=mx.zeros(3))
    hetero = HeteroGraphData(
        edge_index_dict={("a", "to", "a"): mx.array([[0], [1]])},
        num_nodes_dict={"a": 2},
    )
    with pytest.raises(NotImplementedError):
        NeighborLoader(hetero, [2], input_nodes="a", time_attr="time")


@pytest.mark.parametrize("neg", [None, "binary", "triplet"])
def test_link_loader_temporal(neg):
    data, src, dst, time = _temporal_graph(seed=7)
    rng = np.random.default_rng(2)
    eli = rng.integers(0, 120, (2, 50))
    elt = rng.integers(10, 40, 50)
    np.random.seed(0)
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[3, 2],
        edge_label_index=mx.array(eli),
        edge_label_time=mx.array(elt),
        time_attr="time",
        neg_sampling=neg,
        batch_size=16,
    )
    for batch in loader:
        input_id = np.asarray(batch.input_id)
        p = input_id.size
        n_id = np.asarray(batch.n_id)
        bv = np.asarray(batch.batch)
        seed_time = elt[input_id][bv]
        ei = np.asarray(batch.edge_index)
        assert np.all(time[n_id[ei[0]]] <= seed_time[ei[0]])
        if neg == "triplet":
            assert np.array_equal(n_id[np.asarray(batch.src_index)], eli[0, input_id])
            assert np.array_equal(
                n_id[np.asarray(batch.dst_pos_index)], eli[1, input_id]
            )
            neg_nodes = n_id[np.asarray(batch.dst_neg_index)]
            assert np.all(time[neg_nodes] <= elt[input_id])
            assert np.array_equal(np.asarray(batch.seed_time), elt[input_id])
        else:
            label_index = np.asarray(batch.edge_label_index)
            num = label_index.shape[1]
            assert np.array_equal(label_index.reshape(-1), np.arange(2 * num))
            assert np.array_equal(n_id[label_index[:, :p]], eli[:, input_id])
            assert np.asarray(batch.edge_label_time).shape == (num,)
            if neg == "binary":
                assert num == 2 * p
                label = np.asarray(batch.edge_label)
                assert label[:p].sum() == p and label[p:].sum() == 0
                neg_nodes = n_id[label_index[:, p:]]
                assert np.all(time[neg_nodes] <= np.asarray(batch.edge_label_time)[p:])


def test_link_loader_requires_time_pair():
    data, *_ = _temporal_graph()
    with pytest.raises(ValueError, match="conflicting"):
        LinkNeighborLoader(data, [2], time_attr="time")
    with pytest.raises(ValueError, match="conflicting"):
        LinkNeighborLoader(data, [2], edge_label_time=mx.zeros(1500))


# PyG reference (needs pyg-lib) ###############################################


@pytest.mark.parametrize("level", ["node", "edge"])
@pytest.mark.parametrize("fanout", [[3, 2], [5, -1]])
def test_last_strategy_matches_pyg(level, fanout):
    pytest.importorskip("pyg_lib")
    torch = pytest.importorskip("torch")
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader as PyGNeighborLoader

    data, src, dst, time = _temporal_graph(level=level, seed=11)
    x = np.asarray(data.node_features)
    seeds = np.arange(0, 120, 2)
    input_time = (
        np.random.default_rng(3).integers(5, 40, seeds.size)
        if level == "edge"
        else None
    )
    ref = Data(
        x=torch.tensor(x),
        edge_index=torch.tensor(np.stack([src, dst])),
        time=torch.tensor(time),
    )
    kwargs = dict(time_attr="time", temporal_strategy="last", batch_size=16)
    pyg_loader = PyGNeighborLoader(
        ref,
        num_neighbors=fanout,
        input_nodes=torch.tensor(seeds),
        input_time=None if input_time is None else torch.tensor(input_time),
        **kwargs,
    )
    loader = NeighborLoader(
        data,
        num_neighbors=fanout,
        input_nodes=mx.array(seeds),
        input_time=None if input_time is None else mx.array(input_time),
        **kwargs,
    )
    for p, m in zip(pyg_loader, loader):
        for key in ("n_id", "batch", "edge_index", "e_id"):
            assert np.array_equal(getattr(p, key).numpy(), np.asarray(getattr(m, key)))
        assert np.array_equal(p.x.numpy(), np.asarray(m.node_features))
        assert list(p.num_sampled_nodes) == m.num_sampled_nodes
        assert list(p.num_sampled_edges) == m.num_sampled_edges


def test_link_last_strategy_matches_pyg():
    pytest.importorskip("pyg_lib")
    torch = pytest.importorskip("torch")
    from torch_geometric.data import Data
    from torch_geometric.loader import LinkNeighborLoader as PyGLinkNeighborLoader

    data, src, dst, time = _temporal_graph(seed=13)
    rng = np.random.default_rng(4)
    eli = rng.integers(0, 120, (2, 60))
    elt = rng.integers(10, 40, 60)
    ref = Data(
        x=torch.tensor(np.asarray(data.node_features)),
        edge_index=torch.tensor(np.stack([src, dst])),
        time=torch.tensor(time),
    )
    kwargs = dict(
        num_neighbors=[4, 3], batch_size=16, time_attr="time", temporal_strategy="last"
    )
    pyg_loader = PyGLinkNeighborLoader(
        ref,
        edge_label_index=torch.tensor(eli),
        edge_label_time=torch.tensor(elt),
        **kwargs,
    )
    loader = LinkNeighborLoader(
        data, edge_label_index=mx.array(eli), edge_label_time=mx.array(elt), **kwargs
    )
    keys = ("n_id", "batch", "edge_index", "e_id", "edge_label_index", "input_id")
    for p, m in zip(pyg_loader, loader):
        for key in keys + ("edge_label_time",):
            assert np.array_equal(getattr(p, key).numpy(), np.asarray(getattr(m, key)))


def test_disjoint_sampler_direct_call():
    colptr = np.array([0, 2, 3, 3])
    row = np.array([1, 2, 0])
    out = disjoint_neighbor_sample(colptr, row, np.array([0, 0]), [-1, -1])
    # both seeds get the same subgraph: 0 <- {1, 2}, 1 <- 0 (already there)
    assert out.node.tolist() == [0, 0, 1, 2, 1, 2]
    assert out.batch.tolist() == [0, 1, 0, 0, 1, 1]
    assert out.num_sampled_nodes == [2, 4, 0]
    assert out.num_sampled_edges == [4, 2]
