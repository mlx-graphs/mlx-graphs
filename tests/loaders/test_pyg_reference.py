"""Loaders compared with PyG's pure-Python code paths (no pyg-lib,
torch_sparse or torch_cluster needed), and exact structural checks of the
sampled subgraphs."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph as pyg_k_hop_subgraph

from mlx_graphs.data import GraphData
from mlx_graphs.loaders import (
    ClusterData,
    ClusterLoader,
    DynamicBatchSampler,
    ImbalancedSampler,
    LinkNeighborLoader,
    NeighborLoader,
    RandomNodeLoader,
    ShaDowKHopSampler,
)

from .helpers import random_graph

pyg_loader = pytest.importorskip("torch_geometric.loader")


def _pyg_data(data: GraphData) -> Data:
    """PyG copy of ``data`` with node ids as ``x`` and edge ids as ``edge_attr``."""
    return Data(
        edge_index=torch.from_numpy(np.asarray(data.edge_index).astype(np.int64)),
        x=torch.arange(data.num_nodes).view(-1, 1),
        edge_attr=torch.arange(data.num_edges),
        num_nodes=data.num_nodes,
    )


@pytest.mark.parametrize("num_parts", [1, 3, 4, 7])
def test_random_node_loader_matches_pyg(num_parts):
    data = random_graph(40, 200, seed=3)
    pyg = pyg_loader.RandomNodeLoader(_pyg_data(data), num_parts=num_parts)
    ours = RandomNodeLoader(data, num_parts=num_parts)
    assert len(ours) == len(pyg)
    for b, p in zip(ours, pyg):
        n_id = np.asarray(b.n_id)
        np.testing.assert_array_equal(n_id, p.x.view(-1).numpy())
        # Same edges (by original id) with the same local endpoints
        e_id = np.asarray(b.e_id)
        order = np.argsort(e_id)
        np.testing.assert_array_equal(e_id[order], p.edge_attr.numpy())
        np.testing.assert_array_equal(
            np.asarray(b.edge_index)[:, order], p.edge_index.numpy()
        )


def test_cluster_loader_matches_pyg(monkeypatch):
    data = random_graph(40, 200, seed=4)
    partition = np.random.RandomState(0).randint(0, 6, 40)
    # PyG's ClusterData needs METIS (pyg-lib / torch_sparse) only to compute
    # the partition: give it ours, the rest is plain PyTorch.
    monkeypatch.setattr(
        pyg_loader.ClusterData,
        "_metis",
        lambda self, edge_index, num_nodes: torch.from_numpy(partition),
    )
    pyg_cd = pyg_loader.ClusterData(_pyg_data(data), num_parts=6, log=False)
    cluster_data = ClusterData(data, num_parts=6, partition=partition)

    for i in range(6):
        b, p = cluster_data[i], pyg_cd[i]
        assert set(np.asarray(b.n_id).tolist()) == set(p.x.view(-1).tolist())
        assert sorted(np.asarray(b.e_id).tolist()) == sorted(p.edge_attr.tolist())

    for batch_size in (1, 2, 4):
        pyg = pyg_loader.ClusterLoader(pyg_cd, batch_size=batch_size)
        ours = ClusterLoader(cluster_data, batch_size=batch_size)
        assert len(ours) == len(pyg)
        for b, p in zip(ours, pyg):
            n_id = np.asarray(b.n_id)
            assert set(n_id.tolist()) == set(p.x.view(-1).tolist())
            assert sorted(np.asarray(b.e_id).tolist()) == sorted(p.edge_attr.tolist())
            # Same global edges
            ours_edges = set(map(tuple, n_id[np.asarray(b.edge_index)].T.tolist()))
            pyg_edges = set(map(tuple, p.x.view(-1)[p.edge_index].T.tolist()))
            assert ours_edges == pyg_edges


@pytest.mark.parametrize("mode", ["node", "edge"])
@pytest.mark.parametrize("num_steps", [None, 3])
def test_dynamic_batch_sampler_matches_pyg(mode, num_steps):
    rng = np.random.RandomState(0)
    sizes = rng.randint(1, 12, 40)
    graphs = [
        GraphData(
            edge_index=mx.zeros((2, int(n)), dtype=mx.int32),
            node_features=mx.zeros((int(n), 1)),
        )
        for n in sizes
    ]
    pyg_graphs = [
        Data(edge_index=torch.zeros(2, int(n), dtype=torch.long), num_nodes=int(n))
        for n in sizes
    ]
    # A graph larger than the budget is yielded alone
    max_num = 10
    ours = list(DynamicBatchSampler(graphs, max_num, mode, num_steps=num_steps))
    pyg = list(
        iter(
            pyg_loader.DynamicBatchSampler(
                pyg_graphs, max_num, mode, num_steps=num_steps
            )
        )
    )
    assert ours == pyg
    for b in ours:
        assert sizes[b].sum() <= max_num or len(b) == 1
    if num_steps is None:
        assert sorted(i for b in ours for i in b) == list(range(40))


def test_dynamic_batch_sampler_skip_too_big():
    # PyG re-yields graphs after a skipped one (its `num_processed` does not
    # count skipped graphs); here every graph that fits is yielded once.
    sizes = [3, 10, 2, 4, 12, 5]
    graphs = [
        GraphData(
            edge_index=mx.zeros((2, 0), dtype=mx.int32), node_features=mx.zeros((n, 1))
        )
        for n in sizes
    ]
    batches = list(DynamicBatchSampler(graphs, 8, skip_too_big=True))
    assert batches == [[0, 2], [3], [5]]


def test_imbalanced_sampler_matches_pyg():
    y = np.random.RandomState(0).choice(4, 50, p=[0.6, 0.25, 0.1, 0.05])
    mask = np.random.RandomState(1).rand(50) < 0.7
    data = GraphData(
        edge_index=mx.zeros((2, 0), dtype=mx.int32),
        node_features=mx.zeros((50, 1)),
        node_labels=mx.array(y),
    )
    pyg = pyg_loader.ImbalancedSampler(
        Data(y=torch.from_numpy(y), num_nodes=50),
        input_nodes=torch.from_numpy(np.nonzero(mask)[0]),
    )
    ours = ImbalancedSampler(data, input_nodes=mx.array(mask))
    np.testing.assert_allclose(ours.weight, pyg.weights.numpy(), rtol=1e-6)
    assert len(ours) == len(pyg)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_shadow_full_neighborhood_matches_pyg_k_hop_subgraph(depth):
    """With every neighbor kept, each ShaDow subgraph is PyG's
    ``k_hop_subgraph`` of its root: same nodes, same relabeled edges."""
    data = random_graph(40, 100, seed=5)
    edge_index = torch.from_numpy(np.asarray(data.edge_index).astype(np.int64))
    loader = ShaDowKHopSampler(data, depth=depth, num_neighbors=-1, batch_size=7)
    for b in loader:
        n_id = np.asarray(b.n_id)
        ptr = np.asarray(b.ptr)
        batch = np.asarray(b.batch_indices)
        ei = np.asarray(b.edge_index)
        e_id = np.asarray(b.e_id)
        root_n_id = np.asarray(b.root_n_id)
        np.testing.assert_array_equal(
            batch, np.repeat(np.arange(ptr.size - 1), np.diff(ptr))
        )
        for g in range(ptr.size - 1):
            root = int(n_id[root_n_id[g]])
            subset, sub_ei, mapping, edge_mask = pyg_k_hop_subgraph(
                root, depth, edge_index, relabel_nodes=True, num_nodes=40
            )
            np.testing.assert_array_equal(n_id[ptr[g] : ptr[g + 1]], subset.numpy())
            assert root_n_id[g] - ptr[g] == int(mapping)
            in_g = batch[ei[0]] == g
            ours = sorted(zip(e_id[in_g].tolist(), (ei[:, in_g] - ptr[g]).T.tolist()))
            expected_ids = np.nonzero(edge_mask.numpy())[0]
            expected = sorted(zip(expected_ids.tolist(), sub_ei.numpy().T.tolist()))
            assert ours == expected


def _hops(n_id, edge_index, num_seeds, num_hops):
    """Hop of every sampled node, from the order of ``n_id`` (seeds, then
    the nodes discovered at hop 1, hop 2, ...) and the sampled edges."""
    hop = np.full(n_id.size, -1)
    hop[:num_seeds] = 0
    end = num_seeds
    for h in range(1, num_hops + 1):
        sources = edge_index[0, hop[edge_index[1]] == h - 1]
        new = np.unique(sources[hop[sources] < 0])
        # Newly discovered nodes are contiguous, right after the previous hop
        np.testing.assert_array_equal(new, np.arange(end, end + new.size))
        hop[new] = h
        end += new.size
    assert end == n_id.size
    return hop


@pytest.mark.parametrize("replace", [False, True])
def test_neighbor_loader_per_hop_fanout(replace):
    data = random_graph(100, 900, seed=6)
    original = np.asarray(data.edge_index)
    in_deg = np.bincount(original[1], minlength=100)
    num_neighbors = [5, 3, 2]
    loader = NeighborLoader(
        data, num_neighbors, batch_size=8, shuffle=True, replace=replace
    )
    seen = []
    for b in loader:
        n_id = np.asarray(b.n_id)
        ei = np.asarray(b.edge_index)
        seen += n_id[: b.batch_size].tolist()
        np.testing.assert_array_equal(n_id[ei], original[:, np.asarray(b.e_id)])
        hop = _hops(n_id, ei, b.batch_size, len(num_neighbors))
        # Edges always go from hop h + 1 (or an earlier node) to hop h, and
        # every node of hop h < K has min(fanout, in-degree) sampled in-edges
        fan_in = np.bincount(ei[1], minlength=n_id.size)
        for h, k in enumerate(num_neighbors):
            nodes = np.nonzero(hop == h)[0]
            if replace:
                expected = np.where(in_deg[n_id[nodes]] > 0, k, 0)
            else:
                expected = np.minimum(k, in_deg[n_id[nodes]])
            np.testing.assert_array_equal(fan_in[nodes], expected)
        assert np.all(fan_in[hop == len(num_neighbors)] == 0)
        if not replace:
            assert len(set(map(tuple, ei.T.tolist()))) == ei.shape[1]
    assert sorted(seen) == list(range(100))


def test_neighbor_loader_full_fanout_matches_pyg_k_hop_subgraph():
    data = random_graph(60, 240, seed=7)
    original = np.asarray(data.edge_index)
    edge_index = torch.from_numpy(original.astype(np.int64))
    loader = NeighborLoader(data, [-1, -1], batch_size=1)
    for b in loader:
        seed = int(b.n_id[0].item())
        subset, *_ = pyg_k_hop_subgraph(seed, 2, edge_index, num_nodes=60)
        inner, *_ = pyg_k_hop_subgraph(seed, 1, edge_index, num_nodes=60)
        assert set(np.asarray(b.n_id).tolist()) == set(subset.tolist())
        # Sampled edges: every in-edge of the nodes within 1 hop
        expected = np.nonzero(np.isin(original[1], inner.numpy()))[0]
        assert sorted(np.asarray(b.e_id).tolist()) == expected.tolist()


@pytest.mark.parametrize("amount", [1, 2.5])
def test_link_neighbor_loader_binary_labels(amount):
    data = random_graph(50, 300, seed=8)
    original = np.asarray(data.edge_index)
    loader = LinkNeighborLoader(
        data, [3], neg_sampling_ratio=amount, batch_size=32, shuffle=True
    )
    num_pos = num_neg = 0
    seen = []
    for b in loader:
        n_id = np.asarray(b.n_id)
        label_index = n_id[np.asarray(b.edge_label_index)]
        label = np.asarray(b.edge_label)
        input_id = np.asarray(b.input_id)
        seen += input_id.tolist()
        pos = input_id.size
        assert label.size == pos + int(np.ceil(pos * amount))
        assert np.all(label[:pos] == 1) and np.all(label[pos:] == 0)
        np.testing.assert_array_equal(label_index[:, :pos], original[:, input_id])
        # All endpoints (negatives included) are seeds: first in n_id
        seeds = np.unique(label_index)
        np.testing.assert_array_equal(np.sort(n_id[: seeds.size]), seeds)
        num_pos += pos
        num_neg += label.size - pos
    assert sorted(seen) == list(range(300))
    assert num_neg == sum(int(np.ceil(n * amount)) for n in [32] * 9 + [12])


def test_neighbor_loader_boolean_mask_input_nodes():
    # Like PyG, a boolean mask selects the seeds (it used to be cast to 0 / 1
    # node ids).
    data = random_graph()
    mask = np.asarray(data.train_mask)
    loader = NeighborLoader(data, [3], input_nodes=data.train_mask, batch_size=4)
    assert len(loader) == (mask.sum() + 3) // 4
    seeds = np.concatenate([np.asarray(b.n_id)[: b.batch_size] for b in loader])
    np.testing.assert_array_equal(seeds, np.nonzero(mask)[0])

    # With an ImbalancedSampler built on the same mask (its docstring example)
    sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
    loader = NeighborLoader(
        data, [3], input_nodes=data.train_mask, batch_size=4, sampler=sampler
    )
    seeds = np.concatenate([np.asarray(b.n_id)[: b.batch_size] for b in loader])
    assert set(seeds.tolist()) <= set(np.nonzero(mask)[0].tolist())

    with pytest.raises(ValueError):
        NeighborLoader(data, [3], input_nodes=mx.array(mask[:10]))
