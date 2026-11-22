from collections import Counter

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_graphs.data import GraphData, HeteroGraphData
from mlx_graphs.loaders import HGTLoader
from mlx_graphs.nn import GraphSAGE, to_hetero

WRITES = ("author", "writes", "paper")
REV = ("paper", "rev_writes", "author")
CITES = ("paper", "cites", "paper")


def make_graph(seed=0, num_authors=30, num_papers=40, num_edges=120):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, num_authors, num_edges)
    dst = rng.integers(0, num_papers, num_edges)
    pairs = np.unique(np.stack([src, dst], 1), axis=0)
    writes = pairs.T
    cites = np.unique(rng.integers(0, num_papers, (60, 2)), axis=0).T
    return HeteroGraphData(
        edge_index_dict={
            WRITES: mx.array(writes),
            REV: mx.array(writes[::-1].copy()),
            CITES: mx.array(cites),
        },
        node_features_dict={
            "author": mx.array(rng.standard_normal((num_authors, 4)), mx.float32),
            "paper": mx.array(rng.standard_normal((num_papers, 6)), mx.float32),
        },
        edge_features_dict={WRITES: mx.arange(writes.shape[1]).reshape(-1, 1)},
        node_labels_dict={"author": mx.array(rng.integers(0, 3, num_authors))},
        author_train_mask=mx.array(rng.random(num_authors) < 0.5),
    )


def test_batch_structure():
    np.random.seed(0)
    data = make_graph()
    loader = HGTLoader(
        data,
        num_samples=[5, 5],
        input_nodes=("author", data.author_train_mask),
        batch_size=4,
        shuffle=True,
    )
    num_seeds = int(data.author_train_mask.sum().item())
    assert len(loader) == (num_seeds + 3) // 4
    seen = []
    full = {et: np.array(v) for et, v in data.edge_index_dict.items()}
    for batch in loader:
        n = batch.batch_size
        n_id = {t: np.array(v) for t, v in batch.n_id_dict.items()}
        seeds = n_id["author"][:n]
        seen.extend(seeds.tolist())
        assert np.array(data.author_train_mask)[seeds].all()
        for t, ids in n_id.items():
            assert len(np.unique(ids)) == len(ids)
            # at most num_samples new nodes per hop
            counts = batch.num_sampled_nodes_dict[t]
            assert sum(counts) == len(ids)
            assert all(c <= 5 for c in counts[1:])
        assert batch.num_sampled_nodes_dict["author"][0] == n
        # features, labels and masks are sliced
        assert mx.array_equal(
            batch.node_features_dict["paper"],
            data.node_features_dict["paper"][batch.n_id_dict["paper"]],
        )
        assert mx.array_equal(
            batch.node_labels_dict["author"],
            data.node_labels_dict["author"][batch.n_id_dict["author"]],
        )
        assert batch.author_train_mask.shape[0] == len(n_id["author"])
        # all edges are original edges, identified by e_id
        for et, ei in batch.edge_index_dict.items():
            src, _, dst = et
            ei = np.array(ei)
            e_id = np.array(batch.e_id_dict[et])
            assert np.array_equal(n_id[src][ei[0]], full[et][0][e_id])
            assert np.array_equal(n_id[dst][ei[1]], full[et][1][e_id])
            # the subgraph induced by the sampled nodes (all degrees <= 50)
            mask = np.isin(full[et][0], n_id[src]) & np.isin(full[et][1], n_id[dst])
            assert ei.shape[1] == mask.sum()
        assert mx.array_equal(
            batch.edge_features_dict[WRITES].reshape(-1),
            batch.e_id_dict[WRITES].astype(mx.int32),
        )
    assert sorted(seen) == sorted(np.nonzero(np.array(data.author_train_mask))[0])


def test_large_budget_takes_all_reachable_nodes():
    """With budgets larger than the candidates, every node within reach of the
    seeds (along edges pointing towards sampled nodes) is sampled."""
    data = make_graph(seed=1)
    loader = HGTLoader(data, num_samples=[1000], input_nodes=("author", None))
    batch = next(iter(loader))
    seed = int(batch.n_id_dict["author"][0].item())
    rev = np.array(data.edge_index_dict[REV])
    # one hop: the papers pointing to the seed author (rev_writes edges); no
    # edge type points from authors to authors
    expected_papers = set(rev[0][rev[1] == seed].tolist())
    expected_authors = {seed}
    assert set(np.array(batch.n_id_dict["paper"]).tolist()) == expected_papers
    assert set(np.array(batch.n_id_dict["author"]).tolist()) == expected_authors


def test_per_type_budgets_and_missing_hops():
    data = make_graph(seed=2)
    loader = HGTLoader(
        data,
        num_samples={"author": [3], "paper": [2, 4]},
        input_nodes="author",
        batch_size=2,
    )
    batch = next(iter(loader))
    assert batch.num_sampled_nodes_dict["author"][1:] == [
        batch.num_sampled_nodes_dict["author"][1],
        0,
    ]
    assert batch.num_sampled_nodes_dict["author"][1] <= 3
    assert batch.num_sampled_nodes_dict["paper"][1] <= 2
    assert batch.num_sampled_nodes_dict["paper"][2] <= 4


def test_high_degree_nodes_are_capped():
    # paper 0 is cited by 80 papers: at most 50 of them are kept as edges
    cites = np.stack([np.arange(1, 81), np.zeros(80, dtype=np.int64)])
    data = HeteroGraphData(
        edge_index_dict={CITES: mx.array(cites)},
        node_features_dict={"paper": mx.ones((81, 2))},
    )
    loader = HGTLoader(data, num_samples=[100], input_nodes=("paper", mx.array([0])))
    batch = next(iter(loader))
    # 50 neighbors enter the budget, all of them are sampled
    assert len(np.array(batch.n_id_dict["paper"])) == 51
    # the subgraph keeps the edges to another random subset of 50 neighbors
    # (as in PyG), those of them that were sampled
    num_edges = batch.edge_index_dict[CITES].shape[1]
    assert 0 < num_edges <= 50
    assert np.all(np.array(batch.edge_index_dict[CITES][1]) == 0)


def _reference_hgt_sample(colptr, row, edge_types, num_nodes, seeds, input_type, k):
    """Straight transcription of torch-sparse's ``hgt_sample_cpu`` (dict
    budgets, ``choice`` without replacement), for statistical comparison."""
    nodes = {t: [] for t in num_nodes}
    local = {t: {} for t in num_nodes}
    budget = {t: {} for t in num_nodes}

    def update(node_type, samples):
        for et in edge_types:
            src, _, dst = et
            if dst != node_type:
                continue
            for w in samples:
                start, end = colptr[et][w], colptr[et][w + 1]
                neigh = row[et][start:end]
                if end - start > 50:
                    neigh = neigh[np.random.choice(end - start, 50, replace=False)]
                    inv = 1 / 50
                elif end > start:
                    inv = 1 / (end - start)
                for v in neigh:
                    if v not in local[src]:
                        budget[src][v] = budget[src].get(v, 0.0) + inv

    for v in seeds:
        local[input_type][v] = len(nodes[input_type])
        nodes[input_type].append(v)
    update(input_type, seeds)
    for hop in range(len(k)):
        samples_dict = {}
        for t in num_nodes:
            keys = list(budget[t])
            if len(keys) <= k[hop]:
                samples = keys
            else:
                w = np.array([budget[t][v] ** 2 for v in keys])
                samples = list(
                    np.array(keys)[
                        np.random.choice(
                            len(keys), k[hop], replace=False, p=w / w.sum()
                        )
                    ]
                )
            samples_dict[t] = samples
            for v in samples:
                local[t][v] = len(nodes[t])
                nodes[t].append(v)
                del budget[t][v]
        if hop < len(k) - 1:
            for t, samples in samples_dict.items():
                update(t, samples)
    return nodes


def test_sampling_distribution_matches_reference():
    """Inclusion frequencies of every node match a direct transcription of
    PyG's (torch-sparse) algorithm."""
    data = make_graph(seed=3, num_authors=12, num_papers=15, num_edges=40)
    loader = HGTLoader(data, num_samples=[3, 3], input_nodes=("author", mx.array([0])))
    colptr = {et: np.asarray(v) for et, v in loader._colptr_dict.items()}
    row = {et: np.asarray(v) for et, v in loader._row_dict.items()}
    runs = 3000
    ours, ref = Counter(), Counter()
    np.random.seed(0)
    for _ in range(runs):
        n_id, *_ = loader.sample(np.array([0]))
        ours.update((t, int(v)) for t, ids in n_id.items() for v in ids)
        nodes = _reference_hgt_sample(
            colptr,
            row,
            list(data.edge_index_dict),
            data.num_nodes,
            [0],
            "author",
            [3, 3],
        )
        ref.update((t, int(v)) for t, ids in nodes.items() for v in ids)
    keys = set(ours) | set(ref)
    diffs = [abs(ours[k] - ref[k]) / runs for k in keys]
    assert max(diffs) < 0.06, max(diffs)
    assert sum(ours.values()) == pytest.approx(sum(ref.values()), rel=0.02)


def test_errors():
    data = make_graph()
    with pytest.raises(TypeError):
        HGTLoader(
            GraphData(edge_index=mx.array([[0], [1]])), num_samples=[2], input_nodes="a"
        )
    with pytest.raises(ValueError, match="Unknown node types"):
        HGTLoader(data, num_samples={"venue": [2]}, input_nodes="author")
    with pytest.raises(ValueError, match="seed node type"):
        HGTLoader(data, num_samples=[2], input_nodes=None)
    # 30 authors: out-of-range or negative seeds are rejected up front
    for seeds in ([0, 30], [-1]):
        with pytest.raises(IndexError, match="author"):
            HGTLoader(data, num_samples=[2], input_nodes=("author", mx.array(seeds)))
    HGTLoader(data, num_samples=[2], input_nodes=("author", mx.array([0, 29])))


def test_training_with_to_hetero():
    np.random.seed(0)
    data = make_graph(seed=4)
    loader = HGTLoader(
        data, num_samples=[8, 8], input_nodes=("author", None), batch_size=8
    )
    model = to_hetero(
        GraphSAGE(16, 16, num_layers=2, out_features_dim=3),
        data.metadata(),
        input_dims=data.num_node_features,
    )

    def loss_fn(model, batch):
        out = model(batch.edge_index_dict, batch.node_features_dict)["author"]
        n = batch.batch_size
        return nn.losses.cross_entropy(
            out[:n], batch.node_labels_dict["author"][:n], reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for batch in loader:
        loss, _ = loss_and_grad(model, batch)
        assert np.isfinite(loss.item())


def test_capped_neighbors_are_distinct_and_uniform():
    from mlx_graphs.loaders.hgt_loader import MAX_NEIGHBORS, _capped_neighbors

    np.random.seed(1)
    degs = np.array([0, 3, 50, 51, 199, 200, 201, 5000])
    colptr = np.concatenate([[0], np.cumsum(degs)])
    owner, pos, weight = _capped_neighbors(colptr, np.arange(len(degs)))
    assert np.bincount(owner, minlength=len(degs)).tolist() == [
        0,
        3,
        50,
        50,
        50,
        50,
        50,
        50,
    ]
    for i in range(len(degs)):
        p = pos[owner == i]
        assert len(np.unique(p)) == len(p)
        assert np.all((p >= colptr[i]) & (p < colptr[i + 1]))
    assert np.allclose(weight[owner == 1], 1 / 3)
    assert np.allclose(weight[owner == 7], 1 / MAX_NEIGHBORS)

    # every neighbor of a high-degree node is kept with probability 50 / deg
    for deg in (120, 1000):
        counts = np.zeros(deg)
        for _ in range(2000):
            _, pos, _ = _capped_neighbors(np.array([0, deg]), np.array([0]))
            counts[pos] += 1
        freq = counts / 2000
        assert abs(freq.mean() - 50 / deg) < 1e-9
        assert np.abs(freq - 50 / deg).max() < 5 * np.sqrt(50 / deg / 2000)


def test_index_outputs_are_int32():
    data = make_graph()
    for b in HGTLoader(data, num_samples=[4, 4], input_nodes="author", batch_size=8):
        for d in (b.n_id_dict, b.e_id_dict):
            assert all(v.dtype == mx.int32 for v in d.values())
        for key, v in b.edge_index_dict.items():
            assert v.dtype == data.edge_index_dict[key].dtype
