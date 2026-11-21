import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData, GraphDataBatch, unbatch
from mlx_graphs.loaders import (
    DataListLoader,
    Dataloader,
    DenseDataLoader,
    DynamicBatchSampler,
    ImbalancedSampler,
)


def _graphs(n=10):
    return [
        GraphData(
            edge_index=mx.array([[0, 1], [1, 0]]),
            node_features=mx.full((2 + i % 3, 1), i).astype(mx.float32),
            graph_labels=mx.array([i % 2]),
        )
        for i in range(n)
    ]


def test_drop_last():
    graphs = _graphs(10)
    loader = Dataloader(graphs, batch_size=3, drop_last=True)
    batches = list(loader)
    assert len(loader) == 3 == len(batches)
    assert all(b.num_graphs == 3 for b in batches)

    loader = Dataloader(graphs, batch_size=3)
    assert len(loader) == 4 == len(list(loader))

    # An exact multiple is not affected
    assert len(list(Dataloader(graphs, batch_size=5, drop_last=True))) == 2


def test_follow_batch():
    graphs = _graphs(4)
    loader = Dataloader(graphs, batch_size=4, follow_batch=["node_features"])
    b = next(iter(loader))
    sizes = [g.node_features.shape[0] for g in graphs]
    expected = np.repeat(np.arange(4), sizes)
    np.testing.assert_array_equal(np.asarray(b.node_features_batch), expected)
    # Equal to the batch vector derived from edge_index for nodes
    np.testing.assert_array_equal(np.asarray(b.batch_indices), expected)

    # Unbatching still works and returns the original graphs
    back = unbatch(b)
    for g, orig in zip(back, graphs):
        np.testing.assert_array_equal(
            np.asarray(g.node_features), np.asarray(orig.node_features)
        )

    with pytest.raises(ValueError):
        next(iter(Dataloader(graphs, follow_batch=["missing"])))


def test_follow_batch_vectors_are_not_unbatched():
    # As in PyG, `<attr>_batch` vectors belong to the batch only: unbatched
    # graphs have the attributes of the original ones, so they can be batched
    # again, alone or with other graphs.
    graphs = _graphs(4)
    follow = ["node_features", "x", "edge_index"]
    b = next(iter(Dataloader(graphs, batch_size=4, follow_batch=follow)))
    assert b.edge_index_batch.shape == (b.edge_index.shape[1],)
    back = unbatch(b)
    assert set(back[0].to_dict()) == set(graphs[0].to_dict())
    rebatched = GraphDataBatch(back[:2] + graphs[2:])
    assert mx.array_equal(rebatched.edge_index, b.edge_index)


def test_follow_batch_pyg_aliases():
    # `x` / `edge_attr` are aliases of node_features / edge_features, not
    # stored attributes; following them gives `x_batch` / `edge_attr_batch`.
    graphs = _graphs(4)
    for g in graphs:
        g.edge_features = mx.ones((g.edge_index.shape[1], 2))
    b = next(iter(Dataloader(graphs, batch_size=4, follow_batch=["x", "edge_attr"])))
    node_sizes = [g.node_features.shape[0] for g in graphs]
    edge_sizes = [g.edge_index.shape[1] for g in graphs]
    np.testing.assert_array_equal(
        np.asarray(b.x_batch), np.repeat(np.arange(4), node_sizes)
    )
    np.testing.assert_array_equal(
        np.asarray(b.edge_attr_batch), np.repeat(np.arange(4), edge_sizes)
    )
    assert len(unbatch(b)) == 4


def test_transform_and_collate_fn():
    graphs = _graphs(4)

    def double(g):
        return GraphData(edge_index=g.edge_index, node_features=g.node_features * 2)

    b = next(iter(Dataloader(graphs, batch_size=2, transform=double)))
    assert b.node_features[0].item() == 0 and b.node_features[-1].item() == 2

    out = next(iter(Dataloader(graphs, batch_size=2, collate_fn=len)))
    assert out == 2

    with pytest.raises(ValueError):
        Dataloader(graphs, collate_fn=len, follow_batch=["node_features"])


def test_data_list_loader():
    graphs = _graphs(5)
    batches = list(DataListLoader(graphs, batch_size=2))
    assert [len(b) for b in batches] == [2, 2, 1]
    assert all(isinstance(g, GraphData) for b in batches for g in b)
    assert batches[0][1] is graphs[1]


def test_dense_data_loader():
    graphs = [
        GraphData(
            edge_index=mx.zeros((2, 0), dtype=mx.int32),
            node_features=mx.full((5, 3), i).astype(mx.float32),
            adj=mx.eye(5) * i,
        )
        for i in range(6)
    ]
    batches = list(DenseDataLoader(graphs, batch_size=4))
    assert batches[0].adj.shape == (4, 5, 5)
    assert batches[0].node_features.shape == (4, 5, 3)
    assert batches[1].adj.shape == (2, 5, 5)
    assert batches[1].adj[1, 0, 0].item() == 5

    graphs.append(
        GraphData(
            edge_index=mx.zeros((2, 0), dtype=mx.int32),
            node_features=mx.ones((4, 3)),
            adj=mx.eye(4),
        )
    )
    with pytest.raises(ValueError):
        list(DenseDataLoader(graphs, batch_size=7))


def test_imbalanced_sampler_graph_level():
    # 18 graphs of class 0, 2 of class 1
    graphs = [
        GraphData(
            edge_index=mx.array([[0], [0]]),
            node_features=mx.ones((1, 1)),
            graph_labels=mx.array([int(i >= 18)]),
        )
        for i in range(20)
    ]
    sampler = ImbalancedSampler(graphs, num_samples=2000)
    np.testing.assert_allclose(sampler.weight[:18], 1 / 18)
    np.testing.assert_allclose(sampler.weight[18:], 1 / 2)
    assert len(sampler) == 2000

    np.random.seed(0)
    indices = np.array(list(sampler))
    frac_minority = np.mean(indices >= 18)
    assert 0.4 < frac_minority < 0.6

    np.random.seed(0)
    loader = Dataloader(graphs, batch_size=100, sampler=sampler)
    assert len(loader) == 20
    labels = np.concatenate([np.asarray(b.graph_labels).reshape(-1) for b in loader])
    assert labels.size == 2000
    assert 0.4 < labels.mean() < 0.6

    with pytest.raises(ValueError):
        Dataloader(graphs, sampler=sampler, shuffle=True)


def test_imbalanced_sampler_inputs():
    labels = mx.array([0, 0, 0, 1, 2, 2])
    sampler = ImbalancedSampler(labels, input_nodes=mx.array([0, 1, 3]))
    np.testing.assert_allclose(sampler.weight, [0.5, 0.5, 1.0])
    assert len(sampler) == 3
    assert set(sampler) <= {0, 1, 2}

    mask = mx.array([True, False, True, True, False, False])
    sampler = ImbalancedSampler(labels, input_nodes=mask)
    np.testing.assert_allclose(sampler.weight, [0.5, 0.5, 1.0])

    with pytest.raises(ValueError):
        ImbalancedSampler(mx.array([0.5, 1.0]))


def test_dynamic_batch_sampler():
    sizes = [3, 5, 2, 8, 1, 4, 12, 2]
    graphs = [
        GraphData(edge_index=mx.zeros((2, 0), dtype=mx.int32), num_nodes=n)
        for n in sizes
    ]
    sampler = DynamicBatchSampler(graphs, max_num=10)
    batches = list(sampler)
    assert sorted(i for b in batches for i in b) == list(range(len(sizes)))
    for b in batches:
        total = sum(sizes[i] for i in b)
        assert total <= 10 or len(b) == 1
    # The graph larger than max_num is alone in its batch
    assert [6] in batches
    assert batches[0] == [0, 1, 2]

    skipped = list(DynamicBatchSampler(graphs, max_num=10, skip_too_big=True))
    assert 6 not in [i for b in skipped for i in b]
    assert all(sum(sizes[i] for i in b) <= 10 for b in skipped)

    limited = DynamicBatchSampler(graphs, max_num=10, num_steps=2)
    assert len(limited) == 2 and len(list(limited)) == 2
    with pytest.raises(TypeError):
        len(sampler)

    np.random.seed(0)
    shuffled = DynamicBatchSampler(graphs, max_num=10, shuffle=True)
    assert sorted(i for b in shuffled for i in b) == list(range(len(sizes)))

    loader = Dataloader(graphs, batch_sampler=sampler)
    out = list(loader)
    assert isinstance(out[0], GraphDataBatch)
    assert [b.num_graphs for b in out] == [len(b) for b in batches]
    assert [b.num_nodes for b in out] == [sum(sizes[i] for i in b) for b in batches]

    with pytest.raises(ValueError):
        Dataloader(graphs, batch_size=2, batch_sampler=sampler)

    edge_mode = DynamicBatchSampler(
        [GraphData(edge_index=mx.zeros((2, k), dtype=mx.int32)) for k in [2, 2, 3]],
        max_num=4,
        mode="edge",
    )
    assert list(edge_mode) == [[0, 1], [2]]
    with pytest.raises(ValueError):
        DynamicBatchSampler(graphs, max_num=0)
    with pytest.raises(ValueError):
        DynamicBatchSampler(graphs, max_num=3, mode="graph")
