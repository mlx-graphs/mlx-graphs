import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData, HeteroGraphData
from mlx_graphs.datasets import FakeDataset, FakeHeteroDataset
from mlx_graphs.loaders import Dataloader


def test_fake_dataset_node_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = FakeDataset(avg_num_nodes=50, avg_degree=4, num_channels=8, seed=0)
    assert len(dataset) == 1
    graph = dataset[0]
    assert isinstance(graph, GraphData)
    assert 37 <= graph.num_nodes <= 62
    assert graph.node_features.shape == (graph.num_nodes, 8)
    assert graph.node_labels.shape == (graph.num_nodes,)
    assert graph.node_labels.dtype == mx.int32
    assert int(graph.node_labels.max()) < 10
    assert graph.graph_labels is None
    assert graph.edge_index.dtype == mx.int32
    assert graph.is_undirected()
    assert not graph.has_self_loops()
    # nothing is written to disk
    assert list(tmp_path.iterdir()) == []


def test_fake_dataset_graph_task():
    dataset = FakeDataset(num_graphs=5, avg_num_nodes=20, num_classes=3, seed=1)
    assert len(dataset) == 5
    for graph in dataset:
        assert graph.node_labels is None
        assert graph.graph_labels.shape == (1,)
        assert 0 <= int(graph.graph_labels[0]) < 3

    batch = next(iter(Dataloader(dataset, batch_size=5)))
    assert batch.num_graphs == 5
    assert batch.graph_labels.shape == (5,)


def test_fake_dataset_options():
    dataset = FakeDataset(
        avg_num_nodes=30,
        num_channels=0,
        edge_dim=3,
        num_classes=0,
        is_undirected=False,
        seed=2,
        global_features=(2, 4),
    )
    graph = dataset[0]
    assert graph.node_features is None
    assert graph.node_labels is None
    assert graph.num_nodes >= 22
    assert graph.edge_features.shape == (graph.num_edges, 3)
    assert graph.global_features.shape == (2, 4)
    # directed graphs are coalesced: sorted without duplicates
    edges = np.array(graph.edge_index)
    keys = edges[0].astype(np.int64) * graph.num_nodes + edges[1]
    assert np.all(keys[1:] > keys[:-1])

    graph = FakeDataset(avg_num_nodes=30, edge_dim=1, seed=2)[0]
    assert graph.edge_weights.shape == (graph.num_edges,)


def test_fake_dataset_is_deterministic():
    a = FakeDataset(num_graphs=3, avg_num_nodes=30, seed=42)
    b = FakeDataset(num_graphs=3, avg_num_nodes=30, seed=42)
    c = FakeDataset(num_graphs=3, avg_num_nodes=30, seed=43)
    for ga, gb in zip(a, b):
        assert mx.array_equal(ga.edge_index, gb.edge_index)
        assert mx.array_equal(ga.node_features, gb.node_features)
        assert mx.array_equal(ga.graph_labels, gb.graph_labels)
    assert not all(
        ga.edge_index.shape == gc.edge_index.shape
        and mx.array_equal(ga.edge_index, gc.edge_index)
        for ga, gc in zip(a, c)
    )


def test_fake_dataset_invalid_task():
    with pytest.raises(AssertionError):
        FakeDataset(task="edge")


def test_fake_hetero_dataset_node_task():
    dataset = FakeHeteroDataset(
        num_node_types=2, num_edge_types=5, avg_num_nodes=40, seed=0
    )
    assert len(dataset) == 1
    graph = dataset[0]
    assert isinstance(graph, HeteroGraphData)
    assert set(graph.node_types) == {"v0", "v1"}
    assert len(graph.edge_index_dict) == 5
    # 5 edge types over 4 (src, dst) pairs: one pair gets two relations
    assert sorted({rel for _, rel, _ in graph.edge_index_dict}) == ["e0", "e1"]
    for (src, _, dst), edge_index in graph.edge_index_dict.items():
        assert edge_index.dtype == mx.int32
        assert int(edge_index[0].max()) < graph.num_nodes[src]
        assert int(edge_index[1].max()) < graph.num_nodes[dst]
    for node_type, x in graph.node_features_dict.items():
        assert x.shape[0] == graph.num_nodes[node_type]
        assert 48 <= x.shape[1] <= 80
    assert list(graph.node_labels_dict) == ["v0"]
    assert graph.node_labels_dict["v0"].shape == (graph.num_nodes["v0"],)
    assert graph.graph_labels is None


def test_fake_hetero_dataset_graph_task_and_options():
    dataset = FakeHeteroDataset(
        num_graphs=3,
        avg_num_nodes=20,
        avg_num_channels=0,
        edge_dim=2,
        seed=0,
        global_features=3,
    )
    assert len(dataset) == 3
    for graph in dataset:
        assert graph.node_labels_dict is None
        assert graph.node_features_dict is None
        assert graph.graph_labels.shape == (1,)
        assert graph.global_features.shape == (3,)
        for edge_type, edge_index in graph.edge_index_dict.items():
            assert graph.edge_features_dict[edge_type].shape == (
                edge_index.shape[1],
                2,
            )


def test_fake_hetero_dataset_is_deterministic():
    a = FakeHeteroDataset(avg_num_nodes=30, seed=7)[0]
    b = FakeHeteroDataset(avg_num_nodes=30, seed=7)[0]
    assert list(a.edge_index_dict) == list(b.edge_index_dict)
    for key in a.edge_index_dict:
        assert mx.array_equal(a.edge_index_dict[key], b.edge_index_dict[key])
    for key in a.node_features_dict:
        assert mx.array_equal(a.node_features_dict[key], b.node_features_dict[key])
