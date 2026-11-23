import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData, GraphDataBatch
from mlx_graphs.utils.unbatch import unbatch, unbatch_edge_index


def test_unbatch_matches_pyg():
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    src = np.random.default_rng(0).normal(size=(7, 3)).astype(np.float32)
    batch = np.array([0, 0, 0, 1, 1, 3, 3])  # graph 2 is empty
    for batch_size in (None, 5):
        out = unbatch(mx.array(src), mx.array(batch), batch_size=batch_size)
        expected = pyg_utils.unbatch(
            torch.tensor(src), torch.tensor(batch), batch_size=batch_size
        )
        assert len(out) == len(expected)
        for a, b in zip(out, expected):
            assert np.array_equal(np.asarray(a), b.numpy())

    out = unbatch(mx.array(src.T), mx.array(batch), axis=1)
    assert [a.shape for a in out] == [(3, 3), (3, 2), (3, 0), (3, 2)]
    assert unbatch(mx.zeros((0, 2)), mx.zeros((0,), dtype=mx.int32)) == []


def test_unbatch_edge_index_matches_pyg():
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    edges = np.array([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6], [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
    batch = np.array([0, 0, 0, 0, 1, 1, 1])
    for batch_size in (None, 3):
        out = unbatch_edge_index(mx.array(edges), mx.array(batch), batch_size)
        expected = pyg_utils.unbatch_edge_index(
            torch.tensor(edges), torch.tensor(batch), batch_size
        )
        assert len(out) == len(expected)
        for a, b in zip(out, expected):
            assert np.array_equal(np.asarray(a), b.numpy())


def test_unbatch_inverts_graph_data_batch():
    graphs = [
        GraphData(
            edge_index=mx.array([[0, 1, 1], [1, 0, 2]]),
            node_features=mx.arange(3).reshape(3, 1).astype(mx.float32),
        ),
        GraphData(
            edge_index=mx.array([[0, 1], [1, 0]]),
            node_features=mx.arange(3, 5).reshape(2, 1).astype(mx.float32),
        ),
    ]
    batch = GraphDataBatch(graphs)
    features = unbatch(batch.node_features, batch.batch_indices)
    edge_indices = unbatch_edge_index(batch.edge_index, batch.batch_indices)
    for graph, x, e in zip(graphs, features, edge_indices):
        assert mx.array_equal(graph.node_features, x)
        assert mx.array_equal(graph.edge_index, e)
