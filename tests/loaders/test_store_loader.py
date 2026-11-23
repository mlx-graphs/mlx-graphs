"""NeighborLoader over a (FeatureStore, GraphStore) pair (PyG's remote
backend) gives the same mini-batches as over the GraphData."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import (
    GraphData,
    InMemoryFeatureStore,
    InMemoryGraphStore,
    MemmapFeatureStore,
    MemmapGraphStore,
)
from mlx_graphs.loaders import NeighborLoader


def _graph(seed=0, num_nodes=80, num_edges=600):
    rng = np.random.default_rng(seed)
    return GraphData(
        edge_index=mx.array(
            rng.integers(0, num_nodes, (2, num_edges)).astype(np.int32)
        ),
        node_features=mx.array(rng.standard_normal((num_nodes, 6)).astype(np.float32)),
        node_labels=mx.array(rng.integers(0, 4, num_nodes).astype(np.int32)),
        edge_features=mx.array(rng.standard_normal((num_edges, 2)).astype(np.float32)),
        train_mask=mx.array(rng.random(num_nodes) < 0.5),
        time=mx.array(rng.integers(0, 30, num_nodes)),
    )


def _stores(kind, data, tmp_path):
    if kind == "memory":
        return InMemoryFeatureStore.from_data(data), InMemoryGraphStore.from_data(data)
    fs = MemmapFeatureStore(str(tmp_path / "fs"))
    # PyG names: x / y / edge_attr map to node_features / node_labels / ...
    fs["x"] = data.node_features
    fs["y"] = data.node_labels
    fs["edge_attr"] = data.edge_features
    fs["train_mask"] = data.train_mask
    fs["time"] = data.time
    gs = MemmapGraphStore(str(tmp_path / "gs"))
    n = data.num_nodes
    gs.put_edge_index(data.edge_index, layout="coo", size=(n, n))
    return fs, gs


_KEYS = (
    "edge_index",
    "node_features",
    "node_labels",
    "edge_features",
    "train_mask",
    "time",
    "n_id",
    "e_id",
)


def _assert_same(a, b, keys=_KEYS):
    assert a.batch_size == b.batch_size
    for key in keys:
        x, y = getattr(a, key), getattr(b, key)
        assert x.shape == y.shape, key
        assert np.array_equal(np.asarray(x), np.asarray(y)), key


@pytest.mark.parametrize("kind", ["memory", "memmap"])
def test_full_fanout_batches_identical(kind, tmp_path):
    data = _graph()
    stores = _stores(kind, data, tmp_path)
    kwargs = dict(num_neighbors=[-1, -1], input_nodes=data.train_mask, batch_size=8)
    batches = list(NeighborLoader(data, **kwargs))
    store_batches = list(NeighborLoader(stores, **kwargs))
    assert len(batches) == len(store_batches) > 1
    for a, b in zip(batches, store_batches):
        _assert_same(a, b)
        assert b.num_nodes == a.n_id.shape[0]


@pytest.mark.parametrize("kind", ["memory", "memmap"])
@pytest.mark.parametrize("temporal", [False, True])
def test_disjoint_batches_identical(kind, temporal, tmp_path):
    data = _graph(seed=1)
    stores = _stores(kind, data, tmp_path)
    kwargs = dict(num_neighbors=[3, 2], batch_size=10, shuffle=True)
    if temporal:
        kwargs.update(time_attr="time", temporal_strategy="uniform")
    else:
        kwargs.update(disjoint=True)
    np.random.seed(0)
    batches = list(NeighborLoader(data, **kwargs))
    np.random.seed(0)
    store_batches = list(NeighborLoader(stores, **kwargs))
    for a, b in zip(batches, store_batches):
        _assert_same(a, b, _KEYS + ("batch",))


@pytest.mark.parametrize("kind", ["memory", "memmap"])
def test_sampled_rows_match_features(kind, tmp_path):
    # Random fan-out (mlx_cluster sampler): every gathered row is the row of
    # its node / edge.
    data = _graph(seed=2)
    fs, gs = _stores(kind, data, tmp_path)
    x = np.asarray(data.node_features)
    ea = np.asarray(data.edge_features)
    ei = np.asarray(data.edge_index)
    for batch in NeighborLoader((fs, gs), num_neighbors=[4, 3], batch_size=16):
        n_id = np.asarray(batch.n_id)
        e_id = np.asarray(batch.e_id)
        assert np.array_equal(np.asarray(batch.node_features), x[n_id])
        assert np.array_equal(np.asarray(batch.edge_features), ea[e_id])
        local = np.asarray(batch.edge_index)
        assert np.array_equal(n_id[local], ei[:, e_id])


def test_csc_graph_store(tmp_path):
    data = _graph(seed=3)
    fs = InMemoryFeatureStore.from_data(data)
    colptr, row, _ = InMemoryGraphStore.from_data(data).csc()
    n = data.num_nodes
    gs = InMemoryGraphStore()
    gs.put_edge_index((row, colptr), layout="csc", size=(n, n))
    batch = next(iter(NeighborLoader((fs, gs), num_neighbors=[-1], batch_size=5)))
    ref = next(iter(NeighborLoader(data, num_neighbors=[-1], batch_size=5)))
    assert np.array_equal(np.asarray(batch.n_id), np.asarray(ref.n_id))
    assert np.array_equal(
        np.asarray(batch.node_features), np.asarray(ref.node_features)
    )
