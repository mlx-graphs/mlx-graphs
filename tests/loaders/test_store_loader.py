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


@pytest.mark.parametrize("storage", ["memory", "memmap", "memmap_copy"])
@pytest.mark.parametrize("directed", [False, True])
def test_csc_graph_store(tmp_path, monkeypatch, storage, directed):
    data = _graph(seed=3)
    colptr, row, perm = InMemoryGraphStore.from_data(data).csc()
    fs = InMemoryFeatureStore.from_data(data, attrs=["node_features", "node_labels"])
    fs["edge_attr"] = data.edge_features[mx.array(perm)]
    n = data.num_nodes
    root = str(tmp_path / "graph")
    on_disk = storage != "memory"
    gs = MemmapGraphStore(root) if on_disk else InMemoryGraphStore()
    gs.put_edge_index((row, colptr), layout="csc", size=(n, n))
    if on_disk:
        gs = MemmapGraphStore(root)
    if storage == "memmap_copy":
        asarray = mx.asarray

        def without_host_sharing(value, *args, **kwargs):
            if isinstance(value, memoryview) and kwargs.get("copy") is False:
                raise ValueError("Host buffer sharing is unavailable")
            return asarray(value, *args, **kwargs)

        monkeypatch.setattr(mx, "asarray", without_host_sharing)
    kwargs = dict(num_neighbors=[-1, -1], batch_size=5, directed=directed)
    loader = NeighborLoader((fs, gs), **kwargs)
    batch = next(iter(loader))
    ref = next(iter(NeighborLoader(data, **kwargs)))
    assert loader._loader._edge_perm is None
    _assert_same(
        batch,
        ref,
        ("n_id", "edge_index", "node_features", "node_labels", "edge_features"),
    )
    np.testing.assert_array_equal(perm[np.asarray(batch.e_id)], np.asarray(ref.e_id))


def test_csc_sampler_shares_disk_mapping(tmp_path):
    data = _graph(seed=4)
    colptr, row, _ = InMemoryGraphStore.from_data(data).csc()
    gs = MemmapGraphStore(str(tmp_path))
    gs.put_edge_index(
        (row, colptr), layout="csc", size=(data.num_nodes, data.num_nodes)
    )
    mapped_colptr, mapped_row, _ = gs.csc()
    try:
        mx.asarray(memoryview(mapped_row), copy=False)
    except (TypeError, ValueError):
        pytest.skip("This MLX build does not support zero-copy host buffers")

    fs = InMemoryFeatureStore.from_data(data)
    loader = NeighborLoader((fs, gs), num_neighbors=[-1], batch_size=5)._loader
    assert np.shares_memory(np.asarray(loader._colptr), mapped_colptr)
    assert np.shares_memory(np.asarray(loader._row), mapped_row)


def test_csc_sampler_packs_strided_store_arrays(tmp_path, monkeypatch):
    path = tmp_path / "rows.npy"
    np.save(path, np.array([1, 0, 2, 0], dtype=np.int64))
    row = np.load(path, mmap_mode="r")[::2]
    colptr = np.array([0, 2, 2, 2], dtype=np.int64)
    graph = InMemoryGraphStore()
    graph.put_edge_index((row, colptr), layout="csc", size=(3, 3))
    monkeypatch.setattr(graph, "csc", lambda: (colptr, row, None))
    features = InMemoryFeatureStore()
    features["x"] = mx.ones((3, 1))

    loader = NeighborLoader((features, graph), num_neighbors=[-1], input_nodes=[0])
    assert next(iter(loader)).n_id.tolist() == [0, 1, 2]


def test_csc_replacement_preserves_active_loader(tmp_path):
    graph = MemmapGraphStore(str(tmp_path))
    graph.put_edge_index(([1, 2], [0, 2, 2, 2]), layout="csc", size=(3, 3))
    features = InMemoryFeatureStore()
    features["x"] = mx.ones((3, 1))
    kwargs = dict(num_neighbors=[-1], input_nodes=mx.array([0]))
    active = NeighborLoader((features, graph), **kwargs)

    graph.put_edge_index(([2], [0, 1, 1, 1]), layout="csc", size=(3, 3))
    fresh = NeighborLoader((features, graph), **kwargs)
    assert next(iter(active)).n_id.tolist() == [0, 1, 2]
    assert next(iter(fresh)).n_id.tolist() == [0, 2]
