import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import (
    GraphData,
    InMemoryFeatureStore,
    InMemoryGraphStore,
    MemmapFeatureStore,
    MemmapGraphStore,
    TensorAttr,
)


def _graph(num_nodes=20, num_edges=60, seed=0):
    rng = np.random.default_rng(seed)
    edge_index = rng.integers(0, num_nodes, (2, num_edges)).astype(np.int32)
    return GraphData(
        edge_index=mx.array(edge_index),
        node_features=mx.array(rng.standard_normal((num_nodes, 5)).astype(np.float32)),
        node_labels=mx.array(rng.integers(0, 3, num_nodes).astype(np.int32)),
        train_mask=mx.array(rng.random(num_nodes) < 0.5),
    )


@pytest.fixture(params=["memory", "memmap"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryFeatureStore()
    return MemmapFeatureStore(str(tmp_path / "features"))


def test_put_get_index(store):
    x = np.arange(40, dtype=np.float32).reshape(10, 4)
    assert store.put_tensor(mx.array(x), attr_name="x")
    assert store.get_tensor_size(attr_name="x") == (10, 4)
    assert np.array_equal(np.asarray(store.get_tensor(attr_name="x")), x)

    # unsorted, duplicated indices
    index = np.array([7, 2, 2, 9, 0])
    out = store.get_tensor(TensorAttr(None, "x", mx.array(index)))
    assert isinstance(out, mx.array)
    assert np.array_equal(np.asarray(out), x[index])
    # boolean mask, slice, integer
    mask = np.zeros(10, dtype=bool)
    mask[[1, 4]] = True
    assert np.array_equal(np.asarray(store.get_tensor(None, "x", mask)), x[mask])
    assert np.array_equal(np.asarray(store.get_tensor(None, "x", slice(2, 5))), x[2:5])
    assert np.array_equal(np.asarray(store.get_tensor(None, "x", 3)), x[3])
    # dict-like access
    assert np.array_equal(np.asarray(store["x"]), x)
    assert "x" in store and "y" not in store


def test_update_remove_and_attrs(store):
    store["x"] = np.zeros((4, 2), dtype=np.float32)
    store.put_tensor(np.ones(4, dtype=np.int32), group_name="paper", attr_name="y")
    keys = {a.key for a in store.get_all_tensor_attrs()}
    assert keys == {(None, "x"), ("paper", "y")}
    assert len(store) == 2

    # write rows
    store.put_tensor(
        np.full((2, 2), 5.0, dtype=np.float32), None, "x", np.array([1, 3])
    )
    x = np.asarray(store["x"])
    assert np.array_equal(x[[1, 3]], np.full((2, 2), 5.0))
    assert np.array_equal(x[[0, 2]], np.zeros((2, 2)))

    store.update_tensor(np.arange(6, dtype=np.float32).reshape(3, 2), attr_name="x")
    assert store.get_tensor_size(attr_name="x") == (3, 2)
    with pytest.raises(KeyError):
        store.update_tensor(np.zeros(2), attr_name="missing")
    with pytest.raises(KeyError):
        store.get_tensor(attr_name="missing")

    assert store.remove_tensor(attr_name="x")
    assert not store.remove_tensor(attr_name="x")
    del store[("paper", "y")]
    assert len(store) == 0


def test_from_data(tmp_path):
    data = _graph()
    for store in (
        InMemoryFeatureStore.from_data(data),
        MemmapFeatureStore.from_data(data, str(tmp_path / "fs")),
    ):
        names = {a.attr_name for a in store.get_all_tensor_attrs()}
        assert names == {"node_features", "node_labels", "train_mask"}
        for name in names:
            assert np.array_equal(
                np.asarray(store[name]), np.asarray(getattr(data, name))
            )


def test_memmap_persistence_and_chunked_writes(tmp_path):
    root = str(tmp_path / "fs")
    store = MemmapFeatureStore(root)
    out = store.create_tensor("x", (1000, 8), dtype="float16")
    for start in range(0, 1000, 300):
        out[start : start + 300] = np.arange(start, min(start + 300, 1000))[:, None]
    out.flush()
    del out
    store.put_tensor(np.zeros((0, 3), dtype=np.float32), attr_name="empty")
    store.put_tensor(np.arange(5), group_name="author", attr_name="year")

    reopened = MemmapFeatureStore(root)
    assert {a.key for a in reopened.get_all_tensor_attrs()} == {
        (None, "x"),
        (None, "empty"),
        ("author", "year"),
    }
    rows = np.asarray(reopened.get_tensor(None, "x", np.array([999, 0, 500])))
    assert rows.dtype == np.float16
    assert np.array_equal(rows[:, 0], np.array([999, 0, 500], dtype=np.float16))
    assert reopened.get_tensor_size(attr_name="empty") == (0, 3)
    assert np.asarray(reopened["empty"]).shape == (0, 3)
    with pytest.raises(ValueError):
        reopened.put_tensor(np.zeros(2), attr_name="../escape")


def test_graph_store_layouts(tmp_path):
    data = _graph(num_nodes=15, num_edges=50, seed=1)
    ei = np.asarray(data.edge_index).astype(np.int64)
    n = data.num_nodes
    stores = [InMemoryGraphStore.from_data(data)]
    mm = MemmapGraphStore(str(tmp_path / "gs"))
    mm.put_edge_index(data.edge_index, layout="coo", size=(n, n))
    stores.append(mm)
    stores.append(MemmapGraphStore(str(tmp_path / "gs")))  # reopened

    order = np.argsort(ei[1], kind="stable")
    for store in stores:
        assert store.num_nodes() == (n, n)
        assert store.num_edges() == ei.shape[1]
        row, col = store.coo()
        assert np.array_equal(row, ei[0]) and np.array_equal(col, ei[1])
        colptr, row, perm = store.csc()
        assert np.array_equal(perm, order)
        assert np.array_equal(row, ei[0][order])
        assert np.array_equal(np.diff(colptr), np.bincount(ei[1], minlength=n))
        rowptr, col, perm = store.csr()
        assert np.array_equal(col, ei[1][perm])
        assert np.array_equal(np.diff(rowptr), np.bincount(ei[0], minlength=n))

    # A store holding CSC directly
    csc_store = InMemoryGraphStore()
    colptr, row, perm = stores[0].csc()
    csc_store.put_edge_index((row, colptr), layout="csc", size=(n, n))
    c2, r2, p2 = csc_store.csc()
    assert p2 is None and np.array_equal(c2, colptr) and np.array_equal(r2, row)
    r3, c3 = csc_store.coo()
    assert np.array_equal(r3, ei[0][perm]) and np.array_equal(c3, ei[1][perm])

    assert stores[0].remove_edge_index()
    assert not stores[0].remove_edge_index()
    with pytest.raises(KeyError):
        stores[0].csc()


@pytest.mark.parametrize("num_threads", [1, 4])
def test_memmap_threaded_row_reads(tmp_path, num_threads):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5000, 3, 2)).astype(np.float32)
    store = MemmapFeatureStore(str(tmp_path / "fs"), num_threads=num_threads)
    store["x"] = x
    # many runs (thread pool), duplicates, negative ids, 2-D index
    index = rng.integers(-5000, 5000, (40, 50))
    out = np.asarray(store.get_tensor(None, "x", index))
    assert out.shape == (40, 50, 3, 2)
    assert np.array_equal(out, x[index])
    # consecutive runs
    index = np.concatenate([np.arange(10, 400), np.arange(4000, 4100)])
    assert np.array_equal(np.asarray(store.get_tensor(None, "x", index)), x[index])
    assert np.asarray(store.get_tensor(None, "x", np.zeros(0, np.int64))).shape == (
        0,
        3,
        2,
    )
    with pytest.raises(IndexError):
        store.get_tensor(None, "x", np.array([5000]))


@pytest.mark.parametrize(
    "max_gap,max_read", [(1, 1), (24 * 3, 24 * 10), (1 << 16, 1 << 24)]
)
def test_memmap_read_blocks(tmp_path, max_gap, max_read):
    # Sparse (one row per read), medium and dense (whole blocks) requests.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3000, 6)).astype(np.float32)  # 24-byte rows
    store = MemmapFeatureStore(str(tmp_path / "fs"), num_threads=3)
    store["x"] = x
    store._MAX_GAP, store._MAX_READ = max_gap, max_read
    for index in (
        rng.integers(0, 3000, 2000),
        np.arange(100, 900, 3),
        np.array([2999, 0, 1, 2, 7, 8]),
    ):
        assert np.array_equal(np.asarray(store.get_tensor(None, "x", index)), x[index])
