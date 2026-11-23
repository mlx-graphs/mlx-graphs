"""Tests of TemporalData and TemporalDataLoader, with parity tests against
torch_geometric.data.TemporalData and torch_geometric.loader.TemporalDataLoader."""

import copy
import pickle

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import TemporalData
from mlx_graphs.loaders import TemporalDataLoader


def _arrays(num_events=23, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, 5, num_events)
    dst = rng.integers(5, 11, num_events)
    t = np.sort(rng.integers(0, 40, num_events))
    msg = rng.standard_normal((num_events, 3)).astype(np.float32)
    y = rng.integers(0, 2, num_events)
    return src, dst, t, msg, y


def _events(seed=0, **extra):
    src, dst, t, msg, y = _arrays(seed=seed)
    return TemporalData(
        src=mx.array(src),
        dst=mx.array(dst),
        t=mx.array(t),
        msg=mx.array(msg),
        y=mx.array(y),
        **extra,
    )


def _pyg_events(seed=0):
    torch = pytest.importorskip("torch")
    from torch_geometric.data import TemporalData as PyGTemporalData

    src, dst, t, msg, y = _arrays(seed=seed)
    return PyGTemporalData(
        src=torch.tensor(src),
        dst=torch.tensor(dst),
        t=torch.tensor(t),
        msg=torch.tensor(msg),
        y=torch.tensor(y),
    )


def _assert_same(data, pyg_data, keys=("src", "dst", "t", "msg", "y")):
    for key in keys:
        assert np.allclose(np.array(data[key]), pyg_data[key].numpy()), key


def test_temporal_data_basics():
    data = _events(node_feat=mx.zeros((11, 2)))
    assert data.num_events == data.num_edges == len(data) == 23
    assert data.num_nodes == max(data.src.max().item(), data.dst.max().item()) + 1
    assert data.size() == (data.src.max().item(), data.dst.max().item())
    assert data.edge_index.tolist() == [data.src.tolist(), data.dst.tolist()]
    assert set(data.keys()) == {"src", "dst", "t", "msg", "y", "node_feat"}
    assert "y" in data and "foo" not in data
    assert data["src"] is data.src

    # missing core attributes are None, other missing attributes raise
    assert TemporalData().msg is None
    with pytest.raises(AttributeError):
        data.foo

    # attributes are set, replaced and deleted like a dictionary
    data.weight = mx.ones(23)
    data["label"] = mx.zeros(23)
    assert "weight" in data and "label" in data
    del data["label"]
    data.weight = None
    assert "label" not in data and "weight" not in data
    with pytest.raises(AttributeError):
        data.num_events = 3

    assert "src=[23]" in repr(data)
    assert TemporalData.from_dict(data.to_dict()).keys() == data.keys()
    assert [k for k, _ in data("src", "t")] == ["src", "t"]


@pytest.mark.parametrize(
    "idx",
    [
        3,
        -1,
        slice(2, 9),
        slice(None, None, 3),
        [4, 0, 7],
        (1, 2),
        "array",
        "mask",
    ],
)
def test_index_select_matches_pyg(idx):
    torch = pytest.importorskip("torch")
    data, pyg_data = _events(), _pyg_events()
    data.node_feat = mx.zeros((11, 2))  # not per-event: left untouched
    pyg_data.node_feat = torch.zeros((11, 2))
    if idx == "array":
        idx, pyg_idx = mx.array([5, 1, 1]), torch.tensor([5, 1, 1])
    elif idx == "mask":
        mask = np.arange(23) % 3 == 0
        idx, pyg_idx = mx.array(mask), torch.tensor(mask)
    elif idx == -1:
        pyg_idx = 22  # PyG turns ints into slices, without wrapping negatives
    else:
        pyg_idx = idx
    out, expected = data[idx], pyg_data[pyg_idx]
    assert isinstance(out, TemporalData)
    _assert_same(out, expected)
    assert out.node_feat.shape == (11, 2)
    # the original is unchanged
    assert data.num_events == 23


def test_iteration_and_copies():
    data = _events()
    events = list(data)
    assert len(events) == 23 and all(e.num_events == 1 for e in events)
    assert events[4].src.item() == data.src[4].item()

    shallow = copy.copy(data)
    shallow.extra = mx.ones(23)
    assert "extra" not in data
    deep = data.clone()
    assert np.array_equal(np.array(deep.msg), np.array(data.msg))
    loaded = pickle.loads(pickle.dumps(data))
    assert np.array_equal(np.array(loaded.t), np.array(data.t))


@pytest.mark.parametrize("ratios", [(0.15, 0.15), (0.2, 0.1), (0.0, 0.3)])
def test_train_val_test_split_matches_pyg(ratios):
    data, pyg_data = _events(), _pyg_events()
    splits = data.train_val_test_split(*ratios)
    pyg_splits = pyg_data.train_val_test_split(*ratios)
    for split, pyg_split in zip(splits, pyg_splits):
        assert split.num_events == pyg_split.num_events
        _assert_same(split, pyg_split)


def test_batching_matches_pyg():
    torch = pytest.importorskip("torch")
    from torch_geometric.data import Batch

    data_list, pyg_list = [], []
    for seed in range(3):
        data, pyg_data = _events(seed), _pyg_events(seed)
        group = np.random.default_rng(seed).integers(0, 3, 4)
        data.node_batch = mx.array(group)
        pyg_data.node_batch = torch.tensor(group)
        data_list.append(data)
        pyg_list.append(pyg_data)
        # __inc__ / __cat_dim__ as in PyG
        for key in ("src", "dst", "t", "msg", "node_batch"):
            assert data.__cat_dim__(key) == pyg_data.__cat_dim__(key, pyg_data[key])
            expected = pyg_data.__inc__(key, pyg_data[key])
            assert (data.__inc__(key) or 0) == expected

    out = TemporalData.from_data_list(data_list)
    expected = Batch.from_data_list(pyg_list)
    _assert_same(out, expected, ("src", "dst", "t", "msg", "y", "node_batch"))


@pytest.mark.parametrize("batch_size", [1, 5, 23, 50])
@pytest.mark.parametrize("drop_last", [False, True])
def test_temporal_data_loader_matches_pyg(batch_size, drop_last):
    from torch_geometric.loader import TemporalDataLoader as PyGTemporalDataLoader

    data, pyg_data = _events(), _pyg_events()
    loader = TemporalDataLoader(data, batch_size=batch_size, drop_last=drop_last)
    pyg_loader = PyGTemporalDataLoader(
        pyg_data, batch_size=batch_size, drop_last=drop_last
    )
    assert len(loader) == len(pyg_loader)
    batches = list(loader)
    pyg_batches = list(pyg_loader)
    assert len(batches) == len(pyg_batches)
    for batch, pyg_batch in zip(batches, pyg_batches):
        _assert_same(batch, pyg_batch)
        assert batch.n_id.tolist() == pyg_batch.n_id.tolist()
        assert "neg_dst" not in batch


def test_temporal_data_loader_negative_sampling():
    data = _events()
    val = data[10:]
    mx.random.seed(0)
    loader = TemporalDataLoader(val, batch_size=4, neg_sampling_ratio=1.5)
    lo, hi = val.dst.min().item(), val.dst.max().item()
    first = []
    for batch in loader:
        num = batch.dst.shape[0]
        assert batch.neg_dst.shape == (round(1.5 * num),)
        assert batch.neg_dst.dtype == batch.dst.dtype
        neg = np.array(batch.neg_dst)
        assert neg.min() >= lo and neg.max() <= hi
        nodes = np.concatenate([np.array(batch.src), np.array(batch.dst), neg])
        assert batch.n_id.tolist() == np.unique(nodes).tolist()
        first.append(neg)
    # reproducible with mx.random.seed
    mx.random.seed(0)
    again = [np.array(b.neg_dst) for b in loader]
    assert all(np.array_equal(a, b) for a, b in zip(first, again))


@pytest.mark.parametrize("idx", [[0, 23], [-24], np.array([5, 100]), mx.array([30])])
def test_out_of_range_event_indices_raise(idx):
    # MLX gathers do not check bounds: these must not silently return events.
    with pytest.raises(IndexError):
        _events()[idx]
