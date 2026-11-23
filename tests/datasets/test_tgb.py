"""Temporal Graph Benchmark datasets, negative samples and evaluator, compared
with the official ``py-tgb`` package (``pip install py-tgb``) when it is
installed.

The fast tests write small raw files in the format of every dataset and
process them with both libraries; the slow ones (``--datasets``) download the
small datasets. Set ``TGB_TEST_DIR`` to keep the downloads between runs.
"""

import os
import pickle

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import TemporalData
from mlx_graphs.datasets import (
    TGBLinkPropPredDataset,
    TGBNegativeEdgeSampler,
    TGBNodePropPredDataset,
)
from mlx_graphs.metrics import TGBEvaluator, tgb_mrr_hits, tgb_ndcg

# ----------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------


def test_mrr_ties():
    # rank = 1 + (#greater + #greater_or_equal) / 2
    pos = np.array([0.5, 0.5, 0.9, 0.1], dtype=np.float32)
    neg = np.array(
        [
            [0.1, 0.2, 0.3],  # rank 1
            [0.5, 0.5, 0.1],  # two ties: rank 1 + (0 + 2) / 2 = 2
            [1.0, 0.9, 0.9],  # 1 + (1 + 3) / 2 = 3
            [0.2, 0.3, 0.4],  # 4
        ],
        dtype=np.float32,
    )
    mrr, hits = tgb_mrr_hits(mx.array(pos), mx.array(neg), k=2)
    assert mrr == pytest.approx(np.mean([1, 1 / 2, 1 / 3, 1 / 4]))
    assert hits == pytest.approx(0.5)
    # -inf padding never outranks the positive
    padded = np.concatenate([neg, np.full((4, 5), -np.inf, np.float32)], axis=1)
    assert tgb_mrr_hits(pos, padded, k=2) == (mrr, hits)


def test_ndcg_matches_sklearn():
    from sklearn.metrics import ndcg_score

    rng = np.random.default_rng(0)
    y_true = rng.random((50, 30)) * (rng.random((50, 30)) < 0.3)
    y_true[0] = 0  # a node without any interaction scores 0
    # rounded scores to create ties
    y_pred = np.round(rng.random((50, 30)), 1).astype(np.float32)
    for k in (1, 5, 10, 40):
        assert tgb_ndcg(y_true, y_pred, k=k) == pytest.approx(
            ndcg_score(y_true, y_pred, k=k), abs=1e-12
        )
    with pytest.raises(ValueError):
        tgb_ndcg(np.ones((3, 1)), np.ones((3, 1)))
    with pytest.raises(ValueError):
        tgb_ndcg(-np.ones((3, 2)), np.ones((3, 2)))


def test_evaluator_matches_py_tgb():
    pytest.importorskip("tgb")
    from tgb.linkproppred.evaluate import Evaluator as LinkEvaluator
    from tgb.nodeproppred.evaluate import Evaluator as NodeEvaluator

    rng = np.random.default_rng(1)
    for scale in (1.0, 4.0):  # scale 4 rounds the scores: many ties
        pos = np.round(rng.random(500) * scale).astype(np.float32) / scale
        neg = np.round(rng.random((500, 100)) * scale).astype(np.float32) / scale
        ref = LinkEvaluator("tgbl-wiki").eval(
            {"y_pred_pos": pos, "y_pred_neg": neg, "eval_metric": ["mrr"]}
        )
        out = TGBEvaluator("tgbl-wiki").eval(
            {"y_pred_pos": mx.array(pos), "y_pred_neg": mx.array(neg),
             "eval_metric": ["mrr"]}
        )  # fmt: skip
        assert out.keys() == ref.keys()
        for key in ref:
            assert out[key] == pytest.approx(float(ref[key]), abs=1e-7)

    y_true = (rng.random((200, 255)) * (rng.random((200, 255)) < 0.1)).astype(
        np.float32
    )
    y_pred = np.round(rng.normal(size=(200, 255)), 1).astype(np.float32)
    ref = NodeEvaluator("tgbn-trade").eval(
        {"y_true": y_true, "y_pred": y_pred, "eval_metric": ["ndcg"]}
    )
    out = TGBEvaluator("tgbn-trade").eval(
        {"y_true": mx.array(y_true), "y_pred": mx.array(y_pred),
         "eval_metric": ["ndcg"]}
    )  # fmt: skip
    assert out["ndcg"] == pytest.approx(ref["ndcg"], abs=1e-12)


def test_evaluator_errors():
    with pytest.raises(NotImplementedError):
        TGBEvaluator("tgbl-unknown")
    evaluator = TGBEvaluator("tgbl-wiki")
    with pytest.raises(ValueError):
        evaluator.eval({"y_pred_pos": [1], "y_pred_neg": [[0]], "eval_metric": ["x"]})
    with pytest.raises(RuntimeError):
        evaluator.eval({"y_pred_pos": [1], "eval_metric": ["mrr"]})
    mse = TGBEvaluator("tgbn-trade").eval(
        {"y_true": np.zeros((2, 3)), "y_pred": np.ones((2, 3)), "eval_metric": ["mse"]}
    )
    assert mse == {"mse": 1.0, "rmse": 1.0}


# ----------------------------------------------------------------------
# Raw files in the format of every dataset
# ----------------------------------------------------------------------


def _write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def _events(rng, num_events, num_src, num_dst, prefix=("u", "v"), t0=1000):
    t = np.sort(t0 + rng.integers(0, 400, num_events))
    src = [f"{prefix[0]}{i}" for i in rng.integers(0, num_src, num_events)]
    dst = [f"{prefix[1]}{i}" for i in rng.integers(0, num_dst, num_events)]
    return t, src, dst


def _write_link_raw(raw, name, rng, num_events=400):
    """Raw files of a link property prediction dataset: the edge list and
    negative samples of the validation and test events (as TGB pickles)."""
    sfx = "_v2" if name != "tgbl-comment" else ""
    path = os.path.join(raw, f"{name}_edgelist{sfx}.csv")
    if name == "tgbl-wiki":
        t = np.sort(rng.integers(0, 400, num_events)).astype(np.float64)
        rows = [
            [rng.integers(0, 30), rng.integers(0, 20), ti, 0]
            + list(np.round(rng.normal(size=4), 6))
            for ti in t
        ]
        _write_csv(path, "user_id,item_id,timestamp,state_label,features", rows)
    elif name in ("tgbl-review", "tgbl-coin"):
        t, src, dst = _events(rng, num_events, 40, 40, ("n", "n"))
        w = np.round(rng.random(num_events) * 10, 2)
        w[:5] = 0  # zero weights count as one
        rows = list(zip(t, src, dst, w))
        _write_csv(path, "ts,src,dst,weight", rows)
    elif name == "tgbl-comment":
        t, src, dst = _events(rng, num_events, 40, 40, ("n", "n"))
        rows = [
            (ti, s, d, rng.integers(0, 5000), rng.integers(-5, 50))
            for ti, s, d in zip(t, src, dst)
        ]
        _write_csv(path, "ts,src,dst,num_words,score", rows)
    elif name == "tgbl-flight":
        days = np.sort(rng.integers(0, 60, num_events))
        days = [
            str(np.datetime64("2019-01-01") + np.timedelta64(int(d), "D")) for d in days
        ]
        airports = [f"A{i}" for i in range(12)]
        src = rng.choice(airports, num_events)
        dst = rng.choice(airports, num_events)
        callsigns = ["", "AB12", "XYZ9876", "SHORT1", "Q"]
        typecodes = ["", "B738", "A320", "TOOLONGCODE", "E75L"]
        rows = [
            (day, s, d, rng.choice(callsigns), rng.choice(typecodes))
            for day, s, d in zip(days, src, dst)
        ]
        _write_csv(path, "day,src,dst,callsign,typecode", rows)
        # 4 airport types + 7 continents + 7 region characters + 2 coordinates
        types = ["small", "medium", "large", "heliport"]
        continents = ["EU", "NA", "SA", "AS", "AF", "OC", "AN"]
        rows = [
            (a, types[i % 4], continents[i % 7], f"R-{i}",
             round(rng.uniform(-180, 180), 4), round(rng.uniform(-90, 90), 4))
            for i, a in enumerate(airports)
        ] + [("UNUSED", "closed", "XX", "Z-1", 0.0, 0.0)]  # fmt: skip
        _write_csv(
            os.path.join(raw, "airport_node_feat_v2.csv"),
            "airport_code,type,continent,iso_region,longitude,latitude",
            rows,
        )

    # Negative samples for the events of the last 30% of the time.
    return path, sfx


def _write_negatives(raw, name, sfx, src, dst, t, masks, rng):
    for split, mask in zip(("val", "test"), masks):
        eval_set = {}
        for s, d, ti in zip(src[mask], dst[mask], t[mask]):
            num = 7 if rng.random() < 0.8 else 5  # varying numbers of negatives
            eval_set[(np.int64(s), np.int64(d), np.int64(ti))] = rng.integers(
                0, 50, num
            ).astype(np.int64)
        with open(os.path.join(raw, f"{name}_{split}_ns{sfx}.pkl"), "wb") as f:
            pickle.dump(eval_set, f, protocol=pickle.HIGHEST_PROTOCOL)


def _py_tgb_root(tmp_path, dir_name, raw_path):
    """A py-tgb dataset directory with links to our raw files, as the
    ``root`` argument of py-tgb (relative to its package directory)."""
    from tgb.utils.info import PROJ_DIR

    target = os.path.join(tmp_path, "py_tgb", dir_name)
    os.makedirs(target)
    for f in os.listdir(raw_path):
        os.symlink(os.path.join(raw_path, f), os.path.join(target, f))
    return os.path.relpath(os.path.join(tmp_path, "py_tgb"), PROJ_DIR)


def _compare_link(dataset, ref, check_negatives=True):
    data = dataset[0]
    data_ref = ref.get_TemporalData()
    assert isinstance(data, TemporalData)
    assert data.num_events == data_ref.num_events
    for key in ("src", "dst", "t"):
        assert np.array_equal(np.array(data[key]), data_ref[key].numpy()), key
    for key in ("msg", "w", "y"):
        assert np.array_equal(
            np.array(data[key]), data_ref[key].numpy().astype(np.float32)
        ), key
    for split in ("train_mask", "val_mask", "test_mask"):
        assert np.array_equal(
            np.array(getattr(dataset, split)), getattr(ref, split).numpy()
        ), split
    if ref.node_feat is None:
        assert dataset.node_feat is None
    else:
        assert np.array_equal(np.array(dataset.node_feat), ref.node_feat.numpy())
    assert dataset.eval_metric == ref.eval_metric
    assert dataset.min_dst_idx == ref.dataset.min_dst_idx
    assert dataset.max_dst_idx == ref.dataset.max_dst_idx

    if not check_negatives:
        return
    dataset.load_val_ns()
    dataset.load_test_ns()
    ref.load_val_ns()
    ref.load_test_ns()
    sampler, sampler_ref = dataset.negative_sampler, ref.negative_sampler
    for split, mask in (("val", dataset.val_mask), ("test", dataset.test_mask)):
        events = data[mask]
        ref_neg = sampler_ref.query_batch(
            np.array(events.src).astype(np.int64),
            np.array(events.dst).astype(np.int64),
            np.array(events.t),
            split_mode=split,
        )
        neg = sampler.query_batch(events.src, events.dst, events.t, split_mode=split)
        assert neg == ref_neg
        # padded version
        neg_arr, valid = sampler.query_batch_array(
            events.src, events.dst, events.t, split_mode=split
        )
        neg_arr, valid = np.array(neg_arr), np.array(valid)
        assert [row[m].tolist() for row, m in zip(neg_arr, valid)] == ref_neg
        assert (neg_arr[~valid] == -1).all()


def _compare_node(dataset, ref):
    data = dataset[0]
    data_ref = ref.get_TemporalData()
    for key in ("src", "dst", "t"):
        assert np.array_equal(np.array(data[key]), data_ref[key].numpy()), key
    assert np.array_equal(np.array(data.msg), data_ref.msg.numpy())
    for split in ("train_mask", "val_mask", "test_mask"):
        assert np.array_equal(
            np.array(getattr(dataset, split)), getattr(ref, split).numpy()
        ), split
    assert dataset.num_classes == ref.num_classes
    assert dataset.eval_metric == ref.eval_metric

    # The label streams, consumed as in TGB's examples.
    label_ts = ref.dataset.label_ts
    assert np.array_equal(dataset.label_timestamps, label_ts)
    dataset.reset_label_time()
    ref.reset_label_time()
    assert dataset.get_label_time() == ref.get_label_time()
    assert dataset.get_node_label(int(label_ts[0]) - 1) is None
    for ts in label_ts:
        out = dataset.get_node_label(int(ts))
        out_ref = ref.get_node_label(int(ts))
        assert dataset.get_label_time() == ref.get_label_time()
        for a, b in zip(out, out_ref):
            assert np.array_equal(np.array(a), b.numpy())
    assert dataset.get_node_label(int(label_ts[-1]) + 1) is None


@pytest.mark.parametrize(
    "name", ["tgbl-wiki", "tgbl-review", "tgbl-coin", "tgbl-comment", "tgbl-flight"]
)
def test_link_raw_formats(tmp_path, name):
    """Small raw files in the format of each dataset, processed as py-tgb."""
    rng = np.random.default_rng(0)
    dataset_dir = name.replace("-", "_")
    raw = os.path.join(tmp_path, dataset_dir, "raw")
    os.makedirs(raw)
    _, sfx = _write_link_raw(raw, name, rng)

    # Process the edge list first to get the events (with placeholders for
    # the negative samples, which are only read by `load_val_ns`).
    for split in ("val", "test"):
        open(os.path.join(raw, f"{name}_{split}_ns{sfx}.pkl"), "wb").close()
    dataset = TGBLinkPropPredDataset(name, base_dir=str(tmp_path))
    a = dataset.arrays
    masks = (a["val_mask"], a["test_mask"])
    _write_negatives(raw, name, sfx, a["src"], a["dst"], a["t"], masks, rng)
    dataset = TGBLinkPropPredDataset(name, base_dir=str(tmp_path))  # reloaded
    assert repr(dataset) == f"TGBLinkPropPredDataset({name})"

    pytest.importorskip("tgb")
    from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

    root = _py_tgb_root(tmp_path, dataset_dir, raw)
    ref = PyGLinkPropPredDataset(name=name, root=root)
    _compare_link(dataset, ref)


def _write_node_raw(raw, name, rng, num_events=600):
    path = os.path.join(raw, f"{name}_edgelist.csv")
    labels = os.path.join(raw, f"{name}_node_labels.csv")
    if name == "tgbn-trade":
        years = np.sort(rng.integers(1986, 1996, num_events))
        nations = [f"nation {i}" for i in range(25)]
        rows = [(y, rng.choice(nations), rng.choice(nations), rng.random())
                for y in years]  # fmt: skip
        _write_csv(path, "year,nation,trading nation,weight", rows)
        # labels of the next year, with a duplicated (year, nation, partner)
        rows = [(y + 1, s, d, round(w * 2, 5)) for y, s, d, w in rows[::3]]
        rows.append(rows[0][:3] + (0.123,))
        _write_csv(labels, "year,nation,trading nation,weight", rows)
        return
    t, users, items = _events(rng, num_events, 30, 15, ("user", "item"))
    if name == "tgbn-genre":
        rows = [(ti, u, i, round(rng.random(), 4))
                for ti, u, i in zip(t, users, items)]  # fmt: skip
        _write_csv(path, "ts,user_id,genre,weight", rows)
    elif name == "tgbn-reddit":
        rows = [(ti, u, i, rng.integers(1, 100), rng.integers(-3, 30))
                for ti, u, i in zip(t, users, items)]  # fmt: skip
        _write_csv(path, "ts,src,subreddit,num_words,score", rows)
    else:  # tgbn-token
        rows = [(ti, u, i, round(rng.random() * 1e3 + 0.01, 4), rng.integers(0, 2))
                for ti, u, i in zip(t, users, items)]  # fmt: skip
        _write_csv(path, "timestamp,user_address,token_address,value,IsSender", rows)
    rows = [(ti // 50 * 50 + 50, u, i, round(rng.random(), 4))
            for ti, u, i in zip(t[::2], users[::2], items[::2])]  # fmt: skip
    _write_csv(labels, "ts,user,item,weight", rows)


@pytest.mark.parametrize(
    "name", ["tgbn-trade", "tgbn-genre", "tgbn-reddit", "tgbn-token"]
)
def test_node_raw_formats(tmp_path, name):
    rng = np.random.default_rng(0)
    dataset_dir = name.replace("-", "_")
    raw = os.path.join(tmp_path, dataset_dir, "raw")
    os.makedirs(raw)
    _write_node_raw(raw, name, rng)
    dataset = TGBNodePropPredDataset(name, base_dir=str(tmp_path))
    assert repr(dataset) == f"TGBNodePropPredDataset({name})"
    labels = np.array(dataset.node_labels(0)[2])
    assert labels.shape[1] == dataset.label_dim

    pytest.importorskip("tgb")
    from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset

    root = _py_tgb_root(tmp_path, dataset_dir, raw)
    ref = PyGNodePropPredDataset(name=name, root=root)
    if name == "tgbn-token":
        # py-tgb applies the log to the weights once the files are cached
        ref = PyGNodePropPredDataset(name=name, root=root)
    _compare_node(dataset, ref)


def test_unknown_names(tmp_path):
    with pytest.raises(ValueError):
        TGBLinkPropPredDataset("tgbn-trade", base_dir=str(tmp_path))
    with pytest.raises(ValueError):
        TGBNodePropPredDataset("tgbl-wiki", base_dir=str(tmp_path))


def test_negative_sampler_errors(tmp_path):
    path = os.path.join(tmp_path, "ns.pkl")
    with open(path, "wb") as f:
        pickle.dump({(1, 2, 3): np.array([4, 5]), (1, 2, 4): np.array([6])}, f)
    sampler = TGBNegativeEdgeSampler("tgbl-wiki")
    with pytest.raises(ValueError):
        sampler.query_batch([1], [2], [3], split_mode="val")  # not loaded
    sampler.load_eval_set(path, split_mode="val")
    assert sampler.query_batch([1, 1], [2, 2], [4, 3], split_mode="val") == [
        [6],
        [4, 5],
    ]
    neg, mask = sampler.query_batch_array(
        mx.array([1, 1]), mx.array([2, 2]), mx.array([4, 3]), split_mode="val"
    )
    assert np.array(neg).tolist() == [[6, -1], [4, 5]]
    assert np.array(mask).tolist() == [[True, False], [True, True]]
    with pytest.raises(ValueError):
        sampler.query_batch([1], [2], [5], split_mode="val")  # unknown edge
    with pytest.raises(ValueError):
        sampler.query_batch([1], [2], [3], split_mode="train")
    sampler.reset_eval_set("val")
    with pytest.raises(ValueError):
        sampler.query_batch([1], [2], [3], split_mode="val")

    # pickles of anything else than arrays are refused
    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    evil = os.path.join(tmp_path, "evil.pkl")
    with open(evil, "wb") as f:
        pickle.dump({(1, 2, 3): Evil()}, f)
    with pytest.raises(pickle.UnpicklingError):
        sampler.load_eval_set(evil, split_mode="test")


# ----------------------------------------------------------------------
# Downloaded datasets
# ----------------------------------------------------------------------


def _base_dir(tmp_path):
    return os.environ.get("TGB_TEST_DIR", str(tmp_path))


@pytest.mark.slow
@pytest.mark.parametrize("name", ["tgbl-wiki", "tgbl-review"])
def test_tgb_link_dataset(tmp_path, name):
    pytest.importorskip("tgb")
    from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

    dataset = TGBLinkPropPredDataset(name, base_dir=_base_dir(tmp_path))
    root = _py_tgb_root(tmp_path, dataset.name, dataset.raw_path)
    ref = PyGLinkPropPredDataset(name=name, root=root)
    _compare_link(dataset, ref)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["tgbn-trade", "tgbn-genre"])
def test_tgb_node_dataset(tmp_path, name):
    pytest.importorskip("tgb")
    from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset

    dataset = TGBNodePropPredDataset(name, base_dir=_base_dir(tmp_path))
    root = _py_tgb_root(tmp_path, dataset.name, dataset.raw_path)
    ref = PyGNodePropPredDataset(name=name, root=root)
    _compare_node(dataset, ref)


# ----------------------------------------------------------------------
# examples/tgn_tgbl.py: batched one-vs-many scoring
# ----------------------------------------------------------------------


def test_example_batched_mrr_equals_per_edge(tmp_path):
    """The example scores the events of a mini-batch together; TGB's example
    scores them one at a time. In evaluation mode both give the same MRR."""
    import copy
    import importlib.util

    from mlx_graphs.loaders import TemporalDataLoader
    from mlx_graphs.nn.models.tgn import LastNeighborLoader

    path = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
    spec = importlib.util.spec_from_file_location(
        "tgn_tgbl", os.path.join(path, "tgn_tgbl.py")
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    mx.random.seed(0)
    rng = np.random.default_rng(0)
    num_events, num_src, num_dst = 300, 15, 25
    src = rng.integers(0, num_src, num_events)
    dst = rng.integers(num_src, num_src + num_dst, num_events)
    t = np.sort(rng.integers(0, 1000, num_events))
    data = TemporalData(
        src=mx.array(src.astype(np.int32)),
        dst=mx.array(dst.astype(np.int32)),
        t=mx.array(t),
        msg=mx.array(rng.normal(size=(num_events, 4)).astype(np.float32)),
    )
    num_nodes = num_src + num_dst
    eval_set = {
        (np.int64(s), np.int64(d), np.int64(ti)): rng.integers(
            num_src, num_src + num_dst, 10 if rng.random() < 0.7 else 7
        )
        for s, d, ti in zip(src[200:], dst[200:], t[200:])
    }
    pkl = os.path.join(tmp_path, "val_ns.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(eval_set, f)
    sampler = TGBNegativeEdgeSampler("tgbl-wiki")
    sampler.load_eval_set(pkl, split_mode="val")
    evaluator = TGBEvaluator("tgbl-wiki")

    model = example.TGN(num_nodes, 4, 16, 16, 16)
    neighbor_loader = LastNeighborLoader(num_nodes, size=5)
    model.eval()
    history = data[:200]
    model.memory.update_state(history.src, history.dst, history.t, history.msg)
    neighbor_loader.insert(history.src, history.dst)
    memory_state = model.memory.memory_state()
    neighbors = copy.deepcopy(neighbor_loader.__dict__)
    assoc = np.zeros(num_nodes, dtype=np.int64)
    loader = TemporalDataLoader(data[200:], batch_size=40)

    batched = example.test(
        model, data, loader, neighbor_loader, assoc, sampler, evaluator, "val"
    )

    # TGB's loop: one forward pass per positive edge
    model.memory.load_memory_state(memory_state)
    neighbor_loader.__dict__.update(neighbors)
    per_edge = []
    for batch in loader:
        negatives = sampler.query_batch(batch.src, batch.dst, batch.t, split_mode="val")
        for i, neg in enumerate(negatives):
            s = int(batch.src[i].item())
            dsts = np.array([int(batch.dst[i].item())] + neg)
            n_id = np.unique(np.concatenate([[s], dsts]))
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[np.array(n_id)] = np.arange(n_id.size)
            z = model.embed(data, n_id, edge_index, e_id)
            z_src = z[mx.array(np.full(dsts.size, assoc[s], dtype=np.int32))]
            z_dst = z[mx.array(assoc[dsts].astype(np.int32))]
            y = np.array(model.link_pred(z_src, z_dst)).reshape(-1)
            per_edge.append(
                evaluator.eval(
                    {"y_pred_pos": y[:1], "y_pred_neg": y[1:], "eval_metric": ["mrr"]}
                )["mrr"]
            )
        model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
    assert batched == pytest.approx(float(np.mean(per_edge)), abs=1e-6)
