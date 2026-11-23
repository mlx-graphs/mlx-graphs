"""Reddit, Reddit2, Flickr, Yelp, AmazonProducts and PPI: processing of
(small, synthetic) raw files, and comparison with PyG on the real data
(slow tests, ``--datasets``).

The slow tests download the datasets (Reddit: 1.4 GB, AmazonProducts:
2.5 GB) into ``$MLX_GRAPHS_LARGE_DATA_DIR`` if set (``<dir>/<name>`` for
mlx-graphs, ``<dir>/pyg/<Name>`` for PyG), else into a temporary directory.
"""

import json
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.datasets import (
    PPI,
    AmazonProducts,
    Flickr,
    Reddit,
    Reddit2,
    Yelp,
)


def _write_graphsaint_raw(raw, multilabel=False):
    os.makedirs(raw)
    # 4 nodes, CSR adjacency
    indptr = np.array([0, 2, 3, 3, 5], dtype=np.int32)
    indices = np.array([1, 3, 0, 0, 2], dtype=np.int32)
    np.savez(
        os.path.join(raw, "adj_full.npz"),
        indptr=indptr,
        indices=indices,
        data=np.ones(5, dtype=np.float32),
        shape=np.array([4, 4]),
        format=np.array(b"csr"),
    )
    np.save(
        os.path.join(raw, "feats.npy"), np.arange(8, dtype=np.float64).reshape(4, 2)
    )
    if multilabel:
        class_map = {"0": [1, 0], "1": [0, 1], "3": [1, 1]}
    else:
        class_map = {"0": 2, "1": 0, "3": 1}  # node 2 has no label
    with open(os.path.join(raw, "class_map.json"), "w") as f:
        json.dump(class_map, f)
    with open(os.path.join(raw, "role.json"), "w") as f:
        json.dump({"tr": [0, 1], "va": [2], "te": [3]}, f)


@pytest.mark.parametrize("cls,name", [(Flickr, "flickr"), (Reddit2, "reddit2")])
def test_graphsaint_processing(tmp_path, cls, name):
    _write_graphsaint_raw(str(tmp_path / name / "raw"))
    graph = cls(base_dir=str(tmp_path))[0]
    assert np.array_equal(
        np.asarray(graph.edge_index), [[0, 0, 1, 3, 3], [1, 3, 0, 0, 2]]
    )
    assert graph.edge_index.dtype == mx.int32
    assert graph.node_features.dtype == mx.float32
    assert np.array_equal(np.asarray(graph.node_features), np.arange(8).reshape(4, 2))
    assert np.asarray(graph.node_labels).tolist() == [2, 0, -1, 1]
    assert np.asarray(graph.train_mask).tolist() == [True, True, False, False]
    assert np.asarray(graph.val_mask).tolist() == [False, False, True, False]
    assert np.asarray(graph.test_mask).tolist() == [False, False, False, True]

    # Loaded back from the processed (safetensors) file
    again = cls(base_dir=str(tmp_path))[0]
    for key in ("edge_index", "node_features", "node_labels", "train_mask"):
        assert np.array_equal(
            np.asarray(getattr(again, key)), np.asarray(getattr(graph, key))
        )


@pytest.mark.parametrize(
    "cls,name", [(Yelp, "yelp"), (AmazonProducts, "amazon_products")]
)
def test_graphsaint_multilabel_processing(tmp_path, cls, name):
    _write_graphsaint_raw(str(tmp_path / name / "raw"), multilabel=True)
    graph = cls(base_dir=str(tmp_path))[0]
    y = np.asarray(graph.node_labels)
    assert y.dtype == np.float32
    assert y.tolist() == [[1, 0], [0, 1], [-1, -1], [1, 1]]


def test_reddit_processing(tmp_path):
    raw = tmp_path / "reddit" / "raw"
    os.makedirs(raw)
    np.savez(
        raw / "reddit_data.npz",
        feature=np.arange(10, dtype=np.float32).reshape(5, 2),
        label=np.array([0, 1, 2, 1, 0]),
        node_types=np.array([1, 1, 2, 3, 3]),
    )
    # unsorted, with a duplicate edge (removed by coalescing, as in PyG)
    np.savez(
        raw / "reddit_graph.npz",
        row=np.array([3, 0, 1, 0, 4], dtype=np.int32),
        col=np.array([1, 2, 0, 2, 3], dtype=np.int32),
        data=np.ones(5),
        shape=np.array([5, 5]),
        format=np.array(b"coo"),
    )
    graph = Reddit(base_dir=str(tmp_path))[0]
    assert np.asarray(graph.edge_index).tolist() == [[0, 1, 3, 4], [2, 0, 1, 3]]
    assert np.asarray(graph.node_labels).tolist() == [0, 1, 2, 1, 0]
    assert np.asarray(graph.train_mask).tolist() == [True, True, False, False, False]
    assert np.asarray(graph.val_mask).tolist() == [False, False, True, False, False]
    assert np.asarray(graph.test_mask).tolist() == [False, False, False, True, True]


def test_ppi_processing(tmp_path):
    raw = tmp_path / "ppi" / "raw" / "ppi"
    os.makedirs(raw)
    # graphs: train {0,1,2} {3,4}, val {5,6}, test {7,8}; node 1 is isolated
    # inside the first graph, (0, 0) is a self-loop, (4, 3) is a duplicate.
    nodes = [{"id": i, "val": i in (5, 6), "test": i in (7, 8)} for i in range(9)]
    links = [(0, 2), (0, 0), (3, 4), (4, 3), (5, 6), (8, 7)]
    graph = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": nodes,
        "links": [{"source": a, "target": b} for a, b in links],
    }
    with open(raw / "ppi-G.json", "w") as f:
        json.dump(graph, f)
    np.save(raw / "ppi-feats.npy", np.arange(18, dtype=np.float64).reshape(9, 2))
    with open(raw / "ppi-class_map.json", "w") as f:
        json.dump({str(i): [i % 2, 1 - i % 2] for i in range(9)}, f)

    train = PPI("train", base_dir=str(tmp_path))
    assert len(train) == 2
    assert train[0].num_nodes == 3 and train[1].num_nodes == 2
    assert np.asarray(train[0].edge_index).tolist() == [[0, 2], [2, 0]]
    assert np.asarray(train[1].edge_index).tolist() == [[0, 1], [1, 0]]
    assert np.array_equal(np.asarray(train[1].node_features), [[6, 7], [8, 9]])
    assert np.asarray(train[1].node_labels).tolist() == [[1, 0], [0, 1]]
    assert len(PPI("val", base_dir=str(tmp_path))) == 1
    test = PPI("test", base_dir=str(tmp_path))
    assert np.asarray(test[0].edge_index).tolist() == [[0, 1], [1, 0]]
    with pytest.raises(ValueError):
        PPI("valid", base_dir=str(tmp_path))


# Comparison with PyG on the real datasets ####################################


def _data_dir(tmp_path):
    return os.environ.get("MLX_GRAPHS_LARGE_DATA_DIR", str(tmp_path))


def _assert_same_as_pyg(graph, ref):
    assert graph.num_nodes == ref.num_nodes
    assert graph.num_edges == ref.num_edges
    assert np.array_equal(np.asarray(graph.edge_index), ref.edge_index.numpy())
    assert np.array_equal(np.asarray(graph.node_features), ref.x.numpy())
    y = np.asarray(graph.node_labels)
    assert y.shape == tuple(ref.y.shape)
    assert np.array_equal(y, ref.y.numpy().astype(y.dtype))
    for mask in ("train_mask", "val_mask", "test_mask"):
        assert np.array_equal(
            np.asarray(getattr(graph, mask)), getattr(ref, mask).numpy()
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "cls,pyg_name",
    [
        (Flickr, "Flickr"),
        (Reddit2, "Reddit2"),
        (Yelp, "Yelp"),
        (Reddit, "Reddit"),
        (AmazonProducts, "AmazonProducts"),
    ],
)
def test_matches_pyg(tmp_path, cls, pyg_name):
    import torch_geometric.datasets as pyg_datasets

    root = _data_dir(tmp_path)
    graph = cls(base_dir=root)[0]
    ref = getattr(pyg_datasets, pyg_name)(os.path.join(root, "pyg", pyg_name))[0]
    _assert_same_as_pyg(graph, ref)


@pytest.mark.slow
def test_ppi_matches_published_statistics(tmp_path):
    root = _data_dir(tmp_path)
    sizes = {}
    for split in ("train", "val", "test"):
        ds = PPI(split, base_dir=root)
        sizes[split] = (len(ds), sum(g.num_nodes for g in ds))
        for g in ds:
            assert g.node_features.shape[1] == 50
            assert g.node_labels.shape[1] == 121
            assert g.is_undirected() and not g.has_self_loops()
    assert sizes == {"train": (20, 44906), "val": (2, 6514), "test": (2, 5524)}
    assert sum(g.num_edges for g in PPI("train", base_dir=root)) == 1226368
