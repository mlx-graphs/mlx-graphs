import os
import os.path as osp
import subprocess
import sys
import textwrap

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.datasets import OGBMAG, OGBDataset
from mlx_graphs.datasets.ogb_mag import EDGE_TYPES

NUM_NODES = {"paper": 6, "author": 5, "institution": 2, "field_of_study": 3}


def _write_csv(path, array, header=None):
    pd = pytest.importorskip("pandas")

    os.makedirs(osp.dirname(path), exist_ok=True)
    frame = pd.DataFrame(array, columns=header)
    frame.to_csv(path, index=False, header=header is not None, compression="gzip")


def make_raw(root):
    """A tiny ogbn-mag with the layout of the official mag.zip."""
    rng = np.random.default_rng(0)
    raw = osp.join(root, "ogbn-mag", "raw", "mag", "raw")
    x = rng.standard_normal((6, 4)).astype(np.float32)
    _write_csv(osp.join(raw, "node-feat", "paper", "node-feat.csv.gz"), x)
    years = np.array([[2015], [2016], [2017], [2018], [2019], [2017]])
    _write_csv(osp.join(raw, "node-feat", "paper", "node_year.csv.gz"), years)
    labels = np.array([[3], [1], [0], [2], [1], [3]])
    _write_csv(osp.join(raw, "node-label", "paper", "node-label.csv.gz"), labels)
    _write_csv(
        osp.join(raw, "num-node-dict.csv.gz"),
        np.array([[NUM_NODES[t] for t in NUM_NODES]]),
        header=list(NUM_NODES),
    )
    edges = {}
    for src, rel, dst in EDGE_TYPES:
        e = np.stack(
            [rng.integers(0, NUM_NODES[src], 8), rng.integers(0, NUM_NODES[dst], 8)], 1
        )
        edges[(src, rel, dst)] = e
        _write_csv(
            osp.join(raw, "relations", f"{src}___{rel}___{dst}", "edge.csv.gz"), e
        )
    split = osp.join(root, "ogbn-mag", "raw", "mag", "split", "time", "paper")
    for name, idx in (("train", [0, 1, 2, 5]), ("valid", [3]), ("test", [4])):
        _write_csv(osp.join(split, f"{name}.csv.gz"), np.array(idx)[:, None])
    return x, years, labels, edges


def test_process_save_and_load(tmp_path):
    x, years, labels, edges = make_raw(str(tmp_path))
    data = OGBMAG(base_dir=str(tmp_path))[0]
    assert data.num_nodes == NUM_NODES
    assert data.num_nodes["author"] == 5
    assert np.allclose(np.array(data.node_features_dict["paper"]), x)
    assert np.array_equal(np.array(data.node_labels_dict["paper"]), labels[:, 0])
    assert np.array_equal(np.array(data.paper_year), years[:, 0])
    assert np.array(data.paper_train_mask).tolist() == [1, 1, 1, 0, 0, 1]
    assert np.array(data.paper_val_mask).tolist() == [0, 0, 0, 1, 0, 0]
    for edge_type, e in edges.items():
        assert data.edge_index_dict[edge_type].dtype == mx.int32
        assert np.array_equal(np.array(data.edge_index_dict[edge_type]), e.T)

    # loaded back from the processed safetensors
    again = OGBMAG(base_dir=str(tmp_path))[0]
    for edge_type in EDGE_TYPES:
        assert mx.array_equal(
            again.edge_index_dict[edge_type], data.edge_index_dict[edge_type]
        )
    assert mx.array_equal(again.paper_test_mask, data.paper_test_mask)
    assert again.num_nodes == data.num_nodes
    assert mx.array_equal(
        again.node_features_dict["paper"], data.node_features_dict["paper"]
    )


def test_add_reverse_edges(tmp_path):
    make_raw(str(tmp_path))
    data = OGBMAG.add_reverse_edges(OGBMAG(base_dir=str(tmp_path))[0])
    writes = np.array(data.edge_index_dict[("author", "writes", "paper")])
    rev = np.array(data.edge_index_dict[("paper", "rev_writes", "author")])
    assert np.array_equal(rev, writes[::-1])
    cites = np.array(data.edge_index_dict[("paper", "cites", "paper")])
    pairs = set(map(tuple, cites.T))
    assert all((b, a) in pairs for a, b in pairs)
    assert len(pairs) == cites.shape[1]
    dst_types = {dst for _, _, dst in data.edge_types}
    assert dst_types == set(NUM_NODES)


def test_ogb_dataset_points_to_ogbmag():
    with pytest.raises(ValueError, match="OGBMAG"):
        OGBDataset("ogbn-mag")


_PYG_DUMP = textwrap.dedent(
    """
    import os, sys
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from torch_geometric.datasets import OGB_MAG

    root, out = sys.argv[1], sys.argv[2]
    data = OGB_MAG(root)[0]
    np.save(os.path.join(out, "x.npy"), data["paper"].x.numpy())
    np.save(os.path.join(out, "y.npy"), data["paper"].y.numpy())
    np.save(os.path.join(out, "year.npy"), data["paper"].year.numpy())
    for split in ("train", "val", "test"):
        mask = data["paper"][f"{split}_mask"]
        np.save(os.path.join(out, f"{split}.npy"), mask.numpy())
    for i, et in enumerate(data.edge_types):
        np.save(os.path.join(out, f"edge_{'__'.join(et)}.npy"),
                data[et].edge_index.numpy())
    with open(os.path.join(out, "num_nodes.txt"), "w") as f:
        for t in data.node_types:
            f.write(f"{t} {data[t].num_nodes}\\n")
    """
)


@pytest.mark.slow
def test_ogbmag_matches_pyg(tmp_path):
    """Field-by-field comparison with PyG's OGB_MAG (430 MB download). PyG
    reads the same raw files (linked), in a separate process to keep the peak
    memory low."""
    pytest.importorskip("torch_geometric")
    data = OGBMAG(base_dir=str(tmp_path / "mlx"))[0]

    raw = osp.join(str(tmp_path / "mlx"), "ogbn-mag", "raw", "mag")
    pyg_raw = tmp_path / "pyg" / "mag" / "raw"
    pyg_raw.mkdir(parents=True)
    for name in ("node-feat", "node-label", "relations", "num-node-dict.csv.gz"):
        os.symlink(osp.join(raw, "raw", name), pyg_raw / name)
    os.symlink(osp.join(raw, "split"), pyg_raw / "split")
    out = tmp_path / "dump"
    out.mkdir()
    subprocess.run(
        [sys.executable, "-c", _PYG_DUMP, str(tmp_path / "pyg"), str(out)],
        check=True,
    )

    def expected(name):
        return np.load(out / name, mmap_mode="r")

    with open(out / "num_nodes.txt") as f:
        num_nodes = dict(line.split() for line in f)
    assert {t: int(n) for t, n in num_nodes.items()} == data.num_nodes
    assert np.array_equal(np.array(data.node_features_dict["paper"]), expected("x.npy"))
    assert np.array_equal(np.array(data.node_labels_dict["paper"]), expected("y.npy"))
    assert np.array_equal(np.array(data.paper_year), expected("year.npy"))
    for split in ("train", "val", "test"):
        mask = np.array(getattr(data, f"paper_{split}_mask"))
        assert np.array_equal(mask, expected(f"{split}.npy"))
    assert len(data.edge_index_dict) == 4
    for et, edge_index in data.edge_index_dict.items():
        assert np.array_equal(
            np.array(edge_index), expected(f"edge_{'__'.join(et)}.npy")
        ), et
