"""Are the large-graph datasets of mlx-graphs identical to PyG's?

    python benchmarks/run_capped.py --gb 24 -- \\
        python benchmarks/quality/dataset_parity.py --data-dir .mlx_graphs_data \\
        --datasets flickr reddit2 yelp reddit ppi amazon_products

For every dataset, the mlx-graphs version (``<data-dir>/<name>``) and PyG's
(``<data-dir>/pyg/<Name>``) are loaded, one after the other, and compared:
number of nodes and edges, the edges themselves, node features, labels and
split masks (all exactly). Raw files are downloaded on first use, by each
library. Every dataset runs in its own subprocess, so that memory is given
back between datasets.

PPI: PyG reads DGL's ``ppi.zip``, which can no longer be downloaded
(``data.dgl.ai`` is down). mlx-graphs reads the GraphSAGE release instead
and recovers the tissue graphs from the node order (see
:class:`~mlx_graphs.datasets.PPI`). For the comparison, PyG's raw files
are written from the GraphSAGE release with those graph ids, so the PPI row
checks the processing (edges, features, labels, graph sizes) against PyG's
code, and the graph sizes against the published statistics, not the graph
ids against DGL's files.

Results are printed as a markdown table.
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

NAMES = {
    "flickr": ("Flickr", "Flickr"),
    "reddit2": ("Reddit2", "Reddit2"),
    "yelp": ("Yelp", "Yelp"),
    "amazon_products": ("AmazonProducts", "AmazonProducts"),
    "reddit": ("Reddit", "Reddit"),
}


def _equal(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    chunk = 1 << 24
    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    for start in range(0, flat_a.size, chunk):
        if not np.array_equal(
            flat_a[start : start + chunk], flat_b[start : start + chunk]
        ):
            return False
    return True


def compare_single(name: str, data_dir: str) -> dict:
    import mlx_graphs.datasets as mxd

    mx_cls, pyg_cls = NAMES[name]
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        graph = getattr(mxd, mx_cls)(base_dir=data_dir)[0]
    t_mx = time.perf_counter() - t0
    mine = {
        "num_nodes": graph.num_nodes,
        "edge_index": np.asarray(graph.edge_index),
        "x": np.asarray(graph.node_features),
        "y": np.asarray(graph.node_labels),
        "masks": {
            m: np.asarray(getattr(graph, m))
            for m in ("train_mask", "val_mask", "test_mask")
        },
    }
    del graph

    import torch_geometric.datasets as pygd

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        ref = getattr(pygd, pyg_cls)(os.path.join(data_dir, "pyg", pyg_cls))[0]
    t_pyg = time.perf_counter() - t0

    row = {
        "dataset": mx_cls,
        "nodes": f"{mine['num_nodes']:,}",
        "edges": f"{mine['edge_index'].shape[1]:,}",
        "same #nodes/#edges": mine["num_nodes"] == ref.num_nodes
        and mine["edge_index"].shape[1] == ref.edge_index.shape[1],
        "same edges": _equal(mine["edge_index"], ref.edge_index.numpy()),
        "same x": _equal(mine["x"], ref.x.numpy()),
        "same y": _equal(mine["y"], ref.y.numpy().astype(mine["y"].dtype))
        and mine["y"].shape == tuple(ref.y.shape),
        "y": f"{tuple(mine['y'].shape)} {mine['y'].dtype} (PyG {ref.y.dtype})",
        "same masks": all(
            _equal(v, getattr(ref, k).numpy()) for k, v in mine["masks"].items()
        ),
        "split sizes": "/".join(f"{int(v.sum()):,}" for v in mine["masks"].values()),
        "load s (mlx-graphs / PyG)": f"{t_mx:.1f} / {t_pyg:.1f}",
    }
    return row


def write_pyg_ppi_raw(raw_dir: str, mlx_raw_root: str):
    """PyG's raw PPI files (DGL layout) written from the GraphSAGE release,
    with the graph ids recovered by mlx-graphs."""
    from mlx_graphs.datasets.large_graph import _ppi_graph_ids

    os.makedirs(raw_dir, exist_ok=True)
    with open(os.path.join(mlx_raw_root, "ppi-G.json")) as f:
        graph = json.load(f)
    nodes = graph["nodes"]
    n = len(nodes)
    is_val = np.array([n_["val"] for n_ in nodes])
    is_test = np.array([n_["test"] for n_ in nodes])
    src = np.array([e["source"] for e in graph["links"]], dtype=np.int64)
    dst = np.array([e["target"] for e in graph["links"]], dtype=np.int64)
    x = np.load(os.path.join(mlx_raw_root, "ppi-feats.npy"))
    with open(os.path.join(mlx_raw_root, "ppi-class_map.json")) as f:
        class_map = json.load(f)
    y = np.array([class_map[str(i)] for i in range(n)], dtype=np.int64)
    keep = src != dst
    graph_id = _ppi_graph_ids(n, src[keep], dst[keep])
    split = np.where(is_test, 2, np.where(is_val, 1, 0))
    for s, name in enumerate(["train", "valid", "test"]):
        ids = np.nonzero(split == s)[0]
        start = ids[0]
        local = np.full(n, -1)
        local[ids] = np.arange(ids.size)
        sel = (local[src] >= 0) & (local[dst] >= 0)
        links = [
            {"source": int(a), "target": int(b)}
            for a, b in zip(local[src[sel]], local[dst[sel]])
        ]
        g = {
            "directed": False,
            "multigraph": False,
            "graph": {},
            "nodes": [{"id": int(i)} for i in range(ids.size)],
            "links": links,
        }
        with open(os.path.join(raw_dir, f"{name}_graph.json"), "w") as f:
            json.dump(g, f)
        np.save(os.path.join(raw_dir, f"{name}_feats.npy"), x[ids])
        np.save(os.path.join(raw_dir, f"{name}_labels.npy"), y[ids])
        np.save(
            os.path.join(raw_dir, f"{name}_graph_id.npy"),
            graph_id[ids] + 1 - graph_id[start],
        )


def compare_ppi(data_dir: str) -> dict:
    from mlx_graphs.datasets import PPI

    rows = []
    for split in ("train", "val", "test"):
        with contextlib.redirect_stdout(io.StringIO()):
            ds = PPI(split, base_dir=data_dir)
        rows.append(ds)
    raw_dir = os.path.join(data_dir, "pyg", "PPI", "raw")
    if not os.path.exists(os.path.join(raw_dir, "test_graph_id.npy")):
        write_pyg_ppi_raw(raw_dir, os.path.join(data_dir, "ppi", "raw", "ppi"))
    from torch_geometric.datasets import PPI as PyGPPI

    ok_edges = ok_x = ok_y = True
    counts = []
    for split, mine in zip(("train", "val", "test"), rows):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            ref = PyGPPI(os.path.join(data_dir, "pyg", "PPI"), split=split)
        counts.append(len(mine))
        assert len(ref) == len(mine)
        for g, r in zip(mine, ref):
            e = np.asarray(g.edge_index).astype(np.int64)
            re = r.edge_index.numpy()
            order = np.lexsort((re[1], re[0]))
            ok_edges &= _equal(e, re[:, order])
            ok_x &= _equal(np.asarray(g.node_features), r.x.numpy())
            ok_y &= _equal(np.asarray(g.node_labels), r.y.numpy())
    train = rows[0]
    num_nodes = sum(g.num_nodes for g in train)
    num_edges = sum(g.num_edges for g in train)
    return {
        "dataset": "PPI",
        "nodes": f"{num_nodes:,} (train)",
        "edges": f"{num_edges:,} (train)",
        "same #nodes/#edges": num_nodes == 44906 and num_edges == 1226368,
        "same edges": ok_edges,
        "same x": ok_x,
        "same y": ok_y,
        "y": "(n, 121) float32",
        "same masks": "-",
        "split sizes": "/".join(str(c) for c in counts) + " graphs",
        "load s (mlx-graphs / PyG)": "-",
    }


def table(rows):
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(str(r[h]) for h in header) + " |")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default=os.path.join(REPO_ROOT, ".mlx_graphs_data")
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["flickr", "reddit2", "yelp", "ppi"]
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)

    if args.worker:
        row = (
            compare_ppi(data_dir)
            if args.worker == "ppi"
            else compare_single(args.worker, data_dir)
        )
        print("ROW " + json.dumps(row), flush=True)
        return

    rows = []
    for name in args.datasets:
        out = subprocess.run(
            [sys.executable, __file__, "--data-dir", data_dir, "--worker", name],
            capture_output=True,
            text=True,
        )
        lines = [line for line in out.stdout.splitlines() if line.startswith("ROW ")]
        if out.returncode != 0 or not lines:
            print(
                f"{name}: failed\n{out.stdout[-2000:]}\n{out.stderr[-4000:]}",
                flush=True,
            )
            continue
        rows.append(json.loads(lines[-1][4:]))
        print(f"{name}: done", flush=True)
    if rows:
        table(rows)


if __name__ == "__main__":
    main()
