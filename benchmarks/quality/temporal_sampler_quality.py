"""Temporal neighbor sampling of mlx-graphs against PyG (pyg-lib).

    PYTHONPATH=<pyg-lib dir> python benchmarks/quality/temporal_sampler_quality.py

On a random graph with node-level timestamps (and on the same graph with
edge-level timestamps and random seed times):

- ``violations``: sampled edges whose time is later than their seed's (must
  be 0).
- ``last``: fraction of mini-batches identical to PyG's (``n_id``, ``batch``,
  ``edge_index``, ``e_id``) with ``temporal_strategy="last"``.
- ``uniform``: over ``--reps`` epochs, how often every valid in-edge of a
  seed is sampled (one hop, fan-out ``k``), compared with the exact
  probability ``min(1, k / num_valid)``: mean absolute error of the
  frequencies, and the p-value of a chi-square test of the counts of the
  seeds with more valid neighbors than ``k``; the same numbers for PyG.
- time per epoch of the loader (2 hops), both libraries on the CPU host.
"""

import argparse
import os
import sys
import time as time_mod

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import mlx.core as mx  # noqa: E402
import torch  # noqa: E402
from scipy.stats import chisquare  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import NeighborLoader as PyGNeighborLoader  # noqa: E402

from mlx_graphs.data import GraphData  # noqa: E402
from mlx_graphs.loaders import NeighborLoader  # noqa: E402


def make_graph(num_nodes, num_edges, level, seed):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, num_nodes, num_edges)
    dst = rng.integers(0, num_nodes, num_edges)
    time = rng.integers(0, 1000, num_nodes if level == "node" else num_edges)
    return src, dst, time


def loaders(src, dst, time, seeds, input_time, fanout, strategy, batch_size):
    mdata = GraphData(
        edge_index=mx.array(np.stack([src, dst]).astype(np.int32)),
        time=mx.array(time),
        num_nodes=int(max(src.max(), dst.max())) + 1,
    )
    pdata = Data(
        edge_index=torch.tensor(np.stack([src, dst])),
        time=torch.tensor(time),
        num_nodes=mdata.num_nodes,
    )
    kw = dict(time_attr="time", temporal_strategy=strategy, batch_size=batch_size)
    ml = NeighborLoader(
        mdata,
        fanout,
        input_nodes=mx.array(seeds),
        input_time=None if input_time is None else mx.array(input_time),
        **kw,
    )
    pl = PyGNeighborLoader(
        pdata,
        fanout,
        input_nodes=torch.tensor(seeds),
        input_time=None if input_time is None else torch.tensor(input_time),
        **kw,
    )
    return ml, pl


def as_np(batch, key):
    v = getattr(batch, key)
    return v.numpy() if hasattr(v, "numpy") else np.asarray(v)


def violations(batch, level, time, seed_times):
    n_id, e_id, bv = (as_np(batch, k) for k in ("n_id", "e_id", "batch"))
    src_local = as_np(batch, "edge_index")[0]
    t = time[n_id[src_local]] if level == "node" else time[e_id]
    return int((t > seed_times[bv[src_local]]).sum())


def run(level, args):
    src, dst, time = make_graph(args.num_nodes, args.num_edges, level, 0)
    rng = np.random.default_rng(1)
    seeds = rng.choice(args.num_nodes, args.num_seeds, replace=False)
    input_time = rng.integers(200, 1000, seeds.size) if level == "edge" else None
    seed_times = time[seeds] if level == "node" else input_time
    rows = {}

    # last: identical mini-batches, and no violation
    ml, pl = loaders(src, dst, time, seeds, input_time, [10, 5], "last", 512)
    same = total = 0
    for mb, pb in zip(ml, pl):
        total += 1
        ok = all(
            np.array_equal(as_np(mb, k), as_np(pb, k))
            for k in ("n_id", "batch", "edge_index", "e_id")
        )
        same += ok
    rows["last: identical batches"] = f"{same}/{total}"

    # uniform: frequencies and time per epoch
    k = args.k
    ml, pl = loaders(src, dst, time, seeds, input_time, [k], "uniform", 512)
    num_edges = src.size
    # exact probability of every (seed position, edge) pair
    order = np.argsort(dst, kind="stable")
    res = {}
    for name, loader in (("mlx-graphs", ml), ("PyG", pl)):
        counts = {}
        viol = 0
        time_taken = 0.0
        for rep in range(args.reps):
            np.random.seed(rep)
            torch.manual_seed(rep)
            t0 = time_mod.perf_counter()
            pos = 0
            for b in loader:
                bs = int(b.batch_size)
                st = seed_times[pos : pos + bs]
                viol += violations(b, level, time, st)
                e_id = as_np(b, "e_id")
                bv = as_np(b, "batch")[as_np(b, "edge_index")[1]]
                key = (pos + bv) * num_edges + e_id
                u, c = np.unique(key, return_counts=True)
                for kk, cc in zip(u.tolist(), c.tolist()):
                    counts[kk] = counts.get(kk, 0) + cc
                pos += bs
            time_taken += time_mod.perf_counter() - t0
        res[name] = (counts, viol, time_taken / args.reps)

    # expected probabilities
    errs = {name: [] for name in res}
    pvals = {name: [] for name in res}
    for i, s in enumerate(seeds):
        edges = order[
            np.searchsorted(dst[order], s) : np.searchsorted(dst[order], s, "right")
        ]
        et = time[src[edges]] if level == "node" else time[edges]
        valid = edges[et <= seed_times[i]]
        if valid.size == 0:
            continue
        p = min(1.0, k / valid.size)
        for name, (counts, _, _) in res.items():
            c = np.array([counts.get(i * num_edges + e, 0) for e in valid.tolist()])
            errs[name].append(np.abs(c / args.reps - p).mean())
            if valid.size > k and i < 200:
                pvals[name].append(chisquare(c).pvalue)
    for name, (counts, viol, t) in res.items():
        rows[f"{name}: violations"] = viol
        rows[f"{name}: uniform mean abs(freq - p)"] = f"{np.mean(errs[name]):.4f}"
        rows[
            f"{name}: chi-square p < 0.01"
        ] = f"{np.mean(np.array(pvals[name]) < 0.01):.3f} of {len(pvals[name])} seeds"
        rows[f"{name}: s / epoch (1 hop, k={k})"] = f"{t:.3f}"

    # 2-hop loader speed
    for strategy in ("uniform", "last"):
        ml, pl = loaders(src, dst, time, seeds, input_time, [10, 5], strategy, 512)
        for name, loader in (("mlx-graphs", ml), ("PyG", pl)):
            t0 = time_mod.perf_counter()
            for _ in loader:
                pass
            rows[
                f"{name}: s / epoch ([10, 5], {strategy})"
            ] = f"{time_mod.perf_counter() - t0:.3f}"
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-nodes", type=int, default=20000)
    parser.add_argument("--num-edges", type=int, default=400000)
    parser.add_argument("--num-seeds", type=int, default=4096)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = {level: run(level, args) for level in ("node", "edge")}
    keys = list(results["node"])
    print("| check | node-level time | edge-level time |")
    print("|---|---|---|")
    for key in keys:
        print(f"| {key} | {results['node'][key]} | {results['edge'][key]} |")


if __name__ == "__main__":
    main()
