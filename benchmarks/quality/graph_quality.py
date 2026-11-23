"""Training quality of mlx-graphs vs PyTorch Geometric: graph-level,
heterogeneous, link prediction and knowledge graph models.

    python benchmarks/quality/graph_quality.py                  # everything
    python benchmarks/quality/graph_quality.py --groups tu kge  # some groups
    python benchmarks/quality/graph_quality.py --summary        # table only

Each configuration (task, model, dataset, framework, seed) trains in its own
subprocess (``graph_worker.py``) under a memory watchdog that polls the
process footprint every 0.5 s (``top``'s MEM, which, unlike RSS, includes
Metal memory) and kills it above ``--memory-gb``. mlx-graphs runs on the GPU
and PyG on the CPU (``--mlx-procs`` / ``--pyg-procs`` processes at a time).
Results are appended to ``graph_results.jsonl`` (finished configurations are
skipped, so an interrupted run resumes), and the table is written to
``graph_results.md``.

Fairness rules (as in ``benchmarks/pyg_comparison``): same architecture,
hyperparameters and data (loaded once through mlx-graphs, see
``graph_data.py``), same splits, MLX ``Adam(bias_correction=True)``, test
metric taken at the best validation epoch, several seeds.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "pyg_comparison"))
# The process' physical footprint, the MEM column of `top` / Activity Monitor
# (which, unlike RSS, includes Metal memory), read directly from the kernel:
# `top -l 1` itself can take ~10 s on a busy machine, too slow for a watchdog.
from common import footprint_gb  # noqa: E402

RESULTS = os.path.join(HERE, "graph_results.jsonl")
SUMMARY = os.path.join(HERE, "graph_results.md")

# (task, model, dataset, epochs, seeds)
GROUPS = {
    "tu": [
        ("tu", m, ds, 100, 10 if ds == "MUTAG" else 5)
        for ds in ("MUTAG", "PROTEINS")
        for m in (
            "gine",
            "pna",
            "topk",
            "sag",
            "dgcnn",
            "multiaggr",
            "graphnorm",
            "set2set",
            "deepsets",
        )
        if not (m == "gine" and ds != "MUTAG")
    ],
    # NCI1 (4,110 graphs) did not fit in the time budget on a busy machine.
    "hetero": [
        ("han", "han", "IMDB", 200, 5),
        ("hgt", "hgt", "DBLP", 100, 5),
        ("rgat", "rgat", "DBLP-homogeneous", 100, 5),
        ("fastrgcn", "fastrgcn", "DBLP-homogeneous", 100, 5),
    ],
    "link": [
        ("gae", "gae", "Cora", 400, 5),
        ("gae", "vgae", "Cora", 400, 5),
        ("gae", "gae", "CiteSeer", 400, 5),
        ("gae", "vgae", "CiteSeer", 400, 5),
        ("gae", "gae-ttse", "Cora", 400, 5),
        ("lightgcn", "lightgcn", "MovieLens100K", 50, 5),
        ("dgi", "dgi", "Cora", 300, 5),
    ],
    "kge": [
        ("kge", m, "FB15k-237", 40 if m == "rotate" else 20, 5)
        for m in ("transe", "distmult", "complex", "rotate")
    ],
    "pointcloud": [
        ("pointcloud", m, "synthetic", 50, 5) for m in ("edgecnn", "dynedge")
    ],
}


def config_key(cfg, framework, seed):
    task, model, dataset, epochs, _ = cfg
    return f"{task} {model} {dataset} {epochs} {framework} {seed}"


def run_one(cfg, framework, seed, memory_gb, timeout_s):
    """Runs one worker under the memory watchdog; returns (result, status)."""
    task, model, dataset, epochs, _ = cfg
    cmd = [
        sys.executable,
        os.path.join(HERE, "graph_worker.py"),
        "--task", task,
        "--model", model,
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--framework", framework,
        "--seed", str(seed),
    ]  # fmt: skip
    env = dict(os.environ, PYTHONPATH=REPO)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=REPO,
    )
    # drain the pipes in the background so that the worker never blocks
    out, err = [], []
    readers = [
        threading.Thread(target=lambda: out.extend(process.stdout), daemon=True),
        threading.Thread(target=lambda: err.extend(process.stderr), daemon=True),
    ]
    for r in readers:
        r.start()
    start, peak = time.time(), 0.0
    while process.poll() is None:
        used = footprint_gb(process.pid)
        peak = max(peak, used)
        if used > memory_gb:
            process.kill()
            process.wait()
            return None, f"oom (> {memory_gb:.0f} GB)", peak
        if time.time() - start > timeout_s:
            process.kill()
            process.wait()
            return None, f"timeout (> {timeout_s:.0f} s)", peak
        time.sleep(0.5)
    for r in readers:
        r.join()
    if process.returncode != 0:
        lines = "".join(err).strip().splitlines() or ["?"]
        return None, f"error: {lines[-1][:300]}", peak
    lines = [line for line in out if line.startswith("{")]
    return json.loads(lines[-1]), "ok", peak


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 5)
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    return obj


def run_all(
    groups, memory_gb, timeout_s, procs, seeds_override=None, models=None, datasets=None
):
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
    queues = {"mlx": [], "pyg": []}
    for g in groups:
        for cfg in GROUPS[g]:
            if (models and cfg[1] not in models) or (
                datasets and cfg[2] not in datasets
            ):
                continue
            n_seeds = seeds_override or cfg[4]
            for seed in range(n_seeds):
                for fw in queues:
                    if config_key(cfg, fw, seed) not in done:
                        queues[fw].append((cfg, seed))
    print(f"{sum(len(q) for q in queues.values())} runs to do", flush=True)
    lock = threading.Lock()

    def worker(fw):
        while True:
            with lock:
                if not queues[fw]:
                    return
                cfg, seed = queues[fw].pop(0)
            t0 = time.time()
            result, status, peak = run_one(cfg, fw, seed, memory_gb, timeout_s)
            key = config_key(cfg, fw, seed)
            record = {
                "key": key,
                "task": cfg[0],
                "model": cfg[1],
                "dataset": cfg[2],
                "framework": fw,
                "seed": seed,
                "status": status,
                "peak_gb": round(peak, 2),
                **(_round(result) if result else {}),
            }
            with lock:
                with open(RESULTS, "a") as f:
                    f.write(json.dumps(record) + "\n")
                test = record.get("test")
                print(
                    f"{key} -> {status} test={test} ({time.time() - t0:.0f} s, "
                    f"{peak:.1f} GB)",
                    flush=True,
                )

    threads = [
        threading.Thread(target=worker, args=(fw,))
        for fw in queues
        for _ in range(procs[fw])
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

METRIC_NAMES = {
    "accuracy": "test accuracy (%)",
    "auc": "test AUC (%)",
    "recall@20": "test recall@20 (%)",
    "mrr": "test MRR (%)",
}
EXTRA = {"auc": "ap", "recall@20": "precision", "mrr": "hits10"}


HEADER = """# Training quality: graph-level, heterogeneous, link and KG models

mlx-graphs (MLX, GPU) vs PyTorch Geometric 2.8.1 (torch 2.15, CPU), trained
by `graph_quality.py` with the same architectures, hyperparameters, data
(loaded once through mlx-graphs and handed to both) and splits; MLX's Adam
with `bias_correction=True`; the test metric is read at the epoch with the
best validation metric. Mean ± sample std over seeds (each seed is a
different split for the TU, link and point cloud tasks, the same split in
both frameworks). "paired gap" is the mean ± standard error of the per-seed
difference mlx - PyG. Verdict: *pass* when the gap is within 1 point or
within 2 PyG standard deviations.
"""

FINDINGS = """
## Setups

- **TU graph classification** (`graph_tu.py`): MUTAG (10 seeds) and PROTEINS
  (5 seeds), random 80/10/10 split per seed, 100 epochs, Adam (lr 0.01,
  batches of 32; lr 0.0005 / batches of 60 for TopK/SAG as in PyG's
  `proteins_topk_pool.py`; lr 0.001 for DGCNN). Hidden size 64. `graphnorm`
  trains with `Dataloader(drop_last=True, follow_batch=["node_features"])`
  and reads the `node_features_batch` vector (PyG: `follow_batch=["x"]`).
- **HAN / IMDB**: PyG's `han_imdb.py` (metapaths movie-actor-movie and
  movie-director-movie, `HANConv(-1, 128, heads=8, dropout=0.6)`, Adam lr
  0.005, weight decay 0.001, 200 epochs).
- **HGT / DBLP**: PyG's `hgt_dblp.py` with 2 layers (`HGTConv(64, 64,
  heads=2)`), conference nodes with a constant feature, 100 epochs.
- **RGAT / FastRGCN**: DBLP converted with `HeteroGraphData.to_homogeneous()`
  (6 relations, 26,128 nodes; author and paper bag-of-words reduced to 32
  features by a fixed random projection so that PyG's per-edge weight
  tensors fit in memory), 2 layers of 16, authors classified, Adam lr 0.01,
  weight decay 5e-4, 100 epochs.
- **GAE / VGAE**: PyG's `autoencoder.py` (GCN encoder 32 -> 16, inner product
  decoder, Adam lr 0.01, 400 epochs) on splits made by mlx-graphs'
  `RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True)`
  (`gae-ttse`: `train_test_split_edges`); the splits were checked (no test
  or validation edge in the message passing graph, negatives are non-edges).
  AUC / AP by scikit-learn for both; mlx-graphs' own `GAE.test` gives the
  same numbers to 1e-15.
- **LightGCN / MovieLens100K**: u1.base / u1.test ratings >= 4, 64 dims,
  2 layers, BPR loss (lambda 1e-4), one uniform negative movie per positive,
  batches of 1024, Adam lr 0.001, 50 epochs, recall@20 / precision@20 over
  the test users (training items excluded); last epoch reported.
- **DGI / Cora**: PyG's `infomax_transductive.py` (GCN 512 + PReLU, 300
  epochs); embeddings probed by the same scikit-learn logistic regression
  in both frameworks.
- **KGE / FB15k-237**: PyG's `kge_fb15k_237.py` (50 dims, batches of 1000,
  TransE Adam 0.01, DistMult Adam 1e-4 + wd 1e-6, ComplEx Adagrad 1e-3 + wd
  1e-6, RotatE margin 9 Adam 1e-3) for 20 epochs (RotatE 40) instead of
  500; raw MRR / Hits@10 of the true tail among all 14,541 entities on fixed
  random subsets of 2,000 validation / test triplets, ranked by the same
  code in both frameworks.
- **Point clouds**: 1,000 synthetic 64-point clouds of 4 shapes, `EdgeCNN`
  on the 8-NN graph built by mlx-graphs' `knn_graph` (checked against a
  brute-force kNN), and 2 x `DynamicEdgeConv(k=8)`. PyG's `DynamicEdgeConv`
  needs `pyg-lib` / `torch-cluster`, which do not install here, so the PyG
  side runs `EdgeConv` on a torch kNN graph recomputed in feature space at
  every layer (the same graph as PyG's `knn(x, x, k).flip(0)`).

## Discrepancies found and fixed in mlx-graphs

1. **`TUDataset` dropped edge labels** (e.g. MUTAG's 4 bond types): the
   edge-labels-only branch assigned the (missing) edge attributes. Edge
   features were also not reordered when the edges were sorted and
   deduplicated. Fixed (edges and features go through `coalesce` together,
   as in PyG); MUTAG / PROTEINS / NCI1 now equal PyG's arrays exactly.
2. **`IMDB` labels were stored under `node_labels_dict["movies"]`** while the
   node type is `"movie"`, so `to_homogeneous()` dropped them and they could
   not be looked up by node type. Fixed.
3. **`HeteroDictLinear` (used by `HGTConv`) initialized with Glorot weights
   and zero biases**, PyG's with `torch.nn.Linear`'s init: HGTConv's
   projections were up to 1.7x larger and HGT on DBLP trained faster and
   reached 76.9 ± 2.3 % vs PyG's 71.8 ± 1.8 %. A forward pass with PyG's
   weights copied in matched PyG on the full DBLP graph, so only the init
   differed. Fixed; now 73.0 ± 2.5 % (row above).
4. **`GINEConv`'s edge projection initialized with Glorot / zero bias**,
   PyG's like `torch.nn.Linear` (std 1.75x larger for MUTAG's 4 -> 7
   projection). Found by comparing the initial parameter scales of every
   model of this harness in both libraries; fixed. (GINE / MUTAG before the
   fix: 80.0 ± 10.5 %.)

## Remaining, explained differences

- MLX's `Conv1d` starts with zero biases and MLX's `LSTM` has one bias vector
  where torch's has two (MLX core, not mlx-graphs). The harness initializes
  DGCNN's `Conv1d` biases like torch; `Set2Set`'s LSTM keeps MLX's init.
- TopK / SAG pooling: on PROTEINS many scores tie after the `tanh` (it
  saturates to exactly +-1 in float32 for structurally identical nodes),
  and the two libraries break ties differently, so with identical weights
  the pooled graphs differ in a few percent of the nodes. Both are valid;
  training quality is the same.
- MUTAG has only 19-20 test graphs per split, so its accuracies move in steps
  of 5 points and the largest gaps (SAG +6.0, MultiAggregation -5.0, of
  opposite signs) are within 1.5 paired standard errors; on PROTEINS the
  same models are within 2.5 points.
- EdgeCNN's training loss falls faster in mlx-graphs in the middle of
  training (median over 5 seeds). Trained from PyG's weights on the same
  batches without shuffling, both libraries follow the same loss curve
  (within 0.003 after 6 epochs), so this is seed-to-seed variance of when
  the model leaves its initial plateau, not a difference in the model.
- MLX's float32 matmul on this GPU (M5, `applegpu_g17s`) has a ~1e-3
  relative error (numpy / torch CPU: ~1e-7). It is the source of the
  ~1e-3 forward differences, and of gradient differences of up to ~10 % in
  EdgeConv's first layer (the `x_j - x_i` inputs cancel). It is MLX
  behaviour, not mlx-graphs', and did not change any result above.

## Not covered

- NCI1 (did not fit in the time budget on a heavily loaded machine).
- PyG's `DynamicEdgeConv`, `knn_graph` and neighbor sampling need `pyg-lib`
  or `torch-cluster`, which have no wheels here (see the point cloud setup).
- KGE models were trained for 20-40 epochs, not PyG's 500: the comparison is
  of the early trajectories (losses and MRR match to the third decimal), not
  of converged models.
"""


def summarize():
    records = []
    with open(RESULTS) as f:
        records = [json.loads(line) for line in f if line.strip()]
    groups = {}
    for r in records:
        groups.setdefault((r["task"], r["model"], r["dataset"]), {}).setdefault(
            r["framework"], []
        ).append(r)
    order = [(c[0], c[1], c[2]) for g in GROUPS.values() for c in g]
    rows = []
    lines = [
        "| model | dataset | metric | PyG | mlx-graphs | gap | paired gap | "
        "train loss at 10% / 50% / 100% of the epochs, PyG ; mlx | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in order:
        if key not in groups:
            continue
        runs = groups[key]
        ok = {
            fw: [r for r in runs.get(fw, []) if r["status"] == "ok"]
            for fw in ("pyg", "mlx")
        }
        failed = [
            r for fw in ("pyg", "mlx") for r in runs.get(fw, []) if r["status"] != "ok"
        ]
        if not ok["pyg"] or not ok["mlx"]:
            status = failed[0]["status"] if failed else "missing"
            lines.append(f"| {key[1]} | {key[2]} | | | | | | | {status} |")
            continue
        metric = ok["pyg"][0]["metric"]
        vals = {fw: np.array([100 * r["test"] for r in ok[fw]]) for fw in ok}
        mean = {fw: v.mean() for fw, v in vals.items()}
        std = {fw: v.std(ddof=1) if len(v) > 1 else 0.0 for fw, v in vals.items()}
        gap = mean["mlx"] - mean["pyg"]
        # paired by seed: same split / data in both frameworks
        by_seed = {fw: {r["seed"]: 100 * r["test"] for r in ok[fw]} for fw in ok}
        seeds = sorted(set(by_seed["pyg"]) & set(by_seed["mlx"]))
        diffs = np.array([by_seed["mlx"][s] - by_seed["pyg"][s] for s in seeds])
        paired = (
            f"{diffs.mean():+.1f} ± {diffs.std(ddof=1) / np.sqrt(len(diffs)):.1f}"
            if len(diffs) > 1
            else "-"
        )
        # mean training loss at 10%, 50% and 100% of the epochs
        curves = {fw: np.mean([r["loss_curve"] for r in ok[fw]], axis=0) for fw in ok}
        points = [max(0, int(f * len(curves["pyg"])) - 1) for f in (0.1, 0.5, 1.0)]
        loss = {fw: " / ".join(f"{curves[fw][i]:.3g}" for i in points) for fw in ok}
        within = abs(gap) <= 1.0 or abs(gap) <= 2 * max(std["pyg"], 1e-9)
        verdict = "pass" if within else ("**better**" if gap > 0 else "**FAIL**")
        extra = ""
        if metric in EXTRA:
            name = EXTRA[metric]
            ev = {fw: 100 * np.mean([r[f"test_{name}"] for r in ok[fw]]) for fw in ok}
            extra = f" ({name}: {ev['pyg']:.1f} / {ev['mlx']:.1f})"
        n = f"n={len(vals['pyg'])}/{len(vals['mlx'])}"
        lines.append(
            f"| {key[1]} | {key[2]} | {METRIC_NAMES.get(metric, metric)}{extra} | "
            f"{mean['pyg']:.1f} ± {std['pyg']:.1f} | "
            f"{mean['mlx']:.1f} ± {std['mlx']:.1f} | {gap:+.1f} | {paired} | "
            f"{loss['pyg']} ; {loss['mlx']} | {verdict} ({n}) |"
        )
        rows.append((key, verdict))
    table = "\n".join(lines)
    return table, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--groups", nargs="+", default=list(GROUPS), choices=list(GROUPS)
    )
    parser.add_argument("--memory-gb", type=float, default=6.0)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--seeds", type=int, default=None, help="override seed count")
    parser.add_argument("--models", nargs="+", default=None, help="only these models")
    parser.add_argument("--datasets", nargs="+", default=None, help="only these")
    parser.add_argument("--mlx-procs", type=int, default=1, help="concurrent MLX runs")
    parser.add_argument("--pyg-procs", type=int, default=2, help="concurrent PyG runs")
    parser.add_argument("--summary", action="store_true", help="only print the table")
    args = parser.parse_args()
    if not args.summary:
        procs = {"mlx": args.mlx_procs, "pyg": args.pyg_procs}
        run_all(
            args.groups,
            args.memory_gb,
            args.timeout_s,
            procs,
            args.seeds,
            args.models,
            args.datasets,
        )
    table, _ = summarize()
    print(table)
    with open(SUMMARY, "w") as f:
        f.write(HEADER + "\n" + table + "\n" + FINDINGS)


if __name__ == "__main__":
    main()
