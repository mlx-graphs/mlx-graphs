"""Do node classification models built with mlx-graphs train as well as their
PyTorch Geometric equivalents?

    python benchmarks/quality/node_quality.py                    # everything
    python benchmarks/quality/node_quality.py --configs arma gen # some configs
    python benchmarks/quality/node_quality.py --summarize        # table only

For every configuration (a model, its hyperparameters and the datasets it
runs on) the same model is trained in both frameworks (``node_models.py``)
over several seeds, on the public Planetoid splits, with the test accuracy
taken at the epoch of best validation accuracy, and the training loss
recorded at every epoch. Fairness rules are those of
``benchmarks/pyg_comparison``: same architecture, hyperparameters and arrays;
MLX ``Adam(bias_correction=True)``; torch's Adam ``weight_decay`` reproduced
in MLX as the same L2 term added to the gradients.

Every (configuration, dataset, framework, seed) runs in its own subprocess
(``node_worker.py``) under a hard memory cap (``--memory-gb``, 6 GB by
default): the watchdog polls the process' physical footprint (the "Memory"
column of ``top`` / Activity Monitor, which includes Metal allocations,
unlike RSS) and kills it above the cap. Workers run in three pools: PyG on
the CPU (``--cpu-jobs``, one torch thread each), mlx-graphs on the GPU
(``--gpu-jobs``) and, for the slow deep models (GCNII, DeeperGCN), PyG on
MPS (``--mps-jobs``); the device does not matter for accuracy. On top of
that, every worker waits before importing torch / MLX until it holds one of
``NODE_QUALITY_SLOTS`` (10) slots shared by all the workers, and the machine
has ``NODE_QUALITY_MIN_FREE_GB`` (3 GB) of unused memory (``top``), so the
harness backs off when other jobs need the memory.

Results are appended to ``node_results.jsonl``, one line per seed (seeds
already run are skipped), merged per configuration and summarized in
``node_results.md``.
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
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(HERE, "node_results.jsonl")
SUMMARY = os.path.join(HERE, "node_results.md")

PLANETOID = ["cora", "citeseer", "pubmed"]
SEEDS = [0, 1, 2, 3, 4]

# Default hyperparameters: Kipf & Welling's / PyG's examples on Planetoid.
BASE = dict(
    hidden=64, dropout=0.5, lr=0.01, wd=5e-4, epochs=200, eval_every=1
)  # fmt: skip


def cfg(model, desc, datasets=PLANETOID, features="raw", kind="train", **hp):
    per_dataset = hp.pop("per_dataset", {})
    return {
        "model": model,
        "desc": desc,
        "datasets": datasets,
        "features": features,
        "kind": kind,
        "hp": {**BASE, **hp},
        "per_dataset": per_dataset,
    }


CS = dict(
    num_correction_layers=50, correction_alpha=0.8,
    num_smoothing_layers=50, smoothing_alpha=0.8, autoscale=True,
)  # fmt: skip

CONFIGS = {
    # reference: a layer that existed before, to calibrate the harness
    "gcn": cfg("gcn", "GCNConv x2 (reference)", hidden=16),
    # --- new convolution layers -------------------------------------------
    "gcn2": cfg(
        "gcn2",
        "GCNII: Linear + GCN2Conv x64 + Linear (PyG gcn2_cora)",
        layers=64,
        alpha=0.1,
        theta=0.5,
        dropout=0.6,
        wd=5e-4,
        wd_conv=0.01,
        epochs=500,
        eval_every=5,
        per_dataset={
            "citeseer": dict(layers=32, hidden=256, theta=0.6),
            "pubmed": dict(layers=16, hidden=256, theta=0.4, dropout=0.5, wd_conv=5e-4),
        },
    ),  # fmt: skip
    "ssg": cfg(
        "ssg",
        "SSGConv(K=16, alpha=0.05) -> classes (SSGC)",
        K=16,
        alpha=0.05,
        lr=0.2,
        wd=1e-5,
        dropout=0.0,
        epochs=100,
    ),  # fmt: skip
    "lg": cfg("lg", "LGConv x2 (no params) + 2-layer MLP head", K=2),
    "arma": cfg("arma", "ARMAConv x2 (3 stacks, 2 layers; PyG arma.py)", hidden=16),
    "gated": cfg("gated", "Linear + GatedGraphConv(2 steps) + Linear"),
    "resgated": cfg("resgated", "ResGatedGraphConv x2"),
    "nn": cfg(
        "nn", "Linear + NNConv(mean, 2 synthetic edge feats) + Linear", hidden=16
    ),
    "cg": cfg("cg", "Linear + CGConv(mean, 2 synthetic edge feats) + Linear"),
    "mf": cfg("mf", "MFConv x2"),
    "le": cfg("le", "LEConv x2"),
    "film": cfg("film", "FiLMConv x2"),
    "agnn": cfg("agnn", "Linear + AGNNConv x2 + Linear (PyG agnn.py)", hidden=16),
    "clustergcn": cfg("clustergcn", "ClusterGCNConv x2 (full batch)"),
    "gen": cfg("gen", "Linear + GENConv(softmax, learn_t, LN) x2 + Linear"),
    "mixhop": cfg("mixhop", "MixHopConv([0,1,2]) x2 + Linear", hidden=32),
    "fa": cfg(
        "fa",
        "FAGCN: Linear + FAConv x4 (eps 0.3) + Linear",
        layers=4,
        eps=0.3,
        att_dropout=0.4,
        dropout=0.6,
        wd=1e-3,
    ),  # fmt: skip
    "general": cfg("general", "GeneralConv(mean) x2"),
    "general_att": cfg(
        "general_att", "GeneralConv(add, additive attention) x2", datasets=["cora"]
    ),
    "dir_gcn": cfg("dir_gcn", "DirGNNConv(GCNConv) x2"),
    "dir_sage": cfg("dir_sage", "DirGNNConv(SAGEConv) x2"),
    "pna_conv": cfg("pna_conv", "Linear + PNAConv(mean/min/max/std, 3 scalers) x2"),
    "sage_multi": cfg("sage_multi", 'SAGEConv(aggr=["mean", "max"]) x2'),
    "gen_multi": cfg(
        "gen_multi",
        'as gen, aggr=["softmax", "mean", "max"] (+ lin_aggr_out)',
        datasets=["cora", "citeseer"],
    ),
    # --- normalization layers ---------------------------------------------
    "nonorm": cfg(
        "nonorm",
        "GCNConv x4, no norm (baseline of the norms)",
        datasets=["cora", "citeseer"],
        layers=4,
    ),  # fmt: skip
    "graphnorm": cfg(
        "graphnorm", "GCNConv x4 + GraphNorm", datasets=["cora", "citeseer"], layers=4
    ),
    "pairnorm": cfg(
        "pairnorm", "GCNConv x4 + PairNorm", datasets=["cora", "citeseer"], layers=4
    ),
    "diffgroupnorm": cfg(
        "diffgroupnorm",
        "GCNConv x4 + DiffGroupNorm(groups=classes)",
        datasets=["cora", "citeseer"],
        layers=4,
    ),  # fmt: skip
    "gen_msgnorm": cfg(
        "gen_msgnorm",
        "as gen, with MessageNorm (learn_msg_scale)",
        datasets=["cora", "citeseer"],
    ),  # fmt: skip
    # --- models --------------------------------------------------------------
    "label_prop": cfg(
        "mlp",
        "LabelPropagation(50 layers, alpha 0.9), train labels",
        kind="label_prop",
        num_layers=50,
        alpha=0.9,
    ),  # fmt: skip
    "correct_and_smooth": cfg(
        "mlp",
        "MLP + CorrectAndSmooth(50/0.8, 50/0.8, autoscale)",
        kind="correct_and_smooth",
        cs=CS,
    ),  # fmt: skip
    "linkx": cfg("linkx", "LINKX(2 layers)", datasets=["cora", "citeseer"]),
    "deepgcn": cfg(
        "deepgcn",
        "DeeperGCN: 7 x DeepGCNLayer(GENConv, LayerNorm, res+)",
        datasets=["cora", "citeseer"],
        layers=7,
        layer_dropout=0.1,
    ),  # fmt: skip
    "masklabel": cfg(
        "masklabel",
        "Linear + MaskLabel(add, ratio 0.5) + GCNConv x2",
        datasets=["cora", "citeseer"],
        mask_ratio=0.5,
    ),  # fmt: skip
    "pna": cfg("pna", "Linear + PNA model (2 layers)", datasets=["cora", "citeseer"]),
    # --- transforms ----------------------------------------------------------
    "normfeat_gcn": cfg(
        "gcn",
        "NormalizeFeatures + GCNConv x2 (PyG gcn.py)",
        features="normalize",
        hidden=16,
    ),  # fmt: skip
    "sign_mlp": cfg(
        "sign_mlp",
        "NormalizeFeatures + SIGN(K=3) + SIGN-MLP",
        datasets=["cora", "citeseer"],
        features="sign",
        K=3,
    ),  # fmt: skip
    "pe_ldp_gcn": cfg(
        "gcn",
        "x + AddLaplacianEigenvectorPE(8) + LocalDegreeProfile, GCNConv x2",
        datasets=["cora"],
        features="pe_ldp",
        hidden=16,
        pe_k=8,
    ),  # fmt: skip
}

FRAMEWORKS = ("pyg", "mlx")


# ---------------------------------------------------------------------------
# Running, under a memory cap
# ---------------------------------------------------------------------------


def footprint_gb(pid):
    """Physical footprint of a process in GB, as ``top -stats mem`` shows it
    (includes Metal allocations). Same helper as benchmarks/pyg_comparison."""
    sys.path.insert(0, os.path.join(HERE, "..", "pyg_comparison"))
    from common import footprint_gb as fp

    return fp(pid)


def run_one(argv, memory_gb, timeout_s):
    """Runs one worker; returns (result dict or None, status, peak GB).
    The pattern of ``benchmarks/pyg_comparison/run.py``'s ``run_one``."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    cmd = [sys.executable, os.path.join(HERE, "node_worker.py"), *argv]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, cwd=REPO_ROOT,
    )  # fmt: skip
    # drain the pipes in the background so a chatty worker cannot block
    out_chunks, err_chunks = [], []
    readers = [
        threading.Thread(target=lambda: out_chunks.append(process.stdout.read())),
        threading.Thread(target=lambda: err_chunks.append(process.stderr.read())),
    ]
    for r in readers:
        r.start()
    start, peak = time.time(), 0.0
    status = None
    while process.poll() is None:
        used = footprint_gb(process.pid)
        peak = max(peak, used)
        if used > memory_gb:
            process.kill()
            status = f"oom (> {memory_gb:.0f} GB)"
            break
        if time.time() - start > timeout_s:
            process.kill()
            status = f"timeout (> {timeout_s:.0f} s)"
            break
        time.sleep(0.2)
    process.wait()
    for r in readers:
        r.join()
    if status is not None:
        return None, status, peak
    out, err = "".join(out_chunks), "".join(err_chunks)
    if process.returncode != 0:
        lines = err.strip().splitlines() or ["?"]
        return None, f"error: {lines[-1][:300]}", peak
    return json.loads(out.strip().splitlines()[-1]), "ok", peak


def load_results():
    """Merges the records of ``node_results.jsonl`` by (config, dataset,
    framework): one run per seed, the last successful one of each seed.
    Failed attempts are kept (``errors``) only for the seeds that never
    succeeded."""
    merged = {}
    if not os.path.exists(RESULTS):
        return merged
    with open(RESULTS) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    for r in lines:
        key = (r["config"], r["dataset"], r["framework"])
        m = merged.setdefault(key, {"runs": {}, "errors": {}, "devices": set()})
        seeds = r.get("seeds", [])
        if r["status"] == "ok":
            for seed, run in zip(seeds, r["runs"]):
                m["runs"][seed] = run
                m["errors"].pop(seed, None)
            m["devices"].add(r.get("device", "cpu" if key[2] == "pyg" else "gpu"))
        else:
            for seed in seeds:
                if seed not in m["runs"]:
                    m["errors"][seed] = r["status"]
    return merged


def compact(result):
    """Keeps the loss curve every 5 epochs to keep the file small."""
    for run in result.get("runs", []):
        if "loss_curve" in run:
            curve = run.pop("loss_curve")
            run["loss_at"] = {
                str(e): curve[e - 1]
                for e in (1, 10, 25, 50, 100, 200, len(curve))
                if e <= len(curve)
            }
            run["loss_curve_every5"] = curve[4::5]
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def stats(values):
    v = np.asarray(values, dtype=float) * 100
    return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0


def runs_of(record):
    return [record["runs"][s] for s in sorted(record["runs"])]


def mean_loss(record, first, last):
    """Mean training loss over epochs ``first``..``last`` (sampled every 5
    epochs) and seeds: single epochs are noisy (dropout, loss spikes)."""
    vals = [
        v
        for r in runs_of(record)
        for i, v in enumerate(r.get("loss_curve_every5", []))
        if first <= 5 * (i + 1) <= last
    ]
    return float(np.mean(vals)) if vals else None


def verdict(gap, pyg_std, mlx_std, n):
    if abs(gap) <= 1.0 or abs(gap) <= 2 * pyg_std:
        return "pass"
    # also accept a gap within two standard errors of the difference
    se = np.sqrt((pyg_std**2 + mlx_std**2) / max(n, 1))
    return "pass (noise)" if abs(gap) <= 2 * se else "**MISMATCH**"


def summarize(records, notes=None):
    rows = []
    for name, c in CONFIGS.items():
        for ds in c["datasets"]:
            pyg, mlx = records.get((name, ds, "pyg")), records.get((name, ds, "mlx"))
            if pyg is None or mlx is None:
                continue
            if not pyg["runs"] or not mlx["runs"]:
                err = [
                    "; ".join(sorted(set(r["errors"].values()))) or "-"
                    for r in (pyg, mlx)
                ]
                rows.append(
                    f"| {name} | {c['desc']} | {ds} | {err[0]} | {err[1]} "
                    "| | | | | failed |"
                )
                continue
            pruns, mruns = runs_of(pyg), runs_of(mlx)
            pm, ps = stats([r["test_acc"] for r in pruns])
            mm, ms = stats([r["test_acc"] for r in mruns])
            gap = mm - pm
            n = min(len(pruns), len(mruns))
            lp, lm = mean_loss(pyg, 25, 75), mean_loss(mlx, 25, 75)
            fp = np.mean([r.get("final_train_loss", np.nan) for r in pruns])
            fm = np.mean([r.get("final_train_loss", np.nan) for r in mruns])
            loss_cell = (
                f"{lp:.3f} / {lm:.3f} | {fp:.3f} / {fm:.3f}"
                if lp is not None
                else "- | -"
            )
            extra = ""
            if c["kind"] == "correct_and_smooth":
                bp, _ = stats([r["base_test_acc"] for r in pruns])
                bm, _ = stats([r["base_test_acc"] for r in mruns])
                extra = f" (MLP: {bp:.1f} / {bm:.1f})"
            if "mps" in pyg["devices"]:
                extra += " [PyG on MPS]"
            seeds = (
                f"{len(pruns)}"
                if len(pruns) == len(mruns)
                else f"{len(pruns)}/{len(mruns)}"
            )
            rows.append(
                f"| {name} | {c['desc']}{extra} | {ds} | {seeds} | "
                f"{pm:.1f} ± {ps:.1f} | "
                f"{mm:.1f} ± {ms:.1f} | {gap:+.1f} | {loss_cell} | "
                f"{verdict(gap, ps, ms, n)} |"
            )
    header = [
        "# Node classification: training quality, mlx-graphs vs PyG",
        "",
        "Generated by `benchmarks/quality/node_quality.py` (raw results in "
        "`node_results.jsonl`).",
        "",
        "Same model, hyperparameters, arrays and public Planetoid split in both "
        "frameworks; mlx-graphs on the GPU, PyG "
        f"{_version('torch_geometric')} (torch {_version('torch')}) on the CPU "
        "(on MPS where marked: the slow deep models; the device does not "
        "matter for accuracy). Test accuracy (%) at the epoch of best "
        "validation accuracy, mean ± std over the seeds (0-4 unless the "
        "seeds column says otherwise; one deterministic run for "
        "LabelPropagation). Gap = mlx-graphs − PyG. Loss = mean training "
        "loss over epochs 25-75 (every 5th epoch) and over the "
        "last 10 epochs, PyG / mlx-graphs. Verdict: pass if |gap| ≤ 1 point "
        "or ≤ 2 × PyG's std; 'pass (noise)' if within 2 standard errors of "
        "the difference of the means.",
        "",
        "| config | model | dataset | seeds | PyG acc | mlx-graphs acc | gap | "
        "loss ep. 25-75 PyG / mlx | final loss PyG / mlx | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    text = "\n".join(header + rows) + "\n"
    if notes and os.path.exists(notes):
        with open(notes) as f:
            text += "\n" + f.read()
    return text


def _version(pkg):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(pkg)
    except PackageNotFoundError:
        return "?"


# PyG configurations that take minutes per seed on the CPU: run on MPS.
MPS_CONFIGS = {"gcn2", "deepgcn"}
DATASET_COST = {"cora": 1, "citeseer": 1.5, "pubmed": 8}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--frameworks", nargs="+", default=list(FRAMEWORKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--memory-gb", type=float, default=6.0)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--cpu-jobs", type=int, default=6, help="PyG CPU workers")
    parser.add_argument("--gpu-jobs", type=int, default=2, help="MLX workers")
    parser.add_argument("--mps-jobs", type=int, default=2, help="PyG MPS workers")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--no-mps", action="store_true", help="PyG on CPU only")
    parser.add_argument("--rerun", action="store_true", help="ignore old results")
    parser.add_argument("--summarize", action="store_true", help="table only")
    parser.add_argument(
        "--notes", default=os.path.join(HERE, "node_notes.md"),
        help="markdown appended to the summary",
    )  # fmt: skip
    args = parser.parse_args()

    if not os.environ.get("MLX_GRAPHS_BENCH_DATA"):
        print("note: set MLX_GRAPHS_BENCH_DATA to share downloaded datasets")

    records = load_results()
    if not args.summarize:
        # One job per (config, dataset, framework, seed), in three pools:
        # PyG on the CPU, MLX on the GPU, PyG on MPS for the slow configs.
        queues = {"cpu": [], "gpu": [], "mps": []}
        for name in args.configs:
            c = CONFIGS[name]
            seeds = [0] if c["kind"] == "label_prop" else c.get("seeds", args.seeds)
            for ds in c["datasets"]:
                if args.datasets and ds not in args.datasets:
                    continue
                for fw in args.frameworks:
                    done = records.get((name, ds, fw), {"runs": {}})["runs"]
                    for seed in seeds:
                        if not args.rerun and seed in done:
                            continue
                        if fw == "mlx":
                            pool = "gpu"
                        elif name in MPS_CONFIGS and not args.no_mps:
                            pool = "mps"
                        else:
                            pool = "cpu"
                        cost = DATASET_COST.get(ds, 1) * (
                            10 if name in MPS_CONFIGS else 1
                        )
                        queues[pool].append((cost, name, ds, fw, seed))
        for q in queues.values():
            q.sort(key=lambda item: -item[0])  # slowest first
        print({k: len(v) for k, v in queues.items()}, "runs to do")
        lock = threading.Lock()

        def work(pool, item):
            _, name, ds, fw, seed = item
            device = "mps" if pool == "mps" else "cpu"
            argv = ["--config", name, "--framework", fw, "--dataset", ds,
                    "--seeds", str(seed), "--device", device,
                    "--threads", str(args.torch_threads)]  # fmt: skip
            t0 = time.time()
            result, status, peak = run_one(argv, args.memory_gb, args.timeout_s)
            if status != "ok" and pool == "mps":  # e.g. an op missing on MPS
                print(f"{name} {ds} seed {seed}: {status} on MPS, retrying on CPU")
                with lock:
                    queues["cpu"].insert(0, item)
                return
            record = {"config": name, "dataset": ds, "framework": fw}
            record.update(compact(result or {}))
            record.update(
                {"seeds": [seed], "status": status, "peak_gb": round(peak, 2)}
            )
            if fw == "pyg":
                record["device"] = device
            with lock:
                with open(RESULTS, "a") as f:
                    f.write(json.dumps(record) + "\n")
                accs = [round(r["test_acc"] * 100, 1) for r in record.get("runs", [])]
                print(
                    f"{name:20s} {ds:9s} {fw:4s} {device if fw == 'pyg' else 'gpu':4s} "
                    f"seed {seed} {status:10s} {time.time() - t0:6.0f}s "
                    f"{peak:4.1f}GB {accs}",
                    flush=True,
                )

        def worker_loop(pool):
            while True:
                with lock:
                    if not queues[pool]:
                        return
                    item = queues[pool].pop(0)
                work(pool, item)

        sizes = {"cpu": args.cpu_jobs, "gpu": args.gpu_jobs, "mps": args.mps_jobs}
        threads = [
            threading.Thread(target=worker_loop, args=(pool,))
            for pool, n in sizes.items()
            for _ in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # CPU jobs re-queued by MPS failures after the CPU pool ended
        for item in list(queues["cpu"]):
            queues["cpu"].remove(item)
            work("cpu", item)
        records = load_results()

    text = summarize(records, args.notes)
    with open(SUMMARY, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
