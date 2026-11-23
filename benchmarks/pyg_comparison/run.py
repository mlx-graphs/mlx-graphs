"""Benchmarks mlx-graphs against PyTorch Geometric on this Mac.

    python -m benchmarks.pyg_comparison.run                 # everything
    python -m benchmarks.pyg_comparison.run --tasks layer   # one task
    python -m benchmarks.pyg_comparison.run --quick         # smoke test

Every configuration runs in its own subprocess (see ``worker.py``) under a
hard memory cap: a run whose footprint (CPU + GPU memory) goes over
``--memory-gb`` is killed and recorded as out of memory.

Results are appended to ``results/<chip>/<label>/results.jsonl``, where
``<label>`` names the measured version of mlx-graphs (see ``versions.py``).
Workers import ``mlx_graphs`` from the checkout this script is in, so running
it from a worktree of another branch measures that branch. Configurations
already in the file are skipped, so an interrupted run can be resumed.
``plot.py`` draws the figures of one version; ``history.py`` compares versions.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))

LAYER_GRAPHS = [(10_000, 100_000), (50_000, 500_000), (200_000, 2_000_000)]
LAYERS = ["gcn", "sage", "gin", "gat", "gatv2", "transformer"]

NODE_DATASETS = [
    "cora",
    "citeseer",
    "pubmed",
    "amazon-photo",
    "amazon-computers",
    "coauthor-cs",
    "coauthor-physics",
    "ogbn-arxiv",
]
NODE_MODELS = ["gcn", "sage", "gat"]

GRAPH_DATASETS = ["MUTAG", "PROTEINS", "NCI1", "DD"]
GRAPH_MODELS = ["gcn", "gin", "sage"]


def configurations(tasks, quick, seeds=(0,), layers=LAYERS, frameworks=None):
    from benchmarks.common import FRAMEWORKS

    frameworks = FRAMEWORKS if frameworks is None else frameworks

    if "layer" in tasks:
        graphs = LAYER_GRAPHS[:1] if quick else LAYER_GRAPHS
        for (n, e), layer, fw in itertools.product(graphs, layers, frameworks):
            repeats = "5" if quick else ("10" if fw == "pyg-cpu" else "20")
            yield [
                "layer",
                "--framework", fw,
                "--layer", layer,
                "--num-nodes", str(n),
                "--num-edges", str(e),
                "--repeats", repeats,
            ]  # fmt: skip
    if "node" in tasks:
        datasets = NODE_DATASETS[:1] if quick else NODE_DATASETS
        for seed, ds, model, fw in itertools.product(
            seeds, datasets, NODE_MODELS, frameworks
        ):
            epochs = "20" if quick else ("100" if ds == "ogbn-arxiv" else "200")
            yield [
                "node",
                "--framework", fw,
                "--dataset", ds,
                "--model", model,
                "--epochs", epochs,
                "--seed", str(seed),
            ]  # fmt: skip
    if "graph" in tasks:
        datasets = GRAPH_DATASETS[:1] if quick else GRAPH_DATASETS
        for seed, ds, model, fw in itertools.product(
            seeds, datasets, GRAPH_MODELS, frameworks
        ):
            yield [
                "graph",
                "--framework", fw,
                "--dataset", ds,
                "--model", model,
                "--epochs", "5" if quick else "100",
                "--warmup", "1" if quick else "3",
                "--seed", str(seed),
            ]  # fmt: skip


def config_key(argv) -> str:
    return " ".join(argv)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tasks", nargs="+", default=["layer", "node", "graph"],
        choices=["layer", "node", "graph"],
    )  # fmt: skip
    parser.add_argument(
        "--layers", nargs="+", default=LAYERS, choices=LAYERS,
        help="layers of the layer task (default: all)",
    )  # fmt: skip
    parser.add_argument(
        "--frameworks", nargs="+", default=None,
        help="frameworks to measure: mlx-graphs, pyg-mps, pyg-cpu (default: all)",
    )  # fmt: skip
    parser.add_argument("--memory-gb", type=float, default=16.0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--quick", action="store_true", help="tiny smoke test")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="seeds of the node and graph classification runs",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="name of the measured version (default: <version>-<commit of "
        "the last change to mlx_graphs/>, see versions.py)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="results directory (default: results/<chip>/<label>)",
    )
    args = parser.parse_args()
    if not __package__:
        # Retain direct script invocation and paths relative to the caller.
        argv = sys.argv[1:]
        if args.out:
            argv = [*argv, "--out", os.path.abspath(args.out)]
        return subprocess.call(
            [sys.executable, "-m", "benchmarks.pyg_comparison.run", *argv],
            cwd=Path(__file__).resolve().parents[2],
        )

    from benchmarks.common import FRAMEWORKS, run_worker, system_info
    from benchmarks.versions import check_imported_library, library_info

    unknown = set(args.frameworks or ()) - set(FRAMEWORKS)
    if unknown:
        parser.error(f"unknown frameworks {sorted(unknown)}; choose from {FRAMEWORKS}")

    library = library_info()
    if args.label:
        library["label"] = args.label
    library["imported_from"] = check_imported_library(sys.executable)
    info = {**system_info(), "library": library}
    chip_dir = info["chip"].replace(" ", "_") + ("_quick" if args.quick else "")
    out_dir = args.out or os.path.join(HERE, "results", chip_dir, library["label"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "system.json"), "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    print(
        f"measuring mlx-graphs {library['label']} (branch {library['branch']}, "
        f"from {library['imported_from']}) -> {out_dir}"
    )

    results_path = os.path.join(out_dir, "results.jsonl")
    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            done = {json.loads(line)["config"] for line in f if line.strip()}

    todo = [
        c
        for c in configurations(
            args.tasks,
            args.quick,
            [0] if args.quick else args.seeds,
            args.layers,
            args.frameworks,
        )
        if config_key(c) not in done
    ]
    print(f"{len(todo)} configurations to run ({len(done)} already done)")
    for i, argv in enumerate(todo, 1):
        t0 = time.time()
        result, status = run_worker(
            "benchmarks.pyg_comparison.worker", argv, args.memory_gb, args.timeout_s
        )
        record = {"config": config_key(argv), "status": status, **(result or {})}
        if result is None:  # keep the identifying fields of failed runs
            keys = argv[1::2]
            record.update(
                {"task": argv[0]}
                | {k.lstrip("-").replace("-", "_"): v for k, v in zip(keys, argv[2::2])}
            )
        with open(results_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        summary = {
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in (result or {}).items()
            if k.endswith(("_ms", "_acc", "_gb", "_loss"))
        }
        print(
            f"[{i}/{len(todo)}] {config_key(argv)}  ->  {status} "
            f"({time.time() - t0:.0f} s) {summary}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
