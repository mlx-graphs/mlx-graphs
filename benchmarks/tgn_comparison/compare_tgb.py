"""TGN on the Temporal Graph Benchmark: mlx-graphs vs py-tgb's PyG examples.

    python benchmarks/tgn_comparison/compare_tgb.py --data <dir> --epochs 10

Runs, for every seed and one run at a time, each under a memory cap
(``benchmarks/run_capped.py``) and only once macOS reports 30% of the
memory available:

- ``tgbl-wiki`` link prediction: ``examples/tgn_tgbl.py`` (MLX, GPU) and
  ``pyg_tgbl.py`` (py-tgb's TGN example, PyG on the CPU), with the official
  one-vs-many evaluation (MRR against the 999 negatives of TGB). The PyG runs
  score the edges of a mini-batch together (``--eval batched``), which gives
  the same MRR as the edge-by-edge loop of the TGB example and is much
  faster; ``--per_edge_seed`` adds one run with the original loop.
- ``tgbn-trade`` node prediction: ``examples/tgn_tgbn.py`` and
  ``pyg_tgbn.py`` (NDCG@10).

Both implementations use the same hyperparameters (those of the TGB
examples: batch size 200, learning rate 1e-4, memory, time and embedding
dimensions 100, 10 neighbors) and the same seeds, but different random
number generators. ``--data`` is a dataset directory of mlx-graphs (the
datasets are downloaded if missing). The summary of every run is written to
``results_tgb.json`` next to this script.
"""

import json
import os
import subprocess
import sys
import time
from argparse import ArgumentParser

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def wait_for_memory(min_level: int = 30):
    """Waits until macOS reports at least ``min_level`` % of memory available
    (``kern.memorystatus_level``)."""
    while True:
        out = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True
        ).stdout.strip()
        if not out.isdigit() or int(out) >= min_level:
            return
        print("waiting for memory ...", flush=True)
        time.sleep(30)


def commands(data: str, epochs: int, seed: int) -> dict:
    py = sys.executable
    common = ["--epochs", str(epochs), "--seed", str(seed), "--json"]
    return {
        "tgbl-wiki/mlx": [py, os.path.join(REPO, "examples", "tgn_tgbl.py"),
                          "--base_dir", data, *common],
        "tgbl-wiki/pyg": [py, "-W", "ignore", os.path.join(HERE, "pyg_tgbl.py"),
                          "--data", data, "--eval", "batched", *common],
        "tgbn-trade/mlx": [py, os.path.join(REPO, "examples", "tgn_tgbn.py"),
                           "--base_dir", data, *common],
        "tgbn-trade/pyg": [py, "-W", "ignore", os.path.join(HERE, "pyg_tgbn.py"),
                           "--data", data, *common],
    }  # fmt: skip


def run(key, cmd, results, out_path, memory_gb):
    if key in results:
        return
    wait_for_memory()
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "benchmarks", "run_capped.py"),
         "--gb", str(memory_gb), "--", *cmd],
        capture_output=True, text=True, cwd=REPO,
        env=dict(os.environ, PYTHONPATH=REPO),
    )  # fmt: skip
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if not lines:
        print(key, "failed:", proc.stderr[-1000:], flush=True)
        return
    results[key] = json.loads(lines[-1])
    print(key, json.dumps(results[key])[:300], flush=True)
    with open(out_path, "w") as f:
        json.dump({"runs": results}, f, indent=1)


def main(args):
    out_path = os.path.join(HERE, "results_tgb.json")
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f).get("runs", {})
    for seed in args.seeds:
        for name, cmd in commands(args.data, args.epochs, seed).items():
            run(f"{name}/{seed}", cmd, results, out_path, args.memory_gb)
    if args.per_edge_seed is not None:
        cmd = commands(args.data, args.epochs, args.per_edge_seed)["tgbl-wiki/pyg"]
        cmd[cmd.index("batched")] = "per_edge"
        key = f"tgbl-wiki/pyg_per_edge/{args.per_edge_seed}"
        run(key, cmd, results, out_path, args.memory_gb)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="dataset directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--per_edge_seed", type=int, default=None)
    parser.add_argument("--memory_gb", type=float, default=8.0)
    main(parser.parse_args())
