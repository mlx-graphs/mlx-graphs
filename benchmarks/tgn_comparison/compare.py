"""TGN link prediction on JODIE Wikipedia: mlx-graphs vs PyG.

    python benchmarks/tgn_comparison/compare.py --data <dir> --epochs 8

Runs ``examples/tgn_jodie.py`` (MLX, GPU) and ``pyg_tgn.py`` (PyG's
``examples/tgn.py`` on the CPU with 2 threads) with the same hyperparameters
for every seed, one run at a time, each under a 4 GB memory cap
(``benchmarks/run_capped.py``) and only once 4 GB of memory are free.

Two PyG variants are run: ``pyg`` as installed and ``pyg_fixed``, in which
``scatter_argmax`` gives empty groups an out-of-range index as torch-scatter
does. Without torch-scatter, PyG's fallback ``scatter_argmax`` returns
``dim_size - 1`` for empty groups, so its ``LastAggregator`` copies an
unrelated message to the nodes that have none instead of zeros; mlx-graphs
follows the torch-scatter semantics.

``--data`` must contain ``jodie_wikipedia/`` (mlx-graphs) and
``pyg/wikipedia/`` (PyG) dataset folders, which are created (downloaded) if
missing. The final-epoch metrics of every run are written to
``results.json`` next to this script.
"""

import json
import os
import re
import subprocess
import sys
import time
from argparse import ArgumentParser

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def unused_gb() -> float:
    out = subprocess.run(
        ["top", "-l", "1", "-n", "0"], capture_output=True, text=True
    ).stdout
    m = re.search(r"([\d.]+)([MG]) unused", out)
    if not m:
        return 0.0
    return float(m.group(1)) / (1024 if m.group(2) == "M" else 1)


def wait_for_memory(min_gb: float):
    while unused_gb() < min_gb:
        print("waiting for memory ...", flush=True)
        time.sleep(30)


def commands(data: str, epochs: int, seed: int) -> dict:
    py = sys.executable
    pyg = [py, "-W", "ignore", os.path.join(HERE, "pyg_tgn.py"),
           "--root", os.path.join(data, "pyg"),
           "--epochs", str(epochs), "--seed", str(seed)]  # fmt: skip
    return {
        "mlx": [
            py,
            os.path.join(REPO, "examples", "tgn_jodie.py"),
            "--base_dir",
            data,
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--json",
        ],  # fmt: skip
        "pyg": pyg,
        "pyg_fixed": pyg + ["--fix_argmax"],
    }


def main(args):
    out_path = os.path.join(HERE, "results.json")
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f).get("runs", {})
    for seed in args.seeds:
        for name, cmd in commands(args.data, args.epochs, seed).items():
            key = f"{name}/{seed}"
            if key in results:
                continue
            wait_for_memory(args.memory_gb)
            proc = subprocess.run(
                [sys.executable, os.path.join(REPO, "benchmarks", "run_capped.py"),
                 "--gb", str(args.memory_gb), "--", *cmd],
                capture_output=True, text=True, cwd=REPO,
                env=dict(os.environ, PYTHONPATH=REPO),
            )  # fmt: skip
            lines = [line for line in proc.stdout.splitlines() if line.startswith("[")]
            if not lines:
                print(name, seed, "failed:", proc.stderr[-500:], flush=True)
                continue
            results[key] = json.loads(lines[-1])[-1]
            print(name, seed, json.dumps(results[key]), flush=True)
            with open(out_path, "w") as f:
                json.dump({"epochs": args.epochs, "runs": results}, f, indent=1)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="dataset directory")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--memory_gb", type=float, default=4.0)
    main(parser.parse_args())
