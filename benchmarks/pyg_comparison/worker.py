"""Run one framework comparison and print its result as JSON."""

import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path


def main():
    if not __package__:
        # Preserve direct script invocation from older benchmark commands.
        env = os.environ.copy()
        if env.get("MLX_GRAPHS_BENCH_DATA"):
            env["MLX_GRAPHS_BENCH_DATA"] = os.path.abspath(env["MLX_GRAPHS_BENCH_DATA"])
        return subprocess.call(
            [sys.executable, "-m", "benchmarks.pyg_comparison.worker", *sys.argv[1:]],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
        )

    from benchmarks.common import peak_footprint_gb
    from benchmarks.workloads import LAYERS, run_graph, run_layer, run_node

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)

    def common_args(p):
        p.add_argument("--framework", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--warmup", type=int, default=5)

    p = sub.add_parser("layer")
    common_args(p)
    p.add_argument("--layer", choices=LAYERS, required=True)
    p.add_argument("--num-nodes", type=int, required=True)
    p.add_argument("--num-edges", type=int, required=True)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--repeats", type=int, default=20)

    p = sub.add_parser("node")
    common_args(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", choices=LAYERS, required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=10)

    p = sub.add_parser("graph")
    common_args(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", choices=LAYERS, required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    if args.framework == "mlx-graphs":
        import mlx.core as mx

        # keep MLX from holding on to freed buffers without bound
        mx.set_cache_limit(4 * 1024**3)

    result = {"layer": run_layer, "node": run_node, "graph": run_graph}[args.task](args)
    result["peak_memory_gb"] = peak_footprint_gb()
    print(json.dumps({**vars(args), **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
