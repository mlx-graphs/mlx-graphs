"""Compare TransformerConv's edge-score paths: gather, in-tree Metal kernel, and
mlx-cluster's ``edge_scores``."""

import argparse
import json
import sys
import warnings
from math import isfinite
from pathlib import Path
from statistics import fmean, median

from benchmarks.common import peak_footprint_gb, run_worker, system_info
from benchmarks.versions import library_info
from benchmarks.workloads import run_layer, run_node

PATHS = ("reference", "metal", "mlx-cluster")


def workloads(tasks, large):
    if "layer" in tasks:
        graphs = [(10_000, 100_000)]
        if large:
            graphs.append((200_000, 2_000_000))
        for nodes, edges in graphs:
            yield (
                f"{edges:,}-edge layer",
                dict(
                    task="layer", num_nodes=nodes, num_edges=edges, dim=128, repeats=12
                ),
            )
    if "node" in tasks:
        yield (
            "Coauthor Physics",
            dict(
                task="node",
                dataset="coauthor-physics",
                num_layers=2,
                epochs=100,
                eval_every=10,
                hidden=64,
                dropout=0.5,
                lr=0.01,
            ),
        )


def measure(score_path, config):
    warnings.filterwarnings("ignore")

    import mlx.core as mx
    import numpy as np

    from mlx_graphs.nn import is_fused_message_passing_enabled
    from mlx_graphs.nn.conv import transformer_conv
    from mlx_graphs.nn.kernels.transformer_edge_scores import (
        _metal_scores,
        _reference_scores,
    )

    if mx.default_device().type != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("The TransformerConv benchmark requires the Metal GPU")
    if not is_fused_message_passing_enabled():
        raise RuntimeError("The TransformerConv benchmark requires mlx-cluster SpMM")
    # Swap only the scores; SpMM and the rest of the layer stay the same.
    if score_path == "reference":
        transformer_conv.edge_scores = _reference_scores
    elif score_path == "mlx-cluster":
        from mlx_cluster import edge_scores

        transformer_conv.edge_scores = edge_scores

    args = argparse.Namespace(
        framework="mlx-graphs",
        layer="transformer",
        model="transformer",
        warmup=5,
        **config,
    )
    mx.set_cache_limit(4 * 1024**3)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    result = {"layer": run_layer, "node": run_node}[args.task](args)
    cache_info = _metal_scores.cache_info()
    metal_calls = cache_info.hits + cache_info.misses
    if (metal_calls > 0) != (score_path == "metal"):
        raise RuntimeError(f"{score_path} score path did not run as selected")
    record = {
        **vars(args),
        **result,
        "score_path": score_path,
        "metal_score_calls": metal_calls,
        "peak_mlx_gb": mx.get_peak_memory() / 1e9,
        "peak_footprint_gib": peak_footprint_gb(),
    }
    print(json.dumps(record))


def summarize(label, results, is_layer):
    key = "forward_backward_ms" if is_layer else "epoch_ms"
    timing = "forward + backward" if is_layer else "training epoch"
    base = median(row[key] for row in results["reference"])
    print(f"{label} {timing}:")
    for path, rows in results.items():
        value = median(row[key] for row in rows)
        line = f"  {path:>11}: {value:8.2f} ms ({base / value:.2f}× vs reference)"
        if is_layer:
            line += f", forward {rows[0]['forward_ms']:.2f} ms"
        else:
            line += f", test accuracy {fmean(row['test_acc'] for row in rows):.2%}"
        memory = median(row["peak_mlx_gb"] for row in rows)
        print(f"{line}, peak MLX memory {memory:.3f} GB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", choices=("layer", "node"), default=["layer"]
    )
    parser.add_argument("--large", action="store_true", help="add a 2M-edge layer run")
    parser.add_argument(
        "--paths", nargs="+", choices=PATHS, default=list(PATHS), help="score paths"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2], help="training seeds"
    )
    parser.add_argument("--memory-gb", type=float, default=16.0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--out", type=Path, help="save environment and results to JSON")
    args = parser.parse_args()
    if args.large and "layer" not in args.tasks:
        parser.error("--large requires the layer task")
    if "reference" not in args.paths:
        parser.error("--paths must include reference, the baseline")
    if not all(isfinite(n) and n > 0 for n in (args.memory_gb, args.timeout_s)):
        parser.error("memory cap and timeout must be positive and finite")
    if args.out and args.out.exists():
        parser.error(f"output already exists: {args.out}")

    system = system_info()
    print(f"{system['chip']}, MLX {system['mlx']}, mlx-cluster {system['mlx-cluster']}")
    records = []
    failed = False
    for label, config in workloads(args.tasks, args.large):
        is_layer = config["task"] == "layer"
        seeds = [0] if is_layer else args.seeds
        results = {path: [] for path in args.paths}
        for index, seed in enumerate(seeds):
            # rotate the order so no path always runs first
            shift = index % len(args.paths)
            order = args.paths[shift:] + args.paths[:shift]
            for path in order:
                print(f"Running {label}, seed {seed}, {path}", flush=True)
                options = {**config, "seed": seed}
                result, status = run_worker(
                    "benchmarks.kernels.transformer_scores",
                    ["--worker", path, json.dumps(options)],
                    args.memory_gb,
                    args.timeout_s,
                )
                records.append(
                    {
                        "workload": label,
                        "status": status,
                        "score_path": path,
                        **options,
                        **(result or {}),
                    }
                )
                if result is None:
                    failed = True
                    print(f"  {status}", flush=True)
                else:
                    results[path].append(result)
        if all(len(rows) == len(seeds) for rows in results.values()):
            summarize(label, results, is_layer)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as output:
            json.dump(
                {"system": system, "library": library_info(), "results": records},
                output,
                indent=2,
            )
            output.write("\n")
        print(f"Saved results to {args.out}")
    return int(failed)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        measure(sys.argv[2], json.loads(sys.argv[3]))
    else:
        raise SystemExit(main())
