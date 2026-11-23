"""Compare MLX GATv2's reference and Metal edge-score paths."""

import argparse
import json
import sys
import warnings
from math import isfinite
from pathlib import Path
from statistics import fmean, median

from benchmarks.common import peak_footprint_gb, run_worker, system_info
from benchmarks.versions import library_info
from benchmarks.workloads import run_graph, run_layer, run_node


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
    training = dict(hidden=64, dropout=0.5, lr=0.01)
    if "node" in tasks:
        yield (
            "Coauthor Physics",
            dict(
                task="node",
                dataset="coauthor-physics",
                num_layers=2,
                epochs=100,
                eval_every=10,
                **training,
            ),
        )
    if "graph" in tasks:
        yield (
            "DD",
            dict(
                task="graph",
                dataset="DD",
                num_layers=3,
                batch_size=64,
                epochs=40,
                eval_every=5,
                **training,
            ),
        )


def measure(score_path, config):
    warnings.filterwarnings("ignore")

    import mlx.core as mx
    import numpy as np

    from mlx_graphs.nn import is_fused_message_passing_enabled
    from mlx_graphs.nn.conv import gatv2_conv
    from mlx_graphs.nn.kernels.gatv2_edge_scores import _metal_scores, _reference_scores

    if mx.default_device().type != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("The GATv2 benchmark requires the Metal GPU")
    if not is_fused_message_passing_enabled():
        raise RuntimeError("The GATv2 benchmark requires mlx-cluster fused SpMM")
    if score_path == "reference":
        # Keep SpMM and the rest of the layer in the comparison.
        gatv2_conv.gatv2_edge_scores = _reference_scores

    args = argparse.Namespace(
        framework="mlx-graphs", layer="gatv2", model="gatv2", warmup=5, **config
    )
    mx.set_cache_limit(4 * 1024**3)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    result = {"layer": run_layer, "node": run_node, "graph": run_graph}[args.task](args)
    cache_info = _metal_scores.cache_info()
    metal_calls = cache_info.hits + cache_info.misses
    if (metal_calls > 0) != (score_path == "metal"):
        raise RuntimeError(f"GATv2 {score_path} score path did not run as selected")
    record = {
        **vars(args),
        **result,
        "score_path": score_path,
        "metal_score_calls": metal_calls,
        "peak_mlx_gb": mx.get_peak_memory() / 1e9,
        "peak_footprint_gib": peak_footprint_gb(),
    }
    print(json.dumps(record))


def summarize(label, reference, metal, is_layer):
    key = "forward_backward_ms" if is_layer else "epoch_ms"
    timing = "forward + backward" if is_layer else "training epoch"
    before, after = (median(row[key] for row in rows) for rows in (reference, metal))
    print(f"{label} {timing}: {before:.2f} → {after:.2f} ms ({before / after:.2f}×)")
    if is_layer:
        a, b = reference[0]["forward_ms"], metal[0]["forward_ms"]
        print(f"  Forward: {a:.2f} → {b:.2f} ms")
    else:
        a, b = (fmean(row["test_acc"] for row in rows) for rows in (reference, metal))
        print(f"  Mean test accuracy: {a:.2%} → {b:.2%}")
    a, b = (median(row["peak_mlx_gb"] for row in rows) for rows in (reference, metal))
    print(f"  Peak MLX memory: {a:.3f} → {b:.3f} GB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", nargs="+", choices=("layer", "node", "graph"), default=["layer"]
    )
    parser.add_argument("--large", action="store_true", help="add a 2M-edge layer run")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2], help="training seeds"
    )
    parser.add_argument("--memory-gb", type=float, default=16.0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--out", type=Path, help="save environment and results to JSON")
    args = parser.parse_args()
    if args.large and "layer" not in args.tasks:
        parser.error("--large requires the layer task")
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
        paths = {"reference": [], "metal": []}
        for index, seed in enumerate(seeds):
            order = tuple(paths) if index % 2 == 0 else tuple(reversed(paths))
            for path in order:
                print(f"Running {label}, seed {seed}, {path}", flush=True)
                options = {**config, "seed": seed}
                result, status = run_worker(
                    "benchmarks.kernels.gatv2_scores",
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
                    paths[path].append(result)
        if all(len(rows) == len(seeds) for rows in paths.values()):
            summarize(label, paths["reference"], paths["metal"], is_layer)

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
