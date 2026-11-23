"""Compare separate softmax and SpMM with their fused CSR implementation."""

import argparse
import itertools
import json
import sys
from math import isfinite

from benchmarks.common import run_worker, system_info
from benchmarks.workloads import run_layer


def measure(config):
    import mlx.core as mx

    from mlx_graphs.nn import (
        fused_message_passing_mode,
        is_fused_message_passing_enabled,
        message_passing,
    )

    if mx.default_device().type != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("This benchmark requires the Metal GPU")
    if not is_fused_message_passing_enabled():
        raise RuntimeError("This benchmark requires mlx-cluster fused SpMM")

    # Both paths include the same score projections and post-aggregation work.
    calls = 0
    original = message_passing.csr_softmax_spmm

    def count_calls(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    message_passing.csr_softmax_spmm = count_calls
    mx.random.seed(0)
    mx.set_cache_limit(4 * 1024**3)
    args = argparse.Namespace(
        framework="mlx-graphs", dim=128, repeats=12, warmup=5, **config
    )
    with fused_message_passing_mode(True, format="csr", softmax_spmm=args.fuse):
        result = run_layer(args)
    if (calls > 0) != args.fuse:
        raise RuntimeError("The requested softmax SpMM path did not run")
    print(
        json.dumps(
            {
                **vars(args),
                **result,
                "fused_calls": calls,
                "peak_mlx_gb": mx.get_peak_memory() / 1e9,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large", action="store_true", help="include 2M edges")
    parser.add_argument("--memory-gb", type=float, default=8.0)
    args = parser.parse_args()
    if not isfinite(args.memory_gb) or args.memory_gb <= 0:
        parser.error("memory cap must be positive and finite")

    system = system_info()
    print(f"{system['chip']}, MLX {system['mlx']}, mlx-cluster {system['mlx-cluster']}")
    sizes = [(10_000, 100_000)]
    if args.large:
        sizes.append((200_000, 2_000_000))
    failed = False
    for (nodes, edges), layer in itertools.product(sizes, ("gat", "gatv2")):
        print(f"{layer.upper()}, {edges:,} edges", flush=True)
        rows = []
        for enabled in (False, True):
            config = dict(layer=layer, num_nodes=nodes, num_edges=edges, fuse=enabled)
            result, status = run_worker(
                "benchmarks.kernels.csr_softmax_spmm",
                ["--worker", json.dumps(config)],
                args.memory_gb,
                300,
            )
            if result is None:
                failed = True
                print(f"  {status}", flush=True)
                break
            rows.append(result)
        if len(rows) == 2:
            ref, fused = rows
            for label, key in (
                ("Forward (ms)", "forward_ms"),
                ("Forward + backward (ms)", "forward_backward_ms"),
                ("Peak MLX memory (GB)", "peak_mlx_gb"),
            ):
                before, after = ref[key], fused[key]
                print(f"  {label}: {before:.3f} → {after:.3f} ({before / after:.2f}×)")
    return int(failed)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        measure(json.loads(sys.argv[2]))
    else:
        raise SystemExit(main())
