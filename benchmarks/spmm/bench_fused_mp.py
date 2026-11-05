"""Benchmark fused (SpMM) message passing against gather-scatter.

Runs the same mlx-graphs layers with fused message passing off ("gather",
today's path) and on ("coo", "csr"), and records forward time, forward +
backward time and peak memory. Optionally records a Metal GPU trace or a
cProfile of one run for profiling.

Examples (see README.md in this folder for details):

    # default sweep: Planetoid + 1M-edge synthetic graphs, 3 feature dims
    python bench_fused_mp.py

    # bigger graphs, where gather-scatter is expected to run out of memory
    python bench_fused_mp.py --graphs uniform-10m powerlaw-10m --dims 64 256

    # Metal GPU trace of one configuration, to open in Xcode
    MTL_CAPTURE_ENABLED=1 python bench_fused_mp.py --graphs powerlaw-1m \\
        --dims 64 --layers gcn --modes coo --capture spmm_coo.gputrace

    # Python-side profile, to open with snakeviz
    python bench_fused_mp.py --graphs cora --cprofile fused.prof
"""

import argparse
import cProfile
import csv
import os
import platform
import pstats
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from mlx_graphs.nn import (
    GCNConv,
    SAGEConv,
    SimpleConv,
    clear_csr_cache,
    fused_message_passing_mode,
    is_fused_message_passing_enabled,
)

# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------


def synthetic(num_nodes: int, num_edges: int, power_law: bool, seed: int = 0):
    """Random graph. With power_law=True, destination degrees follow a Zipf-like
    law, so a few nodes receive a large share of the edges (hard for COO's
    atomics and for load balancing)."""
    rng = np.random.default_rng(seed)
    src = rng.integers(0, num_nodes, num_edges, dtype=np.int64)
    if power_law:
        p = 1.0 / np.arange(1, num_nodes + 1) ** 1.1
        dst = rng.choice(num_nodes, size=num_edges, p=p / p.sum())
    else:
        dst = rng.integers(0, num_nodes, num_edges, dtype=np.int64)
    return mx.array(np.stack([src, dst]).astype(np.int64)), num_nodes


def planetoid(name: str):
    from mlx_graphs.datasets import PlanetoidDataset

    g = PlanetoidDataset(name=name).graphs[0]
    return g.edge_index.astype(mx.int64), g.num_nodes


def ogbn_arxiv():
    from mlx_graphs.datasets import OGBDataset

    g = OGBDataset("ogbn-arxiv").graphs[0]
    return g.edge_index.astype(mx.int64), g.num_nodes


GRAPHS: Dict[str, Callable[[], Tuple[mx.array, int]]] = {
    "cora": lambda: planetoid("cora"),
    "citeseer": lambda: planetoid("citeseer"),
    "pubmed": lambda: planetoid("pubmed"),
    "arxiv": ogbn_arxiv,  # ~170k nodes, ~1.2M edges (download ~100 MB)
    "uniform-100k": lambda: synthetic(10_000, 100_000, False),
    "uniform-1m": lambda: synthetic(100_000, 1_000_000, False),
    "powerlaw-1m": lambda: synthetic(100_000, 1_000_000, True),
    "uniform-10m": lambda: synthetic(1_000_000, 10_000_000, False),
    "powerlaw-10m": lambda: synthetic(1_000_000, 10_000_000, True),
}
DEFAULT_GRAPHS = ["cora", "pubmed", "uniform-1m", "powerlaw-1m"]

# ---------------------------------------------------------------------------
# Layers (all D -> D so every mode sees identical shapes)
# ---------------------------------------------------------------------------

LAYERS: Dict[str, Callable[[int], object]] = {
    # pure aggregation: isolates the message-passing cost
    "aggr-add": lambda d: SimpleConv(aggr="add"),
    "aggr-mean": lambda d: SimpleConv(aggr="mean"),
    # real layers: aggregation + linear projections + normalisation
    "gcn": lambda d: GCNConv(d, d),
    "sage": lambda d: SAGEConv(d, d),
}
DEFAULT_LAYERS = ["aggr-add", "gcn", "sage"]

MODES = ["gather", "coo", "csr"]

# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass
class Result:
    graph: str
    num_nodes: int
    num_edges: int
    dim: int
    layer: str
    mode: str
    status: str = "ok"
    fwd_ms: Optional[float] = None
    fwd_bwd_ms: Optional[float] = None
    fwd_peak_mb: Optional[float] = None
    fwd_bwd_peak_mb: Optional[float] = None
    csr_build_ms: Optional[float] = None
    max_abs_err: Optional[float] = None
    extra: Dict = field(default_factory=dict)


def mode_ctx(mode: str):
    if mode == "gather":
        return fused_message_passing_mode(False)
    return fused_message_passing_mode(True, format=mode)


def time_ms(fn: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        mx.eval(fn())
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def peak_mb(fn: Callable[[], object]) -> float:
    """Peak memory allocated while evaluating fn(), beyond what was live before."""
    mx.eval(fn())  # make sure lazily created inputs (e.g. cached CSR) exist
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    out = fn()
    mx.eval(out)
    mx.synchronize()
    del out
    return (mx.get_peak_memory() - base) / 2**20


def is_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in ("memory", "resource limit", "allocat"))


def run_config(
    graph: str,
    edge_index: mx.array,
    num_nodes: int,
    dim: int,
    layer_name: str,
    modes: List[str],
    warmup: int,
    repeats: int,
) -> List[Result]:
    mx.random.seed(0)
    layer = LAYERS[layer_name](dim)
    mx.eval(layer.parameters())
    x = mx.random.normal((num_nodes, dim), key=mx.random.key(1))
    mx.eval(x)

    fwd = lambda: layer(edge_index, x)  # noqa: E731
    # value_and_grad forces the forward pass to run, like a real training step
    loss_and_grad = mx.value_and_grad(lambda x_: layer(edge_index, x_).sum())
    fwd_bwd = lambda: loss_and_grad(x)  # noqa: E731

    results, reference = [], None
    for mode in modes:
        r = Result(graph, num_nodes, edge_index.shape[1], dim, layer_name, mode)
        try:
            with mode_ctx(mode):
                if mode == "csr":
                    clear_csr_cache()
                    t0 = time.perf_counter()
                    mx.eval(fwd())  # first call sorts edge_index into CSR
                    r.csr_build_ms = (time.perf_counter() - t0) * 1e3
                out = fwd()
                mx.eval(out)
                if mode == "gather":
                    reference = out
                elif reference is not None:
                    r.max_abs_err = mx.abs(out - reference).max().item()
                del out
                r.fwd_ms = time_ms(fwd, warmup, repeats)
                r.fwd_peak_mb = peak_mb(fwd)
                r.fwd_bwd_ms = time_ms(fwd_bwd, warmup, repeats)
                r.fwd_bwd_peak_mb = peak_mb(fwd_bwd)
        except Exception as e:  # noqa: BLE001
            r.status = "OOM" if is_oom(e) else f"error: {type(e).__name__}: {e}"
        finally:
            mx.clear_cache()
        results.append(r)
        print_row(r, results)
    clear_csr_cache()
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

HEADER = (
    f"{'graph':<13} {'edges':>10} {'D':>4} {'layer':<9} {'mode':<6} "
    f"{'fwd ms':>9} {'f+b ms':>9} {'fwd MB':>9} {'f+b MB':>9} "
    f"{'speedup':>8} {'mem x':>6} {'err':>8}"
)


def fmt(v, spec):
    return format(v, spec) if v is not None else "-"


def print_row(r: Result, siblings: List[Result]):
    base = next((s for s in siblings if s.mode == "gather" and s.status == "ok"), None)
    speed = mem = None
    if base and r.status == "ok" and r.mode != "gather":
        speed = base.fwd_bwd_ms / r.fwd_bwd_ms
        mem = base.fwd_bwd_peak_mb / max(r.fwd_bwd_peak_mb, 1e-6)
    if r.status != "ok":
        tail = r.status[:60]
        print(
            f"{r.graph:<13} {r.num_edges:>10} {r.dim:>4} {r.layer:<9} {r.mode:<6} "
            f"{tail}",
            flush=True,
        )
        return
    print(
        f"{r.graph:<13} {r.num_edges:>10} {r.dim:>4} {r.layer:<9} {r.mode:<6} "
        f"{fmt(r.fwd_ms, '9.2f')} {fmt(r.fwd_bwd_ms, '9.2f')} "
        f"{fmt(r.fwd_peak_mb, '9.1f')} {fmt(r.fwd_bwd_peak_mb, '9.1f')} "
        f"{fmt(speed, '7.2f') + 'x' if speed else '-':>8} "
        f"{fmt(mem, '5.1f') + 'x' if mem else '-':>6} "
        f"{fmt(r.max_abs_err, '8.1e')}",
        flush=True,
    )


def environment() -> Dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    def ver(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return "not installed"

    chip = platform.processor()
    if sys.platform == "darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:  # noqa: BLE001
            pass
    return {
        "mlx": mx.__version__,
        "mlx-graphs": ver("mlx-graphs"),
        "mlx-cluster": ver("mlx_cluster"),
        "device": str(mx.default_device()),
        "chip": chip,
        "python": platform.python_version(),
    }


def write_csv(path: str, results: List[Result], env: Dict[str, str]):
    rows = [{**asdict(r), **env} for r in results]
    for row in rows:
        row.pop("extra", None)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def capture_gpu_trace(
    path: str, graph: str, dim: int, layer_name: str, mode: str, iters: int
):
    """Records a Metal GPU trace of `iters` forward+backward steps."""
    if not path.endswith(".gputrace"):
        sys.exit("--capture path must end with .gputrace")
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        sys.exit("Set MTL_CAPTURE_ENABLED=1 in the environment to record a trace.")
    if os.path.exists(path):
        sys.exit(f"{path} already exists; Metal will not overwrite it.")

    edge_index, num_nodes = GRAPHS[graph]()
    mx.random.seed(0)
    layer = LAYERS[layer_name](dim)
    x = mx.random.normal((num_nodes, dim), key=mx.random.key(1))
    grad_x = mx.grad(lambda x_: layer(edge_index, x_).sum())
    with mode_ctx(mode):
        for _ in range(3):  # warm up: compile kernels, build CSR
            mx.eval(grad_x(x))
        mx.synchronize()
        mx.metal.start_capture(path)
        for _ in range(iters):
            mx.eval(grad_x(x))
        mx.synchronize()
        mx.metal.stop_capture()
    print(f"Wrote {path} ({graph}, D={dim}, {layer_name}, {mode}, {iters} steps).")
    print(f"Open it with: open {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--graphs", nargs="+", default=DEFAULT_GRAPHS, choices=GRAPHS)
    p.add_argument("--dims", nargs="+", type=int, default=[16, 64, 256])
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS, choices=LAYERS)
    p.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    p.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--out", default="results.csv", help="CSV output path")
    p.add_argument(
        "--capture",
        metavar="PATH.gputrace",
        help="record a Metal GPU trace of the FIRST graph/dim/layer/mode instead "
        "of benchmarking (needs MTL_CAPTURE_ENABLED=1)",
    )
    p.add_argument("--capture-iters", type=int, default=2)
    p.add_argument(
        "--cprofile",
        metavar="PATH.prof",
        help="also profile the Python side of the run with cProfile",
    )
    return p.parse_args()


def main():
    args = parse_args()
    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)

    if (
        any(m != "gather" for m in args.modes)
        and not is_fused_message_passing_enabled()
    ):
        sys.exit(
            "Fused kernels not found: install an mlx-cluster with spmm_coo "
            "(pip install -e <path to mlx_cluster>)."
        )

    if args.capture:
        capture_gpu_trace(
            args.capture,
            args.graphs[0],
            args.dims[0],
            args.layers[0],
            args.modes[0],
            args.capture_iters,
        )
        return

    env = environment()
    print("  ".join(f"{k}={v}" for k, v in env.items()))
    print(HEADER)
    print("-" * len(HEADER))

    profiler = cProfile.Profile() if args.cprofile else None
    if profiler:
        profiler.enable()

    results: List[Result] = []
    for g in args.graphs:
        edge_index, num_nodes = GRAPHS[g]()
        mx.eval(edge_index)
        for d in args.dims:
            for layer in args.layers:
                results += run_config(
                    g,
                    edge_index,
                    num_nodes,
                    d,
                    layer,
                    args.modes,
                    args.warmup,
                    args.repeats,
                )
        del edge_index
        mx.clear_cache()

    if profiler:
        profiler.disable()
        profiler.dump_stats(args.cprofile)
        print(f"\nWrote {args.cprofile}. Top functions by cumulative time:")
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(15)

    write_csv(args.out, results, env)
    print(f"\nWrote {args.out}")
    print(
        "speedup / mem x = gather-scatter forward+backward time / peak memory "
        "divided by the fused mode's (higher is better). err = max |fused - gather|."
    )


if __name__ == "__main__":
    main()
