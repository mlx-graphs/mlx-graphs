"""Time GATv2's score function alone (not the whole layer): the gather
reference, the in-tree Metal kernel and mlx_cluster.gatv2_scores, forward and
forward + backward, on each graph.

Inputs match the benchmark's GATv2 layer: 4 heads x 32 channels, int32 edges.
mlx_cluster gets its CSR built once beforehand, as mlx-graphs' cache does.
Each measurement: 5 warm-ups, then the median of 12 timed iterations. Run the
script three times; it rotates the path order on each run (--run 1/2/3).

Run from the mlx-graphs folder:
    python gatv2_score_only.py --run 1
    python gatv2_score_only.py --run 2 amazon-computers coauthor-physics
"""

import argparse
import statistics
import time

import mlx.core as mx
import numpy as np

from benchmarks.common import random_graph
from benchmarks.workloads import load_node_dataset
from mlx_graphs.nn.kernels.gatv2_edge_scores import _reference_scores, gatv2_edge_scores

GRAPHS = {
    "synthetic-100k": (10_000, 100_000),
    "synthetic-2m": (200_000, 2_000_000),
    "ogbn-arxiv": None,
    "amazon-computers": None,
    "coauthor-physics": None,
}
PATHS = ("reference", "metal", "mlx-cluster")
HEADS, CHANNELS, SLOPE = 4, 32, 0.2


def load(name):
    if GRAPHS[name] is not None:
        nodes, edges = GRAPHS[name]
        return random_graph(nodes, edges), nodes
    data = load_node_dataset(name, seed=0)
    return data["edge_index"], data["y"].shape[0]


def timed(fn, warmup=5, iterations=12):
    for _ in range(warmup):
        mx.eval(fn())
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        mx.eval(fn())
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def score_fn(path, edge_index, num_nodes):
    if path == "reference":
        return lambda s, d, a: _reference_scores(s, d, a, edge_index, SLOPE)
    if path == "metal":
        return lambda s, d, a: gatv2_edge_scores(s, d, a, edge_index, SLOPE)
    from mlx_cluster import coo_to_csr, gatv2_scores

    row, col = edge_index[1], edge_index[0]
    csr = coo_to_csr(row, col, num_nodes, num_cols=num_nodes)  # once, like the cache
    return lambda s, d, a: gatv2_scores(row, col, d, s, a, SLOPE, csr=csr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", nargs="*", help=f"default: all of {list(GRAPHS)}")
    parser.add_argument(
        "--run", type=int, default=1, help="1, 2 or 3: rotates path order"
    )
    args = parser.parse_args()
    unknown = [g for g in args.graphs if g not in GRAPHS]
    if unknown:
        parser.error(f"unknown graph(s) {unknown}; choose from {list(GRAPHS)}")
    shift = (args.run - 1) % len(PATHS)
    order = PATHS[shift:] + PATHS[:shift]
    print(f"device: {mx.default_device()}, path order: {', '.join(order)}")

    for name in args.graphs or list(GRAPHS):
        edge_np, num_nodes = load(name)
        edge_index = mx.array(edge_np.astype(np.int32))
        keys = mx.random.split(mx.random.key(0), 4)
        src = mx.random.normal((num_nodes, HEADS, CHANNELS), key=keys[0])
        dst = mx.random.normal((num_nodes, HEADS, CHANNELS), key=keys[1])
        att = mx.random.normal((1, HEADS, CHANNELS), key=keys[2])
        cot = mx.random.normal((edge_np.shape[1], HEADS), key=keys[3])
        mx.eval(edge_index, src, dst, att, cot)

        results = {}
        for path in order:
            fn = score_fn(path, edge_index, num_nodes)

            def forward_backward():
                out, grads = mx.vjp(fn, [src, dst, att], [cot])
                return [*out, *grads]

            results[path] = (timed(lambda: fn(src, dst, att)), timed(forward_backward))

        print(
            f"\n{name}: {num_nodes:,} nodes, {edge_np.shape[1]:,} edges (scores only)"
        )
        metal_fb = results["metal"][1]
        for path in PATHS:
            fwd, fb = results[path]
            print(
                f"  {path:>11}: forward {fwd:8.2f} ms   forward + backward {fb:8.2f} ms"
                f"   ({metal_fb / fb:.2f}x vs metal)"
            )


if __name__ == "__main__":
    main()
