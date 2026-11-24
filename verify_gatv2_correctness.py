"""Do the in-tree Metal kernel and mlx_cluster.gatv2_scores give the gather
reference's layer outputs and gradients, on each graph?

The same GATv2 layer as benchmarks/kernels/gatv2_scores.py (mlx_conv("gatv2",
128, 128): 4 heads x 32 channels, int32 edges, fused SpMM aggregation), run
three times with only the score function swapped. Compares the loss, the layer
output, and the gradients of every parameter and of the input features.

Run from the mlx-graphs folder:
    python verify_gatv2_correctness.py                 # all five graphs
    python verify_gatv2_correctness.py amazon-computers synthetic-100k
"""

import sys

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from benchmarks.common import random_graph
from benchmarks.workloads import load_node_dataset, mlx_conv
from mlx_graphs.nn.conv import gatv2_conv
from mlx_graphs.nn.kernels.gatv2_edge_scores import _reference_scores
from mlx_graphs.nn.message_passing import _csr_cache, clear_csr_cache

GRAPHS = {
    "synthetic-100k": (10_000, 100_000),
    "synthetic-2m": (200_000, 2_000_000),
    "ogbn-arxiv": None,
    "amazon-computers": None,
    "coauthor-physics": None,
}
# relative error ||got - want|| / ||want||, per tensor (float32 sums over up to
# millions of edges, in a different order in each kernel)
TOLERANCE = 1e-4


def load(name):
    if GRAPHS[name] is not None:
        nodes, edges = GRAPHS[name]
        return random_graph(nodes, edges), nodes
    data = load_node_dataset(name, seed=0)
    return data["edge_index"], data["y"].shape[0]


def mlx_cluster_scores(src_feats, dst_feats, att, edge_index, slope, edge_feats=None):
    from mlx_cluster import gatv2_scores

    csr = _csr_cache.get(edge_index, dst_feats.shape[0], src_feats.shape[0])
    return gatv2_scores(
        edge_index[1], edge_index[0], dst_feats, src_feats, att, slope, csr=csr
    )


PATHS = {
    "reference": _reference_scores,
    "metal": gatv2_conv.gatv2_edge_scores,  # the in-tree kernel (Metal on GPU)
    "mlx-cluster": mlx_cluster_scores,
}


def run(edge_index, num_nodes, score):
    gatv2_conv.gatv2_edge_scores = score
    mx.random.seed(0)
    layer = mlx_conv("gatv2", 128, 128)
    ei = mx.array(edge_index.astype(np.int32))
    x = mx.random.normal((num_nodes, 128), key=mx.random.key(1))
    cot = mx.random.normal((num_nodes, 128), key=mx.random.key(2))

    def loss(params, x):
        layer.update(params)
        out = layer(ei, x)
        return (out * cot).sum(), out

    (value, out), (g_params, g_x) = _value_and_grad(loss, layer, x)
    tensors = {"loss": value, "output": out, "grad x": g_x}
    tensors.update({f"grad {k}": v for k, v in tree_flatten(g_params)})
    mx.eval(tensors)
    return tensors


def _value_and_grad(loss, layer, x):
    def scalar(params, x):
        return loss(params, x)[0]

    value, grads = mx.value_and_grad(scalar, argnums=(0, 1))(layer.parameters(), x)
    out = loss(layer.parameters(), x)[1]
    return (value, out), grads


def main():
    names = sys.argv[1:] or list(GRAPHS)
    unknown = [n for n in names if n not in GRAPHS]
    if unknown:
        raise SystemExit(f"unknown graph(s) {unknown}; choose from {list(GRAPHS)}")
    print(f"device: {mx.default_device()}")
    failed = False
    for name in names:
        edge_index, num_nodes = load(name)
        print(f"\n{name}: {num_nodes:,} nodes, {edge_index.shape[1]:,} edges")
        results = {}
        for path, score in PATHS.items():
            clear_csr_cache()  # each path builds its own CSR
            results[path] = run(edge_index, num_nodes, score)
        want = results["reference"]
        for path in ("metal", "mlx-cluster"):
            worst_key, worst = None, 0.0
            for key, ref in want.items():
                got = np.array(results[path][key], dtype=np.float64)
                ref = np.array(ref, dtype=np.float64)
                err = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30)
                if err > worst:
                    worst_key, worst = key, err
            ok = worst < TOLERANCE
            failed |= not ok
            print(
                f"  {path:>11} vs reference: {'OK  ' if ok else 'FAIL'} "
                f"worst relative error {worst:.2e} ({worst_key}), "
                f"{len(want)} tensors compared"
            )
    gatv2_conv.gatv2_edge_scores = PATHS["metal"]
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
