import platform
from argparse import ArgumentParser
from importlib.metadata import version

import torch
from benchmark_layers import (
    benchmark_GATConv,
    benchmark_gather,
    benchmark_gather_batch,
    benchmark_GCNConv,
    benchmark_scatter,
    benchmark_scatter_batch,
)
from benchmark_utils import print_benchmark, run_processes, str2bool

shapes = [
    # 10 nodes
    [(2, 1000), (10, 64)],
    [(2, 10000), (10, 64)],
    [(2, 100000), (10, 64)],
    [(2, 1000000), (10, 64)],
    # 100 nodes
    [(2, 1000), (100, 64)],
    [(2, 10000), (100, 64)],
    [(2, 100000), (100, 64)],
    [(2, 1000000), (100, 64)],
    # 1000 nodes
    [(2, 1000), (1000, 64)],
    [(2, 10000), (1000, 64)],
    [(2, 100000), (1000, 64)],
    [(2, 1000000), (1000, 64)],
    # 10_000 nodes
    [(2, 1000), (10000, 64)],
    [(2, 10000), (10000, 64)],
    [(2, 100000), (10000, 64)],
    [(2, 1000000), (10000, 64)],
]
ops = [
    benchmark_gather,
    benchmark_gather_batch,
    benchmark_scatter,
    benchmark_scatter_batch,
    benchmark_GATConv,
    benchmark_GCNConv,
]
scatter_aggr = ["add", "max"]
batch_sizes = [16, 128, 1024]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--include_cpu", type=str2bool, default="True")
    parser.add_argument("--include_mps", type=str2bool, default="True")
    parser.add_argument("--include_mlx", type=str2bool, default="True")
    args = parser.parse_args()
    print(args)

    if args.include_mps:
        assert torch.backends.mps.is_available(), "MPS backend not available."

    layers = []

    for op in ops:
        for aggr in scatter_aggr:
            for batch_size in batch_sizes:
                for shape in shapes:
                    edge_index_shape, node_features_shape = shape

                    op_args = {
                        "edge_index_shape": edge_index_shape,
                        "node_features_shape": node_features_shape,
                    }
                    if op in [benchmark_scatter, benchmark_scatter_batch]:
                        op_args["scatter_op"] = aggr
                    if op in [benchmark_gather_batch, benchmark_scatter_batch]:
                        op_args["batch_size"] = batch_size
                    if op in [benchmark_GATConv, benchmark_GCNConv]:
                        op_args["in_dim"] = node_features_shape[-1]
                        op_args["out_dim"] = node_features_shape[-1]
                    if (
                        op in [benchmark_gather_batch, benchmark_scatter_batch]
                        and edge_index_shape[1] * node_features_shape[0] * batch_size
                        > 1e9
                    ):  # rm batch ops on very large examples
                        pass
                    else:
                        layers.append((op, op_args))

    all_times = run_processes(layers, args)

    print(f"\nPlatform {platform.platform(terse=True)}")
    print(f"\nmlx version: {version('mlx')}")
    print(f"mlx-graphs version: {version('mlx_graphs')}")
    print(f"torch version: {version('torch')}")
    print(f"torch_geometric version: {version('torch_geometric')}")
    print("\nDetailed benchmark:")
    print_benchmark(all_times, args)
    print("\nAverage benchmark:")
    print_benchmark(all_times, args, reduce_mean=True)
