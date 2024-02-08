import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import gc
from argparse import ArgumentParser
from collections import defaultdict
from distutils.util import strtobool

import mlx.core as mx
import numpy as np
from tqdm import tqdm

try:
    import torch_geometric  # noqa
    import torch
except RuntimeError:
    raise ImportError(
        "To run the benchmark, install torch_geometric: `pip install torch_geometric`"
    )

from benchmark_layers import (
    benchmark_gather,
    benchmark_GCNConv,
    benchmark_scatter,
)
from benchmark_utils import print_benchmark


def run_processes(layers, args, iterations=1):
    """
    Runs all layers in serial, on separate processes.
    Using processes avoids exploding memory within the main process during the bench.
    """
    all_times = defaultdict(dict)
    queue = mp.Queue()

    with tqdm(total=len(layers) * iterations) as pbar:
        for lay in layers:
            lay_times = defaultdict(list)
            lay_name = None

            for _ in range(iterations):
                p = mp.Process(target=run, args=(lay, args, queue))
                p.start()

                times = queue.get()
                p.join()

                for backend, time in list(times.values())[0].items():
                    lay_times[backend].append(time)
                lay_name = list(times.keys())[0]

                pbar.update(1)

            lay_times_mean = {k: np.mean(v) for k, v in lay_times.items()}
            all_times[lay_name] = lay_times_mean

            # NOTE: without this, memory still increases until the end of the bench.
            del lay
            gc.collect()

    print("\nDetailed benchmark:")
    print_benchmark(all_times, args)
    print("\n Average benchmark:")
    print_benchmark(all_times, args, reduce_mean=True)


def run(fn_with_args, args, queue=None):
    """
    Measures runtime of a single fn on all frameworks and devices included in args.
    """
    fn, kwargs = fn_with_args
    times = times = defaultdict(dict)
    args_str = " ".join([f"{k[:3]}={v}" for k, v in kwargs.items()])
    op_name = f"{fn.__name__} / {args_str}"

    # MLX benchmark.
    if args.include_mlx:
        # GPU
        mx.set_default_device(mx.gpu)
        mlx_time = fn(framework="mlx", **kwargs)
        times[op_name]["mlx_gpu"] = mlx_time

        # CPU
        mx.set_default_device(mx.cpu)
        mlx_time = fn(framework="mlx", **kwargs)
        times[op_name]["mlx_cpu"] = mlx_time

    # CPU PyTorch benchmarks.
    if args.include_cpu:
        cpu_time = fn(framework="torch", device=torch.device("cpu"), **kwargs)
        times[op_name]["pyg_cpu"] = cpu_time

    # MPS PyTorch benchmarks.
    if args.include_mps:
        try:
            mps_time = fn(framework="torch", device=torch.device("mps"), **kwargs)
        except Exception:
            mps_time = float("nan")
        times[op_name]["pyg_mps"] = mps_time

    if queue is None:
        return times
    queue.put(times)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--include_cpu", type=strtobool, default="True")
    parser.add_argument("--include_mps", type=strtobool, default="True")
    parser.add_argument("--include_mlx", type=strtobool, default="True")
    args = parser.parse_args()
    print(args)

    if args.include_mps:
        assert torch.backends.mps.is_available(), "MPS backend not available."

    layers = [
        # Scatter add benchmarks
        (
            benchmark_scatter,  # 100k indices, with 100 unique sources and destinations
            {
                "edge_index_shape": (2, 100000),
                "node_features_shape": (100, 64),
                "scatter_op": "add",
            },
        ),
        (
            benchmark_scatter,  # 1M indices, with 10 unique sources and destinations
            {
                "edge_index_shape": (2, 1000000),
                "node_features_shape": (10, 64),
                "scatter_op": "add",
            },
        ),
        (
            benchmark_scatter,  # 10k indices, with 1000 unique sources and destinations
            {
                "edge_index_shape": (2, 10000),
                "node_features_shape": (1000, 64),
                "scatter_op": "add",
            },
        ),
        # Scatter max benchmarks
        (
            benchmark_scatter,  # 100k indices, with 100 unique sources and destinations
            {
                "edge_index_shape": (2, 100000),
                "node_features_shape": (100, 64),
                "scatter_op": "max",
            },
        ),
        (
            benchmark_scatter,  # 1M indices, with 10 unique sources and destinations
            {
                "edge_index_shape": (2, 1000000),
                "node_features_shape": (10, 64),
                "scatter_op": "max",
            },
        ),
        (
            benchmark_scatter,  # 10k indices, with 1000 unique sources and destinations
            {
                "edge_index_shape": (2, 10000),
                "node_features_shape": (1000, 64),
                "scatter_op": "max",
            },
        ),
        (
            benchmark_gather,  # 100k indices, with 100 unique sources and destinations
            {
                "edge_index_shape": (2, 100000),
                "node_features_shape": (100, 64),
            },
        ),
        (
            benchmark_gather,  # 1M indices, with 10 unique sources and destinations
            {
                "edge_index_shape": (2, 1000000),
                "node_features_shape": (10, 64),
            },
        ),
        (
            benchmark_gather,  # 10k indices, with 1000 unique sources and destinations
            {
                "edge_index_shape": (2, 10000),
                "node_features_shape": (1000, 64),
            },
        ),
        (
            benchmark_GCNConv,
            {
                "in_dim": 64,
                "out_dim": 64,
                "edge_index_shape": (2, 100000),
                "node_features_shape": (100, 64),
            },
        ),
        (
            benchmark_GCNConv,
            {
                "in_dim": 64,
                "out_dim": 64,
                "edge_index_shape": (2, 1000000),
                "node_features_shape": (10, 64),
            },
        ),
        (
            benchmark_GCNConv,
            {
                "in_dim": 64,
                "out_dim": 64,
                "edge_index_shape": (2, 10000),
                "node_features_shape": (1000, 64),
            },
        ),
        # (
        #     benchmark_GCNConv,
        #     {
        #         "in_dim": 8,
        #         "out_dim": 16,
        #         "edge_index_shape": (2, 1000),
        #         "node_features_shape": (100, 8),
        #     },
        # ),
        # (
        #     benchmark_GATConv,
        #     {
        #         "in_dim": 64,
        #         "out_dim": 128,
        #         "edge_index_shape": (2, 100000),
        #         "node_features_shape": (100, 64),
        #     },
        # ),
        # (
        #     benchmark_GATConv,
        #     {
        #         "in_dim": 64,
        #         "out_dim": 128,
        #         "edge_index_shape": (2, 1000000),
        #         "node_features_shape": (10, 64),
        #     },
        # ),
        # (
        #     benchmark_GATConv,
        #     {
        #         "in_dim": 64,
        #         "out_dim": 16,
        #         "edge_index_shape": (2, 10000),
        #         "node_features_shape": (1000, 64),
        #     },
        # ),
        # (
        #     benchmark_GATConv,
        #     {
        #         "in_dim": 8,
        #         "out_dim": 16,
        #         "edge_index_shape": (2, 1000),
        #         "node_features_shape": (100, 8),
        #     },
        # ),
    ]

    run_processes(layers, args)
