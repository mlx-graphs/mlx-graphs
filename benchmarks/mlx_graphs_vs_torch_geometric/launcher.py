import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import gc
from argparse import ArgumentParser
from collections import defaultdict

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
    benchmark_fast_gather,
    benchmark_GATConv,
    benchmark_gather,
    benchmark_GCNConv,
    benchmark_scatter,
)
from benchmark_utils import print_benchmark, str2bool


def run_processes(layers, args):
    """
    Runs all layers in serial, on separate processes.
    Using processes avoids exploding memory within the main process during the bench.
    """
    all_times = defaultdict(dict)
    queue = mp.Queue()

    with tqdm(total=len(layers)) as pbar:
        for lay in layers:
            lay_times = defaultdict(list)
            lay_name = None

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
    parser.add_argument("--include_cpu", type=str2bool, default="True")
    parser.add_argument("--include_mps", type=str2bool, default="True")
    parser.add_argument("--include_mlx", type=str2bool, default="True")
    args = parser.parse_args()
    print(args)

    if args.include_mps:
        assert torch.backends.mps.is_available(), "MPS backend not available."

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
    operations = [
        "benchmark_scatter",
        "benchmark_gather",
        "benchmark_fast_gather",
        "benchmark_GCNConv",
        "benchmark_GATConv",
    ]

    layers = []
    for op in operations:
        for shape in shapes:
            edge_index_shape, node_features_shape = shape

            if op == "benchmark_scatter":
                layers.append(
                    (
                        benchmark_scatter,
                        {
                            "edge_index_shape": edge_index_shape,
                            "node_features_shape": node_features_shape,
                            "scatter_op": "add",
                        },
                    )
                )

            if op == "benchmark_gather":
                layers.append(
                    (
                        benchmark_gather,
                        {
                            "edge_index_shape": edge_index_shape,
                            "node_features_shape": node_features_shape,
                        },
                    )
                )

            if op == "benchmark_fast_gather":
                layers.append(
                    (
                        benchmark_fast_gather,
                        {
                            "edge_index_shape": list(edge_index_shape),
                            "node_features_shape": list(node_features_shape),
                        },
                    )
                )

            if op == "benchmark_GCNConv":
                layers.append(
                    (
                        benchmark_GCNConv,
                        {
                            "in_dim": node_features_shape[-1],
                            "out_dim": node_features_shape[-1],
                            "edge_index_shape": edge_index_shape,
                            "node_features_shape": node_features_shape,
                        },
                    )
                )
            if op == "benchmark_GATConv":
                layers.append(
                    (
                        benchmark_GATConv,
                        {
                            "in_dim": node_features_shape[-1],
                            "out_dim": node_features_shape[-1],
                            "edge_index_shape": edge_index_shape,
                            "node_features_shape": node_features_shape,
                        },
                    )
                )

    run_processes(layers, args)
