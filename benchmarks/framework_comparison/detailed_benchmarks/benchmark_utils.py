import math
import multiprocessing as mp
import timeit
from argparse import ArgumentTypeError
from collections import defaultdict

import mlx.core as mx
import numpy as np
import torch

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import gc

import torch_geometric  # noqa
from tqdm import tqdm


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def get_dummy_edge_index(shape, num_nodes, device, framework):
    if framework == "mlx":
        return mx.random.randint(0, num_nodes - 1, shape)
    elif framework == "pyg":
        return torch.randint(0, num_nodes - 1, shape).to(device)
    raise ValueError("Framework should be either mlx or pyg.")


def get_dummy_features(shape, device, framework):
    if framework == "mlx":
        return mx.random.normal(shape).astype(mx.float32)
    elif framework == "pyg":
        return torch.randn(shape, dtype=torch.float32).to(device)
    raise ValueError("Framework should be either mlx or pyg.")


def measure_runtime(fn, repeat=5, iters=2, **kwargs) -> float:
    time = min(timeit.Timer(lambda: fn(**kwargs)).repeat(repeat=repeat, number=iters))
    return time * 1000 / iters


def calculate_speedup(a, compared_to):
    percentage_difference = -((a - compared_to) / a)
    return percentage_difference * 100


def print_benchmark(times, args, reduce_mean=False):
    times = dict(times)

    if reduce_mean:
        new_times = defaultdict(lambda: defaultdict(list))
        for k, v in times.items():
            op = k.split("/")[0]
            for backend, runtime in v.items():
                new_times[op][backend].append(runtime)

        for k, v in new_times.items():
            for backend, runtimes in v.items():
                new_times[k][backend] = np.mean(new_times[k][backend])
        times = new_times

    # Column headers
    headers = []
    if args.include_mlx:
        headers.append("mlx_gpu")
        headers.append("mlx_cpu")
    if args.include_mps:
        headers.append("pyg_mps")
    if args.include_cpu:
        headers.append("pyg_cpu")

    if args.include_cpu and args.include_mlx:
        h = "mlx_gpu/pyg_cpu speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = calculate_speedup(v["mlx_gpu"], compared_to=v["pyg_cpu"])

    if args.include_mps and args.include_mlx:
        h = "mlx_gpu/pyg_mps speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = calculate_speedup(v["mlx_gpu"], compared_to=v["pyg_mps"])

    max_name_length = max(len(name) for name in times.keys())

    header_row = (
        "| Operation" + " " * (max_name_length - 5) + " | " + " | ".join(headers) + " |"
    )
    header_line_parts = ["-" * (max_name_length + 6)] + [
        "-" * max(6, len(header)) for header in headers
    ]
    header_line = "|" + "|".join(header_line_parts) + "|"

    print(header_row)
    print(header_line)

    def add_plus_symbol(x, rounding):
        return (
            f"{'+' if x > 0 else ''}{int(x) if rounding == 0 else round(x, rounding)}"
        )

    def format_value(header):
        return (
            f"{add_plus_symbol(times[header], 0):>6}%"
            if "speedup" in header
            else f"{times[header]:>6.2f}"
        )

    def format_header(header):
        return f"{times[header]}" if math.isnan(times[header]) else format_value(header)

    for op, times in times.items():
        times_str = " | ".join(format_header(header) for header in headers)

        # Formatting each row
        print(f"| {op.ljust(max_name_length)} | {times_str} |")


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

    return all_times


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
