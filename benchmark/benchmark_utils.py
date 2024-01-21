import math
from collections import defaultdict
from time import time

import mlx.core as mx
import numpy as np
import torch


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


def measure_runtime(fn, **kwargs) -> float:
    # Avoid first call due to cold start
    # fn(**kwargs)

    tic = time()
    fn(**kwargs)

    return (time() - tic) * 1000


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
