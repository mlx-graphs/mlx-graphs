import platform
import timeit
from importlib.metadata import version

import mlx.core as mx
import numpy as np
import torch
from setup_mlx import benchmark_mlx_loader, create_mlx_graph
from setup_pyg import benchmark_pyg_loader, create_pyg_graph
from tqdm import tqdm
from utils import to_markdown_table

# Set seeds for reproducibility
mx.random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Use GPU for MLX
mx.set_default_device(mx.gpu)

# Benchmark configurations
CONFIGS = [
    {
        "name": "Small (Cora-sized)",
        "nodes": 2708,
        "edges": 10556,
        "batch": 256,
        "neighbors": [10, 5],
    },
    {
        "name": "Medium",
        "nodes": 10000,
        "edges": 100000,
        "batch": 512,
        "neighbors": [15, 10, 5],
    },
    {
        "name": "Large",
        "nodes": 100000,
        "edges": 1000000,
        "batch": 1024,
        "neighbors": [25, 10],
    },
]

NUM_BATCHES = 50  # Number of batches to iterate per timing run
REPEATS = 5  # Number of timing repeats (take minimum)


def run_benchmarks():
    results = [
        [
            "Config",
            "Graph Size",
            "Batch Size",
            "Neighbors",
            "MLX-Graphs (s)",
            "PyG (s)",
            "Speedup",
        ]
    ]

    for config in tqdm(CONFIGS, desc="Configurations"):
        num_nodes = config["nodes"]
        num_edges = config["edges"]
        batch_size = config["batch"]
        num_neighbors = config["neighbors"]

        print(f"\nBenchmarking: {config['name']}")
        print(f"  Nodes: {num_nodes}, Edges: {num_edges}")
        print(f"  Batch size: {batch_size}, Neighbors: {num_neighbors}")

        # Create graphs for each framework
        mlx_graph = create_mlx_graph(num_nodes, num_edges)
        mx.eval(mlx_graph.edge_index, mlx_graph.node_features)

        pyg_graph = create_pyg_graph(num_nodes, num_edges)

        # Warmup runs
        print("  Warming up MLX...")
        benchmark_mlx_loader(mlx_graph, batch_size, num_neighbors, min(5, NUM_BATCHES))

        print("  Warming up PyG...")
        benchmark_pyg_loader(pyg_graph, batch_size, num_neighbors, min(5, NUM_BATCHES))

        # Benchmark MLX-Graphs
        print("  Benchmarking MLX-Graphs...")
        mlx_times = timeit.Timer(
            lambda: benchmark_mlx_loader(
                mlx_graph, batch_size, num_neighbors, NUM_BATCHES
            )
        ).repeat(repeat=REPEATS, number=1)
        mlx_time = min(mlx_times)

        # Benchmark PyG
        print("  Benchmarking PyG...")
        pyg_times = timeit.Timer(
            lambda: benchmark_pyg_loader(
                pyg_graph, batch_size, num_neighbors, NUM_BATCHES
            )
        ).repeat(repeat=REPEATS, number=1)
        pyg_time = min(pyg_times)

        # Calculate speedup (positive means MLX is faster)
        speedup = pyg_time / mlx_time

        results.append(
            [
                config["name"],
                f"{num_nodes:,} nodes",
                str(batch_size),
                str(num_neighbors),
                f"{mlx_time:.4f}",
                f"{pyg_time:.4f}",
                f"{speedup:.2f}x",
            ]
        )

        print(f"  MLX: {mlx_time:.4f}s, PyG: {pyg_time:.4f}s, Speedup: {speedup:.2f}x")

    return results


def main():
    print("=" * 60)
    print("NeighborLoader Benchmark: MLX-Graphs vs PyTorch Geometric")
    print("=" * 60)

    # Run benchmarks
    results = run_benchmarks()

    # Gather version info
    platform_info = f"Platform: {platform.platform(terse=True)}"
    mlx_ver = f"mlx version: {version('mlx')}"
    mlx_graphs_ver = f"mlx-graphs version: {version('mlx_graphs')}"
    mlx_cluster_ver = f"mlx-cluster version: {version('mlx_cluster')}"
    torch_ver = f"torch version: {version('torch')}"
    pyg_ver = f"torch_geometric version: {version('torch_geometric')}"

    # Create markdown content
    md_table = to_markdown_table(results)

    md_content = f"""# NeighborLoader Benchmark Results

{platform_info}

{mlx_ver}

{mlx_graphs_ver}

{mlx_cluster_ver}

{torch_ver}

{pyg_ver}

## Configuration

- Number of batches per timing run: {NUM_BATCHES}
- Timing repeats (min taken): {REPEATS}

## Results

{md_table}

## Notes

- Speedup > 1.0x means MLX-Graphs is faster than PyG
- Times are minimum of {REPEATS} runs for {NUM_BATCHES} batch iterations
- MLX uses GPU, PyG uses CPU (Apple Silicon does not support CUDA)
"""

    # Write results
    with open("results.md", "w") as f:
        f.write(md_content)

    print("\n" + "=" * 60)
    print("Results saved to results.md")
    print("=" * 60)

    # Also print the table to console
    print("\nResults Table:")
    print(md_table)


if __name__ == "__main__":
    main()
