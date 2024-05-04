import platform
import timeit
from importlib.metadata import version

import mlx.core as mx
import numpy as np
import torch
import torch_geometric.datasets as pyg_datasets
from setup_mlx import mlx_random_walks
from setup_numpy import random_walks_numpy
from setup_torch_cluster import torch_cluster_random_walk
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import to_markdown_table

import mlx_graphs.datasets as mxg_datasets

mx.set_default_device(mx.gpu)

torch.manual_seed(42)
mx.random.seed(42)
np.random.seed(42)

FRAMEWORKS = [
    "mxg",
    "pyg",
    "numpy",
]

frameworks_to_benchmark = {
    "mxg": mlx_random_walks,
    "pyg": torch_cluster_random_walk,
    "numpy": random_walks_numpy,
}

frameworks_to_dataset = {
    "mxg": mxg_datasets.PlanetoidDataset(name="cora", base_dir="~"),
    "pyg": pyg_datasets.Planetoid(root="data/Cora", name="Cora"),
    "numpy": pyg_datasets.Planetoid(root="data/Cora", name="Cora"),
}
COMPILE = {"mxg": [True], "pyg": [False], "numpy": [False]}


def benchmark(framework, edge_index, start_indices, walk_length, compile=None):
    randomwalks = frameworks_to_benchmark[framework]
    randomwalks(edge_index, start_indices, walk_length, compile)


loader = DataLoader(range(2708), batch_size=2000)
start_indices = next(iter(loader))

results = [["Framework", "Time"]]
for framework in tqdm(FRAMEWORKS):
    for compile in COMPILE[framework]:
        dataset = frameworks_to_dataset[framework]

        if framework == "mxg":
            edge_index = dataset.graphs[0].edge_index
        else:
            edge_index = dataset.edge_index

        times = timeit.Timer(
            lambda: benchmark(
                framework,
                edge_index=edge_index,
                start_indices=start_indices,
                walk_length=10000,
                compile=compile,
            )
        ).repeat(repeat=5, number=1)

        time = min(times) / 1

        results.append(
            [
                f"{framework + ('(compiled)' if compile else '')}",
                f"{time:.3f}s",
            ]
        )

platform_info = f"Platform: {platform.platform(terse=True)}"
mlx_version = f"mlx version: {version('mlx')}"
mlx_graphs_version = f"mlx-graphs version: {version('mlx_graphs')}"
torch_version = f"torch version: {version('torch')}"
torch_geometric_version = f"torch_geometric version: {version('torch_geometric')}"
numpy_version = f"Numpy version: {version('numpy')}"
torch_cluster_version = f"torch_cluster version: {version('torch_cluster')}"

md_tab = to_markdown_table(results)

md_content = f"""
# Results

{platform_info}

{mlx_version}

{mlx_graphs_version}

{torch_version}

{torch_geometric_version}

{numpy_version}

{torch_cluster_version}


{md_tab}
"""
with open("results.md", "w") as f:
    f.write(md_content)
