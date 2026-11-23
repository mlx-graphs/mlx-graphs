---
title: Datasets guide
---

# Datasets

MLX-graphs provides several built-in datasets for graph machine learning tasks.

!!! note

    Datasets are automatically downloaded to `.mlx_graphs_data/` in the current working directory upon first use. You can specify a custom directory by passing `base_dir` to the dataset constructor.

## Available Datasets

Each name links the API page of the class that loads it; `Load with` is the call. The table is built from [`docs-site/datasets.yml`](https://github.com/mlx-graphs/mlx-graphs/blob/main/docs-site/datasets.yml).

--8<-- "datasets_table.md"

The last rows are collections and heterogeneous graphs, with one example each and no statistics: their class page lists the other datasets they load.

--8<-- "datasets_other.md"

## Using Datasets

Loading a dataset is straightforward:

```python
from mlx_graphs.datasets import KarateClubDataset, PlanetoidDataset

# Simple built-in dataset
dataset = KarateClubDataset()
graph = dataset[0]
print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")

# Citation network dataset
cora = PlanetoidDataset(name="cora")
print(cora[0])
```

## Computing Dataset Statistics

To compute statistics for any dataset, use the included utility script:

```bash
# Stats for all available datasets
python examples/compute_dataset_stats.py

# Stats for a specific dataset
python examples/compute_dataset_stats.py --dataset karate_club

# Output as markdown table (for this page, or issues)
python examples/compute_dataset_stats.py --format markdown

# Output as RST table
python examples/compute_dataset_stats.py --format rst
```
