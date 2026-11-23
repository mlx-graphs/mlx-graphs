---
title: Datasets guide
---

# Datasets

MLX-graphs provides several built-in datasets for graph machine learning tasks.

!!! note

    Datasets are automatically downloaded to `.mlx_graphs_data/` in the current working directory upon first use. You can specify a custom directory by passing `base_dir` to the dataset constructor.

## Available Datasets

Each name links the API page of the class that loads it; `Load with` is the call. The table is built from [`docs/datasets.yml`](https://github.com/mlx-graphs/mlx-graphs/blob/main/docs/datasets.yml).

<div class="mg-datasets" markdown>

--8<-- "datasets_table.md"

</div>

The average degree is 2 × edges / nodes. The last rows are collections, heterogeneous and temporal graphs, one example each; their class page lists the other datasets they load:

- for a collection, `Graphs` is the number of graphs (of the split that `Load with` loads) and the other numbers are averages per graph (\~);
- a heterogeneous graph counts the nodes and edges of all types: ogbn-mag has 4 node types (paper, author, institution, field of study) and 4 edge types, IMDB 3 node types (movie, director, actor) and 4 edge types, DBLP 4 node types (author, paper, term, conference) and 6 edge types;
- a temporal graph counts each event as an edge.

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

## Add your own dataset

Subclass [`Dataset`][mlx_graphs.datasets.Dataset] and implement two methods: `download`, which fetches the raw files into `self.raw_path`, and `process`, which turns them into a list of [`GraphData`][mlx_graphs.data.data.GraphData] in `self.graphs`. The first instantiation runs both and saves the processed graphs under `base_dir` (`.mlx_graphs_data/` by default); later ones load them. Indexing, slicing, `shuffle`, the `num_*` properties, transforms and the [`Dataloader`][mlx_graphs.loaders.Dataloader] then work as for the built-in datasets.

A made-up dataset, generated instead of downloaded: is a graph a cycle or a path?

```python
import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets import Dataset
from mlx_graphs.loaders import Dataloader


class CyclesOrPaths(Dataset):
    """Graph classification: is the graph a cycle (label 1) or a path (label 0)?"""

    def __init__(self, num_graphs: int = 200, base_dir=None):
        self.num_graphs_to_make = num_graphs
        super().__init__(name="cycles_or_paths", base_dir=base_dir)

    def download(self):
        pass  # nothing to download: the graphs are generated in process()

    def process(self):
        rng = np.random.default_rng(0)
        self.graphs = []
        for i in range(self.num_graphs_to_make):
            n = int(rng.integers(5, 20))
            src = np.arange(n - 1)
            if i % 2:  # close the path into a cycle
                src = np.arange(n)
            dst = (src + 1) % n
            edges = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
            self.graphs.append(
                GraphData(
                    edge_index=mx.array(edges, dtype=mx.int32),
                    node_features=mx.ones((n, 1)),
                    graph_labels=mx.array([i % 2]),
                )
            )


dataset = CyclesOrPaths()  # processed once, then loaded from .mlx_graphs_data/
print(dataset)
print(dataset.num_graphs, dataset.num_graph_classes, dataset[0])
loader = Dataloader(dataset, batch_size=32, shuffle=True)
print(next(iter(loader)).num_graphs)
```

```text
cycles_or_paths(num_graphs=200)
200 2 GraphData(
	edge_index(shape=(2, 32), int32)
	node_features(shape=(17, 1), float32)
	graph_labels(shape=(1,), int32))
32
```

For a dataset that is downloaded, call [`download`][mlx_graphs.datasets.utils.download.download] in `download` (and [`check_sha1`][mlx_graphs.datasets.utils.check_sha1] or [`extract_archive`][mlx_graphs.datasets.utils.extract_archive] as needed), and read the files of `self.raw_path` in `process`. The built-in datasets, in [`mlx_graphs/datasets`](https://github.com/mlx-graphs/mlx-graphs/tree/main/mlx_graphs/datasets), are examples. For a dataset of heterogeneous graphs, subclass [`HeteroDataset`][mlx_graphs.datasets.HeteroDataset].

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
