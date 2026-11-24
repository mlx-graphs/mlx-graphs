---
title: Large graphs guide
description: Train on graphs larger than the memory with feature and graph stores, and sample neighborhoods that respect time.
---

# Large graphs: stores and temporal sampling

Mini-batch training with [`NeighborLoader`][mlx_graphs.loaders.NeighborLoader] only needs, at every step, the features of the sampled nodes. This page shows how to keep the features out of memory with a feature store, and how to sample neighborhoods that respect time.

## Large datasets

[`Reddit`][mlx_graphs.datasets.Reddit] (233K nodes, 115M edges), [`Reddit2`][mlx_graphs.datasets.Reddit2], [`Flickr`][mlx_graphs.datasets.Flickr], [`Yelp`][mlx_graphs.datasets.Yelp], [`AmazonProducts`][mlx_graphs.datasets.AmazonProducts] (1.6M nodes, 264M edges) and [`PPI`][mlx_graphs.datasets.PPI] follow PyG's files, processing and splits. The single-graph datasets are stored as safetensors after the first processing, and load in seconds:

```python
from mlx_graphs.datasets import Reddit
from mlx_graphs.loaders import NeighborLoader

data = Reddit()[0]
loader = NeighborLoader(
    data, num_neighbors=[25, 10], input_nodes=data.train_mask,
    batch_size=1024, shuffle=True,
)
```

The [Datasets guide](datasets.md#available-datasets) lists their sizes.

## Feature and graph stores

A [`FeatureStore`][mlx_graphs.data.FeatureStore] holds named arrays (node features, labels, timestamps, ...) and returns the rows asked for; a [`GraphStore`][mlx_graphs.data.GraphStore] holds the edges and converts them to the CSC layout used by the samplers. Like PyG's remote backend, a `(feature_store, graph_store)` pair can replace the graph in [`NeighborLoader`][mlx_graphs.loaders.NeighborLoader]: the sampler reads the graph store, and only the rows of the sampled nodes (and edges) are gathered from the feature store. Arrays named `x`, `y` and `edge_attr` become the `node_features`, `node_labels` and `edge_features` of the mini-batch; other arrays keep their name.

[`MemmapFeatureStore`][mlx_graphs.data.MemmapFeatureStore] stores every array in a `.npy` file and reads rows with positional reads, so the features can be larger than the memory. Write them in chunks with [`create_tensor`][mlx_graphs.data.MemmapFeatureStore.create_tensor]:

```python
import numpy as np
from mlx_graphs.data import MemmapFeatureStore, InMemoryGraphStore
from mlx_graphs.loaders import NeighborLoader

feature_store = MemmapFeatureStore("features/")
x = feature_store.create_tensor("x", (num_nodes, 1024), "float32")
for start in range(0, num_nodes, 100_000):
    x[start : start + 100_000] = load_feature_chunk(start)
x.flush()
del x
feature_store["y"] = labels

graph_store = InMemoryGraphStore()
graph_store.put_edge_index(edge_index, layout="coo", size=(num_nodes, num_nodes))

loader = NeighborLoader(
    (feature_store, graph_store), num_neighbors=[10, 5], batch_size=1024
)
for batch in loader:
    out = model(batch.edge_index, batch.node_features)
```

An existing graph converts with `InMemoryFeatureStore.from_data(data)` / `MemmapFeatureStore.from_data(data, root)` and `InMemoryGraphStore.from_data(data)`. [`MemmapGraphStore`][mlx_graphs.data.MemmapGraphStore] keeps the edges (e.g. already in CSC) in `.npy` files, to reopen a large graph without sorting its edges again. The [stores API page](../../api/data/stores.md) lists all of them.

For static, non-disjoint sampling, `NeighborLoader` can sample directly from a memory-mapped CSC graph on MLX builds that support sharing host buffers. The `ogbn-products` example prepares both stores from the official archive and trains GraphSAGE on the official split. [Runnable example](https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/sage_ogbn_products.py) · [Benchmark results](https://github.com/mlx-graphs/mlx-graphs/blob/main/benchmarks/quality/large_graph_results.md#disk-backed-graphsage-on-ogbn-products)

[`benchmarks/quality/feature_store_memory.py`](https://github.com/mlx-graphs/mlx-graphs/blob/main/benchmarks/quality/feature_store_memory.py) trains nothing but iterates a `NeighborLoader` over a 6.1 GB features file (250 MB of features per mini-batch): the process peaks under 3 GB of resident memory, the size of a few mini-batches rather than of the file.

## Temporal sampling

With `time_attr`, [`NeighborLoader`][mlx_graphs.loaders.NeighborLoader] only samples the neighbors whose timestamp is not later than the time of their seed, as in PyG:

```python
# node-level time: a neighbor's time is its node's
data.time = node_timestamps  # [num_nodes]
loader = NeighborLoader(
    data, num_neighbors=[10, 5], input_nodes=seeds, batch_size=256,
    time_attr="time", temporal_strategy="last",
)

# edge-level time: the time of the edge; seed times are then required
data.time = edge_timestamps  # [num_edges]
loader = NeighborLoader(
    data, num_neighbors=[10, 5], input_nodes=seeds, input_time=seed_times,
    time_attr="time",
)
```

- `temporal_strategy="uniform"` samples uniformly among the valid neighbors, `"last"` keeps the most recent ones (then the mini-batches are the same as PyG's with pyg-lib).
- Every seed gets its own subgraph (`disjoint=True`): a node sampled for two seeds appears twice, and `batch.batch` gives the seed of every node.
- [`LinkNeighborLoader`][mlx_graphs.loaders.LinkNeighborLoader] takes `time_attr` and `edge_label_time`: both ends of a supervision edge (and its negatives) are sampled at the time of the edge.

Temporal sampling runs on the host, with numpy: the neighbors of every node are sorted by time once, and each hop is a vectorized binary search and sampling step over the whole frontier.

The [validation page](../../validation.md#large-graphs) compares training on these datasets, and the temporal sampler, with PyG.
