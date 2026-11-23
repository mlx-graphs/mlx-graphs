# Large graphs: quality checks

Results of the scripts of this folder for the large-graph datasets, temporal
neighbor sampling and feature stores (Apple M5 Pro, 64 GB, MLX on the GPU,
PyG 2.8.1 with pyg-lib 0.9.0 on the CPU). The machine was shared with other
jobs (load average about 14), so timings are indicative.

## Datasets: identical to PyG

[`dataset_parity.py`](dataset_parity.py) loads every dataset with
mlx-graphs and with PyG and compares them exactly (edges in the same order,
features, labels, split masks):

| dataset | nodes | edges | same edges | same x | same y | y (mlx-graphs) | same masks | train / val / test |
|---|---|---|---|---|---|---|---|---|
| Flickr | 89,250 | 899,756 | ✓ | ✓ | ✓ | `(89250,)` int32 (PyG int64) | ✓ | 44,625 / 22,312 / 22,313 |
| Reddit2 | 232,965 | 23,213,838 | ✓ | ✓ | ✓ | `(232965,)` int32 (PyG int64) | ✓ | 153,932 / 23,699 / 55,334 |
| Yelp | 716,847 | 13,954,819 | ✓ | ✓ | ✓ | `(716847, 100)` float32 (PyG float32) | ✓ | 537,635 / 107,527 / 71,685 |
| Reddit | 232,965 | 114,615,892 | ✓ | ✓ | ✓ | `(232965,)` int32 (PyG int64) | ✓ | 153,431 / 23,831 / 55,703 |
| AmazonProducts | 1,569,960 | 264,339,468 | ✓ | ✓ | ✓ | `(1569960, 107)` float32 (PyG int64) | ✓ | 1,255,968 / 78,498 / 235,494 |
| PPI | 44,906 (train) | 1,226,368 (train) | ✓ | ✓ | ✓ | `(n, 121)` float32 | - | 20 / 2 / 2 graphs |

PPI: PyG reads DGL's `ppi.zip`, which cannot be downloaded any more
(`data.dgl.ai` is down). mlx-graphs reads the GraphSAGE release (same nodes,
features, labels and edges) and recovers the 24 tissue graphs as the node
ranges no edge crosses: this gives exactly PyG's 20 / 2 / 2 graphs, 44,906 /
6,514 / 5,524 nodes and 1,226,368 training edges. The PPI row compares with
PyG's processing of raw files written in DGL's layout from that release
(edges compared as sorted sets: PyG keeps networkx's edge order).

## Temporal sampling

[`temporal_sampler_quality.py`](temporal_sampler_quality.py), random graph
(20,000 nodes, 400,000 edges, timestamps in [0, 1000)), 4,096 seeds, batches
of 512:

| check | node-level time | edge-level time |
|---|---|---|
| `"last"`, `[10, 5]`: batches identical to PyG's (`n_id`, `batch`, `edge_index`, `e_id`) | 8/8 | 8/8 |
| mlx-graphs: sampled edges later than their seed | 0 | 0 |
| PyG: sampled edges later than their seed | 0 | 0 |
| `"uniform"`, k=5, 100 epochs: mean abs(frequency - k / #valid), mlx-graphs | 0.0267 | 0.0315 |
| same, PyG | 0.0266 | 0.0317 |
| chi-square uniformity rejected (p < 0.01), mlx-graphs | 0 of 142 seeds | 0 of 168 seeds |
| same, PyG | 0 of 142 seeds | 0 of 168 seeds |
| s / epoch, `[10, 5]`, uniform: mlx-graphs / PyG | 0.042 / 0.030 | 0.104 / 0.045 |
| s / epoch, `[10, 5]`, last: mlx-graphs / PyG | 0.052 / 0.024 | 0.043 / 0.036 |

The temporal sampler is vectorized numpy; pyg-lib's C++ sampler is 1.2 to
2.3 times faster on this graph. The test suite also checks `"last"` against
PyG exactly for `NeighborLoader` and `LinkNeighborLoader`
(`tests/loaders/test_temporal_sampling.py`, when pyg-lib is installed).

## Memory-mapped feature store

[`feature_store_memory.py`](feature_store_memory.py): 1.5M nodes with 1,024
float32 features (6.1 GB `.npy` file), 15M edges, `NeighborLoader` over
`(MemmapFeatureStore, MemmapGraphStore)` with fan-out `[10, 5]` and batches
of 1,024 seeds (about 61K sampled nodes, 250 MB of features per batch):

| | value |
|---|---|
| features on disk | 6.14 GB |
| mini-batches / rows gathered | 100 / 6.1M (25 GB read in total) |
| peak resident memory (`ru_maxrss`) | 2.81 GB (0.46 GB before iterating) |
| peak physical footprint (`top`) | 2.41 GB |
| time per mini-batch (cold file cache) | 1.42 s |

The process holds the gathered rows of a mini-batch (and their copies:
the numpy gather, the reordering, the `mx.array`), not the file: loading
the same features in a `GraphData` would take 6.1 GB. Rows are read with
`os.preadv` (nearby rows in one request, 8 threads): in a first version
gathering through `np.memmap`, the mapped pages and the read-ahead kept up to
3.6 GB resident, and a cold-cache probe of 61K random rows took 7 s with a
plain mapping against 0.4 to 1.1 s with positional reads.

## GraphSAGE mini-batch training

[`large_graph_training.py`](large_graph_training.py): 2-layer GraphSAGE
(hidden 256, dropout 0.5, Adam lr 0.003), `NeighborLoader` with fan-outs
`[25, 10]` and batches of 1,024 training nodes, test accuracy (sampled
neighborhoods, same fan-outs) at the epoch of best validation accuracy, time
per epoch (sampling, gathering, forward, backward, optimizer step; median
over epochs). PyG on the CPU with 8 threads.

RESULTS_TRAINING
