# Samplers and loaders: quality checks

Results of [`sampler_quality.py`](sampler_quality.py) (Apple M5 Pro, MLX on
the GPU, PyG 2.8.1 on the CPU). PyG's own neighbor samplers, GraphSAINT,
ShaDow, `ClusterData` (METIS) and `knn` / `radius` need compiled extensions
(pyg-lib, torch_sparse, torch_cluster) that are not installed, so they are
checked against references that do not:

- **Exact, in the test suite** (`tests/loaders/test_pyg_reference.py`,
  `tests/utils/test_spatial.py`, `tests/transforms/test_spatial.py`,
  `tests/nn/conv/test_dynamic_edge_conv.py`): `RandomNodeLoader`,
  `ClusterLoader` (PyG's `ClusterData` given the same partition),
  `DynamicBatchSampler` and `ImbalancedSampler` against PyG's pure-Python
  classes; ShaDow subgraphs (all neighbors) and full-fan-out `NeighborLoader`
  against PyG's `k_hop_subgraph` (nodes, relabeled edges, root index, batch
  vector); exact per-hop fan-out of `NeighborLoader`; `LinkNeighborLoader`
  label ratios and supervision edges mapped back through `n_id`; `knn` /
  `radius` against scipy's KD-tree on a few thousand points in unsorted
  batches; `KNNGraph` / `RadiusGraph` against PyG's transforms (with
  torch_cluster replaced by a brute-force version of its documented
  semantics); `DynamicEdgeConv` against PyG's `EdgeConv` on the same k-NN
  graph with copied weights.
- **Statistical and training checks** below.

Reproduce (about 15 minutes):

```
python benchmarks/run_capped.py --gb 6 -- python benchmarks/quality/sampler_quality.py \
    --data-dir .mlx_graphs_data --seeds 0 1 2 3 4
```

## GraphSAINT normalization (Cora)

GraphSAINT makes mini-batch aggregations and losses unbiased with
`edge_norm` (aggregation weight of each edge) and `node_norm` (loss weight of
each node), estimated from `sample_coverage` pre-sampled subgraphs. The paper
defines the aggregation weight of edge `u -> v` as `p_v / p_(u,v)`, with `v`
the node that **aggregates** the edge. PyG computes
`count(source) / count(edge)` instead, i.e. the weight of the reverse edge,
so a node's normalized aggregation is off by `count(u) / count(v)` per
neighbor. mlx-graphs copied PyG's formula; it now uses the target count. Over
3000 fresh subgraphs, the mean of each node's normalized aggregation (when it
is sampled) against its full-graph sum:

| sampler | sample_coverage | edge_norm | nodes | mean rel. err | median | max |
|---|---|---|---|---|---|---|
| Node (batch_size=1000) | 100 | mlx-graphs (target count) | 2708 | 1.3522 | 0.0563 | 901.256 |
| Node (batch_size=1000) | 100 | PyG (source count) | 2708 | 1.1983 | 0.6199 | 11.540 |
| Node (batch_size=1000) | 1000 | mlx-graphs (target count) | 2708 | 0.0474 | 0.0249 | 0.945 |
| Edge (batch_size=400) | 100 | mlx-graphs (target count) | 2708 | 0.0311 | 0.0238 | 0.217 |
| Edge (batch_size=400) | 100 | PyG (source count) | 2708 | 0.8632 | 0.5052 | 6.437 |
| Edge (batch_size=400) | 1000 | mlx-graphs (target count) | 2708 | 0.0134 | 0.0104 | 0.088 |
| RandomWalk (300 roots, length 2) | 100 | mlx-graphs (target count) | 2708 | 0.0292 | 0.0218 | 0.185 |
| RandomWalk (300 roots, length 2) | 100 | PyG (source count) | 2708 | 0.8257 | 0.5015 | 7.130 |
| RandomWalk (300 roots, length 2) | 1000 | mlx-graphs (target count) | 2708 | 0.0128 | 0.0098 | 0.068 |

(The PyG-formula norm is computed from 3000 subgraphs, more than the
library's at `sample_coverage=100`, so it is not handicapped by noisy counts.)
With the target count the error is sampling noise and shrinks with
`sample_coverage`; with PyG's formula it is a bias of about 50% (median) that
does not. The node sampler's large mean / max error at `sample_coverage=100`
comes from a few edges between low-degree nodes (rarely drawn by this
degree-proportional sampler) that were counted once or never while computing
the norms; it vanishes at `sample_coverage=1000`.

Normalized loss `sum(node_norm * loss)` per subgraph vs the full-graph mean
loss (± standard error over 3000 subgraphs):

| sampler | sample_coverage | avg. subgraph nodes | E[normalized loss] | full-graph mean |
|---|---|---|---|---|
| Node (batch_size=1000) | 100 | 742 | 0.50884 ± 0.00034 | 0.50422 |
| Node (batch_size=1000) | 1000 | 742 | 0.50500 ± 0.00033 | 0.50422 |
| Edge (batch_size=400) | 100 | 678 | 0.50924 ± 0.00024 | 0.50422 |
| Edge (batch_size=400) | 1000 | 678 | 0.50459 ± 0.00024 | 0.50422 |
| RandomWalk (300 roots, length 2) | 100 | 671 | 0.50980 ± 0.00027 | 0.50422 |
| RandomWalk (300 roots, length 2) | 1000 | 671 | 0.50480 ± 0.00026 | 0.50422 |

The ~1% excess at `sample_coverage=100` is the estimator's own finite-sample
bias (`E[1 / count] > 1 / E[count]`, about `1 / mean count`, i.e. 1% for
~100 counts per node); it drops tenfold at `sample_coverage=1000`. Same
formula as PyG.

### Edge sampler probabilities: paper vs PyG

The paper (Zeng et al., 2020, sec. 3.3; also its reference implementation)
samples edge `(u, v)` with probability proportional to
`1/deg(u) + 1/deg(v)`, the choice that minimizes the variance of the
normalized aggregation. PyG's `GraphSAINTEdgeSampler` computes
`deg_in = 1 / colcount`, `deg_out = 1 / rowcount` and then
`prob = 1 / deg_in[row] + 1 / deg_out[col]`, i.e.
`in_degree(source) + out_degree(target)`: the inverse weighting, which favors
edges between hubs (it also draws edges with replacement). mlx-graphs uses
the paper's probabilities (`1 / out_degree(source) + 1 / in_degree(target)`,
equal to the paper's for undirected graphs, without replacement); this is the
right choice. One subgraph's estimate of the summed (mean-)aggregation
`sum_(u,v) x_u / deg(v)`, from its edges weighted by `1 / P(edge sampled)`:

| edge probability | edges / subgraph | avg. subgraph nodes | rel. bias | rel. std | edges never sampled |
|---|---|---|---|---|---|
| paper: 1/deg(u) + 1/deg(v) | 400 | 678 | +0.0026 | 0.0269 | 0.0000 |
| PyG: deg(u) + deg(v) | 400 | 495 | +0.0154 | 0.1182 | 0.0000 |
| PyG: deg(u) + deg(v) | 600 | 679 | +0.0127 | 0.0951 | 0.0000 |

At the same number of edges (or of nodes), the paper's probabilities give a
3.5-4.4x smaller standard deviation.

## Statistical properties

ImbalancedSampler on Cora labels (20 epochs of draws; the "imbalanced
subset" keeps all of class 3 and 5% of the other nodes):

| input | labels | class freq. before | class freq. drawn | target | chi2 p (uniform) |
|---|---|---|---|---|---|
| all nodes | 2708 | 0.066-0.302 | 0.140-0.146 | 0.143 | 0.13 |
| imbalanced subset | 921 | 0.015-0.888 | 0.139-0.150 | 0.143 | 0.07 |

DynamicBatchSampler on PROTEINS (1113 graphs, shuffled):

| mode | max_num | skip_too_big | batches | largest batch | mean batch | graphs > max_num | over-budget batches are single graphs | every graph exactly once |
|---|---|---|---|---|---|---|---|---|
| node | 300 | False | 163 | 620 | 267 | 5 | True | True |
| node | 300 | True | 156 | 300 | 264 | 5 | True | True |
| node | 1000 | False | 47 | 1000 | 925 | 0 | True | True |
| node | 1000 | True | 47 | 1000 | 925 | 0 | True | True |
| edge | 2000 | False | 89 | 2098 | 1821 | 2 | True | True |
| edge | 2000 | True | 86 | 2000 | 1837 | 2 | True | True |

("Every graph exactly once" excludes the skipped graphs.) Without
`skip_too_big`, batches match PyG's `DynamicBatchSampler` exactly (test
suite). With `skip_too_big=True`, PyG re-yields graphs that follow a skipped
one (its `num_processed` does not count skipped graphs); mlx-graphs yields
each graph once.

Epoch coverage (each seed / node exactly once per epoch):

| loader | items | covered exactly once |
|---|---|---|
| NeighborLoader (train seeds) | 140 | True |
| LinkNeighborLoader (edges) | 10556 | True |
| ShaDowKHopSampler (train roots) | 140 | True |
| RandomNodeLoader (nodes) | 2708 | True |
| ClusterLoader (nodes) | 2708 | True |

GraphSAINT node coverage over 20 subgraphs:

| sampler | subgraphs | avg. nodes | fraction of nodes seen |
|---|---|---|---|
| Node | 20 | 743 | 0.969 |
| Edge | 20 | 682 | 0.990 |
| RandomWalk | 20 | 673 | 0.989 |

Neighbor sampling uniformity (fan-out 5, 3000 draws, highest in-degree nodes
of Cora; each neighbor should be kept with probability `k / degree`):

| sampler | node | in-degree | k | expected inclusion | observed inclusion | chi2 p (uniform) |
|---|---|---|---|---|---|---|
| NeighborLoader | 1358 | 168 | 5 | 0.030 | 0.023-0.040 | 0.35 |
| ShaDowKHopSampler | 1358 | 168 | 5 | 0.030 | 0.022-0.039 | 0.52 |
| NeighborLoader | 306 | 78 | 5 | 0.064 | 0.053-0.074 | 0.87 |
| ShaDowKHopSampler | 306 | 78 | 5 | 0.064 | 0.055-0.076 | 0.76 |
| NeighborLoader | 1701 | 74 | 5 | 0.068 | 0.060-0.077 | 0.87 |
| ShaDowKHopSampler | 1701 | 74 | 5 | 0.068 | 0.054-0.082 | 0.45 |

Negative sampling (`LinkNeighborLoader`, `NegativeSampling`) draws uniform
(or weighted) node pairs without rejecting existing edges, exactly like PyG
(`neg_sample` has a "TODO: Do not sample false negatives"); on Cora, a
uniform pair is an edge with probability `E / N^2` = 0.14%.

## Node classification (2-layer GraphSAGE, public split)

Same model everywhere (SAGEConv 64 hidden, ReLU, dropout 0.5, Adam lr 0.01,
L2 5e-4), 5 seeds; test accuracy at the best validation accuracy (evaluated
every 5 epochs with full-graph inference). GraphSAINT uses its `node_norm`
loss weights (random walks from 10% of the nodes, length 2, 5 subgraphs per
epoch, `sample_coverage=50`); ClusterLoader uses the dependency-free
balanced partitioner (pymetis is not installed); ShaDow trains on the root
embedding of each subgraph.

| dataset | training | epochs | val acc. (%) | test acc. (%) | time / run (s) |
|---|---|---|---|---|---|
| cora | full-batch (PyG) | 200 | 79.1 ± 0.4 | 80.3 ± 0.7 | 6.2 |
| cora | full-batch | 200 | 78.4 ± 0.4 | 80.2 ± 0.7 | 0.7 |
| cora | NeighborLoader [10, 10] | 50 | 78.3 ± 0.7 | 80.3 ± 0.9 | 1.3 |
| cora | GraphSAINTRandomWalkSampler | 100 | 78.5 ± 0.6 | 80.3 ± 1.1 | 1.8 |
| cora | ClusterLoader (20 parts, 5 per batch) | 100 | 77.5 ± 0.7 | 79.4 ± 1.4 | 1.5 |
| cora | ShaDowKHopSampler (depth 2, 10) | 50 | 78.4 ± 0.3 | 80.2 ± 0.8 | 1.0 |
| pubmed | full-batch (PyG) | 200 | 80.4 ± 0.3 | 76.9 ± 0.8 | 12.5 |
| pubmed | full-batch | 200 | 80.2 ± 0.2 | 77.3 ± 0.8 | 1.9 |
| pubmed | NeighborLoader [10, 10] | 50 | 79.6 ± 0.5 | 76.6 ± 0.4 | 0.5 |
| pubmed | GraphSAINTRandomWalkSampler | 100 | 79.5 ± 0.4 | 77.0 ± 0.9 | 4.0 |
| pubmed | ClusterLoader (20 parts, 5 per batch) | 100 | 80.0 ± 0.4 | 76.9 ± 0.6 | 2.1 |
| pubmed | ShaDowKHopSampler (depth 2, 10) | 50 | 79.8 ± 0.3 | 76.9 ± 0.5 | 0.3 |

Every mini-batch method lands within one standard deviation of full-batch
training (within 0.9 points on Cora, 0.7 on PubMed).

Note for GraphSAINT on graphs with few labeled nodes: `sum(node_norm * loss)`
over the sampled train nodes estimates `sum(train loss) / N`, i.e. the mean
train loss times `num_train / N` (0.3% on PubMed). Combined with weight decay,
that tiny loss scale stopped training altogether on PubMed (41% accuracy,
majority class) until the loss was rescaled by `N / num_train`.

## Link prediction

2-layer GraphSAGE encoder (128 hidden, 64 out, no dropout), dot-product
decoder, Adam lr 0.01, 5 seeds, each with its own 85/5/10 split of the
undirected edges (shared by all methods; validation and test negatives are
fixed non-edges). Full-batch training draws as many uniform negatives as
positives each epoch; `LinkNeighborLoader` uses `neg_sampling_ratio=1.0`,
batches of 256 supervision edges and `[10, 10]` neighbors. Test ROC-AUC at the
best validation AUC (full-graph inference):

| dataset | training | epochs | val AUC (%) | test AUC (%) | time / run (s) |
|---|---|---|---|---|---|
| cora | full-batch (PyG) | 200 | 83.0 ± 4.2 | 83.3 ± 1.6 | 8.5 |
| cora | full-batch | 200 | 83.4 ± 2.2 | 84.4 ± 3.2 | 1.2 |
| cora | LinkNeighborLoader [10, 10], 1 negative / positive | 30 | 85.2 ± 1.3 | 84.3 ± 1.4 | 4.0 |
| pubmed | full-batch (PyG) | 200 | 90.1 ± 0.2 | 90.1 ± 0.3 | 58.0 |
| pubmed | full-batch | 200 | 90.1 ± 0.5 | 90.1 ± 0.5 | 2.8 |
| pubmed | LinkNeighborLoader [10, 10], 1 negative / positive | 30 | 89.1 ± 0.3 | 89.2 ± 0.7 | 49.3 |

`LinkNeighborLoader` matches full-batch training on Cora and is within one
point on PubMed (after 30 epochs). Full-batch link prediction on Cora is noisy
in both frameworks (a 6-seed rerun gave PyG 83.8 ± 2.0 and mlx-graphs
83.4 ± 3.5 test AUC).

On PubMed, `LinkNeighborLoader` batches cover about half of the graph (9.5k
nodes with 500 features); their varying shapes made MLX's buffer cache grow
past 6 GB until the script capped it (`mx.set_cache_limit`). The loader's own
active memory stays constant (50 MB) over epochs.
