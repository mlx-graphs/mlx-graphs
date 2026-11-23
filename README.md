<p align="center">
  <img alt="mlx-graphs" src="https://raw.githubusercontent.com/mlx-graphs/mlx-graphs/main/docs/source/_static/logo.svg" width="420">
</p>

<h3 align="center">Fast graph neural networks on Apple Silicon</h3>

<p align="center">
  <a href="https://pypi.org/project/mlx-graphs/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mlx-graphs?color=2a78d6"></a>
  <a href="https://pypi.org/project/mlx-graphs/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776ab"></a>
  <img alt="macOS on Apple Silicon" src="https://img.shields.io/badge/macOS-Apple%20Silicon-555">
  <a href="https://github.com/mlx-graphs/mlx-graphs/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://github.com/ml-explore/mlx"><img alt="Built on MLX" src="https://img.shields.io/badge/built%20on-MLX-black?logo=apple"></a>
  <a href="https://discord.gg/K3mWFCxxM7"><img alt="Discord" src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white"></a>
</p>

<p align="center">
  <a href="https://mlx-graphs.github.io/mlx-graphs/"><b>Documentation</b></a> ·
  <a href="https://mlx-graphs.github.io/mlx-graphs/tutorials/quickstart.html"><b>Quickstart</b></a> ·
  <a href="#benchmarks-vs-pytorch-geometric"><b>Benchmarks</b></a> ·
  <a href="https://github.com/mlx-graphs/mlx-graphs/tree/main/examples"><b>Examples</b></a> ·
  <a href="https://discord.gg/K3mWFCxxM7"><b>Discord</b></a>
</p>

MLX-graphs is a library for Graph Neural Networks (GNNs) on Apple Silicon, built on Apple's [MLX](https://github.com/ml-explore/mlx).
It runs message passing on your Mac's GPU with fused sparse kernels, which
PyTorch Geometric lacks on MPS.

**On an M5 Pro, mlx-graphs trains GNNs 5.6× faster than PyTorch Geometric on
MPS and 5.8× faster than on CPU, with the same accuracy** (median of 35
benchmarks on 12 datasets, up to 58×).

<img alt="Left: time of PyG on MPS and on CPU relative to mlx-graphs: 12× and 6.7× for one layer on 2 million edges, 5.3× and 7.4× per epoch of node classification, 5.6× and 2.9× per epoch of graph classification. Right: mean test accuracy of mlx-graphs, PyG on MPS and PyG on CPU: 79.2%, 79.0% and 78.9% in node classification, 68.8%, 69.2% and 68.9% in graph classification" src="https://raw.githubusercontent.com/mlx-graphs/mlx-graphs/main/benchmarks/pyg_comparison/results/Apple_M5_Pro/0.1.0-c5f1246/figures/headline_light.png">

<sub>Apple M5 Pro, 64 GB · mlx 0.32.4 · PyTorch 2.15 · PyTorch Geometric 2.8.1 · same models, data and
hyperparameters in every framework, 3 seeds. See [benchmarks](#benchmarks-vs-pytorch-geometric) for the details.</sub>


## Installation
```
pip install mlx-graphs
```

## Features

- **Fast GNN training and inference on Apple Silicon.**
   Message passing runs on the Mac's GPU through MLX and fused sparse
   kernels: neighbours are aggregated with one sparse x dense matmul (SpMM)
   instead of building a `[num_edges, features]` message array, including for
   the attention layers (one SpMM per head): 5–6× faster training than PyTorch
   Geometric on the same Mac, see the [benchmarks](#benchmarks-vs-pytorch-geometric).
- **A PyG-style API.** `MessagePassing`, `GraphData` / batching, datasets,
  transforms and utilities follow PyTorch Geometric's design, so porting a
  PyG model is mostly mechanical. `from_pyg` / `to_pyg` convert graphs both ways.
- **Scalability to large graphs.** With unified memory, the whole memory of
  the Mac is available to the GPU, and fused message passing keeps the peak
  memory of a layer close to the size of its inputs and outputs.

### What's included

Names and signatures follow PyTorch Geometric; see [Coming from PyTorch Geometric](https://mlx-graphs.github.io/mlx-graphs/tutorials/from_pyg.html).

| | |
|---|---|
| **Convolutions** | `GCNConv`, `SAGEConv`, `GINConv`, `GINEConv`, `GATConv`, `GATv2Conv`, `TransformerConv`, `GraphConv`, `GeneralConv`, `SGConv`, `SSGConv`, `APPNP`, `ChebConv`, `TAGConv`, `ARMAConv`, `MixHopConv`, `GCN2Conv`, `FAConv`, `AGNNConv`, `LEConv`, `LGConv`, `MFConv`, `ClusterGCNConv`, `GENConv`, `PNAConv`, `CGConv`, `NNConv`, `ResGatedGraphConv`, `GatedGraphConv`, `FiLMConv`, `EdgeConv`, `DynamicEdgeConv`, `DirGNNConv`, `SimpleConv`, `GeneralizedRelationalConv` |
| **Heterogeneous** | `to_hetero`, `to_hetero_with_bases`, `HeteroConv`, `HANConv`, `HGTConv`, `RGCNConv`, `FastRGCNConv`, `RGATConv`, `HeteroDictLinear`, `HeteroGraphData` |
| **Graph transformers** | `GPSConv`, `SGFormer`, `Polynormer`, `MultiheadAttention`, `PerformerAttention`; positional encodings `AddRandomWalkPE`, `AddLaplacianEigenvectorPE`, `RWSEEncoder`, `LapPEEncoder` |
| **Temporal graphs** | `TemporalData`, `TemporalDataLoader`, `TGNMemory`, `LastNeighborLoader`, `TemporalEncoding`, `JODIEDataset`, TGB datasets (`TGBLinkPropPredDataset`, `TGBNodePropPredDataset`) with the official negative samples and `TGBEvaluator` |
| **Models** | `GCN`, `GraphSAGE`, `GIN`, `GAT`, `PNA`, `EdgeCNN`, `MLP`, `DeepGCNLayer`, `LINKX`, `LightGCN`, `GAE` / `VGAE` / `ARGA` / `ARGVA`, `DeepGraphInfomax`, `RECT_L`, `LabelPropagation`, `CorrectAndSmooth`, knowledge-graph embeddings (`TransE`, `DistMult`, `ComplEx`, `RotatE`), `Node2Vec` |
| **Aggregations** | `aggr="sum"`, `"mean"`, `"max"`, ... or lists of them, and the modules `SoftmaxAggregation`, `PowerMeanAggregation`, `MultiAggregation`, `DegreeScalerAggregation`, `QuantileAggregation`, `MedianAggregation`, `StdAggregation`, `VarAggregation`, `LSTMAggregation`, `GRUAggregation`, `SortAggregation`, `SetTransformerAggregation`, `GraphMultisetTransformer`, `DeepSetsAggregation`, `MLPAggregation`, `EquilibriumAggregation`, `LCMAggregation`, ... |
| **Pooling / readout** | `global_{add,mean,max,sort}_pool`, `TopKPooling`, `SAGPooling`, `AttentionalAggregation`, `Set2Set`, `JumpingKnowledge`, `knn_graph`, `radius_graph` |
| **Normalization** | `BatchNorm`, `LayerNorm`, `InstanceNorm`, `GraphNorm`, `PairNorm`, `DiffGroupNorm`, `MessageNorm`, `GraphSizeNorm`, `MeanSubtractionNorm`, `HeteroBatchNorm`, `HeteroLayerNorm` |
| **Loaders and samplers** | `Dataloader`, `NeighborLoader` and `LinkNeighborLoader` (homogeneous and heterogeneous), `HGTLoader`, `ClusterData` / `ClusterLoader`, GraphSAINT samplers, `ShaDowKHopSampler`, `RandomNodeLoader`, `ImbalancedSampler`, `DynamicBatchSampler`, `DataListLoader`, `DenseDataLoader` |
| **Datasets** | Planetoid, CitationFull, Amazon, Coauthor, WikiCS, WebKB, WikipediaNetwork, Actor, HeterophilousGraphDataset, OGB (incl. OGBMAG), TU datasets, GNNBenchmarkDataset, LRGB, JODIE, Temporal Graph Benchmark (TGB), QM7b, superpixels, Elliptic, MovieLens, IMDB, DBLP, FB15k-237, WordNet18, Karate Club, fake datasets |
| **Transforms** | `Compose`, `ToUndirected`, `AddSelfLoops`, `NormalizeFeatures`, `RandomNodeSplit`, `RandomLinkSplit`, `GCNNorm`, `SIGN`, `VirtualNode`, `LineGraph`, `TwoHop`, `KNNGraph`, `LocalDegreeProfile`, `FeaturePropagation`, `SVDFeatureReduction`, `ToDense`, ... (32 in all) |
| **Metrics** | `LinkPredMRR`, `LinkPredHitRatio`, `LinkPredNDCG`, `LinkPredMAP`, `LinkPredPrecision`, `LinkPredRecall`, `LinkPredF1` and more, `roc_auc_score`, `average_precision_score` |
| **Utilities** | `scatter`, `softmax`, `degree`, `subgraph`, `k_hop_subgraph`, `negative_sampling`, `dropout_edge`, `to_dense_batch`, `to_dense_adj`, `get_laplacian`, `get_ppr`, `homophily`, `assortativity`, `coalesce`, `to_undirected`, graph generators, `from_pyg` / `to_pyg`, `from_networkx` / `to_networkx`, `from_scipy_sparse_matrix`, ... |

Every layer was checked against PyG with copied weights (outputs and gradients),
and the models by training them in both frameworks on the same data: see
[Validation against PyG](https://mlx-graphs.github.io/mlx-graphs/validation.html).

## Benchmarks vs PyTorch Geometric

Apple M5 Pro (64 GB), PyG 2.8.1 / PyTorch 2.15 on MPS and CPU; same models,
hyperparameters and data in every framework, 3 seeds.

### Faster

Speedups against PyG on MPS (the CPU numbers are in the figures):

- **One layer** (forward + backward, 2M edges): **9–31× faster** for GCN, GraphSAGE, GIN and GAT, with up to 3.3× less memory; ~3× for
  GATv2.
- **Node classification** (8 datasets, Cora to ogbn-arxiv): **1.6–58× faster
  per epoch**, median 5.3×. PyG's GraphSAGE runs out of memory (16 GB) on
  Coauthor Physics; mlx-graphs needs 4.7 GB.
- **Graph classification** (MUTAG, PROTEINS, NCI1, DD, batching included):
  **2.7–11× faster**.

<img alt="Time of one forward and backward pass of GCN, GraphSAGE, GIN, GAT and GATv2, relative to mlx-graphs, for mlx-graphs, PyG on MPS and PyG on CPU, on random graphs of 100 thousand, 500 thousand and 2 million edges" src="https://raw.githubusercontent.com/mlx-graphs/mlx-graphs/main/benchmarks/pyg_comparison/results/Apple_M5_Pro/0.1.0-c5f1246/figures/layers_speedup_light.png">

<img alt="Time per training epoch of full-batch node classification with GCN, GraphSAGE and GAT on eight datasets, for mlx-graphs, PyG on MPS and PyG on CPU" src="https://raw.githubusercontent.com/mlx-graphs/mlx-graphs/main/benchmarks/pyg_comparison/results/Apple_M5_Pro/0.1.0-c5f1246/figures/node_classification_light.png">

### Same results

Speed only matters if the models learn the same thing. The training loss of
mlx-graphs and PyG follows the same curve at every epoch, and the test
accuracies match within the spread between seeds (mean over node
classification: 79.2% for mlx-graphs, 79.0% / 78.9% for PyG on MPS / CPU).

<img alt="Training loss per epoch of GCN, GraphSAGE and GAT on eight node classification datasets, mlx-graphs, PyG on MPS and PyG on CPU overlaid: the curves coincide" src="https://raw.githubusercontent.com/mlx-graphs/mlx-graphs/main/benchmarks/pyg_comparison/results/Apple_M5_Pro/0.1.0-c5f1246/figures/node_loss_light.png">

<img alt="Test accuracy of GCN, GraphSAGE and GAT on eight node classification datasets, mean and range over 3 seeds, for mlx-graphs, PyG on MPS and PyG on CPU" src="https://raw.githubusercontent.com/mlx-graphs/mlx-graphs/main/benchmarks/pyg_comparison/results/Apple_M5_Pro/0.1.0-c5f1246/figures/node_accuracy_light.png">

The same figures for graph classification, peak memory and absolute
timings are on the [benchmarks page of the docs](https://mlx-graphs.github.io/mlx-graphs/benchmarks.html)
and in [`results.md`](https://github.com/mlx-graphs/mlx-graphs/blob/main/benchmarks/pyg_comparison/results/Apple_M5_Pro/0.1.0-c5f1246/results.md), which also has every number as a
table. How the comparison is made is in
[`benchmarks/pyg_comparison`](https://github.com/mlx-graphs/mlx-graphs/blob/main/benchmarks/pyg_comparison/README.md). To reproduce
on your Mac:

```bash
pip install -e '.[benchmarks]'
python benchmarks/pyg_comparison/run.py && python benchmarks/pyg_comparison/plot.py
```

## Usage

### Tutorial guides

We provide some notebooks to practice `mlx-graphs`.

- Quickstart guide: [notebook](https://mlx-graphs.github.io/mlx-graphs/tutorials/quickstart.html)
- Graph classification guide: [notebook](https://mlx-graphs.github.io/mlx-graphs/tutorials/graph_classification.html)

Runnable scripts are in [`examples/`](https://github.com/mlx-graphs/mlx-graphs/tree/main/examples), e.g.
[`node_classification.py`](https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/node_classification.py) and
[`link_prediction.py`](https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/link_prediction.py).

### Training a model

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_graphs.datasets import PlanetoidDataset
from mlx_graphs.nn import GCN

graph = PlanetoidDataset("cora")[0]
train = mx.array(np.nonzero(np.array(graph.train_mask))[0])
labels = graph.node_labels

model = GCN(graph.num_node_features, 64, num_layers=2, out_features_dim=7, dropout=0.5)
# bias_correction=True: the same update rule as torch.optim.Adam
optimizer = optim.Adam(learning_rate=0.01, bias_correction=True)


def loss_fn(model):
    logits = model(graph.edge_index, graph.node_features)
    return nn.losses.cross_entropy(logits[train], labels[train], reduction="mean")


loss_and_grad = nn.value_and_grad(model, loss_fn)
for epoch in range(200):
    loss, grads = loss_and_grad(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
```

> [!TIP]
> Create MLX's `Adam`/`AdamW` with `bias_correction=True`. MLX's default
> (no bias correction) takes much larger first steps than PyTorch's Adam;
> in our runs it cost GAT about 4 points of accuracy on Cora.

### Writing a layer
Building new GNN layers is straightforward by implementing the `MessagePassing` class. This approach ensures that all operations related to message passing are properly handled and processed efficiently on your Mac's GPU. As a result, you can focus exclusively on the GNN logic, without worrying about the underlying message passing mechanics.

Here is an example of a custom [GraphSAGE](https://proceedings.neurips.cc/paper_files/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf) convolutional layer that considers edge weights:

```python
import mlx.core as mx
from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


@fused_message_passing  # aggregate with one SpMM instead of gather-scatter
class SAGEConv(MessagePassing):
    def __init__(self, node_features_dim: int, out_features_dim: int, bias: bool = True):
        super().__init__(aggr="mean")
        self.neigh_proj = Linear(node_features_dim, out_features_dim, bias=False)
        self.self_proj = Linear(node_features_dim, out_features_dim, bias=bias)

    def __call__(self, edge_index: mx.array, node_features: mx.array, edge_weights: mx.array) -> mx.array:
        # Message passing directly on GPU. The default message is
        # edge_weights * source features, which the fused kernel computes.
        neigh_features = self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_weights": edge_weights},
        )
        return self.self_proj(node_features) + self.neigh_proj(neigh_features)
```

Layers that override `message()` with something other than a weighted sum
of neighbour features (e.g. attention computed per edge) simply drop the
decorator, and `propagate` gathers per-edge messages and scatters them.

## Contributing
### Why contributing?

We are at an early stage of the development of the lib, which means your contributions can have a large impact!
Everyone is welcome to contribute, just open an issue 📝 with your idea 💡 and we'll work together on the implementation ✨.

> [!NOTE]
> Contributions such as the implementation of new layers and datasets would be very valuable for the library.

### Installing test, dev, benchmaks, docs dependencies
Extra dependencies are specified in the `pyproject.toml`.
To install those required for testing, development and building documentation, you can run any of the following
```
pip install -e '.[test]'
pip install -e '.[dev]'
pip install -e '.[benchmarks]'
pip install -e '.[docs]'
```
For dev purposes you may want to install the current version of `mlx` via `pip install git+https://github.com/ml-explore/mlx.git`

### Testing
We encourage to write tests for all components.
Please run `pytest` to ensure breaking changes are not introduced.

> Note: CI is in place to automatically run tests upon opening a PR.

### Pre-commit hooks (optional)
To ensure code quality you can run [pre-commit](https://pre-commit.com) hooks. Simply install them by running
```
pre-commit install
```
and run via `pre-commit run --all-files`.

> Note: CI is in place to verify code quality, so pull requests that don't meet those requirements won't pass CI tests.


## Why running GNNs on my Mac?

Other frameworks like PyG and DGL also benefit from efficient GNN operations parallelized on GPU. However, they are not fully optimized to leverage the Mac's GPU capabilities, often defaulting to CPU execution.

In contrast, `mlx-graphs` is specifically designed to leverage the power of Mac's hardware, delivering optimal performance for Mac users. By taking advantage of Apple Silicon, `mlx-graphs` enables accelerated GPU computation and benefits from unified memory. This approach removes the need for data transfers between devices and allows for the use of the entire memory space available on the Mac's GPU. Consequently, users can manage large graphs directly on the GPU, enhancing performance and efficiency.
