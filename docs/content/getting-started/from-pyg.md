# Coming from PyTorch Geometric

mlx-graphs follows the design of [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) (PyG): graphs in COO format, a `MessagePassing` base class, batching by concatenating graphs, and layers with the same names. Porting a PyG model is mostly renaming. This page lists the differences and walks through porting a model and its training loop.

## Names at a glance

| PyG                                                                | mlx-graphs                                                                                                                           |
|--------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `Data(x, edge_index, edge_attr, y)`                                | `GraphData(edge_index, node_features, edge_features, node_labels / graph_labels)`; `data.x`, `data.edge_attr` and `data.y` also work |
| `DataLoader(dataset, batch_size)`                                  | [`Dataloader`][mlx_graphs.loaders.Dataloader] `(dataset, batch_size)`, with `shuffle`, `drop_last`, `follow_batch`, `sampler` and `batch_sampler`                     |
| `batch.batch`, `batch.ptr`                                         | `batch.batch_indices` (`batch.batch` and `batch.ptr` also work)                                                                      |
| `GCNConv(in_channels, out_channels)`                               | `GCNConv(node_features_dim, out_features_dim)`                                                                                       |
| `conv(x, edge_index)`                                              | `conv(edge_index, node_features)`: **edges first**                                                                                   |
| `global_mean_pool(x, batch)`                                       | `global_mean_pool(node_features, batch_indices)`                                                                                     |
| `torch_geometric.nn.models.GCN`                                    | [`GCN`][mlx_graphs.nn.GCN] (also `GraphSAGE`, `GIN`, `GAT`, `MLP`)                                                                                 |
| `Planetoid(root, "Cora")`                                          | `PlanetoidDataset("cora")`                                                                                                           |
| `SAGEConv(..., aggr=["mean", "max"])`, `aggr=SoftmaxAggregation()` | the same: aggregation names, lists and modules ([`mlx_graphs.nn.aggr`][mlx_graphs.nn.aggr])                                                                        |
| `BatchNorm`, `LayerNorm`, `GraphNorm`, `PairNorm`...               | the same names (aliases of `BatchNormalization`, ...)                                                                                |
| `message(self, x_j, x_i)`                                          | `message(self, src_features, dst_features, **kwargs)`                                                                                |
| `.to(device)`                                                      | nothing: memory is shared by the CPU and the GPU                                                                                     |

The tensors are `mx.array` instead of `torch.Tensor`.

## Where PyG features live

Beyond layers and datasets, most of PyG's toolbox has a counterpart with the same name:

| PyG                                                                                                                                                                            | mlx-graphs                                                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `torch_geometric.loader`: `NeighborLoader`, `LinkNeighborLoader`, `HGTLoader`, `ClusterLoader`, GraphSAINT, `ShaDowKHopSampler`, `ImbalancedSampler`, `DynamicBatchSampler`... | [`mlx_graphs.loaders`][mlx_graphs.loaders], same names. The samplers are built in (no `pyg-lib` / `torch-sparse` needed). |
| `torch_geometric.nn.to_hetero`, `to_hetero_with_bases`, `HeteroConv`, `HANConv`, `HGTConv`                                                                                     | [`to_hetero()`][mlx_graphs.nn.to_hetero], [`to_hetero_with_bases()`][mlx_graphs.nn.to_hetero_with_bases] and the same layers, on [`HeteroGraphData`][mlx_graphs.data.data.HeteroGraphData]                             |
| `TemporalData`, `TemporalDataLoader`, `TGNMemory`                                                                                                                              | [`TemporalData`][mlx_graphs.data.temporal.TemporalData], [`TemporalDataLoader`][mlx_graphs.loaders.TemporalDataLoader], [`TGNMemory`][mlx_graphs.nn.TGNMemory] (see the TGN tutorial)                             |
| `GPSConv`, `SGFormer`, `Polynormer`, `AddLaplacianEigenvectorPE`, `AddRandomWalkPE`                                                                                            | the same names in [`mlx_graphs.nn`][mlx_graphs.nn] and [`mlx_graphs.transforms`][mlx_graphs.transforms]                                             |
| `torch_geometric.transforms`: `NormalizeFeatures`, `ToUndirected`, `RandomLinkSplit`...                                                                                        | [`mlx_graphs.transforms`][mlx_graphs.transforms], same names                                                                    |
| `torch_geometric.metrics` (link prediction), AUC / AP                                                                                                                          | [`mlx_graphs.metrics`][mlx_graphs.metrics]                                                                                |
| `torch_geometric.utils`                                                                                                                                                        | [`mlx_graphs.utils`][mlx_graphs.utils], mostly the same names                                                         |

## Converting data

[`from_pyg()`][mlx_graphs.utils.from_pyg] converts a PyG `Data` object to a [`GraphData`][mlx_graphs.data.data.GraphData], and [`to_pyg()`][mlx_graphs.utils.to_pyg] converts back. `x`, `edge_attr` and `y` are renamed, and every other attribute (masks, positions...) keeps its name:

```python
from torch_geometric.datasets import Planetoid

from mlx_graphs.utils import from_pyg

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]
graph = from_pyg(data)
>>> GraphData(
        edge_index(shape=(2, 10556), int64)
        node_features(shape=(2708, 1433), float32)
        node_labels(shape=(2708,), int64)
        train_mask(shape=(2708,), bool)
        val_mask(shape=(2708,), bool)
        test_mask(shape=(2708,), bool))
```

A whole PyG dataset converts graph by graph, and the list goes straight into a [`Dataloader`][mlx_graphs.loaders.Dataloader]:

```python
from torch_geometric.datasets import TUDataset

from mlx_graphs.loaders import Dataloader

graphs = [from_pyg(data) for data in TUDataset(root="data/TU", name="MUTAG")]
loader = Dataloader(graphs, batch_size=64, shuffle=True)
```

Many datasets are also built in ([Datasets](../api/datasets/index.md)), with no PyG needed.

## Porting a model

A 2-layer GCN for node classification, in PyG:

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)
```

And in mlx-graphs:

```python
import mlx.core as mx
import mlx.nn as nn
from mlx_graphs.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=0.5)

    def __call__(self, edge_index, node_features):
        h = self.dropout(node_features)
        h = nn.relu(self.conv1(edge_index, h))
        h = self.dropout(h)
        return self.conv2(edge_index, h)
```

The differences: `mlx.nn.Module` and `__call__` instead of `forward`, `edge_index` first in the calls, and dropout as a module (it follows `model.train()` / `model.eval()` as in PyTorch).

The ready-made [`GCN`][mlx_graphs.nn.GCN] model does the same in one line:

```python
from mlx_graphs.nn import GCN

model = GCN(1433, 64, num_layers=2, out_features_dim=7, dropout=0.5)
```

## Porting the training loop

PyG trains with autograd and an optimizer that holds the parameters:

```python
model = GCN(dataset.num_features, 64, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

MLX computes gradients with a function transformation, `nn.value_and_grad`, and evaluates lazily: `mx.eval` runs the step. Here with the ready-made model on the Cora graph converted above:

```python
import mlx.optimizers as optim
import numpy as np

from mlx_graphs.nn import GCN

model = GCN(graph.num_node_features, 64, num_layers=2, out_features_dim=7, dropout=0.5)
# bias_correction=True: the update rule of torch.optim.Adam
optimizer = optim.Adam(learning_rate=0.01, bias_correction=True)
train = mx.array(np.nonzero(np.array(graph.train_mask))[0])

def loss_fn(model):
    out = model(graph.edge_index, graph.node_features)
    return nn.losses.cross_entropy(
        out[train], graph.node_labels[train], reduction="mean"
    )

loss_and_grad = nn.value_and_grad(model, loss_fn)
model.train()
for epoch in range(200):
    loss, grads = loss_and_grad(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

model.eval()
pred = model(graph.edge_index, graph.node_features).argmax(axis=1)
```

Boolean masks are turned into index arrays once (`train`), since MLX indexes with integer arrays.

!!! warning

    Create MLX's `Adam` / `AdamW` with `bias_correction=True` when porting from PyTorch. MLX's default has no bias correction and takes much larger first steps, which changes how training goes (about 4 points of accuracy for GAT on Cora in our runs).

## Porting a custom layer

A PyG layer that weights neighbour features by an edge weight:

```python
from torch_geometric.nn import MessagePassing

class WeightedConv(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="add")
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=self.lin(x), edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j
```

In mlx-graphs, `propagate` takes the node features and passes extra arguments to `message` through `message_kwargs`. `x_j` and `x_i` become `src_features` and `dst_features`:

```python
from mlx_graphs.nn import Linear, MessagePassing

class WeightedConv(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="add")
        self.lin = Linear(in_dim, out_dim)

    def __call__(self, edge_index, node_features, edge_weights):
        return self.propagate(
            edge_index,
            self.lin(node_features),
            message_kwargs={"edge_weights": edge_weights},
        )

    def message(self, src_features, dst_features, edge_weights=None, **kwargs):
        return edge_weights.reshape(-1, 1) * src_features
```

This message is a weighted sum of neighbour features, which is what the default `message` already computes. Dropping the `message` override and decorating the class with [`fused_message_passing()`][mlx_graphs.nn.fused_message_passing] aggregates with one sparse x dense matmul on the GPU instead of building the `[num_edges, features]` array of messages, which is how the built-in layers get their speed (see the [Benchmarks](../benchmarks.md)):

```python
from mlx_graphs.nn import fused_message_passing

@fused_message_passing
class WeightedConv(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="add")
        self.lin = Linear(in_dim, out_dim)

    def __call__(self, edge_index, node_features, edge_weights):
        return self.propagate(
            edge_index,
            self.lin(node_features),
            message_kwargs={"edge_weights": edge_weights},
        )
```

## Things to watch for

- **Self-loops.** `GCNConv`, `GATConv` and `GATv2Conv` add self-loops by default, as in PyG. Before this release they didn't: pass `add_self_loops=False` to reproduce older mlx-graphs results.
- **No lazy sizes.** MLX layers need their input sizes when they are built: there is no `in_channels=-1`. [`to_hetero()`][mlx_graphs.nn.to_hetero] takes the input size of each node type with `input_dims` (e.g. `data.num_node_features`).
- **Lazy evaluation.** MLX builds a graph of operations and runs it when a value is needed. Call `mx.eval` on the parameters and optimizer state at the end of each training step, and `.item()` to read a loss.
- **Optimizer.** `bias_correction=True` for `Adam` / `AdamW`, as above.
- **Unused parameters.** MLX returns a zero gradient for a parameter that doesn't contribute to the loss, where PyTorch leaves its `.grad` at `None` and the optimizer skips it. With weight decay or an L2 term, such parameters then shrink at every step. To keep part of a model unused for a while (e.g. a warm-up), `freeze()` it and call `optimizer.init(...)` after unfreezing.
- **Index dtypes.** The built-in datasets and loaders use int32 indices (`edge_index`, `batch`, the `n_id` of sampled subgraphs...), where PyG uses int64. [`from_pyg()`][mlx_graphs.utils.from_pyg] keeps PyG's dtypes; both work.
- **Memory with varying shapes.** MLX caches buffers by size. With batches of graphs of different sizes, cap the cache with `mx.set_cache_limit(...)` to keep memory bounded.
- **Compilation.** Most layers work under `mx.compile`. Layers that need the number of nodes per graph on the host (the LSTM, sort and set transformer aggregations, `TopKPooling` / `SAGPooling`, and `GPSConv` / `SGFormer` / `Polynormer` unless given a precomputed `layout`) work with `nn.value_and_grad` but not inside `mx.compile`.
- **Initialization.** Layers are initialized as in PyG (Glorot for GCN and GAT, PyTorch's default `Linear` initialization for GraphSAGE and GraphConv), so a ported model starts from the same distribution.
