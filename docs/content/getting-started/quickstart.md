# Quickstart

Train a graph neural network on your Mac's GPU in a few lines, then see how graphs, mini-batches and layers work in mlx-graphs. Install it first: `pip install mlx-graphs`.

## Train a GCN on Cora

Node classification on the Cora citation graph (2,708 papers, 7 topics) with a 2-layer GCN, full batch:

```python
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_graphs.datasets import PlanetoidDataset
from mlx_graphs.nn import GCN
from mlx_graphs.transforms import NormalizeFeatures
from mlx_graphs.utils import mask_to_index

mx.random.seed(0)
graph = NormalizeFeatures()(PlanetoidDataset("cora")[0])  # downloaded on first use
train, test = mask_to_index(graph.train_mask), mask_to_index(graph.test_mask)

model = GCN(graph.num_node_features, 64, num_layers=2, out_features_dim=7, dropout=0.5)
# bias_correction=True: the same update rule as torch.optim.Adam
optimizer = optim.Adam(learning_rate=0.01, bias_correction=True)


def loss_fn(model):
    logits = model(graph.edge_index, graph.node_features)
    return nn.losses.cross_entropy(logits[train], graph.node_labels[train], reduction="mean")


loss_and_grad = nn.value_and_grad(model, loss_fn)
start = time.perf_counter()
for epoch in range(200):
    loss, grads = loss_and_grad(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)  # MLX is lazy: run the step
ms_per_epoch = (time.perf_counter() - start) / 200 * 1e3

model.eval()
predictions = model(graph.edge_index, graph.node_features).argmax(axis=1)
accuracy = (predictions[test] == graph.node_labels[test]).mean().item()
print(f"test accuracy: {accuracy:.3f}, {ms_per_epoch:.2f} ms per epoch")
```

On an Apple M5 Pro (the time per epoch varied from 0.9 to 1.4 ms between our runs):

```text
test accuracy: 0.786, 1.06 ms per epoch
```

The whole graph lives in unified memory, and the message passing of the built-in layers runs as fused sparse kernels on the GPU. The [Benchmarks](../benchmarks.md) compare training times and accuracy with PyTorch Geometric on the same Macs.

!!! tip

    Create MLX's `Adam` / `AdamW` with `bias_correction=True`. MLX's default (no bias correction) takes much larger first steps than PyTorch's Adam, which can cost accuracy (about 4 points for GAT on Cora in our runs).

## Graphs in mlx-graphs

A graph is a [`GraphData`][mlx_graphs.data.data.GraphData]: an `edge_index` of shape `[2, num_edges]` (sources, then targets) and optional node, edge and graph features and labels. Any other keyword becomes an attribute. Several graphs [`batch`][mlx_graphs.data.batch.batch] into one [`GraphDataBatch`][mlx_graphs.data.batch.GraphDataBatch], a single disconnected graph whose `batch_indices` give each node's graph:

```python
import mlx.core as mx

from mlx_graphs.data import GraphData, batch

graph = GraphData(
    edge_index=mx.array([[0, 1, 2, 3], [1, 2, 3, 0]]),  # [2, num_edges]: sources, targets
    node_features=mx.random.normal((4, 8)),  # [num_nodes, num_features]
    node_labels=mx.array([0, 1, 1, 0]),
    timestamps=mx.arange(4),  # any other attribute is kept as is
)
print(graph)

# several graphs as one disconnected graph, for mini-batches of graphs
graphs = batch([graph, graph, graph])
print(graphs.num_graphs, graphs.num_nodes, graphs.batch_indices)
print(graphs[1])
```

```text
GraphData(
	edge_index(shape=(2, 4), int32)
	node_features(shape=(4, 8), float32)
	node_labels(shape=(4,), int32)
	timestamps(shape=(4,), int32))
3 12 array([0, 0, 0, ..., 2, 2, 2], dtype=int32)
GraphData(
	edge_index(shape=(2, 4), int32)
	node_features(shape=(4, 8), float32)
	node_labels(shape=(4,), int32)
	timestamps(shape=(4,), int32))
```

The [`GraphData` reference][mlx_graphs.data.data.GraphData] lists all its arguments and methods. Graph utilities (self-loops, subgraphs, conversions to and from PyG, NetworkX and SciPy...) are in [`mlx_graphs.utils`][mlx_graphs.utils].

## Classify molecules: MUTAG

Graph classification on MUTAG (188 molecules, mutagenic or not): a [`Dataloader`][mlx_graphs.loaders.Dataloader] batches the graphs, a [`GIN`][mlx_graphs.nn.GIN] embeds the atoms, [`global_mean_pool`][mlx_graphs.nn.global_mean_pool] averages them per molecule and a linear layer classifies:

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_graphs.datasets import TUDataset
from mlx_graphs.loaders import Dataloader
from mlx_graphs.nn import GIN, Linear, global_mean_pool

mx.random.seed(0)
dataset = TUDataset("MUTAG").shuffle()  # 188 molecules, 2 classes
train_loader = Dataloader(dataset[:150], batch_size=32, shuffle=True)
test_loader = Dataloader(dataset[150:], batch_size=64)


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()
        self.gin = GIN(in_dim, hidden, num_layers=3)
        self.head = Linear(hidden, num_classes)

    def __call__(self, batch):
        h = self.gin(batch.edge_index, batch.node_features)
        return self.head(global_mean_pool(h, batch.batch_indices))  # one row per graph


model = Classifier(dataset.num_node_features, 64, dataset.num_graph_classes)
optimizer = optim.Adam(learning_rate=0.01, bias_correction=True)
loss_and_grad = nn.value_and_grad(
    model, lambda m, b: nn.losses.cross_entropy(m(b), b.graph_labels, reduction="mean")
)

for epoch in range(50):
    for batch in train_loader:
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

correct = sum((model(b).argmax(axis=1) == b.graph_labels).sum().item() for b in test_loader)
print(f"test accuracy: {correct / len(dataset[150:]):.3f}")
```

```text
test accuracy: 0.816
```

The 38 test molecules are few, and GPU reductions are not bitwise deterministic, so the accuracy moves a little between runs (0.816 to 0.868 in ours). The [graph classification tutorial](../tutorials/notebooks/graph_classification.md) goes through the same task step by step.

## A layer of your own

A layer is a [`MessagePassing`][mlx_graphs.nn.MessagePassing] subclass that calls `propagate`. This one adds the mean of each node's neighbors to the node itself; `@fused_message_passing` aggregates with one sparse-dense matrix product on the GPU instead of building a message per edge:

```python
import mlx.core as mx

from mlx_graphs.nn import Linear, MessagePassing, fused_message_passing


@fused_message_passing  # aggregate with one sparse-dense matmul on the GPU
class MeanConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="mean")
        self.root = Linear(in_dim, out_dim)
        self.neighbors = Linear(in_dim, out_dim)

    def __call__(self, edge_index: mx.array, node_features: mx.array) -> mx.array:
        aggregated = self.propagate(edge_index, node_features)  # mean of the neighbors
        return self.root(node_features) + self.neighbors(aggregated)


conv = MeanConv(8, 16)
edge_index = mx.array([[0, 1, 2, 3], [1, 2, 3, 0]])
print(conv(edge_index, mx.random.normal((4, 8))).shape)
```

```text
(4, 16)
```

[Writing your own layers](../tutorials/guides/writing-layers.md) covers messages, aggregations, attention and testing a layer against PyG; [Custom GPU kernels](../tutorials/guides/custom-kernels.md) shows how to fuse an operation of your own in Metal.

## Next steps

- **Large graphs**: train on subgraphs with [`NeighborLoader`][mlx_graphs.loaders.NeighborLoader] and the other samplers, in the [mini-batch training tutorial](../tutorials/notebooks/minibatch_training.md), and keep the features on disk with the [large graphs guide](../tutorials/guides/large-graphs.md).
- **Coming from PyTorch Geometric**: the same names and how a PyG model ports, in [Coming from PyG](from-pyg.md).
- **More examples**: the [tutorials](../tutorials/index.md) (link prediction, heterogeneous and temporal graphs, graph transformers...) and the built-in [datasets](../tutorials/guides/datasets.md), including how to add your own.
- **Everything else**: the [API reference](../api/index.md).
