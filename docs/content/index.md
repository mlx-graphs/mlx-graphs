---
title: Home
description: MLX-graphs is a library for Graph Neural Networks (GNNs) built upon Apple's MLX, with PyTorch Geometric's API.
hide:
  - navigation
  - toc
  - path
---

<div class="mg-hero" markdown>

<div class="mg-hero__text" markdown>

--8<-- "version.md"

<h1 class="mg-hero__title">Graph neural networks,<br><span>fast on Apple silicon</span></h1>

<p class="mg-hero__lead">MLX-graphs is a library for Graph Neural Networks (GNNs) built upon Apple's <a href="https://github.com/ml-explore/mlx">MLX</a>, with the API of PyTorch Geometric.</p>

```bash
pip install mlx-graphs
```

<div class="mg-hero__actions" markdown>
[Quickstart :lucide-arrow-right:](getting-started/quickstart.md){ .md-button .md-button--primary }
[Tutorials](tutorials/index.md){ .md-button }
[API reference](api/index.md){ .md-button }
[:fontawesome-brands-discord: Join the Discord](https://discord.gg/K3mWFCxxM7){ .md-button .mg-discord target="_blank" rel="noopener" }
</div>

</div>

<div class="mg-hero__art" markdown>
![mlx-graphs](assets/images/logo.svg#only-light){ .mg-hero__logo }
![mlx-graphs](assets/images/logo_dark.svg#only-dark){ .mg-hero__logo }
</div>

</div>

<div class="mg-stats" markdown>
<div class="mg-stat"><span class="mg-stat__value">5.6&times;</span><span class="mg-stat__label">faster than PyTorch Geometric on the GPU (MPS)</span></div>
<div class="mg-stat"><span class="mg-stat__value">5.8&times;</span><span class="mg-stat__label">faster than PyTorch Geometric on the CPU</span></div>
<div class="mg-stat"><span class="mg-stat__value">58&times;</span><span class="mg-stat__label">faster at best, over 35 training benchmarks</span></div>
<div class="mg-stat"><span class="mg-stat__value">31&times;</span><span class="mg-stat__label">faster at best for one GNN layer than PyG on the Mac's GPU (M5 Pro)</span></div>
</div>

<section class="mg-section" markdown>

## Faster than PyG, at the same accuracy

**On a Mac, MLX-graphs trains GNNs 5.6× faster than PyTorch Geometric on the GPU (MPS)
and 5.8× faster than PyTorch Geometric on the CPU, and the models reach the same accuracy**
(medians over 35 training benchmarks, up to 58× faster). See the [benchmarks](benchmarks.md).

<figure class="mg-figure" markdown>
![Time of PyG on MPS and on CPU relative to mlx-graphs for one layer, node classification and graph classification, and the test accuracy of each framework](assets/benchmarks/Apple_M5_Pro/0.1.0-c5f1246/headline_light.png#only-light)
![Time of PyG on MPS and on CPU relative to mlx-graphs for one layer, node classification and graph classification, and the test accuracy of each framework](assets/benchmarks/Apple_M5_Pro/0.1.0-c5f1246/headline_dark.png#only-dark)
</figure>

</section>

<section class="mg-section" markdown>

## Features

<div class="grid cards mg-features" markdown>

-   :lucide-zap:{ .mg-icon } **Fast GNN training and inference on Apple Silicon**

    ---

    `MLX-graphs` has been designed to run *fast* on Apple Silicon chips. All GNN operations
    fully leverage the GPU and CPU hardware of Macs thanks to the efficient low-level primitives
    available within the MLX core library, and message passing aggregates neighbours with fused
    sparse kernels. On an M5 Pro, one GNN layer runs up to 31× faster than PyTorch Geometric on
    the Mac's GPU (MPS). See the [benchmarks](benchmarks.md).

-   :lucide-maximize-2:{ .mg-icon } **Scalability to large graphs**

    ---

    With unified memory architecture, objects live in a shared memory accessible by both the CPU and GPU.
    This setup allows Macs to leverage their entire memory capacity for storing graphs.
    Consequently, Macs equipped with substantial memory can efficiently train GNNs on large graphs,
    spanning tens of gigabytes, directly using the Mac's GPU.

-   :lucide-arrow-left-right:{ .mg-icon } **PyTorch Geometric's API, on MLX**

    ---

    The layers, models, samplers and loaders, transforms and datasets of PyG are available under
    the same names: mini-batch training with `NeighborLoader` and `ClusterLoader`,
    heterogeneous models with `to_hetero`, temporal graphs with TGN, and graph transformers
    with `GPSConv`. Each was checked against PyG. See [Coming from PyG](getting-started/from-pyg.md).

-   :lucide-cpu:{ .mg-icon } **Multi-device**

    ---

    Unified memory eliminates the need for time-consuming device-to-device transfers.
    This architecture also enables specific operations to be run explicitly on either the CPU or GPU
    without incurring any overhead, facilitating more efficient computation and resource utilization.

</div>

</section>

<section class="mg-section mg-example" markdown>

<div class="mg-example__text" markdown>

## Example usage

This library has been designed to build GNNs with ease and efficiency. Building new GNN layers is
straightforward by implementing the [`MessagePassing`][mlx_graphs.nn.MessagePassing] class. This approach ensures
that all operations related to message passing are properly handled and processed efficiently on
your Mac's GPU. As a result, you can focus exclusively on the GNN logic, without worrying about the
underlying message passing mechanics.

Here is an example of a custom
[GraphSAGE](https://proceedings.neurips.cc/paper_files/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)
convolutional layer that considers edge weights:

</div>

<div class="mg-example__code" markdown>

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


# 3 nodes, 3 weighted edges, 8 features -> 16
edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
conv = SAGEConv(8, 16)
out = conv(edge_index, mx.random.normal((3, 8)), mx.array([1.0, 0.5, 2.0]))
print(out.shape)  # (3, 16)
```

</div>

<div class="mg-example__text" markdown>

Layers that override `message()` with something other than a weighted sum of
neighbour features (e.g. attention computed per edge) simply drop the
decorator, and `propagate` gathers per-edge messages and scatters them.

[Write your own layers :lucide-arrow-right:](getting-started/writing-layers.md){ .md-button }

</div>

</section>

<section class="mg-section" markdown>

## Tutorials

Notebooks that train complete models, most of them side by side with PyTorch Geometric.
[All tutorials :lucide-arrow-right:](tutorials/index.md)

--8<-- "tutorial_cards_home.md"

</section>
