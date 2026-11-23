---
hide:
  - toc
---

# Tutorials

Guides to the parts of mlx-graphs you build on, and Jupyter notebooks that train complete models, most of them side by side with PyTorch Geometric on accuracy and runtime. New to mlx-graphs? Start with the [Quickstart](../getting-started/quickstart.md).

## Guides

<div class="grid cards mg-guides" markdown>

-   :lucide-layers:{ .mg-icon } **[Writing your own layers](guides/writing-layers.md)**

    ---

    The `MessagePassing` contract, aggregations, attention layers, fused message passing, and how to check a layer against PyG.

-   :lucide-cpu:{ .mg-icon } **[Custom GPU kernels](guides/custom-kernels.md)**

    ---

    Fuse an operation of your own in Metal, with gradients, a fallback, tests and a benchmark: a per-edge score whose forward pass runs about 5.5x faster than with MLX operations on 2 million edges (M5 Pro).

-   :lucide-database:{ .mg-icon } **[Datasets](guides/datasets.md)**

    ---

    The built-in datasets, their sizes and how to load them, and how to add your own.

</div>

## Notebooks

Each page shows the notebook with the outputs of its last run; download it to run it yourself.

--8<-- "tutorial_cards.md"

## Example scripts

The [`examples/`](https://github.com/mlx-graphs/mlx-graphs/tree/main/examples)
folder of the repository has runnable scripts that train models end to end
from the command line: node classification, GCN and GAT on Cora, link
prediction, heterogeneous GraphSAGE, MetaPath2Vec, TGN on JODIE and TGB, GPS and more.
[See all the example scripts](examples.md).
