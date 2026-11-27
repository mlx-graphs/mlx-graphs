---
hide:
  - toc
---

# API reference

Every public class and function of mlx-graphs, generated from the docstrings.
The names follow PyTorch Geometric: `mlx_graphs.nn` is `torch_geometric.nn`,
`mlx_graphs.loaders` is `torch_geometric.loader`, and so on. Use the search
(++slash++) to jump to a name.

<div class="grid cards mg-api-cards" markdown>

-   :lucide-boxes: **[Data](data/index.md)**

    ---

    `GraphData`, `HeteroGraphData`, `TemporalData` and batches of graphs.

    <span class="mg-count"><!-- count data --> objects</span>

-   :lucide-database: **[Datasets](datasets/index.md)**

    ---

    Planetoid, TU, OGB, LRGB, JODIE, knowledge graphs and more, downloaded on first use.

    <span class="mg-count"><!-- count datasets --> objects</span>

-   :lucide-layers: **[Loaders](loaders/index.md)**

    ---

    `Dataloader`, `NeighborLoader`, `LinkNeighborLoader`, ClusterGCN, GraphSAINT and other samplers.

    <span class="mg-count"><!-- count loaders --> objects</span>

-   :lucide-wand-sparkles: **[Transforms](transforms/index.md)**

    ---

    Graph transforms: splits, self-loops, positional encodings, normalization.

    <span class="mg-count"><!-- count transforms --> objects</span>

-   :lucide-brain-circuit: **[Neural networks](nn/index.md)**

    ---

    Message passing, 40+ convolution layers, models, aggregations, pooling, normalization.

    <span class="mg-count"><!-- count nn --> objects</span>

-   :lucide-wrench: **[Utils](utils/index.md)**

    ---

    Edge index and adjacency helpers, scatter, sparse, generators, conversions.

    <span class="mg-count"><!-- count utils --> objects</span>

-   :lucide-route: **[Algorithms](algorithms/index.md)**

    ---

    Node2Vec and MetaPath2Vec embeddings.

    <span class="mg-count"><!-- count algorithms --> objects</span>

-   :lucide-gauge: **[Metrics](metrics/index.md)**

    ---

    Link prediction and ranking metrics, ROC-AUC and average precision.

    <span class="mg-count"><!-- count metrics --> objects</span>

-   :lucide-search: **[Explainability](explain/index.md)**

    ---

    `Explainer`, GNNExplainer, Saliency, Integrated Gradients and fidelity metrics.

    <span class="mg-count"><!-- count explain --> objects</span>

</div>
