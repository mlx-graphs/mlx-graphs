# Feature and graph stores

PyG-style stores holding the features and the edges of a graph outside a
`GraphData`. A `(FeatureStore, GraphStore)` pair can be passed to
[`NeighborLoader`][mlx_graphs.loaders.NeighborLoader], which then only gathers
the rows of the sampled nodes. See the
[large graphs guide](../../tutorials/guides/large-graphs.md).

<!-- api mlx_graphs.data.feature_store -->
TensorAttr
FeatureStore
InMemoryFeatureStore
MemmapFeatureStore
<!-- /api -->

<!-- api mlx_graphs.data.graph_store -->
EdgeAttr
GraphStore
InMemoryGraphStore
MemmapGraphStore
<!-- /api -->
