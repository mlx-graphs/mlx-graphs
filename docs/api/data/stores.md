# Feature and graph stores

PyG-style stores holding the features and the edges of a graph outside a
[`GraphData`][mlx_graphs.data.GraphData]. A `(FeatureStore, GraphStore)` pair
can be passed to [`NeighborLoader`][mlx_graphs.loaders.NeighborLoader], which
then only gathers the rows of the sampled nodes. See the
[large graphs guide](../../tutorials/guides/large-graphs.md).

## Feature stores

<!-- api mlx_graphs.data -->
TensorAttr
FeatureStore
InMemoryFeatureStore
MemmapFeatureStore
<!-- /api -->

## Graph stores

<!-- api mlx_graphs.data -->
EdgeAttr
GraphStore
InMemoryGraphStore
MemmapGraphStore
<!-- /api -->
