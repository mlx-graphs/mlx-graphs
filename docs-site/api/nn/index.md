# Neural networks { #mlx_graphs.nn }

Layers, models, aggregations, normalization and pooling of `mlx_graphs.nn`, under the names of `torch_geometric.nn`.

## Building blocks

<!-- api mlx_graphs.nn -->
Linear
HeteroDictLinear
MessagePassing
GraphNetworkBlock
<!-- /api -->

## Convolution layers

<!-- api mlx_graphs.nn -->
GCNConv
SAGEConv
GINConv
GATConv
GATv2Conv
TransformerConv
GraphConv
SGConv
APPNP
ChebConv
TAGConv
EdgeConv
RGCNConv
FastRGCNConv
RGATConv
GENConv
MixHopConv
FAConv
GeneralConv
DirGNNConv
HANConv
HGTConv
SimpleConv
HeteroConv
GeneralizedRelationalConv
GINEConv
GCN2Conv
SSGConv
LGConv
ARMAConv
GatedGraphConv
ResGatedGraphConv
NNConv
CGConv
MFConv
LEConv
FiLMConv
AGNNConv
ClusterGCNConv
DynamicEdgeConv
PNAConv
GPSConv
<!-- /api -->

## Models

Ready-to-use multi-layer models, with optional normalization, dropout and
Jumping Knowledge.

<!-- api mlx_graphs.nn -->
GCN
GraphSAGE
GIN
GAT
MLP
BasicGNN
PNA
<!-- /api -->

Other models and building blocks:

<!-- api mlx_graphs.nn -->
EdgeCNN
LINKX
DeepGCNLayer
MaskLabel
RECT_L
DeepGraphInfomax
LightGCN
SGFormer
Polynormer
<!-- /api -->

## Graph auto-encoders

<!-- api mlx_graphs.nn -->
InnerProductDecoder
GAE
VGAE
ARGA
ARGVA
<!-- /api -->

## Label propagation

<!-- api mlx_graphs.nn -->
LabelPropagation
CorrectAndSmooth
<!-- /api -->

## Knowledge graph embeddings

<!-- api mlx_graphs.nn -->
KGEModel
TransE
DistMult
ComplEx
RotatE
KGTripletLoader
<!-- /api -->

## Normalization

<!-- api mlx_graphs.nn -->
BatchNormalization
HeteroBatchNormalization
InstanceNormalization
LayerNormalization
HeteroLayerNormalization
GraphNormalization
PairNormalization
GraphSizeNormalization
MeanSubtractionNormalization
MessageNormalization
DiffGroupNormalization
<!-- /api -->

Each normalization layer is also available under its PyG name, as an alias
of the same class:

| PyG name | Class |
|---|---|
| `BatchNorm` | [`BatchNormalization`][mlx_graphs.nn.BatchNormalization] |
| `HeteroBatchNorm` | [`HeteroBatchNormalization`][mlx_graphs.nn.HeteroBatchNormalization] |
| `InstanceNorm` | [`InstanceNormalization`][mlx_graphs.nn.InstanceNormalization] |
| `LayerNorm` | [`LayerNormalization`][mlx_graphs.nn.LayerNormalization] |
| `HeteroLayerNorm` | [`HeteroLayerNormalization`][mlx_graphs.nn.HeteroLayerNormalization] |
| `GraphNorm` | [`GraphNormalization`][mlx_graphs.nn.GraphNormalization] |
| `PairNorm` | [`PairNormalization`][mlx_graphs.nn.PairNormalization] |
| `GraphSizeNorm` | [`GraphSizeNormalization`][mlx_graphs.nn.GraphSizeNormalization] |
| `MeanSubtractionNorm` | [`MeanSubtractionNormalization`][mlx_graphs.nn.MeanSubtractionNormalization] |
| `MessageNorm` | [`MessageNormalization`][mlx_graphs.nn.MessageNormalization] |
| `DiffGroupNorm` | [`DiffGroupNormalization`][mlx_graphs.nn.DiffGroupNormalization] |

## Pooling and readout

<!-- api mlx_graphs.nn -->
JumpingKnowledge
AttentionalAggregation
Set2Set
TopKPooling
SAGPooling
<!-- /api -->

<!-- api mlx_graphs.nn -->
global_add_pool
global_mean_pool
global_max_pool
global_sort_pool
topk
filter_adj
knn
knn_graph
radius
radius_graph
<!-- /api -->

## Aggregations { #mlx_graphs.nn.aggr }

Aggregation operators of `mlx_graphs.nn.aggr` (as `torch_geometric.nn.aggr`).
They reduce a set of elements given by an `index` (`aggr(x, index,
dim_size)`) and can be used as readouts, or passed as `aggr` to any
[`MessagePassing`][mlx_graphs.nn.MessagePassing] layer, alone or as a list (e.g.
`SAGEConv(16, 32, aggr=["mean", "max"])`).

<!-- api mlx_graphs.nn.aggr -->
Aggregation
SumAggregation
MeanAggregation
MaxAggregation
MinAggregation
MulAggregation
VarAggregation
StdAggregation
SoftmaxAggregation
PowerMeanAggregation
MedianAggregation
QuantileAggregation
MultiAggregation
DegreeScalerAggregation
SortAggregation
DeepSetsAggregation
LSTMAggregation
GRUAggregation
SetTransformerAggregation
GraphMultisetTransformer
MLPAggregation
LCMAggregation
EquilibriumAggregation
VariancePreservingAggregation
<!-- /api -->

Set transformer building blocks:

<!-- api mlx_graphs.nn.aggr -->
MultiheadAttentionBlock
SetAttentionBlock
InducedSetAttentionBlock
PoolingByMultiheadAttention
<!-- /api -->

Dense layout of grouped elements (used by the set and graph transformers):

<!-- api mlx_graphs.nn.aggr -->
DenseLayout
<!-- /api -->

<!-- api mlx_graphs.nn.aggr -->
aggregation_resolver
<!-- /api -->

## Positional encodings

Encodings of positions and times (`torch_geometric.nn.encoding`), and the
projections of the random-walk / Laplacian positional encodings used by graph
transformers such as [`GPSConv`][mlx_graphs.nn.GPSConv].

<!-- api mlx_graphs.nn -->
PositionalEncoding
TemporalEncoding
RWSEEncoder
LapPEEncoder
<!-- /api -->

## Attention { #mlx_graphs.nn.attention }

Global attention mechanisms of `mlx_graphs.nn.attention` (as
`torch_geometric.nn.attention`), applied to dense batches of node features
`[batch_size, num_nodes, channels]` with a mask of the real nodes. They are
also exported by `mlx_graphs.nn`, as in PyG.

<!-- api mlx_graphs.nn.attention -->
MultiheadAttention
PerformerAttention
SGFormerAttention
PolynormerAttention
<!-- /api -->

## Fused message passing

Layers decorated with [`fused_message_passing()`][mlx_graphs.nn.fused_message_passing] aggregate neighbours
with a single sparse-dense matmul (SpMM) from `mlx-cluster` instead of
gather-scatter, so the `[num_edges, features]` message array is never
created. `GCNConv`, `SAGEConv`, `GINConv`, `GATConv`, `GATv2Conv`,
`GraphConv`, `SGConv`, `APPNP`, `ChebConv`, `TAGConv` and
`SimpleConv` use it out of the box (the attention layers aggregate with one
SpMM per head). Layers whose messages can't be fused (e.g. GINE with edge
features) fall back to gather-scatter automatically.

The default `format="auto"` starts with the COO kernel and switches to the
faster, deterministic CSR kernel once converting the graph pays off: right
away for graphs with a million edges or more, and after a few calls on the
same `edge_index` otherwise (e.g. full-batch training). Conversions are
cached by `edge_index` identity; call [`clear_csr_cache()`][mlx_graphs.nn.clear_csr_cache] after
modifying an `edge_index` in place.

```python
import mlx_graphs.nn as gnn

# Turn it off globally, or force a format for every layer
gnn.set_fused_message_passing(True, format="csr")

# Or only for a block of code (handy for comparing both paths)
with gnn.fused_message_passing_mode(False):
    reference = model(edge_index, x)
```

<!-- api mlx_graphs.nn -->
fused_message_passing
set_fused_message_passing
fused_message_passing_mode
is_fused_message_passing_enabled
clear_csr_cache
gcn_norm
clear_norm_cache
<!-- /api -->

## Temporal graph networks

The TGN memory model and its building blocks (`mlx_graphs.nn.models.tgn`, as
`torch_geometric.nn.models.tgn`). See `examples/tgn_jodie.py`.

<!-- api mlx_graphs.nn -->
TGNMemory
<!-- /api -->

<!-- api mlx_graphs.nn.models.tgn -->
IdentityMessage
LastAggregator
MeanAggregator
TimeEncoder
GRUCell
LastNeighborLoader
<!-- /api -->

## Heterogeneous models

[`to_hetero()`][mlx_graphs.nn.to_hetero] converts a homogeneous model (e.g. a [`GraphSAGE`][mlx_graphs.nn.GraphSAGE], or
a model written with message passing layers, `Linear` layers and
activations) into a heterogeneous one, with one copy of every message passing
layer per edge type and of every other layer per node type, like PyG's
`to_hetero`. The converted model takes dicts keyed by node and edge types.

```python
from mlx_graphs.nn import GraphSAGE, to_hetero

model = to_hetero(
    GraphSAGE(64, 64, num_layers=2, out_features_dim=4),
    data.metadata(),
    input_dims=data.num_node_features,  # no lazy sizes in MLX
)
out = model(data.edge_index_dict, data.node_features_dict)
```

<!-- api mlx_graphs.nn -->
to_hetero
to_hetero_with_bases
ToHeteroModel
<!-- /api -->
