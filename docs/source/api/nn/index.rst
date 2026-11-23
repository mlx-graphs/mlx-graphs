.. _nn:

Neural Networks
===============

.. currentmodule:: mlx_graphs.nn


.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	Linear
	HeteroDictLinear
	MessagePassing
	GraphNetworkBlock

Convolution layers
------------------

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

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

Point cloud and mesh layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Layers for point clouds and meshes, which take node positions ``pos`` (and
normals or pseudo-coordinates), e.g. from
:class:`~mlx_graphs.datasets.ModelNet` with
:class:`~mlx_graphs.transforms.SamplePoints`. ``PointConv`` is an alias of
:class:`PointNetConv`.

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	PointNetConv
	PointTransformerConv
	PPFConv
	XConv
	SplineConv
	GMMConv
	FeaStConv

Dense convolution layers
------------------------

Layers on batched dense adjacency matrices ``[batch_size, num_nodes,
num_nodes]`` (e.g. from :func:`~mlx_graphs.utils.to_dense_adj` or a
:class:`~mlx_graphs.loaders.DenseDataLoader`), called as ``conv(adj,
node_features, mask)``.

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	DenseGCNConv
	DenseSAGEConv
	DenseGINConv
	DenseGraphConv

Models
------

Ready-to-use multi-layer models, with optional normalization, dropout and
Jumping Knowledge.

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	GCN
	GraphSAGE
	GIN
	GAT
	MLP
	BasicGNN
	PNA

Other models and building blocks:

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	EdgeCNN
	LINKX
	DeepGCNLayer
	MaskLabel
	RECT_L
	DeepGraphInfomax
	LightGCN
	SGFormer
	Polynormer

Graph auto-encoders
-------------------

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	InnerProductDecoder
	GAE
	VGAE
	ARGA
	ARGVA

Label propagation
-----------------

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	LabelPropagation
	CorrectAndSmooth

Knowledge graph embeddings
--------------------------

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	KGEModel
	TransE
	DistMult
	ComplEx
	RotatE
	KGTripletLoader

Normalization
-------------

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

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

Each normalization layer is also available under its PyG name, as an alias
of the same class:

.. list-table::
   :header-rows: 1

   * - PyG name
     - Class
   * - ``BatchNorm``
     - :class:`~mlx_graphs.nn.BatchNormalization`
   * - ``HeteroBatchNorm``
     - :class:`~mlx_graphs.nn.HeteroBatchNormalization`
   * - ``InstanceNorm``
     - :class:`~mlx_graphs.nn.InstanceNormalization`
   * - ``LayerNorm``
     - :class:`~mlx_graphs.nn.LayerNormalization`
   * - ``HeteroLayerNorm``
     - :class:`~mlx_graphs.nn.HeteroLayerNormalization`
   * - ``GraphNorm``
     - :class:`~mlx_graphs.nn.GraphNormalization`
   * - ``PairNorm``
     - :class:`~mlx_graphs.nn.PairNormalization`
   * - ``GraphSizeNorm``
     - :class:`~mlx_graphs.nn.GraphSizeNormalization`
   * - ``MeanSubtractionNorm``
     - :class:`~mlx_graphs.nn.MeanSubtractionNormalization`
   * - ``MessageNorm``
     - :class:`~mlx_graphs.nn.MessageNormalization`
   * - ``DiffGroupNorm``
     - :class:`~mlx_graphs.nn.DiffGroupNormalization`

Pooling and readout
-------------------

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	JumpingKnowledge
	AttentionalAggregation
	Set2Set
	TopKPooling
	SAGPooling

.. autosummary::
	:toctree: _autosummary

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

Point cloud sampling and cluster pooling (``fps``, ``voxel_grid``,
``graclus`` and ``nearest`` are also in :mod:`mlx_graphs.utils`):

.. autosummary::
	:toctree: _autosummary

	fps
	voxel_grid
	graclus
	nearest
	avg_pool
	max_pool
	avg_pool_x
	max_pool_x

Hierarchical and dense pooling
------------------------------

Dense pooling learns soft cluster assignments on dense batched graphs and
returns auxiliary losses; sparse hierarchical pooling coarsens a batch of
graphs given by ``edge_index`` and ``batch_indices``.

.. autosummary::
	:toctree: _autosummary

	dense_diff_pool
	dense_mincut_pool

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	DMoNPooling
	EdgePooling
	ASAPooling
	MemPooling
	PANPooling
	PANConv
	GraphUNet

Aggregations
------------

Aggregation operators of ``mlx_graphs.nn.aggr`` (as ``torch_geometric.nn.aggr``).
They reduce a set of elements given by an ``index`` (``aggr(x, index,
dim_size)``) and can be used as readouts, or passed as ``aggr`` to any
:class:`MessagePassing` layer, alone or as a list (e.g.
``SAGEConv(16, 32, aggr=["mean", "max"])``).

.. currentmodule:: mlx_graphs.nn.aggr

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

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

Set transformer building blocks:

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	MultiheadAttentionBlock
	SetAttentionBlock
	InducedSetAttentionBlock
	PoolingByMultiheadAttention

Dense layout of grouped elements (used by the set and graph transformers):

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	DenseLayout

.. autosummary::
	:toctree: _autosummary

	aggregation_resolver

.. currentmodule:: mlx_graphs.nn

Positional encodings
--------------------

Encodings of positions and times (``torch_geometric.nn.encoding``), and the
projections of the random-walk / Laplacian positional encodings used by graph
transformers such as :class:`GPSConv`.

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	PositionalEncoding
	TemporalEncoding
	RWSEEncoder
	LapPEEncoder

Attention
---------

Global attention mechanisms of ``mlx_graphs.nn.attention`` (as
``torch_geometric.nn.attention``), applied to dense batches of node features
``[batch_size, num_nodes, channels]`` with a mask of the real nodes. They are
also exported by ``mlx_graphs.nn``, as in PyG.

.. currentmodule:: mlx_graphs.nn.attention

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	MultiheadAttention
	PerformerAttention
	SGFormerAttention
	PolynormerAttention

.. currentmodule:: mlx_graphs.nn

Fused message passing
---------------------

Layers decorated with :func:`fused_message_passing` aggregate neighbours
with a single sparse-dense matmul (SpMM) from ``mlx-cluster`` instead of
gather-scatter, so the ``[num_edges, features]`` message array is never
created. ``GCNConv``, ``SAGEConv``, ``GINConv``, ``GATConv``, ``GATv2Conv``,
``GraphConv``, ``SGConv``, ``APPNP``, ``ChebConv``, ``TAGConv`` and
``SimpleConv`` use it out of the box (the attention layers aggregate with one
SpMM per head). Layers whose messages can't be fused (e.g. GINE with edge
features) fall back to gather-scatter automatically.

The default ``format="auto"`` starts with the COO kernel and switches to the
faster, deterministic CSR kernel once converting the graph pays off: right
away for graphs with a million edges or more, and after a few calls on the
same ``edge_index`` otherwise (e.g. full-batch training). Conversions are
cached by ``edge_index`` identity; call :func:`clear_csr_cache` after
modifying an ``edge_index`` in place.

.. code-block:: python

    import mlx_graphs.nn as gnn

    # Turn it off globally, or force a format for every layer
    gnn.set_fused_message_passing(True, format="csr")

    # Or only for a block of code (handy for comparing both paths)
    with gnn.fused_message_passing_mode(False):
        reference = model(edge_index, x)

.. autosummary::
	:toctree: _autosummary

	fused_message_passing
	set_fused_message_passing
	fused_message_passing_mode
	is_fused_message_passing_enabled
	clear_csr_cache
	gcn_norm
	clear_norm_cache

Temporal graph networks
-----------------------

The TGN memory model and its building blocks (``mlx_graphs.nn.models.tgn``, as
``torch_geometric.nn.models.tgn``). See ``examples/tgn_jodie.py``.

.. currentmodule:: mlx_graphs.nn

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	TGNMemory

.. currentmodule:: mlx_graphs.nn.models.tgn

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	IdentityMessage
	LastAggregator
	MeanAggregator
	TimeEncoder
	GRUCell
	LastNeighborLoader

Heterogeneous models
--------------------

:func:`to_hetero` converts a homogeneous model (e.g. a :class:`GraphSAGE`, or
a model written with message passing layers, ``Linear`` layers and
activations) into a heterogeneous one, with one copy of every message passing
layer per edge type and of every other layer per node type, like PyG's
``to_hetero``. The converted model takes dicts keyed by node and edge types.

.. code-block:: python

    from mlx_graphs.nn import GraphSAGE, to_hetero

    model = to_hetero(
        GraphSAGE(64, 64, num_layers=2, out_features_dim=4),
        data.metadata(),
        input_dims=data.num_node_features,  # no lazy sizes in MLX
    )
    out = model(data.edge_index_dict, data.node_features_dict)

.. currentmodule:: mlx_graphs.nn

.. autosummary::
	:toctree: _autosummary

	to_hetero
	to_hetero_with_bases
	ToHeteroModel
