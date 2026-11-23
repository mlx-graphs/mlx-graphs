.. _feature_and_graph_stores:

Feature and Graph Stores
========================

PyG-style stores holding the features and the edges of a graph outside a
:class:`~mlx_graphs.data.GraphData`. A ``(FeatureStore, GraphStore)`` pair can
be passed to :class:`~mlx_graphs.loaders.NeighborLoader`, which then only
gathers the rows of the sampled nodes. See :doc:`/guide/large_graphs`.

.. currentmodule:: mlx_graphs.data.feature_store

.. autosummary::
	:toctree: _autosummary

	TensorAttr
	FeatureStore
	InMemoryFeatureStore
	MemmapFeatureStore

.. currentmodule:: mlx_graphs.data.graph_store

.. autosummary::
	:toctree: _autosummary

	EdgeAttr
	GraphStore
	InMemoryGraphStore
	MemmapGraphStore
