.. _spatial:


Spatial graphs
==============

.. currentmodule:: mlx_graphs.utils

Nearest neighbors and radius search, batch-aware, to build graphs from point
coordinates or node features. Also available from :mod:`mlx_graphs.nn`.

.. autosummary::
	:toctree: _autosummary_spatial

	knn
	knn_graph
	radius
	radius_graph

Sampling and clustering of point clouds and graphs:

.. autosummary::
	:toctree: _autosummary_spatial

	fps
	voxel_grid
	nearest
	graclus
	consecutive_cluster
