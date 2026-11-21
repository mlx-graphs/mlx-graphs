.. _graph_utils:

Graph utilities
===============

.. currentmodule:: mlx_graphs.utils

Subgraphs, negative sampling, dropout, dense conversions and graph statistics.

.. autosummary::
	:toctree: _autosummary_graph

	subgraph
	k_hop_subgraph
	negative_sampling
	structured_negative_sampling
	dropout_edge
	dropout_node
	to_dense_batch
	to_dense_adj
	get_laplacian
	homophily
	add_remaining_self_loops
	segregate_self_loops
	remove_isolated_nodes
	contains_self_loops
	contains_isolated_nodes
	bipartite_subgraph
	batched_negative_sampling
	dropout_path
	shuffle_node
	mask_feature
	add_random_edge
	unbatch
	unbatch_edge_index
	assortativity
	get_ppr
	normalized_cut
	train_test_split_edges
