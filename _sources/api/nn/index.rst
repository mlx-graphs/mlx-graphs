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
	SimpleConv
	GATConv
	GATv2Conv
	GCNConv
	GINConv
	SAGEConv
	HeteroConv
	GeneralizedRelationalConv
	BatchNormalization
	HeteroBatchNormalization
	InstanceNormalization
	LayerNormalization
	HeteroLayerNormalization
	global_add_pool
	global_max_pool
	global_mean_pool

Fused message passing
---------------------

Layers decorated with :func:`fused_message_passing` aggregate neighbours
with a single sparse-dense matmul (SpMM) from ``mlx-cluster`` instead of
gather-scatter, so the ``[num_edges, features]`` message array is never
created. ``GCNConv``, ``SAGEConv``, ``GINConv`` and ``SimpleConv`` use it
out of the box; layers whose messages can't be fused (e.g. ``GATConv``)
fall back to gather-scatter automatically.

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
