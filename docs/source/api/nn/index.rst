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
	SimpleConv
	HeteroConv
	GeneralizedRelationalConv

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

Pooling and readout
-------------------

.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

	JumpingKnowledge
	AttentionalAggregation
	Set2Set

.. autosummary::
	:toctree: _autosummary

	global_add_pool
	global_max_pool
	global_mean_pool

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
	clear_norm_cache
