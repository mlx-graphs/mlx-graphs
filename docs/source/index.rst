MLX-graphs
==========

MLX-graphs is a library for Graph Neural Networks (GNNs) built upon Apple's `MLX <https://github.com/ml-explore/mlx>`_.


.. caution::
   We are very early in the development of this library and there may be breaking changes in upcoming versions.


Usage
-----


Examples showing the functionalities of this library are available `here <https://github.com/TristanBilot/mlx-graphs/tree/main/examples>`_.


Graph data model
^^^^^^^^^^^^^^^^

A graph is defined by a set of (optional) attributes

#. `edge_index`: an array of size `[2, num_edges]` which specifies the topology of the graph. The i-th column in `edge_index` defines the source and destination nodes of the i-th edge
#. `node_features`: an array of size `[num_nodes, num_node_features]` defining the features associated to each node (if any). The i-th row contains the features of the i-th node
#. `edge_features`:  an array of size `[num_edges, num_edge_features]` defining the features associated to each edge (if any). The i-th row contains the features of the i-th edge
#. `graph_features`: an array of size `[num_graph_features]` defining the features associated to the graph itself

We adopt the above convention across the entire library both in terms of shapes of the attributes and the order in which they're provided to functions.

.. toctree::
   :caption: Install
   :maxdepth: 1

   install


.. toctree::
   :caption: API
   :maxdepth: 1

   api/data/index.rst
   api/datasets/index.rst
   api/nn/index.rst
   api/utils/index.rst






.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
