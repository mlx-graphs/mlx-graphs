.. _api:

API reference
=============

The public API of mlx-graphs, by subpackage. Most names follow
`PyTorch Geometric <https://pyg.org>`_, so ``torch_geometric.nn.GCNConv``
is :class:`mlx_graphs.nn.GCNConv`. Use the search (:kbd:`Ctrl` + :kbd:`K`)
to jump to any class or function.

.. grid:: 1 2 2 3
   :gutter: 3
   :class-container: api-grid

   .. grid-item-card:: :octicon:`stack;1.2em;feature-icon` Data
      :link: data/index
      :link-type: doc

      Graph objects (``GraphData``, ``HeteroGraphData``,
      ``TemporalData``), batching and data utilities.

      .. card-footer::

         ``mlx_graphs.data``

   .. grid-item-card:: :octicon:`database;1.2em;feature-icon` Datasets
      :link: datasets/index
      :link-type: doc

      The ``Dataset`` base classes and the built-in
      datasets.

      .. card-footer::

         ``mlx_graphs.datasets``

   .. grid-item-card:: :octicon:`iterations;1.2em;feature-icon` Loaders
      :link: loaders/index
      :link-type: doc

      Data loaders and neighbor, cluster, GraphSAINT
      and temporal samplers.

      .. card-footer::

         ``mlx_graphs.loaders``

   .. grid-item-card:: :octicon:`sync;1.2em;feature-icon` Transforms
      :link: transforms/index
      :link-type: doc

      Graph transforms, from self-loops to
      positional encodings.

      .. card-footer::

         ``mlx_graphs.transforms``

   .. grid-item-card:: :octicon:`cpu;1.2em;feature-icon` Neural networks
      :link: nn/index
      :link-type: doc

      Message passing, convolution layers, models,
      pooling, aggregations, normalization and attention.

      .. card-footer::

         ``mlx_graphs.nn``

   .. grid-item-card:: :octicon:`tools;1.2em;feature-icon` Utils
      :link: utils/index
      :link-type: doc

      Scatter operations, sparse and graph utilities,
      conversions and generators.

      .. card-footer::

         ``mlx_graphs.utils``

   .. grid-item-card:: :octicon:`share-android;1.2em;feature-icon` Algorithms
      :link: algorithms/index
      :link-type: doc

      Random-walk embeddings (``Node2Vec``,
      ``MetaPath2Vec``).

      .. card-footer::

         ``mlx_graphs.algorithms``

   .. grid-item-card:: :octicon:`meter;1.2em;feature-icon` Metrics
      :link: metrics/index
      :link-type: doc

      Link prediction and ranking metrics.

      .. card-footer::

         ``mlx_graphs.metrics``

.. toctree::
   :hidden:
   :maxdepth: 2

   data/index.rst
   datasets/index.rst
   loaders/index.rst
   transforms/index.rst
   nn/index.rst
   utils/index.rst
   algorithms/index.rst
   metrics/index.rst
