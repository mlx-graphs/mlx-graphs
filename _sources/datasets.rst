Datasets
========

MLX-graphs provides several built-in datasets for graph machine learning tasks.

.. note::
   Datasets are automatically downloaded to ``.mlx_graphs_data/`` in the current
   working directory upon first use. You can specify a custom directory by
   passing ``base_dir`` to the dataset constructor.

Available Datasets
------------------

.. list-table:: Dataset Statistics
   :header-rows: 1
   :widths: 20 10 10 12 12 15 15

   * - Dataset
     - Graphs
     - Nodes
     - Edges
     - Avg Degree
     - Features
     - Labels
   * - Karate Club
     - 1
     - 34
     - 156
     - 9.18
     - node
     - node
   * - Cora
     - 1
     - 2,708
     - 10,556
     - 7.80
     - node
     - node
   * - CiteSeer
     - 1
     - 3,312
     - 9,104
     - 5.50
     - node
     - node
   * - PubMed
     - 1
     - 19,717
     - 88,651
     - 8.99
     - node
     - node
   * - QM7b
     - 7,211
     - ~15
     - ~130
     - ~8.67
     - node, edge
     - graph
   * - Elliptic Bitcoin
     - 1
     - 203,769
     - 234,355
     - 2.30
     - node
     - node
   * - MovieLens 100K
     - 1
     - 1,682
     - 100,000
     - 118.90
     - node
     - edge

.. _dataset-classes:

Dataset Classes
---------------

.. currentmodule:: mlx_graphs.datasets

.. autosummary::
   :toctree: generated/
   :nosignatures:

   KarateClubDataset
   PlanetoidDataset
   QM7bDataset
   TUDataset
   SuperPixelDataset
   OGBDataset
   EllipticBitcoinDataset
   MovieLens100K
   IMDB
   DBLP

KarateClubDataset
^^^^^^^^^^^^^^^^^

.. autoclass:: KarateClubDataset
   :members:
   :show-inheritance:

PlanetoidDataset
^^^^^^^^^^^^^^^^

.. autoclass:: PlanetoidDataset
   :members:
   :show-inheritance:

QM7bDataset
^^^^^^^^^^^

.. autoclass:: QM7bDataset
   :members:
   :show-inheritance:

TUDataset
^^^^^^^^^

.. autoclass:: TUDataset
   :members:
   :show-inheritance:

SuperPixelDataset
^^^^^^^^^^^^^^^^^

.. autoclass:: SuperPixelDataset
   :members:
   :show-inheritance:

OGBDataset
^^^^^^^^^^

.. autoclass:: OGBDataset
   :members:
   :show-inheritance:

EllipticBitcoinDataset
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: EllipticBitcoinDataset
   :members:
   :show-inheritance:

MovieLens100K
^^^^^^^^^^^^^

.. autoclass:: MovieLens100K
   :members:
   :show-inheritance:

IMDB
^^^^

.. autoclass:: IMDB
   :members:
   :show-inheritance:

DBLP
^^^^

.. autoclass:: DBLP
   :members:
   :show-inheritance:

Using Datasets
--------------

Loading a dataset is straightforward:

.. code-block:: python

   from mlx_graphs.datasets import KarateClubDataset, PlanetoidDataset

   # Simple built-in dataset
   dataset = KarateClubDataset()
   graph = dataset[0]
   print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")

   # Citation network dataset
   cora = PlanetoidDataset(name="cora")
   print(cora[0])

Computing Dataset Statistics
----------------------------

To compute statistics for any dataset, use the included utility script:

.. code-block:: bash

   # Stats for all available datasets
   python examples/compute_dataset_stats.py

   # Stats for a specific dataset
   python examples/compute_dataset_stats.py --dataset karate_club

   # Output as markdown table (for use in docs/issues)
   python examples/compute_dataset_stats.py --format markdown

   # Output as RST table (for Sphinx docs)
   python examples/compute_dataset_stats.py --format rst
