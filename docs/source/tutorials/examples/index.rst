.. _examples:

Examples
========

Runnable scripts in the `examples/
<https://github.com/mlx-graphs/mlx-graphs/tree/main/examples>`_ folder of the
repository. Each one trains a model end to end and takes its options on the
command line (``python examples/<script>.py --help``). Datasets are downloaded
on first use.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Script
     - What it shows
   * - `node_classification.py
       <https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/node_classification.py>`_
     - Full-batch node classification with the ready-made ``GCN``,
       ``GraphSAGE`` and ``GAT`` models, on Planetoid, Amazon and Coauthor
       datasets.
   * - `gcn.py <https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/gcn.py>`_,
       `gat.py <https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/gat.py>`_
     - A GCN and a GAT written layer by layer on Cora, with early stopping
       (Cora is loaded through PyG, which must be installed).
   * - `link_prediction.py
       <https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/link_prediction.py>`_
     - A GCN encoder and dot-product decoder on edges split with
       ``RandomLinkSplit``.
   * - `hetero_sage_dblp.py
       <https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/hetero_sage_dblp.py>`_
     - A heterogeneous GraphSAGE written by hand on DBLP, trained with the
       heterogeneous ``NeighborLoader``.
   * - `to_hetero_dblp.py
       <https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/to_hetero_dblp.py>`_
     - The same task with a homogeneous model converted by ``to_hetero`` or
       ``to_hetero_with_bases``.
   * - `tgn_jodie.py
       <https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/tgn_jodie.py>`_
     - A Temporal Graph Network on a JODIE dataset, after PyG's
       ``examples/tgn.py``.
   * - `graph_gps_peptides.py
       <https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/graph_gps_peptides.py>`_
     - The GPS graph transformer with random-walk encodings on Peptides-func,
       after PyG's ``examples/graph_gps.py``.
   * - `dictionary_lookup.py
       <https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/dictionary_lookup.py>`_
     - The synthetic DictionaryLookup benchmark from "How Attentive Are Graph
       Attention Networks?": ``GATConv`` against ``GATv2Conv``.
   * - `graph_network_block.py
       <https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/graph_network_block.py>`_
     - A ``GraphNetworkBlock`` with custom edge, node and global models.
   * - `compute_dataset_stats.py
       <https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/compute_dataset_stats.py>`_
     - Statistics (graphs, nodes, edges, features, labels) of the built-in
       datasets.
