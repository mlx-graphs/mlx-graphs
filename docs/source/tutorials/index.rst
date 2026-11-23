.. _tutorials:

Tutorials
=========

Notebooks that train models end to end with mlx-graphs, several of them side
by side with PyTorch Geometric. The outputs shown are the ones stored in the
notebooks: download a notebook from the top of its page to run it yourself.

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: tutorial-grid

   .. grid-item-card:: Graph classification
      :link: notebooks/graph_classification
      :link-type: doc
      :img-top: ../_static/tutorials/graph_classification.svg
      :img-alt:

      A GCN on the MUTAG molecules, from batching to evaluation.

   .. grid-item-card:: Node2Vec
      :link: notebooks/node2vec
      :link-type: doc
      :img-top: ../_static/tutorials/node2vec.svg
      :img-alt:

      Node embeddings from biased random walks on Cora.

   .. grid-item-card:: MetaPath2Vec
      :link: notebooks/metapath2vec
      :link-type: doc
      :img-top: ../_static/tutorials/metapath2vec.svg
      :img-alt:

      Embeddings of a heterogeneous graph from meta-path guided random walks,
      on DBLP.

   .. grid-item-card:: Mini-batch training on large graphs
      :link: notebooks/minibatch_training
      :link-type: doc
      :img-top: ../_static/tutorials/minibatch_training.svg
      :img-alt:

      ``NeighborLoader``, ``ClusterLoader`` and GraphSAINT on ogbn-arxiv.

   .. grid-item-card:: Link prediction
      :link: notebooks/link_prediction
      :link-type: doc
      :img-top: ../_static/tutorials/link_prediction.svg
      :img-alt:

      A GCN encoder and a dot-product decoder on Cora, compared with PyG.

   .. grid-item-card:: Temporal graph networks
      :link: notebooks/temporal_graph_networks
      :link-type: doc
      :img-top: ../_static/tutorials/temporal_graph_networks.svg
      :img-alt:

      A TGN with node memory on the JODIE Wikipedia event stream.

   .. grid-item-card:: Heterogeneous graphs with ``to_hetero``
      :link: notebooks/heterogeneous_to_hetero
      :link-type: doc
      :img-top: ../_static/tutorials/heterogeneous_to_hetero.svg
      :img-alt:

      Turn a homogeneous GraphSAGE or GAT into a heterogeneous model, on DBLP.

   .. grid-item-card:: Graph transformers with GPS
      :link: notebooks/graph_transformers_gps
      :link-type: doc
      :img-top: ../_static/tutorials/graph_transformers_gps.svg
      :img-alt:

      ``GPSConv`` with random-walk encodings on Peptides-func.

   .. grid-item-card:: GNNs for cybersecurity
      :link: notebooks/cybersecurity
      :link-type: doc
      :img-top: ../_static/tutorials/provenance_intrusion_detection.svg
      :img-alt:

      Intrusion detection, anti-money laundering and more, on four public
      security datasets.

   .. grid-item-card:: Graph RAG with G-reasoner
      :link: notebooks/graph_rag_g_reasoner
      :link-type: doc
      :img-top: ../_static/tutorials/graph_rag_g_reasoner.svg
      :img-alt:

      Multi-hop retrieval over a knowledge graph, answered by a local LLM.

   .. grid-item-card:: Quantum-chemical properties
      :link: notebooks/qm9_tutorial
      :link-type: doc
      :img-top: ../_static/tutorials/qm9_tutorial.png
      :img-alt:

      Predict properties of small molecules from the QM9 dataset.


Example scripts
---------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: :octicon:`terminal;1.2em;feature-icon` Runnable examples
      :link: examples/index
      :link-type: doc

      Scripts in the ``examples/`` folder of the repository that train a
      model end to end from the command line: node classification, GCN and
      GAT, link prediction, heterogeneous GNNs, TGN, GPS and more.


.. toctree::
   :hidden:
   :maxdepth: 1

   notebooks/graph_classification.ipynb
   notebooks/node2vec.ipynb
   notebooks/metapath2vec.ipynb
   notebooks/minibatch_training.ipynb
   notebooks/link_prediction.ipynb
   notebooks/temporal_graph_networks.ipynb
   notebooks/heterogeneous_to_hetero.ipynb
   notebooks/graph_transformers_gps.ipynb
   notebooks/cybersecurity.ipynb
   notebooks/graph_rag_g_reasoner.ipynb
   notebooks/qm9_tutorial.ipynb
   examples/index.rst
