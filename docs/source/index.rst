MLX-graphs
==========

MLX-graphs is a library for Graph Neural Networks (GNNs) built upon Apple's `MLX <https://github.com/ml-explore/mlx>`_.

**On a Mac, MLX-graphs trains GNNs 5.6× faster than PyTorch Geometric on the GPU (MPS)
and 5.8× faster than PyTorch Geometric on the CPU, and the models reach the same accuracy**
(medians over 35 training benchmarks, up to 58× faster). See the :doc:`benchmarks`.

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M5_Pro/0.1.0-c5f1246/figures/headline_light.png
   :alt: Time of PyG on MPS and on CPU relative to mlx-graphs for one layer, node classification and graph classification, and the test accuracy of each framework
   :class: only-light
   :width: 100%

.. image:: /../../benchmarks/pyg_comparison/results/Apple_M5_Pro/0.1.0-c5f1246/figures/headline_dark.png
   :alt: Time of PyG on MPS and on CPU relative to mlx-graphs for one layer, node classification and graph classification, and the test accuracy of each framework
   :class: only-dark
   :width: 100%




Features
----------------

- Fast GNN training and inference on Apple Silicon
   ``MLX-graphs`` has been designed to run *fast* on Apple Silicon chips. All GNN operations
   fully leverage the GPU and CPU hardware of Macs thanks to the efficient low-level primitives
   available within the MLX core library, and message passing aggregates neighbours with fused
   sparse kernels. On an M5 Pro, one GNN layer runs up to 31× faster than PyTorch Geometric on
   the Mac's GPU (MPS). See the :doc:`benchmarks`.
- Scalability to large graphs
   With unified memory architecture, objects live in a shared memory accessible by both the CPU and GPU.
   This setup allows Macs to leverage their entire memory capacity for storing graphs.
   Consequently, Macs equipped with substantial memory can efficiently train GNNs on large graphs, spanning tens of gigabytes, directly using the Mac's GPU.
- PyTorch Geometric's API, on MLX
   The layers, models, samplers and loaders, transforms and datasets of PyG are available under
   the same names: mini-batch training with ``NeighborLoader`` and ``ClusterLoader``,
   heterogeneous models with ``to_hetero``, temporal graphs with TGN, and graph transformers
   with ``GPSConv``. Each was checked against PyG. See :doc:`tutorials/from_pyg`.
- Multi-device
   Unified memory eliminates the need for time-consuming device-to-device transfers.
   This architecture also enables specific operations to be run explicitly on either the CPU or GPU without incurring any overhead, facilitating more efficient computation and resource utilization.


Example usage
-------------

This library has been designed to build GNNs with ease and efficiency. Building new GNN layers is straightforward by implementing the `MessagePassing` class. This approach ensures that all operations related to message passing are properly handled and processed efficiently on your Mac's GPU. As a result, you can focus exclusively on the GNN logic, without worrying about the underlying message passing mechanics.

Here is an example of a custom `GraphSAGE <https://proceedings.neurips.cc/paper_files/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf>`_ convolutional layer that considers edge weights:

.. code-block:: python

   import mlx.core as mx
   from mlx_graphs.nn.linear import Linear
   from mlx_graphs.nn.message_passing import MessagePassing, fused_message_passing


   @fused_message_passing  # aggregate with one SpMM instead of gather-scatter
   class SAGEConv(MessagePassing):
       def __init__(self, node_features_dim: int, out_features_dim: int, bias: bool = True):
           super().__init__(aggr="mean")
           self.neigh_proj = Linear(node_features_dim, out_features_dim, bias=False)
           self.self_proj = Linear(node_features_dim, out_features_dim, bias=bias)

       def __call__(self, edge_index: mx.array, node_features: mx.array, edge_weights: mx.array) -> mx.array:
           # Message passing directly on GPU. The default message is
           # edge_weights * source features, which the fused kernel computes.
           neigh_features = self.propagate(
               edge_index=edge_index,
               node_features=node_features,
               message_kwargs={"edge_weights": edge_weights},
           )
           return self.self_proj(node_features) + self.neigh_proj(neigh_features)

Layers that override ``message()`` with something other than a weighted sum of
neighbour features (e.g. attention computed per edge) simply drop the
decorator, and ``propagate`` gathers per-edge messages and scatters them.


.. toctree::
   :caption: Getting started
   :maxdepth: 1

   install
   tutorials/quickstart.rst
   tutorials/from_pyg.rst


.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   tutorials/notebooks/graph_classification.ipynb
   tutorials/notebooks/node2vec.ipynb
   tutorials/notebooks/minibatch_training.ipynb
   tutorials/notebooks/link_prediction.ipynb
   tutorials/notebooks/temporal_graph_networks.ipynb
   tutorials/notebooks/heterogeneous_to_hetero.ipynb
   tutorials/notebooks/graph_transformers_gps.ipynb


.. toctree::
   :caption: User guide
   :maxdepth: 1

   guide/message_passing
   datasets


.. toctree::
   :caption: Examples
   :maxdepth: 1

   tutorials/examples/index.rst


.. toctree::
   :caption: Benchmarks and validation
   :maxdepth: 1

   benchmarks
   validation


.. toctree::
   :caption: API
   :maxdepth: 1

   api/data/index.rst
   api/datasets/index.rst
   api/loaders/index.rst
   api/transforms/index.rst
   api/nn/index.rst
   api/utils/index.rst
   api/algorithms/index.rst
   api/metrics/index.rst


.. toctree::
   :caption: Project
   :maxdepth: 1

   changelog


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
