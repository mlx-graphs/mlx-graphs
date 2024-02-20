MLX-graphs
==========

MLX-graphs is a library for Graph Neural Networks (GNNs) built upon Apple's `MLX <https://github.com/ml-explore/mlx>`_.



Features
----------------

- Fast GNN training and inference on Apple Silicon
   ``MLX-graphs`` has been designed to run *fast* on Apple Silicon chips. All GNN operations
   fully leverage the GPU and CPU hardware of Macs thanks to the efficient low-level primitives
   available within the MLX core library. Initial benchmarks show an up to 10x speed improvement
   with respect to other frameworks on large datasets.
- Scalability to large graphs
   With unified memory architecture, objects live in a shared memory accessible by both the CPU and GPU.
   This setup allows Macs to leverage their entire memory capacity for storing graphs.
   Consequently, Macs equipped with substantial memory can efficiently train GNNs on large graphs, spanning tens of gigabytes, directly using the Mac's GPU.
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
   from mlx_graphs.nn.message_passing import MessagePassing

   class SAGEConv(MessagePassing):
       def __init__(
           self, node_features_dim: int, out_features_dim: int, bias: bool = True, **kwargs
       ):
           super(SAGEConv, self).__init__(aggr="mean", **kwargs)

           self.node_features_dim = node_features_dim
           self.out_features_dim = out_features_dim

           self.neigh_proj = Linear(node_features_dim, out_features_dim, bias=False)
           self.self_proj = Linear(node_features_dim, out_features_dim, bias=bias)

       def __call__(self, edge_index: mx.array, node_features: mx.array, edge_weights: mx.array) -> mx.array:
            """Forward layer of the custom SAGE layer."""
            neigh_features = self.propagate( # Message passing directly on GPU
               edge_index=edge_index,
               node_features=node_features,
               message_kwargs={"edge_weights": edge_weights},
            )
            neigh_features = self.neigh_proj(neigh_features)

           out_features = self.self_proj(node_features) + neigh_features
           return out_features

      def message(self, src_features: mx.array, dst_features: mx.array, **kwargs) -> mx.array:
            """Message function called by propagate(). Computes messages for all edges in the graph."""
           edge_weights = kwargs.get("edge_weights", None)

           return edge_weights.reshape(-1, 1) * src_features


.. toctree::
   :caption: Install
   :maxdepth: 1

   install


.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   tutorials/quickstart.rst
   tutorials/notebooks/graph_classification.ipynb
   tutorials/notebooks/benchmark_pyg_dgl_mxg.ipynb


.. toctree::
   :caption: API
   :maxdepth: 1

   api/data/index.rst
   api/datasets/index.rst
   api/loaders/index.rst
   api/nn/index.rst
   api/utils/index.rst






.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
