MLX-graphs
==========

MLX-graphs is a library for Graph Neural Networks (GNNs) built upon Apple's `MLX <https://github.com/ml-explore/mlx>`_.


.. caution::
   We are very early in the development of this library and there may be breaking changes in upcoming versions.


Features
----------------

- Fast GNN training and inference on Apple Silicon
   ``MLX-graphs`` has been designed to run *fast* on Apple Silicon chips. All GNN operations
   fully leverage the GPU and CPU hardware of Macs thanks to the efficient low-level primitives
   available within the MLX core library.
- Scalability to large graphs
   With unified memory architecture, objects live in a shared memory accessible by both the CPU and GPU.
   This setup allows Macs to leverage their entire memory capacity for storing graphs.
   Consequently, Macs equipped with substantial memory can efficiently train GNNs on large graphs, spanning tens of gigabytes, directly using the Mac's GPU.
- Multi-device
   Unified memory eliminates the need for time-consuming device-to-device transfers.
   This architecture also enables specific operations to be run explicitly on either the CPU or GPU without incurring any overhead, facilitating more efficient computation and resource utilization.

Usage
-----


Examples showing the functionalities of this library are available `here <https://github.com/TristanBilot/mlx-graphs/tree/main/examples>`_.


.. toctree::
   :caption: Install
   :maxdepth: 1

   install


.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   tutorials/quickstart.rst
   .. tutorials/tutorials.rst


.. toctree::
   :caption: API
   :maxdepth: 1

   api/data/index.rst
   api/nn/index.rst
   api/utils/index.rst






.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
