.. _installation:


Installation
============

`mlx-graphs` is available on Pypi and supports Python 3.10 to 3.13.

macOS (Apple Silicon)
---------------------

.. code-block:: shell

	pip install mlx-graphs

MLX and ``mlx-cluster``, which provides the sparse GPU kernels of the fused
message passing, neighbor sampling and random walks, are installed with it.

Linux
-----

On Linux, MLX needs a backend, which you choose with an extra (see the
`MLX installation guide <https://ml-explore.github.io/mlx/build/html/install.html>`_
for the requirements of each):

.. code-block:: shell

	pip install "mlx-graphs[cpu]"     # MLX CPU backend
	pip install "mlx-graphs[cuda]"    # MLX CUDA backend, CUDA 12
	pip install "mlx-graphs[cuda13]"  # MLX CUDA backend, CUDA 13

The test suite runs on Linux with the CPU backend in CI. The CUDA backend is
supported by MLX but not tested in CI.

What changes on Linux
~~~~~~~~~~~~~~~~~~~~~

Two accelerations of mlx-graphs exist only on macOS: custom Metal kernels
(GATv2 attention scores, fused neighbor softmax) and ``mlx-cluster``, which has
no Linux wheels. Without them, every layer, loader and algorithm uses a
portable implementation with the same results:

* message passing uses MLX gather-scatter instead of fused SpMM kernels (see
  the :ref:`message passing guide <message_passing_guide>`), and the Metal kernels
  are replaced by the equivalent MLX operations;
* ``NeighborLoader``, ``LinkNeighborLoader`` and ``HGTLoader`` sample
  neighbors with NumPy, and ``Node2Vec`` / ``MetaPath2Vec`` sample random
  walks with MLX / NumPy (``mlx_graphs.sampling``). Walks are identical given
  the same random numbers; neighbor sampling draws from the same distribution.

To check what is available, or to run the portable code paths on a Mac (e.g.
to test them), use :mod:`mlx_graphs.backend`:

.. code-block:: python

	from mlx_graphs import backend

	backend.HAS_MLX_CLUSTER            # mlx-cluster installed and enabled
	backend.metal_kernels_available()  # custom Metal kernels can run

.. code-block:: shell

	# Behave as on Linux: no Metal kernels, no mlx-cluster
	MLX_GRAPHS_DISABLE_METAL_KERNELS=1 MLX_GRAPHS_DISABLE_MLX_CLUSTER=1 pytest
	# Run the test suite on the CPU device
	MLX_GRAPHS_TEST_DEVICE=cpu pytest


Build from source
-----------------

To build and install `mlx-graphs` from source start by cloning the github repo

.. code-block:: shell

	git clone git@github.com:mlx-graphs/mlx-graphs.git && cd mlx-graphs

Then build and install using `pip` (we suggest using a separate virtual environment)

.. code-block:: shell

	pip install -e .



Installing test, dev, benchmarks, docs dependencies
---------------------------------------------------

To install any extra dependencies for testing, development and building documentation, you can run any of the following

.. code-block:: shell

	pip install -e '.[test]'
	pip install -e '.[dev]'
	pip install -e '.[benchmarks]'
	pip install -e '.[docs]'


For dev purposes you may want to install the current version of `mlx` via `pip install mlx @ git+https://github.com/ml-explore/mlx.git`
