.. _installation:


Installation
============

.. caution::
	We currently only support building from source.

Build from source
-----------------

To build and install `mlx-graphs` from source start by cloning the github repo

.. code-block:: shell

	git clone git@github.com:TristanBilot/mlx-graphs.git && cd mlx-graphs

Then build and install using `pip` (we suggest using a separate virtual environment)

.. code-block:: shell

	pip install -e .



Installing test, dev, docs dependencies
---------------------------------------

To install any extra dependencies for testing, development and building documentation, you can run any of the following

.. code-block:: shell

	pip install -e '.[test]'
	pip install -e '.[dev]'
	pip install -e '.[docs]'
