# Installation

<span class="title-ref">mlx-graphs</span> is available on Pypi. To install run

```shell
pip install mlx-graphs
```

It requires a Mac with Apple Silicon and Python 3.10 or later. MLX and `mlx-cluster`, which provides the sparse GPU kernels of the fused message passing, are installed with it.

## Build from source

To build and install <span class="title-ref">mlx-graphs</span> from source start by cloning the github repo

```shell
git clone git@github.com:mlx-graphs/mlx-graphs.git && cd mlx-graphs
```

Then build and install using <span class="title-ref">pip</span> (we suggest using a separate virtual environment)

```shell
pip install -e .
```

## Installing test, dev, benchmarks, docs dependencies

To install any extra dependencies for testing, development and building documentation, you can run any of the following

```shell
pip install -e '.[test]'
pip install -e '.[dev]'
pip install -e '.[benchmarks]'
pip install -e '.[docs]'
```

For dev purposes you may want to install the current version of <span class="title-ref">mlx</span> via <span class="title-ref">pip install mlx @ git+https://github.com/ml-explore/mlx.git</span>
