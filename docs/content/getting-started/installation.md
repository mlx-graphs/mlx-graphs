# Installation

`mlx-graphs` is available on PyPI. To install it, run

```shell
pip install mlx-graphs
```

It requires a Mac with Apple Silicon and Python 3.10 or later. MLX and `mlx-cluster`, which provides the sparse GPU kernels of the fused message passing, are installed with it, together with NumPy, `requests`, `fsspec` and `tqdm`.

## Build from source

To build and install `mlx-graphs` from source, start by cloning the GitHub repository

```shell
git clone git@github.com:mlx-graphs/mlx-graphs.git && cd mlx-graphs
```

Then build and install it with `pip` (we suggest using a separate virtual environment)

```shell
pip install -e .
```

## Installing test, dev, benchmarks, docs dependencies

To install the extra dependencies for testing, development, benchmarks and building the documentation, you can run any of the following

```shell
pip install -e '.[test]'        # pytest, scipy, networkx, PyTorch, PyG, pandas, scikit-learn
pip install -e '.[dev]'         # pre-commit
pip install -e '.[benchmarks]'  # scipy, matplotlib, ogb, PyTorch, PyG
pip install -e '.[docs]'        # this documentation site (Zensical)
```

For dev purposes you may want to install the current version of `mlx` via `pip install "mlx @ git+https://github.com/ml-explore/mlx.git"`.

## Building the documentation

The site is built with [Zensical](https://zensical.org). The API reference is generated from the docstrings by static analysis, so MLX does not need to be importable:

```shell
pip install -e '.[docs]'
python docs/scripts/gen.py   # API pages, notebooks, changelog, benchmark sections
zensical serve                    # live preview on http://localhost:8000/mlx-graphs/
zensical build --strict --clean   # static site in site/
```
