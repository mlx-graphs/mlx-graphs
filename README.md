# MLX-graphs

[Documentation](https://tristanbilot.github.io/mlx-graphs/)

MLX-graphs is a library for Graph Neural Networks (GNNs) built upon Apple's MLX.

## Work in progress 🚧

We just started the development of this lib, with the aim to integrate it within [ml-explore](https://github.com/ml-explore).

The lib follows the Message Passing Neural Network ([MPNN](https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf)) architecture to build arbitrary GNNs on top of it, similarly as in [PyG](https://github.com/pyg-team/pytorch_geometric).

### Installation
`mlx-graphs` is available on Pypi. To install run
```
pip install mlx-graphs
```
### Build from source

To build and install `mlx-graphs` from source start by cloning the github repo
```
git clone git@github.com:TristanBilot/mlx-graphs.git && cd mlx-graphs
```
Create a new virtual environment and install the requirements
```
pip install -e .
```

### Usage
#### Graph data model
A graph is defined by a set of (optional) attributes
  1. `edge_index`: an array of size `[2, num_edges]` which specifies the topology of the graph. The i-th column in `edge_index` defines the source and destination nodes of the i-th edge
  2. `node_features`: an array of size `[num_nodes, num_node_features]` defining the features associated to each node (if any). The i-th row contains the features of the i-th node
  3. `edge_features`:  an array of size `[num_edges, num_edge_features]` defining the features associated to each edge (if any). The i-th row contains the features of the i-th edge
  4. `graph_features`: an array of size `[num_graph_features]` defining the features associated to the graph itself

We adopt the above convention across the entire library both in terms of shapes of the attributes and the order in which they're provided to functions.

### Contributing
#### Installing test, dev, benchmaks, docs dependencies
Extra dependencies are specified in the `pyproject.toml`.
To install those required for testing, development and building documentation, you can run any of the following
```
pip install -e '.[test]'
pip install -e '.[dev]'
pip install -e '.[benchmarks]'
pip install -e '.[docs]'
```
For dev purposes you may want to install the current version of `mlx` via `pip install mlx @ git+https://github.com/ml-explore/mlx.git`

#### Testing
We encourage to write tests for all components. CI is currently not in place as runners with Apple Silicon are required.
Please run `pytest` to ensure breaking changes are not introduced.


#### Pre-commit hooks (optional)
To ensure code quality you can run [pre-commit](https://pre-commit.com) hooks. Simply install them by running
```
pre-commit install
```
and run via `pre-commit run --all-files`.

> Note: CI is in place to verify code quality, so pull requests that don't meet those requirements won't pass CI tests.


## Why running GNNs on my Mac?

By leveraging Apple Silicon, running GNNs enjoys accelerated GPU computation ⚡️ and unified memory architecture. This eliminates device data transfers and enables leveraging all available memory to store large graphs directly on your Mac's GPU.

## Why contributing?

We are at an early stage of the development of the lib, which means your contributions can have a large impact!
Everyone is welcome to contribute, just open an issue 📝 with your idea 💡 and we'll work together on the implementation ✨.
