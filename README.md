# MLX-graphs

MLX-graphs is a library for Graph Neural Networks (GNNs) built upon Apple's MLX.

## Work in progress 🚧

We just started the development of this lib, with the aim to integrate it within [ml-explore](https://github.com/ml-explore).

The lib follows the Message Passing Neural Network ([MPNN](https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf)) architecture to build arbitrary GNNs on top of it, similarly as in [PyG](https://github.com/pyg-team/pytorch_geometric).

### Installation (build from source)
To build and install `mlx-graphs` from source start by cloning the github repo
```
git clone git@github.com:TristanBilot/mlx-graphs.git && cd mlx-graphs
```
Create a new virtual environment and install the requirements
```
pip install -r requirements.txt
```
For dev, install `mlx-graphs` locally and install `requirements-dev.txt`
```
pip install -r requirements-dev.txt
```

### Contributing

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
