![mlx-graphs logo](docs/source/_static/logo.svg)
______________________________________________________________________

**[Documentation](https://mlx-graphs.github.io/mlx-graphs/)**

MLX-graphs is a library for Graph Neural Networks (GNNs) built upon Apple's MLX.


## Features

- **Fast GNN training and inference on Apple Silicon**

   ``MLX-graphs`` has been designed to run *fast* on Apple Silicon chips. All GNN operations
   fully leverage the GPU and CPU hardware of Macs thanks to the efficient low-level primitives
   available within the MLX core library.
- **Scalability to large graphs**

   With unified memory architecture, objects live in a shared memory accessible by both the CPU and GPU.
   This setup allows Macs to leverage their entire memory capacity for storing graphs.
   Consequently, Macs equipped with substantial memory can efficiently train GNNs on large graphs, spanning tens of gigabytes, directly using the Mac's GPU.
- **Multi-device**

   Unified memory eliminates the need for time-consuming device-to-device transfers.
   This architecture also enables specific operations to be run explicitly on either the CPU or GPU without incurring any overhead, facilitating more efficient computation and resource utilization.


## Installation
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

## Contributing
### Installing test, dev, benchmaks, docs dependencies
Extra dependencies are specified in the `pyproject.toml`.
To install those required for testing, development and building documentation, you can run any of the following
```
pip install -e '.[test]'
pip install -e '.[dev]'
pip install -e '.[benchmarks]'
pip install -e '.[docs]'
```
For dev purposes you may want to install the current version of `mlx` via `pip install git+https://github.com/ml-explore/mlx.git`

### Testing
We encourage to write tests for all components. CI is currently not in place as runners with Apple Silicon are required.
Please run `pytest` to ensure breaking changes are not introduced.


### Pre-commit hooks (optional)
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
