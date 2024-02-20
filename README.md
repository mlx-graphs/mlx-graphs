![mlx-graphs logo](docs/source/_static/logo.svg)
______________________________________________________________________

**[Documentation](https://mlx-graphs.github.io/mlx-graphs/)** | **[Quickstart](https://mlx-graphs.github.io/mlx-graphs/tutorials/quickstart.html)**

MLX-graphs is a library for Graph Neural Networks (GNNs) built upon Apple's [MLX](https://github.com/ml-explore/mlx).


## Features

- **Fast GNN training and inference on Apple Silicon**

   ``mlx-graphs`` has been designed to run GNNs and graph algorithms *fast* on Apple Silicon chips. All GNN operations
   fully leverage the GPU and CPU hardware of Macs thanks to the efficient low-level primitives
   available within the MLX core library. Initial benchmarks show an up to 10x speed improvement
   with respect to other frameworks on large datasets.
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
git clone git@github.com:mlx-graphs/mlx-graphs.git && cd mlx-graphs
```
Create a new virtual environment and install the requirements
```
pip install -e .
```

## Usage

### Tutorial guides

We provide some notebooks to practice `mlx-graphs`.

- Quickstart guide: [notebook](https://mlx-graphs.github.io/mlx-graphs/tutorials/quickstart.html)
- Graph classification guide: [notebook](https://mlx-graphs.github.io/mlx-graphs/tutorials/graph_classification.html)

### Example
This library has been designed to build GNNs with ease and efficiency. Building new GNN layers is straightforward by implementing the `MessagePassing` class. This approach ensures that all operations related to message passing are properly handled and processed efficiently on your Mac's GPU. As a result, you can focus exclusively on the GNN logic, without worrying about the underlying message passing mechanics.

Here is an example of a custom [GraphSAGE](https://proceedings.neurips.cc/paper_files/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf) convolutional layer that considers edge weights:

```python
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
```

## Contributing
### Why contributing?

We are at an early stage of the development of the lib, which means your contributions can have a large impact!
Everyone is welcome to contribute, just open an issue 📝 with your idea 💡 and we'll work together on the implementation ✨.

> [!NOTE]
> Contributions such as the implementation of new layers and datasets would be very valuable for the library.

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

Other frameworks like PyG and DGL also benefit from efficient GNN operations parallelized on GPU. However, they are not fully optimized to leverage the Mac's GPU capabilities, often defaulting to CPU execution.

In contrast, `mlx-graphs` is specifically designed to leverage the power of Mac's hardware, delivering optimal performance for Mac users. By taking advantage of Apple Silicon, `mlx-graphs` enables accelerated GPU computation and benefits from unified memory. This approach removes the need for data transfers between devices and allows for the use of the entire memory space available on the Mac's GPU. Consequently, users can manage large graphs directly on the GPU, enhancing performance and efficiency.
