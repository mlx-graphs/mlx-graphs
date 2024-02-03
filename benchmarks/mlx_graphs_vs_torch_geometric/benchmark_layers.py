import mlx.core as mx
import mlx.nn as mlx_nn
import torch
import torch_geometric.nn as pyg_nn

import mlx_graphs.nn as mlg_nn

try:
    from torch_scatter import scatter as scatter_torch
except ImportError:
    raise ImportError("To run this benchmark, run `pip install torch_scatter`")

from benchmark_utils import (
    get_dummy_edge_index,
    get_dummy_features,
    measure_runtime,
)

from mlx_graphs.utils.scatter import scatter


def sync_mps_if_needed(device):
    """
    Call this function after every torch implementation to ensure
    the mps execution has finished.
    """
    if device == torch.device("mps"):
        torch.mps.synchronize()


class MLXLayer(mlx_nn.Module):
    def __init__(self, layer_cls, **kwargs):
        super().__init__()
        self.layer = layer_cls(**kwargs)
        self.y = []

    def __call__(self, **kwargs):
        for _ in range(1):
            self.y.append(self.layer(**kwargs))
        mx.eval(self.y)


class PyGLayer(torch.nn.Module):
    def __init__(self, layer_cls, **kwargs):
        super().__init__()
        self.layer = layer_cls(**kwargs)
        self.y = []

    def forward(self, **kwargs):
        with torch.no_grad():
            for _ in range(1):
                self.y.append(self.layer(**kwargs))

            device = next(self.parameters()).device
            sync_mps_if_needed(device)


def benchmark_GCNConv(framework, device=None, **kwargs):
    def run_benchmark(in_dim, out_dim, edge_index_shape, node_features_shape):
        if framework == "mlx":
            edge_index = get_dummy_edge_index(
                edge_index_shape, node_features_shape[0], device, "mlx"
            )
            node_features = get_dummy_features(node_features_shape, device, "mlx")
            layer = MLXLayer(
                mlg_nn.GCNConv, node_features_dim=in_dim, out_features_dim=out_dim
            )

            return measure_runtime(
                layer, node_features=node_features, edge_index=edge_index
            )
        else:
            edge_index = get_dummy_edge_index(
                edge_index_shape, node_features_shape[0], device, "pyg"
            )
            node_features = get_dummy_features(node_features_shape, device, "pyg")
            layer = PyGLayer(
                pyg_nn.GCNConv, in_channels=in_dim, out_channels=out_dim
            ).to(device)

            return measure_runtime(layer, x=node_features, edge_index=edge_index)

    return run_benchmark(**kwargs)


def benchmark_GATConv(framework, device=None, **kwargs):
    def run_benchmark(in_dim, out_dim, edge_index_shape, node_features_shape):
        if framework == "mlx":
            edge_index = get_dummy_edge_index(
                edge_index_shape, node_features_shape[0], device, "mlx"
            )
            node_features = get_dummy_features(node_features_shape, device, "mlx")
            layer = MLXLayer(
                mlg_nn.GATConv, node_features_dim=in_dim, out_features_dim=out_dim
            )

            return measure_runtime(
                layer, node_features=node_features, edge_index=edge_index
            )
        else:
            edge_index = get_dummy_edge_index(
                edge_index_shape, node_features_shape[0], device, "pyg"
            )
            node_features = get_dummy_features(node_features_shape, device, "pyg")
            layer = PyGLayer(
                pyg_nn.GATConv, in_channels=in_dim, out_channels=out_dim
            ).to(device)

            return measure_runtime(layer, x=node_features, edge_index=edge_index)

    return run_benchmark(**kwargs)


def benchmark_scatter(framework, device=None, **kwargs):
    def run_benchmark(edge_index_shape, node_features_shape, scatter_op):
        if framework == "mlx":

            def mlx_scatter(node_features_mlx, edge_index_mlx, scatter_op):
                return mx.eval(
                    scatter(
                        node_features_mlx,
                        edge_index_mlx,
                        edge_index_mlx.max().item() + 2,
                        scatter_op,
                    )
                )

            edge_index = get_dummy_edge_index(
                edge_index_shape, node_features_shape[0], device, "mlx"
            )
            node_features = get_dummy_features(node_features_shape, device, "mlx")
            node_features = node_features[edge_index[0]]

            return measure_runtime(
                mlx_scatter,
                node_features_mlx=node_features,
                edge_index_mlx=edge_index[1],
                scatter_op=scatter_op,
            )
        else:

            def pyg_scatter(node_features, edge_index, scatter_op, device):
                if scatter_op == "sum":
                    scatter_op = "add"
                _ = scatter_torch(node_features, edge_index, 0, reduce=scatter_op)
                sync_mps_if_needed(device)
                # return scatter_sum(node_features, edge_index, 0)

            edge_index = get_dummy_edge_index(
                edge_index_shape, node_features_shape[0], device, "pyg"
            )
            node_features = get_dummy_features(node_features_shape, device, "pyg")
            node_features = node_features[edge_index[0]]

            return measure_runtime(
                pyg_scatter,
                node_features=node_features,
                edge_index=edge_index[1],
                scatter_op=scatter_op,
                device=device,
            )

    return run_benchmark(**kwargs)


def benchmark_gather(framework, device=None, **kwargs):
    def run_benchmark(edge_index_shape, node_features_shape):
        if framework == "mlx":

            def mlx_gather(node_features, edge_index):
                src_val = node_features[edge_index[0]]
                dst_val = node_features[edge_index[1]]
                mx.eval(src_val, dst_val)

            edge_index = get_dummy_edge_index(
                edge_index_shape, node_features_shape[0], device, "mlx"
            )
            node_features = get_dummy_features(node_features_shape, device, "mlx")

            return measure_runtime(
                mlx_gather,
                node_features=node_features,
                edge_index=edge_index,
            )
        else:

            def pyg_gather(node_features, edge_index, device):
                _ = node_features[edge_index[0]]
                _ = node_features[edge_index[1]]
                sync_mps_if_needed(device)

            edge_index = get_dummy_edge_index(
                edge_index_shape, node_features_shape[0], device, "pyg"
            )
            node_features = get_dummy_features(node_features_shape, device, "pyg")

            return measure_runtime(
                pyg_gather,
                node_features=node_features,
                edge_index=edge_index,
                device=device,
            )

    return run_benchmark(**kwargs)


"""
scatter_sum operation used in PyG. Will be removed once benchmarks
of scatter operations is done.
"""


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(
    src: torch.Tensor, index: torch.Tensor, dim: int = -1, out=None, dim_size=None
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)
