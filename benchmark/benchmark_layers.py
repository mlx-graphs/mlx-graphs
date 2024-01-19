import torch
import mlx.core as mx
import mlx.nn as mlx_nn
import mlx_graphs.nn as mlg_nn
import torch_geometric.nn as pyg_nn

from benchmark_utils import (
    get_dummy_edge_index,
    get_dummy_features,
    measure_runtime,
)


class MLXLayer(mlx_nn.Module):
    def __init__(self, layer_cls, **kwargs):
        super().__init__()
        self.layer = layer_cls(**kwargs)
        self.y = []

    def __call__(self, **kwargs):
        for _ in range(5):
            self.y.append(self.layer(**kwargs))
        mx.eval(self.y)


class PyGLayer(torch.nn.Module):
    def __init__(self, layer_cls, **kwargs):
        super().__init__()
        self.layer = layer_cls(**kwargs)
        self.y = []

    def forward(self, **kwargs):
        with torch.no_grad():
            for _ in range(5):
                self.y.append(self.layer(**kwargs))

            self.sync_mps_if_needed()

    def sync_mps_if_needed(self):
        """
        Call this function after every torch implementation to ensure
        the mps execution has finished.
        """
        device = next(self.parameters()).device
        if device == torch.device("mps"):
            torch.mps.synchronize()


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
