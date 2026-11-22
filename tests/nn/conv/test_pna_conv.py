import mlx.core as mx
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten
from torch_geometric.data import Data

from mlx_graphs.data import GraphData
from mlx_graphs.nn import PNAConv

NUM_NODES, DIM, EDGE_DIM = 15, 8, 3
# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)

AGGREGATORS = ["mean", "min", "max", "std"]
SCALERS = ["identity", "amplification", "attenuation"]


def make_graph(seed=0):
    """Random graph with two isolated nodes (13, 14)."""
    rng = np.random.default_rng(seed)
    edge_index = rng.integers(0, NUM_NODES - 2, (2, 50))
    x = rng.standard_normal((NUM_NODES, DIM)).astype(np.float32)
    edge_features = rng.standard_normal((50, EDGE_DIM)).astype(np.float32)
    return edge_index, x, edge_features


def copy_linear(mlx_lin, torch_lin):
    mlx_lin.weight = mx.array(torch_lin.weight.detach().numpy())
    mlx_lin.bias = mx.array(torch_lin.bias.detach().numpy())


def copy_weights(conv, pyg_conv):
    for mlx_seqs, torch_seqs in [
        (conv.pre_nns, pyg_conv.pre_nns),
        (conv.post_nns, pyg_conv.post_nns),
    ]:
        for mlx_seq, torch_seq in zip(mlx_seqs, torch_seqs):
            for mlx_layer, torch_layer in zip(mlx_seq.layers, torch_seq):
                if hasattr(torch_layer, "weight"):
                    copy_linear(mlx_layer, torch_layer)
    copy_linear(conv.lin, pyg_conv.lin)
    if "edge_encoder" in conv:
        copy_linear(conv.edge_encoder, pyg_conv.edge_encoder)


def degree_histogram(edge_index):
    return PNAConv.get_degree_histogram([(mx.array(edge_index), NUM_NODES)])


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"towers": 2},
        {"towers": 4, "divide_input": True},
        {"pre_layers": 2, "post_layers": 3},
        {"edge_dim": EDGE_DIM, "towers": 2},
        {"train_norm": True},
    ],
)
def test_matches_pyg(kwargs):
    edge_index, x, edge_features = make_graph()
    deg = degree_histogram(edge_index)
    out_dim = 8

    torch.manual_seed(0)
    pyg_conv = pyg_nn.PNAConv(
        DIM, out_dim, AGGREGATORS, SCALERS, torch.tensor(np.array(deg)), **kwargs
    )
    mlx_kwargs = dict(kwargs)
    if "edge_dim" in mlx_kwargs:
        mlx_kwargs["edge_features_dim"] = mlx_kwargs.pop("edge_dim")
    conv = PNAConv(DIM, out_dim, AGGREGATORS, SCALERS, deg, **mlx_kwargs)
    copy_weights(conv, pyg_conv)

    use_edges = "edge_dim" in kwargs
    out = conv(
        mx.array(edge_index),
        mx.array(x),
        mx.array(edge_features) if use_edges else None,
    )
    expected = pyg_conv(
        torch.tensor(x),
        torch.tensor(edge_index),
        torch.tensor(edge_features) if use_edges else None,
    )
    assert out.shape == (NUM_NODES, out_dim)
    np.testing.assert_allclose(np.array(out), expected.detach().numpy(), **TOL)


def test_degree_histogram_matches_pyg():
    graphs = [make_graph(seed)[0] for seed in range(3)]
    pyg_hist = pyg_nn.PNAConv.get_degree_histogram(
        [Data(edge_index=torch.tensor(e), num_nodes=NUM_NODES) for e in graphs]
    )
    hist = PNAConv.get_degree_histogram(
        [GraphData(edge_index=mx.array(e), num_nodes=NUM_NODES) for e in graphs]
    )
    np.testing.assert_array_equal(np.array(hist), pyg_hist.numpy())
    single = PNAConv.get_degree_histogram(
        GraphData(edge_index=mx.array(graphs[0]), num_nodes=NUM_NODES)
    )
    assert int(np.array(single).sum()) == NUM_NODES


def test_parameters_and_training():
    edge_index, x, edge_features = make_graph()
    deg = degree_histogram(edge_index)
    conv = PNAConv(
        DIM, 8, AGGREGATORS, SCALERS, deg, towers=2, edge_features_dim=EDGE_DIM
    )
    keys = [k for k, _ in tree_flatten(conv.trainable_parameters())]
    assert "edge_encoder.weight" in keys
    assert sum(k.startswith("pre_nns.") for k in keys) == 4  # 2 towers x (w, b)
    assert not any(k.startswith("aggr_module") for k in keys)

    edge_index, x, e = mx.array(edge_index), mx.array(x), mx.array(edge_features)

    def loss(params):
        conv.update(params)
        return (conv(edge_index, x, e) ** 2).mean()

    grads = mx.grad(loss)(conv.trainable_parameters())
    assert all(np.isfinite(np.array(g)).all() for _, g in tree_flatten(grads))


def test_invalid_arguments():
    deg = [0, 3, 2]
    with pytest.raises(ValueError):
        PNAConv(DIM, 6, AGGREGATORS, SCALERS, deg, towers=4)
    with pytest.raises(ValueError):
        PNAConv(6, 8, AGGREGATORS, SCALERS, deg, towers=4, divide_input=True)
    conv = PNAConv(DIM, 8, AGGREGATORS, SCALERS, deg)
    edge_index, x, edge_features = make_graph()
    with pytest.raises(ValueError):
        conv(mx.array(edge_index), mx.array(x), mx.array(edge_features))
