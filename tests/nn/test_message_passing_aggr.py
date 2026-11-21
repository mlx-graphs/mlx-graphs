"""`MessagePassing(aggr=...)` with lists of aggregations and Aggregation modules."""

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import (
    MessagePassing,
    SAGEConv,
    SimpleConv,
    fused_message_passing_mode,
)
from mlx_graphs.nn.aggr import (
    MultiAggregation,
    PowerMeanAggregation,
    SoftmaxAggregation,
)
from mlx_graphs.utils import scatter

NUM_NODES, DIM = 12, 5
TOL = dict(rtol=1e-4, atol=1e-5)


@pytest.fixture
def graph():
    rng = np.random.default_rng(0)
    edge_index = rng.integers(0, NUM_NODES - 2, (2, 40))  # nodes 10, 11 isolated
    x = rng.standard_normal((NUM_NODES, DIM)).astype(np.float32)
    return edge_index, x


def test_string_aggregation_is_unchanged():
    conv = SimpleConv(aggr="mean")
    assert conv.aggr == "mean"
    assert "aggr_module" not in conv
    assert tree_flatten(conv.parameters()) == []
    # PyG's name of "add"
    assert SimpleConv(aggr="sum").aggr == "add"


@pytest.mark.parametrize("reduce", ["mul", "var", "std", "median"])
def test_new_string_reductions(graph, reduce):
    edge_index, x = graph
    edge_index, x = mx.array(edge_index), mx.array(x)
    out = SimpleConv(aggr=reduce)(edge_index, x)
    expected = scatter(x[edge_index[0]], edge_index[1], NUM_NODES, reduce)
    np.testing.assert_allclose(np.array(out), np.array(expected), **TOL)


@pytest.mark.parametrize(
    "aggr",
    [
        ["mean", "max"],
        ["sum", "std", "min"],
        "powermean",
        ["median", "var"],
    ],
)
def test_matches_pyg(graph, aggr):
    edge_index, x = graph
    conv = SimpleConv(aggr=aggr)
    pyg_conv = pyg_nn.SimpleConv(aggr=aggr)
    if aggr == "powermean":
        x = np.abs(x) + 0.1
    out = conv(mx.array(edge_index), mx.array(x))
    expected = pyg_conv(torch.tensor(x), torch.tensor(edge_index))
    assert "aggr_module" in conv
    np.testing.assert_allclose(np.array(out), expected.numpy(), **TOL)


def test_softmax_string_is_softmax_aggregation(graph):
    # As in PyG (and as in a list), "softmax" is SoftmaxAggregation, with one
    # output per node, not scatter's per-message normalization.
    edge_index, x = graph
    conv = SimpleConv(aggr="softmax")
    assert isinstance(conv.aggr_module, SoftmaxAggregation)
    out = conv(mx.array(edge_index), mx.array(x))
    expected = pyg_nn.SimpleConv(aggr="softmax")(
        torch.tensor(x), torch.tensor(edge_index)
    )
    assert out.shape == (NUM_NODES, DIM)
    np.testing.assert_allclose(np.array(out), expected.numpy(), **TOL)

    # a layer with a projection after aggregating
    sage = SAGEConv(DIM, 3, aggr="softmax")
    assert sage(mx.array(edge_index), mx.array(x)).shape == (NUM_NODES, 3)


def test_aggr_kwargs_of_string_reduction_are_used(graph):
    # The arguments were silently ignored for scatter reductions.
    edge_index, x = graph
    conv = SimpleConv(aggr="softmax", aggr_kwargs={"t": 0.3, "learn": True})
    params = dict(tree_flatten(conv.trainable_parameters()))
    assert conv.aggr_module.learn and "aggr_module.t" in params
    expected = pyg_nn.SimpleConv(aggr="softmax", aggr_kwargs={"t": 0.3})(
        torch.tensor(x), torch.tensor(edge_index)
    )
    out = conv(mx.array(edge_index), mx.array(x))
    np.testing.assert_allclose(np.array(out), expected.detach().numpy(), **TOL)

    conv = SimpleConv(aggr="std", aggr_kwargs={"semi_grad": True})
    assert conv.aggr_module.semi_grad
    with pytest.raises(TypeError):
        SimpleConv(aggr="mean", aggr_kwargs={"bad_argument": 1})


def test_list_is_concatenation_of_strings(graph):
    edge_index, x = graph
    edge_index, x = mx.array(edge_index), mx.array(x)
    conv = SimpleConv(aggr=["mean", "max"])
    assert conv.aggr == ["mean", "max"]
    assert isinstance(conv.aggr_module, MultiAggregation)
    expected = mx.concatenate(
        [SimpleConv(aggr="mean")(edge_index, x), SimpleConv(aggr="max")(edge_index, x)],
        axis=-1,
    )
    np.testing.assert_allclose(np.array(conv(edge_index, x)), np.array(expected), **TOL)


def test_aggr_kwargs(graph):
    edge_index, x = graph
    edge_index, x = mx.array(edge_index), mx.array(x)
    conv = SimpleConv(aggr=["mean", "max"], aggr_kwargs={"mode": "sum"})
    expected = SimpleConv(aggr="mean")(edge_index, x) + SimpleConv(aggr="max")(
        edge_index, x
    )
    np.testing.assert_allclose(np.array(conv(edge_index, x)), np.array(expected), **TOL)

    conv = SimpleConv(aggr="powermean", aggr_kwargs={"p": 2.0})
    assert isinstance(conv.aggr_module, PowerMeanAggregation)


def test_learnable_aggregation_module(graph):
    edge_index, x = graph
    edge_index, x = mx.array(edge_index), mx.array(x)
    mx.random.seed(0)
    conv = SAGEConv(DIM, 3, aggr=SoftmaxAggregation(learn=True))
    assert isinstance(conv.aggr, str)
    keys = [k for k, _ in tree_flatten(conv.trainable_parameters())]
    assert "aggr_module.t" in keys

    def loss(params):
        conv.update(params)
        return (conv(edge_index, x) ** 2).sum()

    grads = mx.grad(loss)(conv.trainable_parameters())
    assert abs(grads["aggr_module"]["t"].item()) > 0


def test_fused_path_only_for_plain_strings(graph, monkeypatch):
    edge_index, x = graph
    edge_index, x = mx.array(edge_index), mx.array(x)
    calls = {"n": 0}
    original = MessagePassing.message_and_aggregate

    def spy(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MessagePassing, "message_and_aggregate", spy)

    for aggr in [["mean"], SoftmaxAggregation(), "std", "mul", "median", "var"]:
        conv = SimpleConv(aggr=aggr)
        with fused_message_passing_mode(True):
            out = conv(edge_index, x)
        with fused_message_passing_mode(False):
            expected = conv(edge_index, x)
        np.testing.assert_allclose(np.array(out), np.array(expected), **TOL)
    assert calls["n"] == 0

    if mp.HAS_FUSED_KERNELS:
        with fused_message_passing_mode(True):
            SimpleConv(aggr="mean")(edge_index, x)
        assert calls["n"] == 1


def test_invalid_aggregation():
    with pytest.raises(ValueError):
        SimpleConv(aggr="unknown")
    with pytest.raises(ValueError):
        SimpleConv(aggr=3)
    with pytest.raises(ValueError):
        SimpleConv(aggr=["mean", "unknown"])


def test_custom_layer_three_dimensional_messages(graph):
    """Aggregation modules reduce along the first axis of any message shape."""
    edge_index, x = graph

    class Heads(MessagePassing):
        def __call__(self, edge_index, node_features):
            return self.propagate(edge_index, node_features)

    x3 = mx.array(x).reshape(NUM_NODES, 1, DIM)
    x3 = mx.concatenate([x3, 2 * x3], axis=1)
    out = Heads(aggr=["mean", "std"])(mx.array(edge_index), x3)
    assert out.shape == (NUM_NODES, 2, 2 * DIM)
    flat = Heads(aggr=["mean", "std"])(mx.array(edge_index), x3[:, 0])
    np.testing.assert_allclose(np.array(out[:, 0]), np.array(flat), **TOL)


def _concat_aggr_layers():
    import mlx.nn as nn

    import mlx_graphs.nn as gnn

    return {
        "APPNP": lambda **k: gnn.APPNP(2, 0.1, **k),
        "ARMAConv": lambda **k: gnn.ARMAConv(4, 8, **k),
        "CGConv": lambda **k: gnn.CGConv(4, **k),
        "ChebConv": lambda **k: gnn.ChebConv(4, 8, 2, **k),
        "ClusterGCNConv": lambda **k: gnn.ClusterGCNConv(4, 8, **k),
        "FAConv": lambda **k: gnn.FAConv(4, **k),
        "FiLMConv": lambda **k: gnn.FiLMConv(4, 8, **k),
        "GCN2Conv": lambda **k: gnn.GCN2Conv(4, 0.1, **k),
        "GCNConv": lambda **k: gnn.GCNConv(4, 8, **k),
        "GINConv": lambda **k: gnn.GINConv(nn.Linear(4, 8), **k),
        "GINEConv": lambda **k: gnn.GINEConv(nn.Linear(4, 8), **k),
        "GatedGraphConv": lambda **k: gnn.GatedGraphConv(8, 2, **k),
        "GraphConv": lambda **k: gnn.GraphConv(4, 8, **k),
        "LEConv": lambda **k: gnn.LEConv(4, 8, **k),
        "MFConv": lambda **k: gnn.MFConv(4, 8, **k),
        "MixHopConv": lambda **k: gnn.MixHopConv(4, 8, **k),
        "NNConv": lambda **k: gnn.NNConv(4, 8, nn.Linear(4, 32), **k),
        "ResGatedGraphConv": lambda **k: gnn.ResGatedGraphConv(4, 8, **k),
        "SGConv": lambda **k: gnn.SGConv(4, 8, **k),
        "SSGConv": lambda **k: gnn.SSGConv(4, 8, 0.1, **k),
        "TAGConv": lambda **k: gnn.TAGConv(4, 8, **k),
        "TransformerConv": lambda **k: gnn.TransformerConv(4, 8, **k),
        "GATConv": lambda **k: gnn.GATConv(4, 8, **k),
        "GATv2Conv": lambda **k: gnn.GATv2Conv(4, 8, **k),
    }


@pytest.mark.parametrize("name", sorted(_concat_aggr_layers()))
def test_concat_aggr_unsupported_layers_raise_clearly(name):
    """Regression: layers whose projections are sized for one aggregation
    (all PyG layers but SAGEConv and GENConv) failed with a shape error in
    the forward pass, or, for GATConv / GATv2Conv, silently returned twice
    as many rows. They now refuse a concatenating list when built."""
    build = _concat_aggr_layers()[name]
    with pytest.raises(ValueError, match="does not support a list of aggregations"):
        build(aggr=["mean", "std"])
    # combined without changing the size: fine
    conv = build(aggr=["mean", "max"], aggr_kwargs={"mode": "sum"})
    assert not conv.supports_concat_aggr


@pytest.mark.parametrize("name", ["GeneralConv", "RGATConv"])
def test_scatter_only_layers_reject_aggregation_modules(name):
    import mlx_graphs.nn as gnn

    build = {
        "GeneralConv": lambda **k: gnn.GeneralConv(4, 8, **k),
        "RGATConv": lambda **k: gnn.RGATConv(4, 8, 2, **k),
    }[name]
    with pytest.raises(ValueError, match="single scatter aggregation"):
        build(aggr=["mean", "max"])
    build(aggr="mean")


def test_concat_aggr_supported_layers():
    import mlx_graphs.nn as gnn

    rng = np.random.default_rng(0)
    edge_index = mx.array(rng.integers(0, 10, (2, 40)))
    x = mx.array(rng.normal(size=(10, 4)).astype(np.float32))
    # no projection after the aggregation: the output is concatenated
    assert SimpleConv(aggr=["mean", "max"])(edge_index, x).shape == (10, 8)
    # the projection is sized for the concatenation
    assert gnn.SAGEConv(4, 8, aggr=["mean", "max"])(edge_index, x).shape == (10, 8)
    assert gnn.GENConv(4, 8, aggr=["mean", "max"])(edge_index, x).shape == (10, 8)
