"""Fused (SpMM) message passing must match gather-scatter exactly.

Every decorated layer is run twice on the same parameters, once with
fused message passing and once with it disabled, comparing outputs and
gradients. Skipped when mlx-cluster with SpMM kernels is not installed.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import (
    GATConv,
    GCNConv,
    GINConv,
    MessagePassing,
    SAGEConv,
    SimpleConv,
    clear_csr_cache,
    fused_message_passing,
    fused_message_passing_mode,
)

pytestmark = pytest.mark.skipif(
    not mp.HAS_FUSED_KERNELS, reason="mlx_cluster with SpMM kernels not installed"
)

NUM_NODES, NUM_EDGES, DIM = 60, 400, 16


@pytest.fixture
def graph():
    rng = np.random.default_rng(0)
    edge_index = mx.array(rng.integers(0, NUM_NODES, (2, NUM_EDGES)))
    x = mx.random.normal((NUM_NODES, DIM), key=mx.random.key(0))
    weights = mx.random.uniform(shape=(NUM_EDGES,), key=mx.random.key(1))
    return edge_index, x, weights


@pytest.fixture
def count_fused(monkeypatch):
    """Counts calls to message_and_aggregate, to prove which path ran."""
    calls = {"n": 0}
    original = MessagePassing.message_and_aggregate

    def spy(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MessagePassing, "message_and_aggregate", spy)
    return calls


def make_layer(name):
    mx.random.seed(0)
    return {
        "gcn": lambda: GCNConv(DIM, 8),
        "gcn_self_loops": lambda: GCNConv(DIM, 8, add_self_loops=True),
        "sage_down": lambda: SAGEConv(DIM, 8),  # projection before propagate
        "sage_up": lambda: SAGEConv(DIM, 32),  # projection after propagate
        "simple_add": lambda: SimpleConv(aggr="add"),
        "simple_max": lambda: SimpleConv(aggr="max"),
        "simple_min": lambda: SimpleConv(aggr="min"),
        "simple_mean_self_loop": lambda: SimpleConv(
            aggr="mean", combine_root_func="self_loop"
        ),
        "gin": lambda: GINConv(nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU())),
    }[name]()


LAYERS = [
    "gcn",
    "gcn_self_loops",
    "sage_down",
    "sage_up",
    "simple_add",
    "simple_max",
    "simple_min",
    "simple_mean_self_loop",
    "gin",
]


def close(a, b):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("fmt", ["coo", "csr"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("name", LAYERS)
def test_layer_matches_gather_scatter(graph, count_fused, name, weighted, fmt):
    if name == "gcn_self_loops" and weighted:
        pytest.skip(
            "pre-existing GCNConv bug: add_self_loops does not extend edge_weights"
        )
    edge_index, x, weights = graph
    layer = make_layer(name)
    w = weights if weighted else None
    has_params = len(nn.utils.tree_flatten(layer.trainable_parameters())) > 0
    clear_csr_cache()

    def loss(model, x):
        return (model(edge_index, x, edge_weights=w) ** 2).sum()

    def grad_fn(model, x):
        if not has_params:
            return None, {}
        return nn.value_and_grad(model, loss)(model, x)

    x_grad_fn = mx.grad(lambda x: loss(layer, x))

    with fused_message_passing_mode(True, format=fmt):
        out_f = layer(edge_index, x, edge_weights=w)
        _, g_f = grad_fn(layer, x)
        gx_f = x_grad_fn(x)
    assert count_fused["n"] > 0, "fused path was not taken"

    count_fused["n"] = 0
    with fused_message_passing_mode(False):
        out_r = layer(edge_index, x, edge_weights=w)
        _, g_r = grad_fn(layer, x)
        gx_r = x_grad_fn(x)
    assert count_fused["n"] == 0, "gather-scatter path was not taken"

    close(out_f, out_r)
    close(gx_f, gx_r)
    for (k, a), (_, b) in zip(nn.utils.tree_flatten(g_f), nn.utils.tree_flatten(g_r)):
        close(a, b)


def test_falls_back_for_unsupported_aggr(graph, count_fused):
    edge_index, x, _ = graph
    out = SimpleConv(aggr="softmax")(edge_index, x)
    assert count_fused["n"] == 0
    with fused_message_passing_mode(False):
        close(out, SimpleConv(aggr="softmax")(edge_index, x))


def test_falls_back_for_edge_features(graph, count_fused):
    # GINE: messages are relu(x_j + proj(e)), not a weighted sum
    edge_index, x, _ = graph
    mx.random.seed(0)
    layer = GINConv(
        nn.Sequential(nn.Linear(DIM, DIM)), node_features_dim=DIM, edge_features_dim=4
    )
    edge_features = mx.random.normal((NUM_EDGES, 4), key=mx.random.key(3))
    out = layer(edge_index, x, edge_features=edge_features)
    assert count_fused["n"] == 0
    with fused_message_passing_mode(False):
        close(out, layer(edge_index, x, edge_features=edge_features))


def test_disabled_globally(graph, count_fused):
    edge_index, x, _ = graph
    with fused_message_passing_mode(False):
        GCNConv(DIM, 8)(edge_index, x)
    assert count_fused["n"] == 0


def test_undecorated_layer_unchanged(graph, count_fused):
    edge_index, x, _ = graph
    GATConv(DIM, 8)(edge_index, x)
    assert count_fused["n"] == 0


def test_decorator_rejects_message_override():
    with pytest.raises(TypeError, match="overrides message"):

        @fused_message_passing
        class Bad(MessagePassing):
            def message(self, src_features, dst_features, **kwargs):
                return src_features * dst_features


def test_decorator_accepts_custom_fused_impl():
    @fused_message_passing
    class Custom(MessagePassing):
        def message(self, src_features, dst_features, **kwargs):
            return 2 * src_features

        def message_and_aggregate(self, edge_index, node_features, **kwargs):
            return 2 * super().message_and_aggregate(edge_index, node_features)

    assert Custom._fused_format == "coo"


def test_decorator_rejects_non_message_passing():
    with pytest.raises(TypeError):

        @fused_message_passing
        class NotALayer:
            pass


def test_decorator_bad_format():
    with pytest.raises(ValueError):
        fused_message_passing(format="ell")


def test_csr_cache_reused(graph):
    edge_index, x, _ = graph
    clear_csr_cache()
    layer = SimpleConv(aggr="add")
    with fused_message_passing_mode(True, format="csr"):
        layer(edge_index, x)
        layer(edge_index, x)
    assert len(mp._csr_cache._entries) == 1


@pytest.mark.parametrize("fmt", ["coo", "csr"])
@pytest.mark.parametrize(
    "make",
    [
        lambda: SAGEConv(8, 5),
        lambda: SimpleConv(aggr="add"),
        lambda: SimpleConv(aggr="mean"),
        lambda: SimpleConv(aggr="max"),
    ],
    ids=["sage", "simple_add", "simple_mean", "simple_max"],
)
def test_bipartite_matches_gather_scatter(count_fused, fmt, make):
    # 7 source nodes -> 4 destination nodes: output has one row per dst node
    rng = np.random.default_rng(0)
    edge_index = mx.array(np.stack([rng.integers(0, 7, 30), rng.integers(0, 4, 30)]))
    xs = mx.random.normal((7, 8), key=mx.random.key(0))
    xd = mx.random.normal((4, 8), key=mx.random.key(1))
    mx.random.seed(0)
    layer = make()
    clear_csr_cache()
    with fused_message_passing_mode(True, format=fmt):
        out_f = layer(edge_index, (xs, xd))
        gx_f = mx.grad(lambda a: layer(edge_index, (a, xd)).sum())(xs)
    assert count_fused["n"] > 0
    with fused_message_passing_mode(False):
        out_r = layer(edge_index, (xs, xd))
        gx_r = mx.grad(lambda a: layer(edge_index, (a, xd)).sum())(xs)
    assert out_f.shape == (4, out_r.shape[1])
    close(out_f, out_r)
    close(gx_f, gx_r)
