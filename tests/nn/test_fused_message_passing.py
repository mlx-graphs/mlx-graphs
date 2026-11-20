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
    # Both paths end in a dense matmul (the layer's Linear). MLX's GPU matmul
    # is only accurate to ~1e-3 relative on some chips, and it turns the
    # float-rounding differences between the two aggregation orders into
    # differences of that size, so the tolerance cannot be tighter than this.
    np.testing.assert_allclose(np.array(a), np.array(b), atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("fmt", ["coo", "csr"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("name", LAYERS)
def test_layer_matches_gather_scatter(graph, count_fused, name, weighted, fmt):
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
    class Plain(MessagePassing):
        def __call__(self, edge_index, node_features):
            return self.propagate(edge_index, node_features)

    edge_index, x, _ = graph
    Plain(aggr="add")(edge_index, x)
    assert count_fused["n"] == 0


def test_multi_head_weights_are_fused(graph, count_fused):
    # GAT-style: [num_nodes, heads, channels] features with one weight per
    # (edge, head) run as one SpMM per head.
    edge_index, x, _ = graph
    heads = 4
    x = x.reshape(NUM_NODES, heads, DIM // heads)
    alpha = mx.random.uniform(shape=(NUM_EDGES, heads), key=mx.random.key(2))
    layer = SimpleConv(aggr="add")

    def run(x, alpha):
        return layer.propagate(edge_index, x, message_kwargs={"edge_weights": alpha})

    def loss(x, alpha):
        return (run(x, alpha) ** 2).sum()

    results = {}
    for fmt in ("coo", "csr"):
        clear_csr_cache()
        with fused_message_passing_mode(True, format=fmt):
            results[fmt] = (run(x, alpha), *mx.grad(loss, argnums=[0, 1])(x, alpha))
    assert count_fused["n"] > 0
    count_fused["n"] = 0
    with fused_message_passing_mode(False):
        reference = (run(x, alpha), *mx.grad(loss, argnums=[0, 1])(x, alpha))
    assert count_fused["n"] == 0

    assert reference[0].shape == (NUM_NODES, heads, DIM // heads)
    for fmt in ("coo", "csr"):
        for a, b in zip(results[fmt], reference):
            np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-4, rtol=1e-4)


def test_auto_format_switches_to_csr(graph, monkeypatch):
    edge_index, x, _ = graph
    layer = SimpleConv(aggr="add")
    clear_csr_cache()
    with fused_message_passing_mode(True, format="auto"):
        outs = [layer(edge_index, x) for _ in range(mp._AUTO_CSR_MIN_USES + 1)]
    entry = next(iter(mp._csr_cache._entries.values()))
    assert entry.uses == mp._AUTO_CSR_MIN_USES + 1
    assert entry.csr is not None  # built once the edge_index was reused enough
    for out in outs[1:]:
        np.testing.assert_allclose(np.array(out), np.array(outs[0]), atol=1e-5)

    # a large graph converts on first use
    clear_csr_cache()
    monkeypatch.setattr(mp, "_AUTO_CSR_MIN_EDGES", NUM_EDGES)
    with fused_message_passing_mode(True, format="auto"):
        layer(edge_index, x)
    assert next(iter(mp._csr_cache._entries.values())).csr is not None


def test_auto_format_inside_compile(graph, monkeypatch):
    # The CSR conversion cannot run while tracing; "auto" must fall back to COO.
    edge_index, x, _ = graph
    layer = SimpleConv(aggr="add")
    clear_csr_cache()
    monkeypatch.setattr(mp, "_AUTO_CSR_MIN_EDGES", 1)
    with fused_message_passing_mode(True, format="auto"):
        out = mx.compile(lambda e, x: layer(e, x))(edge_index, x)
        mx.eval(out)
    with fused_message_passing_mode(False):
        close(out, layer(edge_index, x))


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

    assert Custom._fused_format == "auto"


class DoubledConv(SimpleConv):
    """Sends twice the source features along each edge."""

    def message(self, src_features, dst_features, **kwargs):
        return 2 * src_features


def test_subclass_message_override_falls_back(graph, count_fused):
    edge_index, x, _ = graph
    assert DoubledConv._fused_format is None
    assert SimpleConv._fused_format == "auto"  # parent untouched
    out = DoubledConv(aggr="add")(edge_index, x)
    assert count_fused["n"] == 0
    with fused_message_passing_mode(False):
        close(out, 2 * SimpleConv(aggr="add")(edge_index, x))


def test_subclass_aggregate_override_falls_back():
    class MaxConv(SimpleConv):
        def aggregate(self, messages, indices, **kwargs):
            return super().aggregate(messages, indices, **kwargs)

    assert MaxConv._fused_format is None


def test_subclass_without_override_stays_fused(graph, count_fused):
    class PlainGCN(GCNConv):
        pass

    class PlainGIN(GINConv):  # GINConv's own allowed message override
        pass

    assert PlainGCN._fused_format == "auto"
    assert PlainGIN._fused_format == "auto"
    edge_index, x, _ = graph
    mx.random.seed(0)
    PlainGCN(DIM, 8)(edge_index, x)
    assert count_fused["n"] == 1


def test_subclass_with_custom_fused_impl_stays_fused():
    class Custom(SimpleConv):
        def message(self, src_features, dst_features, **kwargs):
            return 2 * src_features

        def message_and_aggregate(self, edge_index, node_features, **kwargs):
            return 2 * super().message_and_aggregate(edge_index, node_features)

    assert Custom._fused_format == "auto"


def test_redecorated_subclass_uses_decorator_format():
    @fused_message_passing(format="csr", allow_message_override=True)
    class Redecorated(SimpleConv):
        def message(self, src_features, dst_features, **kwargs):
            return src_features

    assert Redecorated._fused_format == "csr"


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
