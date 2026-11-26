"""Which kernel computes GATv2Conv's attention scores, and that both agree.

On the fused path, GATv2Conv uses mlx_cluster.gatv2_scores on the Metal GPU
for graphs with at least 1M edges, the in-tree Metal kernel otherwise;
set_gatv2_score_kernel / gatv2_score_kernel_mode override the choice. The
threshold and the "on the Metal GPU" check are patched here so every branch
runs on any machine.
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

import mlx_graphs.nn.conv.gatv2_conv as gatv2
import mlx_graphs.nn.message_passing as mp
from mlx_graphs.nn import (
    GATv2Conv,
    fused_message_passing_mode,
    gatv2_score_kernel_mode,
    set_gatv2_score_kernel,
)

needs_gatv2_scores = pytest.mark.skipif(
    not (mp.HAS_FUSED_KERNELS and gatv2.HAS_GATV2_SCORES),
    reason="needs an mlx-cluster with gatv2_scores",
)


@pytest.fixture
def calls(monkeypatch):
    """Counts the calls of each score kernel (and still runs them)."""
    counts = {"mlx_cluster": 0, "metal": 0}
    original_metal = gatv2.gatv2_edge_scores
    original_cluster = getattr(gatv2, "_mlx_cluster_gatv2_scores", None)

    def metal(*args, **kwargs):
        counts["metal"] += 1
        return original_metal(*args, **kwargs)

    def cluster(*args, **kwargs):
        counts["mlx_cluster"] += 1
        return original_cluster(*args, **kwargs)

    monkeypatch.setattr(gatv2, "gatv2_edge_scores", metal)
    if original_cluster is not None:
        monkeypatch.setattr(gatv2, "_mlx_cluster_gatv2_scores", cluster)
    mp.clear_csr_cache()
    yield counts
    mp.clear_csr_cache()


def graph(num_nodes=40, num_edges=300, seed=0):
    rng = np.random.default_rng(seed)
    edge_index = mx.array(rng.integers(0, num_nodes, (2, num_edges)).astype(np.int32))
    x = mx.random.normal((num_nodes, 16), key=mx.random.key(seed))
    return edge_index, x


def layer(**kwargs):
    mx.random.seed(0)
    return GATv2Conv(16, 8, heads=4, **kwargs)


def test_invalid_kernel():
    with pytest.raises(ValueError, match="'auto', 'metal' or 'mlx_cluster'"):
        set_gatv2_score_kernel("cuda")


def test_mlx_cluster_kernel_needs_gatv2_scores(monkeypatch):
    monkeypatch.setattr(gatv2, "HAS_GATV2_SCORES", False)
    with pytest.raises(ValueError, match="gatv2_scores"):
        set_gatv2_score_kernel("mlx_cluster")
    set_gatv2_score_kernel("metal")  # always available
    set_gatv2_score_kernel("auto")


def test_mode_restores_the_previous_kernel():
    assert gatv2._score_kernel == "auto"
    with pytest.raises(RuntimeError):
        with gatv2_score_kernel_mode("metal"):
            assert gatv2._score_kernel == "metal"
            raise RuntimeError
    assert gatv2._score_kernel == "auto"


@needs_gatv2_scores
@pytest.mark.parametrize(
    "case, kernel, num_edges, on_metal, expected",
    [
        ("auto, small graph", "auto", 300, True, "metal"),
        ("auto, large graph", "auto", 1000, True, "mlx_cluster"),
        ("auto, large graph, not on Metal", "auto", 1000, False, "metal"),
        ("forced metal, large graph", "metal", 1000, True, "metal"),
        ("forced mlx_cluster, small graph", "mlx_cluster", 300, False, "mlx_cluster"),
    ],
)
def test_kernel_choice(calls, monkeypatch, case, kernel, num_edges, on_metal, expected):
    monkeypatch.setattr(gatv2, "_MLX_CLUSTER_MIN_EDGES", 1000)  # 1M in real use
    monkeypatch.setattr(gatv2, "_on_metal", lambda: on_metal)
    edge_index, x = graph(num_edges=num_edges)
    with gatv2_score_kernel_mode(kernel):
        mx.eval(layer(add_self_loops=False)(edge_index, x))
    assert calls == {
        "mlx_cluster": int(expected == "mlx_cluster"),
        "metal": int(expected == "metal"),
    }, case


@needs_gatv2_scores
def test_edge_features_use_the_in_tree_kernel(calls):
    edge_index, x = graph()
    edge_features = mx.random.normal((edge_index.shape[1], 3), key=mx.random.key(1))
    conv = layer(edge_features_dim=3, add_self_loops=False)
    with gatv2_score_kernel_mode("mlx_cluster"):
        mx.eval(conv(edge_index, x, edge_features))
    assert calls == {"mlx_cluster": 0, "metal": 1}


@needs_gatv2_scores
def test_coo_format_uses_the_in_tree_kernel(calls):
    edge_index, x = graph()
    with gatv2_score_kernel_mode("mlx_cluster"), fused_message_passing_mode(
        True, format="coo"
    ):
        mx.eval(layer(add_self_loops=False)(edge_index, x))
    assert calls == {"mlx_cluster": 0, "metal": 1}


@needs_gatv2_scores
def test_unfused_path_uses_neither_kernel(calls):
    edge_index, x = graph()
    with gatv2_score_kernel_mode("mlx_cluster"), fused_message_passing_mode(False):
        mx.eval(layer()(edge_index, x))
    assert calls == {"mlx_cluster": 0, "metal": 0}


@needs_gatv2_scores
@pytest.mark.parametrize("add_self_loops", [True, False])
@pytest.mark.parametrize("bipartite", [False, True])
def test_kernels_agree(calls, add_self_loops, bipartite):
    """Outputs and gradients (parameters and features) on both kernels and on
    the gather-scatter reference."""
    if bipartite and add_self_loops:
        pytest.skip("self-loops are not supported on bipartite inputs")
    edge_index, x = graph()
    inputs = (x, x[:30]) if bipartite else x
    if bipartite:
        edge_index = mx.stack([edge_index[0], edge_index[1] % 30])

    def run(context):
        conv = layer(add_self_loops=add_self_loops)
        cot = mx.random.normal((30 if bipartite else 40, 32), key=mx.random.key(2))

        def loss(params, inputs):
            conv.update(params)
            return (conv(edge_index, inputs) * cot).sum()

        with context:
            value, grads = mx.value_and_grad(loss, argnums=(0, 1))(
                conv.parameters(), inputs
            )
            mx.eval(value, grads)
        return value, grads

    results = {
        "mlx_cluster": run(gatv2_score_kernel_mode("mlx_cluster")),
        "metal": run(gatv2_score_kernel_mode("metal")),
        "reference": run(fused_message_passing_mode(False)),
    }
    assert calls["mlx_cluster"] >= 1 and calls["metal"] >= 1

    def flat(result):
        value, grads = result
        return [("loss", value), *tree_flatten(grads)]

    want = flat(results["reference"])
    for name in ("mlx_cluster", "metal"):
        for (key, got), (_, expected) in zip(flat(results[name]), want):
            np.testing.assert_allclose(
                np.array(got),
                np.array(expected),
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"{name}: {key}",
            )


@needs_gatv2_scores
def test_mlx_cluster_shares_the_aggregation_csr(calls, monkeypatch):
    """The scores and the aggregation of a call use one CSR, built once for a
    reused edge_index: the mlx_cluster path adds no sort."""
    conversions = []
    original = mp._coo_to_csr
    monkeypatch.setattr(
        mp, "_coo_to_csr", lambda *a, **k: conversions.append(1) or original(*a, **k)
    )
    edge_index, x = graph()
    conv = layer(add_self_loops=False)
    with gatv2_score_kernel_mode("mlx_cluster"):
        for _ in range(3):
            mx.eval(conv(edge_index, x))
    assert calls["mlx_cluster"] == 3
    assert len(conversions) == 1
