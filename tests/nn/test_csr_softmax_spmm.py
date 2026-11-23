import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_graphs.nn import (
    GATConv,
    GATv2Conv,
    clear_csr_cache,
    fused_message_passing_mode,
    is_fused_message_passing_enabled,
    message_passing,
)
from mlx_graphs.nn.kernels.csr_softmax_spmm import csr_softmax_spmm
from mlx_graphs.utils import scatter


@pytest.mark.parametrize(
    "channels, reorder, device",
    [(1, False, mx.gpu), (17, True, mx.gpu), (65, True, mx.gpu), (5, True, mx.cpu)],
)
def test_csr_softmax_spmm_outputs_and_gradients(channels, reorder, device):
    if device == mx.gpu and not mx.metal.is_available():
        pytest.skip("Metal required")
    rng = np.random.default_rng(41)
    degrees = [0, 72, 2, 0, 9, 1, 0]
    rows = len(degrees)
    edges, sources, heads = sum(degrees), 16, 3

    with mx.stream(device):
        rowptr = mx.array([0, *np.cumsum(degrees).tolist()], dtype=mx.int64)
        row = mx.array(np.repeat(np.arange(rows), degrees), dtype=mx.int32)
        col = mx.array(rng.integers(sources - 2, size=edges), dtype=mx.int64)
        perm = mx.array(rng.permutation(edges), dtype=mx.int64) if reorder else None
        logits = mx.array(rng.normal(size=(heads, edges)), dtype=mx.float32).T + 1e4
        # Mask a few edges, as in the GAT self-loop path.
        logits = logits.at[9:12].add(-float("inf"))
        values = mx.array(
            rng.normal(size=(sources, heads, channels * 2)), dtype=mx.float32
        )[..., ::2]
        cotangent = mx.array(rng.normal(size=(rows, heads, channels)), dtype=mx.float32)

        def fused(scores, feats):
            return csr_softmax_spmm(rowptr, col, scores, feats, permutation=perm)

        def reference(scores, feats):
            scores = scores if perm is None else scores[perm]
            weights = scatter(scores, row, rows, aggr="softmax")
            return scatter(weights[..., None] * feats[col], row, rows)

        out, grads = mx.vjp(fused, (logits, values), (cotangent,))
        expected, expected_grads = mx.vjp(reference, (logits, values), (cotangent,))
        for actual, want in zip([*out, *grads], [*expected, *expected_grads]):
            np.testing.assert_allclose(actual, want, atol=3e-6, rtol=3e-5)
        np.testing.assert_array_equal(np.array(out[0])[[0, 3, 6]], 0)
        np.testing.assert_array_equal(np.array(grads[1])[-2:], 0)


def test_csr_softmax_spmm_empty_graph():
    rowptr = mx.zeros((4,), dtype=mx.int32)
    col = mx.zeros((0,), dtype=mx.int32)
    logits = mx.zeros((0, 2))
    values = mx.ones((5, 2, 8))
    out = csr_softmax_spmm(rowptr, col, logits, values)
    assert out.shape == (3, 2, 8)
    np.testing.assert_array_equal(out, 0)


@pytest.mark.parametrize("layer", [GATConv, GATv2Conv])
@pytest.mark.parametrize("bipartite", [False, True])
def test_attention_layer_softmax_spmm(layer, bipartite, monkeypatch):
    if not mx.metal.is_available() or not is_fused_message_passing_enabled():
        pytest.skip("Metal and fused SpMM required")
    conv = layer(12, 16, heads=3, add_self_loops=not bipartite)
    src = mx.random.normal((20, 12), key=mx.random.key(5))
    inputs = (src, src[:10]) if bipartite else src
    rng = np.random.default_rng(67)
    edges = mx.array(np.stack((rng.integers(20, size=137), rng.integers(10, size=137))))

    calls = []
    original = message_passing.csr_softmax_spmm

    def track(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(message_passing, "csr_softmax_spmm", track)

    def loss(model):
        return model(edges, inputs).square().mean()

    clear_csr_cache()
    with fused_message_passing_mode(True, format="csr"):
        expected = conv(edges, inputs)
        _, expected_grads = nn.value_and_grad(conv, loss)(conv)
    assert not calls

    with fused_message_passing_mode(True, format="csr", softmax_spmm=True):
        actual = conv(edges, inputs)
        _, grads = nn.value_and_grad(conv, loss)(conv)
    assert calls
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    for (name, grad), (_, expected_grad) in zip(
        nn.utils.tree_flatten(grads), nn.utils.tree_flatten(expected_grads)
    ):
        np.testing.assert_allclose(
            grad, expected_grad, rtol=3e-3, atol=3e-5, err_msg=name
        )


@pytest.mark.parametrize(
    "aggr, dropout, fmt", [("add", 0.3, "csr"), ("mean", 0, "csr"), ("add", 0, "coo")]
)
def test_attention_layer_keeps_unfused_semantics(aggr, dropout, fmt, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("The experimental kernel must preserve this aggregation path")

    monkeypatch.setattr(message_passing, "csr_softmax_spmm", unexpected)
    conv = GATConv(6, 8, aggr=aggr, dropout=dropout, add_self_loops=False)
    edges = mx.array([[0, 1, 2], [1, 1, 2]])
    values = mx.ones((4, 6))
    with fused_message_passing_mode(True, format=fmt, softmax_spmm=True):
        mx.eval(conv(edges, values))
