"""TransformerConv's per-edge attention scores (``scale * (a[row] * b[col]).sum(-1)``).

The cases of ``mlx_cluster``'s ``edge_scores`` tests, for the in-tree kernel:
the Metal path against the reference, the dispatch to the reference, and the
dispatch through ``message_passing.edge_scores``.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.nn import fused_message_passing_mode, message_passing
from mlx_graphs.nn.kernels import transformer_edge_scores as kernels
from mlx_graphs.nn.kernels.transformer_edge_scores import (
    _metal_scores,
    _reference_scores,
    transformer_edge_scores,
)

metal_only = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
TOL = dict(rtol=1e-5, atol=1e-5)


def _assert_close(got, want, **tol):
    np.testing.assert_allclose(np.array(got), np.array(want), **(tol or TOL))


def _edges(num_a, num_b, num_edges, seed=0, dtype=mx.int64):
    rng = np.random.default_rng(seed)
    row = mx.array(rng.integers(0, num_a, num_edges)).astype(dtype)
    col = mx.array(rng.integers(0, num_b, num_edges)).astype(dtype)
    return row, col


def _features(*shapes):
    return [mx.random.normal(s, key=mx.random.key(i)) for i, s in enumerate(shapes)]


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("channels", [7, 32, 77])
def test_scores_and_gradients_match_reference(heads, channels):
    """Whatever path the dispatcher picks on this machine."""
    row, col = _edges(40, 40, 300)
    a, b = _features((40, heads, channels), (40, heads, channels))
    scale = 1.0 / math.sqrt(channels)
    cotangent = mx.random.normal((300, heads), key=mx.random.key(9))

    def scores(a, b):
        return transformer_edge_scores(row, col, a, b, scale)

    def reference(a, b):
        return _reference_scores(row, col, a, b, scale)

    out, grads = mx.vjp(scores, (a, b), (cotangent,))
    want, want_grads = mx.vjp(reference, (a, b), (cotangent,))
    assert out[0].shape == (300, heads)
    for got, expected in zip([*out, *grads], [*want, *want_grads]):
        _assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_two_dimensional_inputs():
    row, col = _edges(25, 25, 120)
    a, b = _features((25, 16), (25, 16))
    out = transformer_edge_scores(row, col, a, b)
    assert out.shape == (120,)
    _assert_close(out, _reference_scores(row, col, a, b), rtol=1e-4, atol=1e-4)


def test_bipartite():
    # a and b have different numbers of rows
    row, col = _edges(10, 50, 200)
    a, b = _features((10, 2, 8), (50, 2, 8))
    _assert_close(
        transformer_edge_scores(row, col, a, b),
        _reference_scores(row, col, a, b),
        rtol=1e-4,
        atol=1e-4,
    )


def test_no_edges():
    row = mx.zeros((0,), dtype=mx.int64)
    (a,) = _features((5, 3, 4))
    out = transformer_edge_scores(row, row, a, a)
    assert out.shape == (0, 3)
    grad = mx.grad(lambda a: transformer_edge_scores(row, row, a, a).sum())(a)
    _assert_close(grad, mx.zeros_like(a))


@metal_only
@pytest.mark.parametrize("channels", [1, 31, 32, 33, 96])
@pytest.mark.parametrize("index_dtype", [mx.int32, mx.int64, mx.uint32])
@pytest.mark.parametrize("scale", [1.0, 0.125])
def test_metal_scores_and_gradients(channels, index_dtype, scale):
    # duplicate edges and self loops, where atomics meet in the backward pass
    row, col = _edges(9, 7, 64, seed=1, dtype=index_dtype)
    row = mx.concatenate([row, row[:8], mx.arange(5).astype(index_dtype)])
    col = mx.concatenate([col, col[:8], mx.arange(5).astype(index_dtype)])
    a, b = _features((9, 3, channels), (7, 3, channels))
    cotangent = mx.random.normal((row.shape[0], 3), key=mx.random.key(5))
    metal = _metal_scores(scale)

    def reference(a, b):
        return _reference_scores(row, col, a, b, scale)

    out, grads = mx.vjp(lambda a, b: metal(row, col, a, b), (a, b), (cotangent,))
    want, want_grads = mx.vjp(reference, (a, b), (cotangent,))
    for got, expected in zip([*out, *grads], [*want, *want_grads]):
        _assert_close(got, expected)


@metal_only
def test_metal_negative_indices_wrap():
    row = mx.array([-1, 0, -3])
    col = mx.array([1, -2, 0])
    a, b = _features((3, 2, 8), (4, 2, 8))
    wrapped_row, wrapped_col = mx.array([2, 0, 0]), mx.array([1, 2, 0])
    _assert_close(
        _metal_scores(1.0)(row, col, a, b),
        _reference_scores(wrapped_row, wrapped_col, a, b),
    )


@metal_only
def test_metal_score_transforms():
    """Compile, forward mode and second order, the latter two through the
    reference rules."""
    row, col = _edges(6, 6, 30, seed=2)
    a, b = _features((6, 2, 8), (6, 2, 8))
    metal = _metal_scores(0.5)

    def fused(a, b):
        return metal(row, col, a, b)

    def reference(a, b):
        return _reference_scores(row, col, a, b, 0.5)

    def transformed(scores):
        def loss(a, b):
            return scores(a, b).square().sum()

        def grad_sum(a):
            return mx.grad(loss)(a, b).sum()

        primals = (a, b)
        tangents = [mx.ones_like(x) for x in primals]
        _, (jvp,) = mx.jvp(scores, primals, tangents)
        return mx.compile(scores)(*primals), jvp, mx.grad(grad_sum)(a)

    for got, want in zip(transformed(fused), transformed(reference)):
        _assert_close(got, want)


@pytest.mark.parametrize(
    "case", ["cpu", "float16", "empty", "mixed-index-dtypes", "different-channels"]
)
def test_dispatch_falls_back(monkeypatch, case):
    def unexpected_metal(scale):
        pytest.fail("The Metal score kernel was selected for a fallback case")

    monkeypatch.setattr(kernels, "_metal_scores", unexpected_metal)
    device = mx.cpu if case == "cpu" else mx.default_device()
    dtype = mx.float16 if case == "float16" else mx.float32
    with mx.stream(device):
        row, col = mx.array([0, 1, 1]), mx.array([1, 0, 0])
        if case == "empty":
            row, col = row[:0], col[:0]
        if case == "mixed-index-dtypes":
            col = col.astype(mx.int32)
        a = mx.arange(12).reshape(2, 2, 3).astype(dtype)
        b = a - 8
        if case == "different-channels":
            # broadcastable, so the reference still runs
            b = b[..., :1]
        out = transformer_edge_scores(row, col, a, b, 0.5)
        _assert_close(out, _reference_scores(row, col, a, b, 0.5))


def test_message_passing_edge_scores_follows_the_fused_switch(monkeypatch):
    calls = []

    def spy(*args):
        calls.append(args)
        return _reference_scores(*args)

    monkeypatch.setattr(message_passing, "transformer_edge_scores", spy)
    row, col = _edges(5, 5, 12)
    a, b = _features((5, 2, 4), (5, 2, 4))
    want = _reference_scores(row, col, a, b, 0.5)

    _assert_close(message_passing.edge_scores(row, col, a, b, 0.5), want)
    assert len(calls) == 1
    with fused_message_passing_mode(False):
        _assert_close(message_passing.edge_scores(row, col, a, b, 0.5), want)
    assert len(calls) == 1
