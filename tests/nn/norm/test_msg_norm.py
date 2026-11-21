import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import MessageNorm as PyGMessageNorm

from mlx_graphs.nn import MessageNormalization


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("learn_scale", [False, True])
def test_message_norm_matches_pyg(p, learn_scale):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 4)).astype(np.float32)
    msg = rng.normal(size=(6, 4)).astype(np.float32)
    msg[2] = 0.0  # zero message: no division by zero

    pyg_norm = PyGMessageNorm(learn_scale)
    with torch.no_grad():
        pyg_norm.scale.fill_(1.7)
        expected = pyg_norm(torch.from_numpy(x), torch.from_numpy(msg), p).numpy()

    norm = MessageNormalization(learn_scale)
    norm.scale = mx.array([1.7])
    out = norm(mx.array(x), mx.array(msg), p)
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("learn_scale", [False, True])
def test_message_norm_trainable_scale(learn_scale):
    norm = MessageNormalization(learn_scale)
    params = dict(tree_flatten(norm.trainable_parameters()))
    assert set(params) == ({"scale"} if learn_scale else set())
    assert repr(norm) == f"MessageNormalization(learn_scale={learn_scale})"

    if learn_scale:
        x, msg = mx.random.normal((4, 3)), mx.random.normal((4, 3))
        _, grads = nn.value_and_grad(norm, lambda: norm(x, msg).sum())()
        assert mx.all(mx.isfinite(grads["scale"])).item()


def _zero_row_graph():
    """A small graph and features where node 2's features are all zero (as
    after a ReLU or dropout)."""
    edge_index = np.array([[0, 1, 2, 3, 1, 2], [1, 2, 3, 0, 0, 0]])
    x = np.random.default_rng(0).normal(size=(4, 6)).astype(np.float32)
    x[2] = 0.0
    return edge_index, x


def _grads_match_pyg(mlx_fn, torch_fn, x):
    """Input gradients of sum(f(x) ** 2) in both frameworks."""

    g = mx.grad(lambda h: (mlx_fn(h) ** 2).sum())(mx.array(x))
    h = torch.from_numpy(x).requires_grad_()
    (torch_fn(h) ** 2).sum().backward()
    assert mx.all(mx.isfinite(g)).item(), "NaN / inf gradient"
    np.testing.assert_allclose(np.array(g), h.grad.numpy(), rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("p", [2.0, 1.0, 3.0])
def test_message_norm_zero_rows_gradient_matches_pyg(p):
    """Regression: the norms of all-zero rows of x or of the messages gave
    NaN gradients (GENConv(msg_norm=True) then trained much worse than in
    PyG). PyG's gradients are finite."""

    _, x = _zero_row_graph()
    norm = MessageNormalization(learn_scale=True)
    pyg_norm = PyGMessageNorm(learn_scale=True)
    msg = np.random.default_rng(1).normal(size=x.shape).astype(np.float32)
    msg[1] = 0.0
    _grads_match_pyg(
        lambda h: norm(h, h * 2 + mx.array(msg), p),
        lambda h: pyg_norm(h, h * 2 + torch.from_numpy(msg), p),
        x,
    )
