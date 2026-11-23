"""PyG-parity tests of the dense pooling operators: DiffPool, MinCutPool and
DMoN (outputs, auxiliary losses and gradients)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

from mlx_graphs.nn import DMoNPooling, dense_diff_pool, dense_mincut_pool
from tests.pyg_copy import load_torch, torch_grads, torch_state

B, N, DIM, K = 3, 8, 5, 3
# MLX's GPU matmul is only accurate to ~1e-3 relative
TOL = 5e-3


def assert_close(actual, expected, tol=TOL, name=""):
    """Tolerance relative to the largest expected value."""
    actual, expected = np.array(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    assert np.all(np.isfinite(expected)), name
    scale = max(1.0, float(np.abs(expected).max(initial=0.0)))
    np.testing.assert_allclose(
        actual, expected, rtol=tol, atol=tol * scale, err_msg=name
    )


def make_inputs(seed=0, symmetric=False):
    """Weighted adjacency, features, assignment logits and a mask where the
    graphs have 8, 6 and 4 real nodes."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(B, N, DIM)).astype(np.float32)
    adj = rng.uniform(0.2, 1.5, size=(B, N, N)) * (rng.uniform(size=(B, N, N)) < 0.5)
    if symmetric:
        adj = np.triu(adj, 1)
        adj = adj + adj.transpose(0, 2, 1)
    mask = np.arange(N)[None, :] < np.array([8, 6, 4])[:, None]
    # every real node has an edge (non-zero degrees)
    adj[:, np.arange(N - 1), np.arange(1, N)] += 0.5
    adj[:, np.arange(1, N), np.arange(N - 1)] += 0.5
    adj = (adj * (mask[:, :, None] & mask[:, None, :])).astype(np.float32)
    s = rng.normal(size=(B, N, K)).astype(np.float32)
    return x, adj, s, mask


def _loss_weights(outputs, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=tuple(o.shape)).astype(np.float32) for o in outputs]


def compare_function(mlx_fn, pyg_fn, use_mask, squeeze=False, **kwargs):
    x, adj, s, mask = make_inputs()
    if squeeze:
        x, adj, s, mask = x[0], adj[0], s[0], None
    elif not use_mask:
        mask = None
    tx, tadj, ts = (torch.tensor(a, requires_grad=True) for a in (x, adj, s))
    tmask = None if mask is None else torch.tensor(mask)
    expected = pyg_fn(tx, tadj, ts, tmask, **kwargs)
    weights = _loss_weights(expected)
    sum((o * torch.tensor(w)).sum() for o, w in zip(expected, weights)).backward()

    mmask = None if mask is None else mx.array(mask)

    def loss_fn(x, adj, s):
        out = mlx_fn(adj, x, s, mmask, **kwargs)
        loss = sum((o * mx.array(w)).sum() for o, w in zip(out, weights))
        return loss, out

    (_, out), grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))(
        mx.array(x), mx.array(adj), mx.array(s)
    )
    names = ["x", "adj", "loss_1", "loss_2"]
    for name, a, b in zip(names, out, expected):
        assert_close(a, b.detach().numpy(), name=name)
    for name, g, t in zip(["x", "adj", "s"], grads, (tx, tadj, ts)):
        assert_close(g, t.grad.numpy(), name=f"grad {name}")
    return out


@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_dense_diff_pool(use_mask, normalize):
    out = compare_function(
        dense_diff_pool, pyg_nn.dense_diff_pool, use_mask, normalize=normalize
    )
    assert out[0].shape == (B, K, DIM) and out[1].shape == (B, K, K)
    assert out[2].ndim == out[3].ndim == 0


def test_dense_diff_pool_single_graph():
    out = compare_function(dense_diff_pool, pyg_nn.dense_diff_pool, False, True)
    assert out[0].shape == (1, K, DIM)


@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("temp", [1.0, 0.5])
def test_dense_mincut_pool(use_mask, temp):
    out = compare_function(
        dense_mincut_pool, pyg_nn.dense_mincut_pool, use_mask, temp=temp
    )
    # zero diagonal of the pooled adjacency
    assert not np.any(np.diagonal(np.array(out[1]), axis1=1, axis2=2))


def test_dense_mincut_pool_single_graph():
    compare_function(dense_mincut_pool, pyg_nn.dense_mincut_pool, False, True)


@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("hidden", [None, 6])
def test_dmon_pooling(use_mask, hidden):
    torch.manual_seed(0)
    dims = DIM if hidden is None else [DIM, hidden]
    ref = pyg_nn.DMoNPooling(dims, K)
    module = DMoNPooling(dims, K)
    load_torch(module, torch_state(ref))

    x, adj, _, mask = make_inputs(symmetric=True)
    if not use_mask:
        mask = None
    tx, tadj = torch.tensor(x, requires_grad=True), torch.tensor(adj)
    expected = ref(tx, tadj, None if mask is None else torch.tensor(mask))
    weights = _loss_weights(expected)
    sum((o * torch.tensor(w)).sum() for o, w in zip(expected, weights)).backward()

    mmask = None if mask is None else mx.array(mask)

    def loss_fn(x):
        out = module(mx.array(adj), x, mmask)
        return sum((o * mx.array(w)).sum() for o, w in zip(out, weights)), out

    (_, out), gx = mx.value_and_grad(loss_fn)(mx.array(x))
    _, grads = nn.value_and_grad(module, lambda: loss_fn(mx.array(x))[0])()

    names = ["s", "x", "adj", "spectral_loss", "ortho_loss", "cluster_loss"]
    for name, a, b in zip(names, out, expected):
        assert_close(a, b.detach().numpy(), name=name)
    assert_close(gx, tx.grad.numpy(), name="grad x")
    grads = dict(tree_flatten(grads))
    expected_grads = torch_grads(ref)
    assert set(grads) == set(expected_grads)
    for name, grad in grads.items():
        assert_close(grad, expected_grads[name], name=name)
    assert repr(module) == f"DMoNPooling({DIM}, num_clusters={K})"


def test_dmon_pooling_dropout():
    module = DMoNPooling(DIM, K, dropout=0.5)
    x, adj, _, _ = make_inputs()
    module.train()
    s_train = module(mx.array(adj), mx.array(x))[0]
    module.eval()
    s_eval = module(mx.array(adj), mx.array(x))[0]
    assert s_train.shape == s_eval.shape == (B, N, K)
    np.testing.assert_allclose(np.array(s_eval.sum(-1)), 1.0, rtol=1e-5)


def test_dmon_pooling_large_features_finite_gradients():
    """The SELU of the pooled features has a finite gradient for large
    inputs (``mlx.nn.selu`` evaluates ``exp`` on both branches, whose
    gradient overflows to NaN)."""
    module = DMoNPooling(DIM, K)
    x, adj, _, _ = make_inputs()
    x = mx.array(100.0 * x)

    def loss_fn(x):
        return module(mx.array(adj), x)[1].sum()

    grad = mx.grad(loss_fn)(x)
    assert mx.all(mx.isfinite(grad)).item()
