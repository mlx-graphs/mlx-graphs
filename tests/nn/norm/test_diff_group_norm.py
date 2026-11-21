import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import DiffGroupNorm as PyGDiffGroupNorm

from mlx_graphs.nn import DiffGroupNormalization

# The soft assignment uses a matmul, which is only accurate to ~1e-3 relative
# on the GPU of some machines.
TOL = dict(rtol=2e-3, atol=2e-3)

DIM, GROUPS, NUM_NODES = 6, 3, 20


def _pair(lamda=0.5, **kwargs):
    pyg_norm = PyGDiffGroupNorm(DIM, GROUPS, lamda=lamda, **kwargs)
    norm = DiffGroupNormalization(DIM, GROUPS, lamda=lamda, **kwargs)
    norm.lin.weight = mx.array(pyg_norm.lin.weight.detach().numpy())
    if kwargs.get("affine", True):
        rng = np.random.default_rng(5)
        weight = rng.normal(size=GROUPS * DIM).astype(np.float32)
        bias = rng.normal(size=GROUPS * DIM).astype(np.float32)
        with torch.no_grad():
            pyg_norm.norm.weight.copy_(torch.from_numpy(weight))
            pyg_norm.norm.bias.copy_(torch.from_numpy(bias))
        norm.norm.weight, norm.norm.bias = mx.array(weight), mx.array(bias)
    return pyg_norm, norm


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("track_running_stats", [True, False])
def test_diff_group_norm_matches_pyg(affine, track_running_stats):
    rng = np.random.default_rng(0)
    pyg_norm, norm = _pair(affine=affine, track_running_stats=track_running_stats)

    # two training steps (updating the running statistics), then evaluation
    for step in range(3):
        x = rng.normal(loc=1.0, size=(NUM_NODES, DIM)).astype(np.float32)
        if step == 2:
            pyg_norm.eval()
            norm.eval()
        with torch.no_grad():
            expected = pyg_norm(torch.from_numpy(x)).numpy()
        out = norm(mx.array(x))
        np.testing.assert_allclose(np.array(out), expected, **TOL)

    if track_running_stats:
        np.testing.assert_allclose(
            np.array(norm.norm.running_mean),
            pyg_norm.norm.running_mean.numpy(),
            **TOL,
        )
        np.testing.assert_allclose(
            np.array(norm.norm.running_var), pyg_norm.norm.running_var.numpy(), **TOL
        )


def test_diff_group_norm_gradients_and_repr():
    norm = DiffGroupNormalization(DIM, GROUPS)
    x = mx.random.normal((NUM_NODES, DIM))
    _, grads = nn.value_and_grad(norm, lambda: (norm(x) ** 2).sum())()
    grads = dict(tree_flatten(grads))
    assert set(grads) == {"lin.weight", "norm.weight", "norm.bias"}
    for grad in grads.values():
        assert mx.all(mx.isfinite(grad)).item()
    assert repr(norm) == f"DiffGroupNormalization({DIM}, groups={GROUPS})"


def test_group_distance_ratio_matches_pyg():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(15, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=15)
    x[y == 1] += 3.0

    expected = PyGDiffGroupNorm.group_distance_ratio(
        torch.from_numpy(x), torch.from_numpy(y)
    )
    out = DiffGroupNormalization.group_distance_ratio(mx.array(x), mx.array(y))
    assert out == pytest.approx(expected, rel=1e-4)
