"""The "mul", "var", "std" and "median" reductions of `scatter`, against PyG."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from torch_geometric.nn import aggr as pyg_aggr
from torch_geometric.utils import scatter as pyg_scatter

from mlx_graphs.utils import scatter
from mlx_graphs.utils.scatter import scatter_quantile

NUM_GROUPS = 7  # groups 2 and 6 are empty


def make_data(seed=0, dims=(3,)):
    rng = np.random.default_rng(seed)
    index = np.array([0, 0, 0, 1, 3, 3, 4, 4, 4, 4, 5, 0, 3, 1], dtype=np.int64)
    x = rng.standard_normal((index.size, *dims)).astype(np.float32)
    return x, index


def pyg_reference(x, index, reduce):
    x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    index = torch.tensor(index)
    if reduce == "mul":
        return pyg_scatter(x, index, 0, NUM_GROUPS, "mul")
    if reduce == "var":
        return pyg_aggr.VarAggregation()(x, index, dim_size=NUM_GROUPS)
    if reduce == "std":
        return pyg_aggr.StdAggregation()(x, index, dim_size=NUM_GROUPS)
    return pyg_aggr.MedianAggregation()(x, index, dim_size=NUM_GROUPS)


@pytest.mark.parametrize("reduce", ["mul", "var", "std", "median"])
def test_matches_pyg(reduce):
    x, index = make_data()
    out = scatter(mx.array(x), mx.array(index), NUM_GROUPS, reduce)
    expected = pyg_reference(x, index, reduce).numpy()
    assert out.shape == (NUM_GROUPS, 3)
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("reduce", ["mul", "var", "std", "median"])
def test_three_dimensional_inputs(reduce):
    x, index = make_data(dims=(2, 3))
    out = scatter(mx.array(x), mx.array(index), NUM_GROUPS, reduce)
    expected = pyg_reference(x.reshape(len(index), -1), index, reduce).numpy()
    assert out.shape == (NUM_GROUPS, 2, 3)
    np.testing.assert_allclose(
        np.array(out).reshape(NUM_GROUPS, -1), expected, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("reduce", ["mul", "var", "std", "median"])
def test_gradients_match_pyg(reduce):
    x, index = make_data(seed=1)
    x[2, 1] = 0.0  # one zero in group 0
    x[4, 2] = x[5, 2] = 0.0  # two zeros in group 3
    weights = np.arange(NUM_GROUPS * 3, dtype=np.float32).reshape(NUM_GROUPS, 3)

    def loss(values):
        out = scatter(values, mx.array(index), NUM_GROUPS, reduce)
        return (out * mx.array(weights)).sum()

    grad = mx.grad(loss)(mx.array(x))

    tx = torch.tensor(x, requires_grad=True)
    (pyg_reference(tx, index, reduce) * torch.tensor(weights)).sum().backward()
    np.testing.assert_allclose(np.array(grad), tx.grad.numpy(), rtol=1e-4, atol=1e-4)


def test_empty_groups():
    x, index = make_data()
    x, index = mx.array(x), mx.array(index)
    assert (scatter(x, index, NUM_GROUPS, "mul")[2] == 1).all().item()
    for reduce in ["var", "std", "median"]:
        assert (scatter(x, index, NUM_GROUPS, reduce)[6] == 0).all().item()


def test_std_of_constant_group_is_zero():
    x = mx.array([[2.0], [2.0], [2.0], [1.0]])
    index = mx.array([0, 0, 0, 1])
    np.testing.assert_allclose(np.array(scatter(x, index, 2, "std")), [[0.0], [0.0]])
    grad = mx.grad(lambda v: scatter(v, index, 2, "std").sum())(x)
    assert np.isfinite(np.array(grad)).all()


@pytest.mark.parametrize(
    "q, interpolation",
    [
        (0.5, "linear"),
        (0.3, "lower"),
        (0.3, "higher"),
        (0.7, "nearest"),
        (0.4, "midpoint"),
        ([0.0, 0.25, 1.0], "linear"),
    ],
)
def test_quantile_matches_pyg(q, interpolation):
    x, index = make_data(seed=2)
    out = scatter_quantile(mx.array(x), mx.array(index), NUM_GROUPS, q, interpolation)
    expected = pyg_aggr.QuantileAggregation(q, interpolation)(
        torch.tensor(x), torch.tensor(index), dim_size=NUM_GROUPS
    )
    np.testing.assert_allclose(np.array(out), expected.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("interpolation", ["nearest", "linear", "lower", "higher"])
def test_quantile_ties_match_pyg(interpolation):
    """Positions exactly between two values (even groups and q = 0.5), at odd
    and even offsets: "nearest" rounds half to even like torch.round."""
    x, index = make_data(seed=3)
    out = scatter_quantile(mx.array(x), mx.array(index), NUM_GROUPS, 0.5, interpolation)
    expected = pyg_aggr.QuantileAggregation(0.5, interpolation)(
        torch.tensor(x), torch.tensor(index), dim_size=NUM_GROUPS
    )
    np.testing.assert_allclose(np.array(out), expected.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.slow
def test_quantile_more_than_2_24_values():
    """Positions above 2**24 are exact (float32 positions would read the last
    value of the previous group). Slow: about 1.6 GB of GPU memory."""
    n = (1 << 24) + 3
    index = mx.concatenate([mx.zeros((n - 2,), mx.int32), mx.ones((2,), mx.int32)])
    values = mx.concatenate([mx.zeros((n - 2, 1)), mx.array([[5.0], [7.0]])])
    for q, expected in [(0.0, 5.0), (1.0, 7.0)]:
        out = scatter_quantile(values, index, 2, q=q, interpolation="lower")
        assert out[1].item() == expected


def test_quantile_unsorted_index_and_no_values():
    values = mx.array([[1.0], [5.0], [2.0], [3.0]])
    index = mx.array([0, 1, 0, 0])
    np.testing.assert_allclose(
        np.array(scatter(values, index, 3, "median")), [[2.0], [5.0], [0.0]]
    )
    empty = scatter_quantile(mx.zeros((0, 2)), mx.zeros((0,), mx.int32), 3)
    assert empty.shape == (3, 2)


def test_invalid_reduction():
    with pytest.raises(ValueError):
        scatter(mx.ones((2, 2)), mx.array([0, 1]), 2, "prod")
    with pytest.raises(ValueError):
        scatter_quantile(mx.ones((2, 2)), mx.array([0, 1]), 2, q=1.5)
