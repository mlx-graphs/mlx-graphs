import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils import degree, invert_sqrt_degree, scatter


@pytest.mark.parametrize(
    "src, index, num_nodes, aggr, expected",
    [
        # Test case for test_scatter
        (
            mx.array([1.0, 1.0, 1.0, 1.0]),
            mx.array([0, 0, 1, 2]),
            None,
            "softmax",
            mx.array([0.5, 0.5, 1, 1]),
        ),
        (
            mx.array([1.0, 1.0, 1.0, 1.0]),
            mx.array([0, 0, 1, 2]),
            3,
            "add",
            mx.array([2, 1, 1]),
        ),
        (
            mx.array([1.0, 1.0, 1.0, 1.0]),
            mx.array([0, 0, 1, 2]),
            3,
            "max",
            mx.array([1, 1, 1]),
        ),
        (
            mx.array([1.0, 1.0, 1.0, 1.0]),
            mx.array([0, 0, 1, 2]),
            3,
            "mean",
            mx.array([1, 1, 1]),
        ),
        # Max of all-negative groups, min of all-pos groups
        (
            mx.array([-3.0, -1.0, -5.0]),
            mx.array([0, 0, 1]),
            None,
            "max",
            mx.array([-1, -5]),
        ),
        (
            mx.array([3.0, 1.0, 5.0]),
            mx.array([0, 0, 1]),
            None,
            "min",
            mx.array([1, 5]),
        ),
        # Test case for test_scatter_2D
        (
            mx.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]),
            mx.array([0, 0, 1, 2]),
            None,
            "softmax",
            mx.array([[0.5, 0.269], [0.5, 0.731], [1, 1], [1, 1]]),
        ),
        (
            mx.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]),
            mx.array([0, 0, 1, 2]),
            3,
            "add",
            mx.array([[2, 5], [1, 4], [1, 5]]),
        ),
        (
            mx.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]),
            mx.array([0, 0, 1, 2]),
            3,
            "max",
            mx.array([[1, 3], [1, 4], [1, 5]]),
        ),
        (
            mx.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]),
            mx.array([0, 0, 1, 2]),
            3,
            "mean",
            mx.array([[1, 2.5], [1, 4], [1, 5]]),
        ),
    ],
)
def test_scatter(src, index, num_nodes, aggr, expected):
    if num_nodes is None:
        num_nodes = src.shape[0] if aggr == "softmax" else index.max().item() + 1
    y_hat = scatter(src, index, num_nodes, aggr=aggr)

    if aggr == "softmax":
        assert mx.array_equal(
            mx.round(y_hat, 3), mx.round(expected, 3)
        ), f"Scatter {aggr} failed"
    else:
        assert mx.array_equal(y_hat, expected), f"Scatter {aggr} failed"


def test_degree():
    index = mx.array([0, 0, 1, 2])

    assert mx.array_equal(degree(index, 3), mx.array([2, 1, 1])), "degree failed"
    assert mx.array_equal(
        degree(index), mx.array([2, 1, 1])
    ), "degree with automatic `num_nodes` failed"
    assert mx.array_equal(
        degree(index, 4), mx.array([2, 1, 1, 0])
    ), "degree with missing node failed"

    index = mx.array([[0, 0, 1, 2], [1, 2, 0, 1]])

    with pytest.raises(ValueError):
        degree(index, 3)


def test_inv_degree():
    deg = mx.array([1, 2, 3, 0])
    expected = mx.array([1, 0.707107, 0.57735, 0])
    out = invert_sqrt_degree(deg)
    assert mx.allclose(out, expected), "Incorrect degree inversion"


@pytest.mark.parametrize("aggr", ["max", "min"])
def test_scatter_max_min_gradient_splits_ties(aggr):
    # The gradient of a maximum (minimum) is split equally among the values
    # equal to it, as torch's scatter_reduce (used by PyG without
    # torch_scatter). MLX's own gradient gave each tied value the full
    # gradient. No exact zeros: torch then also counts its zero initial value.
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    index = rng.integers(0, 6, 50)
    values = np.round(rng.uniform(0.1, 1.0, (50, 3)), 1).astype(np.float32)
    values[40:] = values[:10]  # duplicates: exact ties within their group
    index[40:] = index[:10]
    cot = rng.normal(size=(8, 3)).astype(np.float32)

    def loss(v):
        return (scatter(v, mx.array(index), 8, aggr) * mx.array(cot)).sum()

    v = torch.tensor(values, requires_grad=True)
    out = torch.zeros(8, 3).scatter_reduce(
        0,
        torch.tensor(index).view(-1, 1).expand(-1, 3),
        v,
        reduce="a" + aggr,
        include_self=False,
    )
    (out * torch.tensor(cot)).sum().backward()

    for grad_fn in (mx.grad(loss), mx.compile(mx.grad(loss))):
        grad = grad_fn(mx.array(values))
        np.testing.assert_allclose(np.array(grad), v.grad.numpy(), rtol=1e-5, atol=1e-6)
