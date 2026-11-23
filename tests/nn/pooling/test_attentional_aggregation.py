import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import AttentionalAggregation as PyGAttentionalAggregation

from mlx_graphs.nn.pooling.attentional_aggregation import AttentionalAggregation

# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)

DIM, OUT_DIM = 6, 8
# 4 graphs with 5, 1, 7 and 3 nodes
BATCH = np.repeat(np.arange(4), [5, 1, 7, 3]).astype(np.int64)
NUM_NODES = len(BATCH)


def _make_modules(gate_dim, with_nn):
    """Same gate (2-layer MLP) and optional node transform in torch and MLX."""
    torch.manual_seed(0)
    torch_gate = torch.nn.Sequential(
        torch.nn.Linear(DIM, 16), torch.nn.ReLU(), torch.nn.Linear(16, gate_dim)
    )
    mlx_gate = nn.Sequential(nn.Linear(DIM, 16), nn.ReLU(), nn.Linear(16, gate_dim))
    torch_nn = torch.nn.Linear(DIM, OUT_DIM) if with_nn else None
    mlx_nn = nn.Linear(DIM, OUT_DIM) if with_nn else None

    pyg_pool = PyGAttentionalAggregation(torch_gate, torch_nn)
    pool = AttentionalAggregation(mlx_gate, mlx_nn)

    # copied after building the PyG module, which re-initializes its submodules
    pairs = [(mlx_gate.layers[0], torch_gate[0]), (mlx_gate.layers[2], torch_gate[2])]
    if with_nn:
        pairs.append((mlx_nn, torch_nn))
    for mlx_linear, torch_linear in pairs:
        mlx_linear.weight = mx.array(torch_linear.weight.detach().numpy())
        mlx_linear.bias = mx.array(torch_linear.bias.detach().numpy())

    return pool, pyg_pool


@pytest.mark.parametrize(
    "gate_dim, with_nn, expected_dim",
    [(1, False, DIM), (1, True, OUT_DIM), (OUT_DIM, True, OUT_DIM)],
)
def test_attentional_aggregation_shapes(gate_dim, with_nn, expected_dim):
    pool, _ = _make_modules(gate_dim, with_nn)
    node_features = mx.random.normal((NUM_NODES, DIM))

    assert pool(node_features).shape == (1, expected_dim)
    assert pool(node_features, mx.array(BATCH)).shape == (4, expected_dim)
    assert pool(node_features, mx.array(BATCH), num_graphs=6).shape == (
        6,
        expected_dim,
    )


@pytest.mark.parametrize(
    "gate_dim, with_nn",
    [(1, False), (1, True), (OUT_DIM, True), (DIM, False)],
)
def test_attentional_aggregation_matches_pyg(gate_dim, with_nn):
    rng = np.random.default_rng(1)
    # large features give sharp, far from uniform attention
    x = (3 * rng.normal(size=(NUM_NODES, DIM))).astype(np.float32)

    pool, pyg_pool = _make_modules(gate_dim, with_nn)

    with torch.no_grad():
        expected = pyg_pool(torch.from_numpy(x), torch.from_numpy(BATCH)).numpy()
        expected_single = pyg_pool(
            torch.from_numpy(x), torch.zeros(NUM_NODES, dtype=torch.long)
        ).numpy()

    out = pool(mx.array(x), mx.array(BATCH))
    assert out.shape == expected.shape
    np.testing.assert_allclose(np.array(out), expected, **TOL)
    assert np.abs(expected).max() > 0.2

    # without batch indices, all the nodes belong to one graph
    out_single = pool(mx.array(x))
    assert out_single.shape == expected_single.shape
    np.testing.assert_allclose(np.array(out_single), expected_single, **TOL)


def test_attentional_aggregation_unsorted_batch_and_empty_graphs():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(NUM_NODES, DIM)).astype(np.float32)
    pool, _ = _make_modules(1, True)

    reference = np.array(pool(mx.array(x), mx.array(BATCH)))

    # the result does not depend on the order of the nodes
    perm = rng.permutation(NUM_NODES)
    shuffled = np.array(pool(mx.array(x[perm]), mx.array(BATCH[perm])))
    np.testing.assert_allclose(shuffled, reference, **TOL)

    # graphs without nodes get zeros
    padded = np.array(pool(mx.array(x), mx.array(BATCH), num_graphs=6))
    np.testing.assert_allclose(padded[:4], reference, **TOL)
    assert (padded[4:] == 0).all()

    # a graph with a single node returns the (transformed) features of the node
    single = np.array(pool.nn(mx.array(x[5:6])))
    np.testing.assert_allclose(reference[1:2], single, **TOL)


@pytest.mark.parametrize("with_nn", [False, True])
def test_attentional_aggregation_gradients(with_nn):
    rng = np.random.default_rng(3)
    node_features = mx.array(rng.normal(size=(NUM_NODES, DIM)).astype(np.float32))
    batch_indices = mx.array(BATCH)
    pool, _ = _make_modules(1, with_nn)

    def loss_fn():
        return (pool(node_features, batch_indices) ** 2).sum()

    _, grads = nn.value_and_grad(pool, loss_fn)()
    params = dict(tree_flatten(pool.trainable_parameters()))
    grads = dict(tree_flatten(grads))

    assert set(grads) == set(params)
    assert ("nn.weight" in params) == with_nn
    for name, grad in grads.items():
        assert grad.shape == params[name].shape, name
        assert mx.all(mx.isfinite(grad)).item(), f"non-finite gradient for {name}"
        # the softmax over the nodes of a graph does not depend on the bias of
        # the last gate layer: its gradient is zero
        if name != "gate_nn.layers.2.bias":
            assert mx.any(grad != 0).item(), f"zero gradient for {name}"
