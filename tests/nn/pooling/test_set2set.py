import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import Set2Set as PyGSet2Set

from mlx_graphs.nn.pooling.set2set import Set2Set

# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)

DIM = 6
# 4 graphs with 5, 1, 7 and 3 nodes
BATCH = np.repeat(np.arange(4), [5, 1, 7, 3]).astype(np.int64)
NUM_NODES = len(BATCH)


def _copy_weights(pool, pyg_pool):
    """PyTorch and MLX LSTMs use the same gate order (i, f, g, o). PyTorch has
    two biases (input and hidden), MLX a single one."""
    lstm = pyg_pool.lstm
    pool.lstm.Wx = mx.array(lstm.weight_ih_l0.detach().numpy())
    pool.lstm.Wh = mx.array(lstm.weight_hh_l0.detach().numpy())
    pool.lstm.bias = mx.array((lstm.bias_ih_l0 + lstm.bias_hh_l0).detach().numpy())


@pytest.mark.parametrize("processing_steps", [1, 3])
def test_set2set_shapes(processing_steps):
    pool = Set2Set(DIM, processing_steps)
    node_features = mx.random.normal((NUM_NODES, DIM))

    assert pool(node_features).shape == (1, 2 * DIM)
    assert pool(node_features, mx.array(BATCH)).shape == (4, 2 * DIM)
    assert pool(node_features, mx.array(BATCH), num_graphs=6).shape == (6, 2 * DIM)
    assert pool.out_features_dim == 2 * DIM


@pytest.mark.parametrize("processing_steps", [1, 2, 4])
def test_set2set_matches_pyg(processing_steps):
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(NUM_NODES, DIM)).astype(np.float32)

    pyg_pool = PyGSet2Set(DIM, processing_steps)
    with torch.no_grad():
        # larger LSTM weights give sharper attention, so that a wrong gate
        # mapping is visible
        for param in pyg_pool.parameters():
            param.mul_(3.0)
        expected = pyg_pool(torch.from_numpy(x), torch.from_numpy(BATCH)).numpy()
        expected_single = pyg_pool(
            torch.from_numpy(x), torch.zeros(NUM_NODES, dtype=torch.long)
        ).numpy()

    pool = Set2Set(DIM, processing_steps)
    _copy_weights(pool, pyg_pool)

    out = pool(mx.array(x), mx.array(BATCH))
    assert out.shape == expected.shape
    np.testing.assert_allclose(np.array(out), expected, **TOL)
    assert np.abs(expected).max() > 0.2

    # without batch indices, all the nodes belong to one graph
    out_single = pool(mx.array(x))
    assert out_single.shape == expected_single.shape
    np.testing.assert_allclose(np.array(out_single), expected_single, **TOL)


def test_set2set_unsorted_batch_and_empty_graphs():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(NUM_NODES, DIM)).astype(np.float32)
    mx.random.seed(0)
    pool = Set2Set(DIM, processing_steps=3)

    reference = np.array(pool(mx.array(x), mx.array(BATCH)))

    # the result does not depend on the order of the nodes
    perm = rng.permutation(NUM_NODES)
    shuffled = np.array(pool(mx.array(x[perm]), mx.array(BATCH[perm])))
    np.testing.assert_allclose(shuffled, reference, **TOL)

    # extra graphs without nodes do not change the others, and have an empty
    # readout (second half of the output)
    padded = np.array(pool(mx.array(x), mx.array(BATCH), num_graphs=6))
    np.testing.assert_allclose(padded[:4], reference, **TOL)
    assert (padded[4:, DIM:] == 0).all()

    # the readout of a graph with a single node is the features of the node
    np.testing.assert_allclose(reference[1, DIM:], x[5], **TOL)


def test_set2set_gradients():
    rng = np.random.default_rng(3)
    node_features = mx.array(rng.normal(size=(NUM_NODES, DIM)).astype(np.float32))
    batch_indices = mx.array(BATCH)
    mx.random.seed(0)
    pool = Set2Set(DIM, processing_steps=3)

    def loss_fn():
        return (pool(node_features, batch_indices) ** 2).sum()

    _, grads = nn.value_and_grad(pool, loss_fn)()
    params = dict(tree_flatten(pool.trainable_parameters()))
    grads = dict(tree_flatten(grads))

    assert set(grads) == set(params) == {"lstm.Wx", "lstm.Wh", "lstm.bias"}
    for name, grad in grads.items():
        assert grad.shape == params[name].shape, name
        assert mx.all(mx.isfinite(grad)).item(), f"non-finite gradient for {name}"
        assert mx.any(grad != 0).item(), f"zero gradient for {name}"
