import mlx.core as mx
import numpy as np
import pytest
import torch
from torch_geometric.nn import global_add_pool as pyg_global_add_pool
from torch_geometric.nn import global_max_pool as pyg_global_max_pool
from torch_geometric.nn import global_mean_pool as pyg_global_mean_pool
from torch_geometric.nn.aggr import SortAggregation

from mlx_graphs.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    global_sort_pool,
)

# 3 graphs with 4, 1 and 5 nodes
BATCH = np.repeat(np.arange(3), [4, 1, 5]).astype(np.int64)


@pytest.mark.parametrize(
    "pool, pyg_pool",
    [
        (global_add_pool, pyg_global_add_pool),
        (global_mean_pool, pyg_global_mean_pool),
        (global_max_pool, pyg_global_max_pool),
    ],
)
def test_global_pools_num_graphs(pool, pyg_pool):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(len(BATCH), 3)).astype(np.float32)

    for num_graphs in (None, 3, 5):
        expected = pyg_pool(torch.from_numpy(x), torch.from_numpy(BATCH), num_graphs)
        out = pool(mx.array(x), mx.array(BATCH), num_graphs)
        assert out.shape == tuple(expected.shape)
        np.testing.assert_allclose(np.array(out), expected.numpy(), rtol=1e-5)

    # backward compatible positional call
    assert pool(mx.array(x), mx.array(BATCH)).shape == (3, 3)


@pytest.mark.parametrize("k", [1, 3, 4, 7])
def test_global_sort_pool_matches_pyg(k):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(len(BATCH), 3)).astype(np.float32)

    expected = SortAggregation(k)(torch.from_numpy(x), torch.from_numpy(BATCH))
    out = global_sort_pool(mx.array(x), mx.array(BATCH), k)
    assert out.shape == (3, 3 * k)
    np.testing.assert_allclose(np.array(out), expected.numpy(), rtol=1e-6)

    # nodes do not need to be grouped by graph
    perm = rng.permutation(len(BATCH))
    shuffled = global_sort_pool(mx.array(x[perm]), mx.array(BATCH[perm]), k)
    np.testing.assert_allclose(np.array(shuffled), expected.numpy(), rtol=1e-6)

    # without batch: a single graph
    expected_single = SortAggregation(k)(torch.from_numpy(x))
    single = global_sort_pool(mx.array(x), None, k)
    np.testing.assert_allclose(np.array(single), expected_single.numpy(), rtol=1e-6)


def test_global_sort_pool_num_graphs_and_gradients():
    x = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    batch = mx.array([0, 0, 1])
    out = global_sort_pool(x, batch, k=2, num_graphs=3)
    assert out.tolist() == [[3, 4, 1, 2], [5, 6, 0, 0], [0, 0, 0, 0]]
    with pytest.raises(ValueError):
        global_sort_pool(x, batch, k=2, num_graphs=1)

    grad = mx.grad(lambda x: global_sort_pool(x, batch, k=1).sum())(x)
    # only the first node of each graph after sorting is kept
    assert grad.tolist() == [[0, 0], [1, 1], [1, 1]]
