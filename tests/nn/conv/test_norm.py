"""The edge normalizations must match PyG's `gcn_norm` and `get_laplacian`.

No dense matmul is involved here, so the comparison is tight.
"""

import mlx.core as mx
import numpy as np
import pytest
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm as pyg_gcn_norm
from torch_geometric.utils import get_laplacian

from mlx_graphs.nn.conv._norm import gcn_norm, laplacian

NUM_NODES = 30


def make_graph(seed=0):
    """Random directed graph with duplicate edges, a self-loop on nodes 0-2 and
    five isolated nodes (25-29)."""
    rng = np.random.default_rng(seed)
    edges = rng.integers(0, NUM_NODES - 5, (2, 120))
    edges = edges[:, edges[0] != edges[1]]
    loops = np.array([[0, 1, 2], [0, 1, 2]])
    edges = np.concatenate([edges, edges[:, :10], loops], axis=1)
    weights = rng.uniform(0.1, 2.0, edges.shape[1]).astype(np.float32)
    return edges, weights


def dense(edge_index, weights, num_nodes=NUM_NODES):
    """Dense [src, dst] matrix, summing duplicate edges."""
    out = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    np.add.at(out, (np.array(edge_index[0]), np.array(edge_index[1])), weights)
    return out


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("add_self_loops", [True, False])
def test_gcn_norm_matches_pyg(weighted, add_self_loops):
    edges, weights = make_graph()
    weights = weights if weighted else None

    ref_index, ref_weights = pyg_gcn_norm(
        torch.tensor(edges),
        None if weights is None else torch.tensor(weights),
        NUM_NODES,
        add_self_loops=add_self_loops,
    )
    out_index, out_weights = gcn_norm(
        mx.array(edges),
        NUM_NODES,
        None if weights is None else mx.array(weights),
        add_self_loops=add_self_loops,
    )

    num_out = edges.shape[1] + (NUM_NODES if add_self_loops else 0)
    assert out_index.shape == (2, num_out)
    assert out_weights.shape == (num_out,)
    np.testing.assert_allclose(
        dense(out_index, np.array(out_weights)),
        dense(ref_index.numpy(), ref_weights.numpy()),
        rtol=1e-5,
        atol=1e-6,
    )


def test_gcn_norm_keeps_existing_self_loops():
    # node 0 has a self-loop of weight 3, node 1 has none, node 2 is isolated
    edge_index = mx.array([[0, 0, 1], [0, 1, 0]])
    weights = mx.array([3.0, 1.0, 2.0])

    out_index, out_weights = gcn_norm(edge_index, 3, weights)

    # A + remaining loops, as [src, dst]
    adj = np.array([[3.0, 1.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    deg = adj.sum(axis=0)
    expected = adj / np.sqrt(np.outer(deg, deg))
    np.testing.assert_allclose(
        dense(out_index, np.array(out_weights), 3), expected, rtol=1e-5, atol=1e-6
    )


def test_gcn_norm_sums_duplicate_self_loops():
    edge_index = mx.array([[0, 0, 1], [0, 0, 0]])
    weights = mx.array([1.0, 2.0, 1.0])

    out_index, out_weights = gcn_norm(edge_index, 2, weights)

    adj = np.array([[3.0, 0.0], [1.0, 1.0]])
    deg = adj.sum(axis=0)
    expected = adj / np.sqrt(np.outer(deg, deg))
    np.testing.assert_allclose(
        dense(out_index, np.array(out_weights), 2), expected, rtol=1e-5, atol=1e-6
    )


def test_gcn_norm_column_weights_and_empty_graph():
    edges, weights = make_graph()
    _, flat = gcn_norm(mx.array(edges), NUM_NODES, mx.array(weights))
    _, column = gcn_norm(mx.array(edges), NUM_NODES, mx.array(weights)[:, None])
    assert column.shape == flat.shape
    # scatter-add sums the weights in no fixed order, so not bit-identical
    np.testing.assert_allclose(np.array(column), np.array(flat), rtol=1e-5)

    out_index, out_weights = gcn_norm(mx.zeros((2, 0), dtype=mx.int32), 4)
    assert out_index.shape == (2, 4)
    np.testing.assert_allclose(np.array(out_weights), np.ones(4))


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("normalization", [None, "sym", "rw"])
def test_laplacian_matches_pyg(weighted, normalization):
    edges, weights = make_graph()
    weights = weights if weighted else None

    ref_index, ref_weights = get_laplacian(
        torch.tensor(edges),
        None if weights is None else torch.tensor(weights),
        normalization,
        num_nodes=NUM_NODES,
    )
    out_index, out_weights = laplacian(
        mx.array(edges),
        NUM_NODES,
        None if weights is None else mx.array(weights),
        normalization,
    )

    assert out_index.shape == (2, edges.shape[1] + NUM_NODES)
    np.testing.assert_allclose(
        dense(out_index, np.array(out_weights)),
        dense(ref_index.numpy(), ref_weights.numpy()),
        rtol=1e-5,
        atol=1e-6,
    )
    # The diagonal is the last `num_nodes` entries, in node order.
    np.testing.assert_array_equal(
        np.array(out_index[:, -NUM_NODES:]), np.tile(np.arange(NUM_NODES), (2, 1))
    )


def test_laplacian_invalid_normalization():
    with pytest.raises(ValueError):
        laplacian(mx.array([[0], [1]]), 2, normalization="row")
