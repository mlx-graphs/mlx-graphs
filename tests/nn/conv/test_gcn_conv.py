import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.nn.conv.gcn_conv import GCNConv

mx.random.seed(42)


def test_gcn_conv():
    conv = GCNConv(8, 20)

    node_features = mx.random.uniform(0, 1, [6, 8])
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    y_hat1 = conv(edge_index, node_features)

    node_features = mx.random.uniform(-1, 1, [6, 8])
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    y_hat2 = conv(edge_index, node_features)

    conv = GCNConv(16, 32)
    node_features = mx.random.uniform(0, 1, [100, 16])
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    y_hat3 = conv(edge_index, node_features)

    conv = GCNConv(16, 32, bias=False)
    node_features = mx.random.uniform(0, 1, [100, 16])
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    y_hat4 = conv(
        edge_index,
        node_features,
        edge_weights=mx.random.normal(
            [
                5,
            ]
        ),
    )

    conv = GCNConv(8, 16, add_self_loops=True)
    node_features = mx.random.uniform(0, 1, [6, 8])
    edge_index = mx.array([[0, 1, 2], [1, 2, 3]])
    y_hat5 = conv(edge_index, node_features)

    assert y_hat1.shape == (6, 20), "Simple GCNConv failed"
    assert y_hat2.shape == (6, 20), "GCNConv with negative values failed"
    assert y_hat3.shape == (100, 32), "GCNConv with different shapes failed"
    assert y_hat4.shape == (100, 32), "GCNConv without bias failed"
    assert y_hat5.shape == (6, 16), "GCNConv with self-loops failed"


# ---------------------------------------------------------------------------
# Numerical parity with PyG
# ---------------------------------------------------------------------------


def _random_graph(num_nodes=40, num_edges=300, dim=12, seed=0):
    rng = np.random.default_rng(seed)
    # duplicate edges and isolated nodes (the last 5 nodes) on purpose
    edge_index = rng.integers(0, num_nodes - 5, (2, num_edges))
    # Keep one self-loop per node at most: PyG collapses repeated self-loops
    # of a node into one, mlx-graphs keeps them like any other repeated edge.
    is_loop = edge_index[0] == edge_index[1]
    _, first = np.unique(edge_index[0][is_loop], return_index=True)
    keep = ~is_loop
    keep[np.nonzero(is_loop)[0][first]] = True
    edge_index = edge_index[:, keep]
    num_edges = edge_index.shape[1]
    x = rng.standard_normal((num_nodes, dim)).astype(np.float32)
    weights = rng.uniform(0.1, 2.0, num_edges).astype(np.float32)
    return edge_index, x, weights


@pytest.mark.parametrize("add_self_loops", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_gcn_conv_matches_pyg(add_self_loops, weighted, normalize):
    torch = pytest.importorskip("torch")
    pyg_nn = pytest.importorskip("torch_geometric.nn")

    if add_self_loops and not normalize:
        pytest.skip("not supported by PyG")
    edge_index, x, weights = _random_graph()
    weights = weights if weighted else None

    ref = pyg_nn.GCNConv(
        x.shape[1], 8, add_self_loops=add_self_loops, normalize=normalize
    )
    torch.nn.init.normal_(ref.bias)
    conv = GCNConv(x.shape[1], 8, add_self_loops=add_self_loops)
    conv.linear.weight = mx.array(ref.lin.weight.detach().numpy())
    conv.bias = mx.array(ref.bias.detach().numpy())

    expected = ref(
        torch.tensor(x),
        torch.tensor(edge_index),
        None if weights is None else torch.tensor(weights),
    )
    for _ in range(2):  # the second call reads the cached normalization
        out = conv(
            mx.array(edge_index),
            mx.array(x),
            edge_weights=None if weights is None else mx.array(weights),
            normalize=normalize,
        )
        # tolerance limited by the accuracy of MLX's GPU matmul (~1e-3 relative)
        np.testing.assert_allclose(
            np.array(out), expected.detach().numpy(), rtol=2e-2, atol=2e-2
        )


def test_gcn_conv_norm_cache():
    from mlx_graphs.nn.conv import _norm as gcn_conv

    gcn_conv.clear_norm_cache()
    edge_index, x, _ = _random_graph()
    edge_index, x = mx.array(edge_index), mx.array(x)
    conv1, conv2 = GCNConv(12, 8), GCNConv(8, 4)

    out = conv2(edge_index, conv1(edge_index, x))
    assert len(gcn_conv._norm_cache._entries) == 1  # shared by both layers

    # a different graph (same shape) must not reuse the cached normalization
    other = mx.array(_random_graph(seed=1)[0])
    out_other = conv2(other, conv1(other, x))
    assert len(gcn_conv._norm_cache._entries) == 2
    assert not np.allclose(np.array(out), np.array(out_other), atol=1e-3)

    # edge weights bypass the cache
    conv1(edge_index, x, edge_weights=mx.ones((edge_index.shape[1],)))
    assert len(gcn_conv._norm_cache._entries) == 2


def test_gcn_conv_empty_graph_with_self_loops():
    conv = GCNConv(4, 3, add_self_loops=True)
    x = mx.random.normal((5, 4))
    out = conv(mx.zeros((2, 0), dtype=mx.int32), x)
    # only self-loops: every node keeps its own projected features
    np.testing.assert_allclose(
        np.array(out), np.array(conv.linear(x) + conv.bias), rtol=2e-2, atol=2e-2
    )


def test_gcn_conv_adds_self_loops_by_default():
    # As PyG's GCNConv (mlx-graphs 0.0.11 and earlier: add_self_loops=False).
    torch = pytest.importorskip("torch")
    pyg_nn = pytest.importorskip("torch_geometric.nn")
    assert GCNConv(4, 3)._add_self_loops

    rng = np.random.default_rng(0)
    edge_index = rng.integers(0, 10, (2, 30))
    x = rng.standard_normal((12, 4)).astype(np.float32)
    ref = pyg_nn.GCNConv(4, 3)
    conv = GCNConv(4, 3)
    conv.linear.weight = mx.array(ref.lin.weight.detach().numpy())
    conv.bias = mx.array(ref.bias.detach().numpy())
    with torch.no_grad():
        expected = ref(torch.tensor(x), torch.tensor(edge_index)).numpy()
    out = conv(mx.array(edge_index), mx.array(x))
    np.testing.assert_allclose(np.array(out), expected, rtol=2e-2, atol=2e-2)
