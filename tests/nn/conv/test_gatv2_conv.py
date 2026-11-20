import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_graphs.nn.conv.gatv2_conv import GATv2Conv

mx.random.seed(42)


def test_gat_conv():
    conv = GATv2Conv(8, 20, heads=1)

    node_features = mx.random.uniform(0, 1, [6, 8])
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    y_hat1 = conv(edge_index, node_features)

    node_features = mx.random.uniform(-1, 1, [6, 8])
    edge_index = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
    y_hat2 = conv(edge_index, node_features)

    conv = GATv2Conv(16, 32, heads=1)
    node_features = mx.random.uniform(0, 1, [100, 16])
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    y_hat3 = conv(edge_index, node_features)

    conv = GATv2Conv(16, 32, heads=3, concat=True)
    node_features = mx.random.uniform(0, 1, [100, 16])
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    y_hat4 = conv(edge_index, node_features)

    conv = GATv2Conv(16, 32, heads=3, concat=False)
    node_features = mx.random.uniform(0, 1, [100, 16])
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    y_hat5 = conv(edge_index, node_features)

    conv = GATv2Conv(16, 32, heads=3, concat=False, edge_features_dim=10)
    node_features = mx.random.uniform(0, 1, [100, 16])
    edge_features = mx.random.uniform(0, 1, [5, 10])
    edge_index = mx.array([[0, 1, 2, 3, 50], [0, 0, 1, 1, 99]])
    y_hat6 = conv(edge_index, node_features, edge_features=edge_features)

    assert y_hat1.shape == (6, 20), "Simple GATv2Conv failed"
    assert y_hat2.shape == (6, 20), "GATv2Conv with negative values failed"
    assert y_hat3.shape == (100, 32), "GATv2Conv with different shapes failed"
    assert y_hat4.shape == (100, 32 * 3), "GATv2Conv with multiple heads concat failed"
    assert y_hat5.shape == (
        100,
        32,
    ), "GATv2Conv with multiple heads without concat failed"
    assert y_hat6.shape == (100, 32), "GATv2Conv with edge features failed"


# ---------------------------------------------------------------------------
# Numerical parity with PyG, and fused vs gather-scatter aggregation
# ---------------------------------------------------------------------------


def _random_graph(num_nodes=40, num_edges=300, dim=12, seed=0):
    rng = np.random.default_rng(seed)
    # duplicate edges and isolated nodes (the last 5 nodes) on purpose
    edge_index = rng.integers(0, num_nodes - 5, (2, num_edges))
    x = rng.standard_normal((num_nodes, dim)).astype(np.float32)
    return edge_index, x


@pytest.mark.parametrize("heads, concat", [(1, True), (4, True), (3, False)])
@pytest.mark.parametrize("edge_dim", [None, 5])
def test_gatv2_conv_matches_pyg(heads, concat, edge_dim):
    torch = pytest.importorskip("torch")
    pyg_nn = pytest.importorskip("torch_geometric.nn")

    edge_index, x = _random_graph()
    edge_features = None
    if edge_dim is not None:
        edge_features = (
            np.random.default_rng(1)
            .standard_normal((edge_index.shape[1], edge_dim))
            .astype(np.float32)
        )

    ref = pyg_nn.GATv2Conv(
        x.shape[1],
        8,
        heads=heads,
        concat=concat,
        add_self_loops=False,
        edge_dim=edge_dim,
    )
    # mlx-graphs' projections have no bias
    torch.nn.init.zeros_(ref.lin_l.bias)
    torch.nn.init.zeros_(ref.lin_r.bias)

    conv = GATv2Conv(
        x.shape[1], 8, heads=heads, concat=concat, edge_features_dim=edge_dim
    )
    conv.W_src.weight = mx.array(ref.lin_l.weight.detach().numpy())
    conv.W_dst.weight = mx.array(ref.lin_r.weight.detach().numpy())
    conv.att = mx.array(ref.att.detach().numpy())
    conv.bias = mx.array(ref.bias.detach().numpy())
    if edge_dim is not None:
        conv.edge_lin_proj.weight = mx.array(ref.lin_edge.weight.detach().numpy())

    expected = ref(
        torch.tensor(x),
        torch.tensor(edge_index),
        None if edge_features is None else torch.tensor(edge_features),
    )
    out = conv(
        mx.array(edge_index),
        mx.array(x),
        None if edge_features is None else mx.array(edge_features),
    )
    # tolerance limited by the accuracy of MLX's GPU matmul (~1e-3 relative)
    np.testing.assert_allclose(
        np.array(out), expected.detach().numpy(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize("fmt", ["coo", "csr", "auto"])
def test_gatv2_conv_fused_matches_gather_scatter(fmt):
    from mlx_graphs.nn import clear_csr_cache, fused_message_passing_mode

    edge_index, x = _random_graph()
    edge_index, x = mx.array(edge_index), mx.array(x)
    conv = GATv2Conv(x.shape[1], 8, heads=4)

    def loss(model, x):
        return (model(edge_index, x) ** 2).sum()

    clear_csr_cache()
    results = []
    for enabled in (True, False):
        with fused_message_passing_mode(enabled, format=fmt if enabled else None):
            for _ in range(6):
                out = conv(edge_index, x)
                _, grads = nn.value_and_grad(conv, loss)(conv, x)
            results.append((out, grads))

    (out_f, g_f), (out_r, g_r) = results
    np.testing.assert_allclose(np.array(out_f), np.array(out_r), rtol=5e-3, atol=5e-3)
    for (_, a), (_, b) in zip(nn.utils.tree_flatten(g_f), nn.utils.tree_flatten(g_r)):
        np.testing.assert_allclose(np.array(a), np.array(b), rtol=2e-2, atol=2e-2)
