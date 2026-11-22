import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_graphs.nn.conv.gatv2_conv import GATv2Conv
from mlx_graphs.nn.kernels import gatv2_edge_scores
from mlx_graphs.nn.kernels.gatv2_edge_scores import _metal_scores, _reference_scores

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


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
@pytest.mark.parametrize(
    "channels, index_dtype, slope",
    [(3, mx.int32, 0.0), (32, mx.uint32, 0.2), (65, mx.int64, 0.4)],
)
def test_gatv2_metal_scores_and_gradients(channels, index_dtype, slope):
    rng = np.random.default_rng(23)
    # Include exact zeros in the activation, as well as repeated edges and a
    # partial final threadgroup. Use different node counts and strided inputs.
    src = rng.standard_normal((7, 2, channels * 2)).astype(np.float32)
    dst = rng.standard_normal((8, 2, channels)).astype(np.float32)
    src[..., 0] = dst[..., 0] = 0
    att = rng.standard_normal((1, 2, channels)).astype(np.float32)
    edge_pairs = rng.integers(0, 7, (37, 2))
    if index_dtype == mx.int64:
        edge_pairs[0] = (-1, -1)
    edges = mx.array(edge_pairs, dtype=index_dtype).T
    cotangent = mx.array(rng.standard_normal((2, 37)).astype(np.float32)).T
    primals = (mx.array(src)[..., ::2], mx.array(dst), mx.array(att))
    metal = _metal_scores(slope)

    def reference(s, d, a):
        return _reference_scores(s, d, a, edges, slope)

    scores, grads = mx.vjp(lambda s, d, a: metal(s, d, a, edges), primals, [cotangent])
    expected, expected_grads = mx.vjp(reference, primals, [cotangent])
    for got, want in zip([*scores, *grads], [*expected, *expected_grads]):
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("case", ["cpu", "float16", "edge-features", "empty"])
def test_gatv2_score_dispatch_falls_back(monkeypatch, case):
    def unexpected_metal(slope):
        pytest.fail("The Metal score kernel was selected for a fallback case")

    monkeypatch.setattr(gatv2_edge_scores, "_metal_scores", unexpected_metal)
    device = mx.cpu if case == "cpu" else mx.default_device()
    dtype = mx.float16 if case == "float16" else mx.float32
    with mx.stream(device):
        edges = mx.array([[0, 1, 1], [1, 0, 0]])
        if case == "empty":
            edges = edges[:, :0]
        src = mx.arange(12).reshape(2, 2, 3).astype(dtype)
        dst = src - 8
        weights = mx.ones((1, 2, 3), dtype=dtype)
        edge_feats = (
            mx.ones((3, 2, 3), dtype=dtype) if case == "edge-features" else None
        )
        args = (src, dst, weights, edges, 0.2, edge_feats)
        out = gatv2_edge_scores.gatv2_edge_scores(*args)
        np.testing.assert_array_equal(np.array(out), np.array(_reference_scores(*args)))


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
def test_gatv2_metal_score_transforms():
    edges = mx.array([[0, 1, 1, 2], [1, 2, 2, 0]])
    src, dst, att = [
        mx.random.normal(shape, key=mx.random.key(i))
        for i, shape in enumerate(((3, 2, 8), (3, 2, 8), (1, 2, 8)))
    ]
    metal = _metal_scores(0.2)

    def fused(s, d, a):
        return metal(s, d, a, edges)

    def reference(s, d, a):
        return _reference_scores(s, d, a, edges, 0.2)

    def transformed(scores):
        def loss(s, d, a):
            return scores(s, d, a).square().sum()

        def grad_sum(s):
            return mx.grad(loss)(s, dst, att).sum()

        primals = (src, dst, att)
        tangents = [mx.ones_like(x) for x in primals]
        _, (jvp,) = mx.jvp(scores, primals, tangents)
        return mx.compile(scores)(*primals), jvp, mx.grad(grad_sum)(src)

    for got, want in zip(transformed(fused), transformed(reference)):
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-5, atol=1e-5)
