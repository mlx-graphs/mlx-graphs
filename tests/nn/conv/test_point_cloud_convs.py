"""Point cloud and mesh layers compared with PyG (copied weights): outputs and
gradients of the inputs and of every parameter."""

import contextlib
import itertools
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

from mlx_graphs.nn import (
    FeaStConv,
    GMMConv,
    PointConv,
    PointNetConv,
    PointTransformerConv,
    PPFConv,
    SplineConv,
    XConv,
)
from mlx_graphs.nn.conv.spline_conv import spline_basis, spline_weighting
from tests.pyg_copy import load_torch, torch_state

TOL = 2e-2


def assert_close(actual, expected, tol=TOL, name=""):
    actual, expected = np.array(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    scale = max(1.0, float(np.abs(expected).max()))
    np.testing.assert_allclose(
        actual, expected, rtol=tol, atol=tol * scale, err_msg=name
    )


def point_cloud(seed=0, num_nodes=40, dim=8, k=6):
    """Random points with a k-NN graph (plus a few self-loops) and two
    isolated destination points. There are no duplicate edges: they tie in max
    aggregations, whose gradients PyTorch splits between the ties."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(num_nodes, 3)).astype(np.float32)
    x = rng.normal(size=(num_nodes, dim)).astype(np.float32)
    d = ((pos[:, None] - pos[None]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nbr = np.argsort(d, axis=1)[:, :k]
    edges = np.stack([nbr.reshape(-1), np.repeat(np.arange(num_nodes), k)])
    edges = edges[:, edges[1] < num_nodes - 2]
    loops = np.array([[0, 3], [0, 3]])
    edges = np.concatenate([edges, loops], axis=1)
    return pos, x, edges


def compare_grads(ref, layer, torch_inputs, mlx_inputs, torch_fn, mlx_fn, renames):
    """Gradients of ``sum(out * w)`` w.r.t. every parameter and the inputs."""
    out_t = torch_fn(*torch_inputs)
    w = np.random.default_rng(1).normal(size=tuple(out_t.shape)).astype(np.float32)
    (out_t * torch.tensor(w)).sum().backward()

    def loss(params, *inputs):
        layer.update(params)
        return (mlx_fn(*inputs) * mx.array(w)).sum()

    argnums = tuple(range(len(mlx_inputs) + 1))
    grads = mx.grad(loss, argnums=argnums)(layer.trainable_parameters(), *mlx_inputs)
    param_grads = dict(tree_flatten(grads[0]))
    expected = {}
    for name, p in ref.named_parameters():
        key = name
        for pattern, repl in renames:
            key = re.sub(pattern, repl, key)
        expected[key] = p.grad.numpy()
    assert set(param_grads) == set(expected), (set(param_grads), set(expected))
    for name, g in param_grads.items():
        assert_close(g, expected[name], name=name)
    for i, (t, g) in enumerate(zip(torch_inputs, grads[1:])):
        assert_close(g, t.grad.numpy(), name=f"input {i}")


@contextlib.contextmanager
def exact_device(enabled=True):
    """Runs MLX on the CPU: max aggregations pick one message per channel, and
    the ~1e-3 error of the GPU matmuls can flip near-ties (gaps of ~1e-4 with
    these inputs), which moves the whole gradient to another edge."""
    if not enabled:
        yield
        return
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def leaf(a):
    return torch.tensor(a, requires_grad=True)


def mlp_pair(sizes):
    layers_t, layers_m = [], []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers_t.append(torch.nn.Linear(a, b))
        layers_m.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers_t.append(torch.nn.ReLU())
            layers_m.append(nn.ReLU())
    return torch.nn.Sequential(*layers_t), nn.Sequential(*layers_m)


SEQ = [(r"(local_nn|global_nn|pos_nn|attn_nn)\.(\d+)", r"\1.layers.\2")]


# ---------------------------------------------------------------- PointNetConv


def test_point_conv_alias():
    assert PointConv is PointNetConv


@pytest.mark.parametrize("with_x", [True, False])
@pytest.mark.parametrize("aggr", ["max", "add", "mean"])
@pytest.mark.parametrize("add_self_loops", [True, False])
def test_point_net_conv_matches_pyg(with_x, aggr, add_self_loops):
    torch.manual_seed(0)
    pos, x, edges = point_cloud()
    in_dim = (x.shape[1] if with_x else 0) + 3
    local_t, local_m = mlp_pair([in_dim, 16, 16])
    global_t, global_m = mlp_pair([16, 12])
    ref = pyg_nn.PointNetConv(
        local_t, global_t, add_self_loops=add_self_loops, aggr=aggr
    )
    layer = PointNetConv(local_m, global_m, add_self_loops=add_self_loops, aggr=aggr)
    load_torch(layer, torch_state(ref, SEQ))

    xt, post = leaf(x), leaf(pos)
    ei = torch.tensor(edges)
    em = mx.array(edges)
    with exact_device(aggr == "max"):
        _compare_point_net(ref, layer, with_x, x, pos, xt, post, ei, em)
    expected = ref(xt if with_x else None, post, ei).detach().numpy()
    out = layer(em, mx.array(x) if with_x else None, mx.array(pos))
    assert_close(out, expected)


def _compare_point_net(ref, layer, with_x, x, pos, xt, post, ei, em):
    if with_x:
        compare_grads(
            ref,
            layer,
            [xt, post],
            [mx.array(x), mx.array(pos)],
            lambda a, p: ref(a, p, ei),
            lambda a, p: layer(em, a, p),
            SEQ,
        )
    else:
        compare_grads(
            ref,
            layer,
            [post],
            [mx.array(pos)],
            lambda p: ref(None, p, ei),
            lambda p: layer(em, None, p),
            SEQ,
        )


def test_point_net_conv_bipartite_matches_pyg():
    """PointNet++ set abstraction: sampled centers aggregate their neighbors."""
    torch.manual_seed(0)
    pos, x, _ = point_cloud(num_nodes=30)
    rng = np.random.default_rng(3)
    idx = np.sort(rng.choice(30, 10, replace=False))
    row = np.repeat(np.arange(10), 4)
    col = rng.integers(0, 30, 40)
    edges = np.stack([col, row])
    local_t, local_m = mlp_pair([x.shape[1] + 3, 16])
    ref = pyg_nn.PointNetConv(local_t)
    layer = PointNetConv(local_m)
    load_torch(layer, torch_state(ref, SEQ))
    expected = ref(
        (torch.tensor(x), None),
        (torch.tensor(pos), torch.tensor(pos[idx])),
        torch.tensor(edges),
    )
    out = layer(
        mx.array(edges), (mx.array(x), None), (mx.array(pos), mx.array(pos[idx]))
    )
    assert out.shape == (10, 16)
    assert_close(out, expected.detach().numpy())


def test_point_net_conv_without_edges():
    layer = PointNetConv(
        nn.Sequential(nn.Linear(3, 4), nn.ReLU()), add_self_loops=False
    )
    out = layer(mx.zeros((2, 0), dtype=mx.int32), None, mx.ones((5, 3)))
    assert out.shape == (5, 4) and mx.all(out == 0).item()


# -------------------------------------------------------- PointTransformerConv


@pytest.mark.parametrize("nets", [False, True])
def test_point_transformer_conv_matches_pyg(nets):
    torch.manual_seed(0)
    pos, x, edges = point_cloud()
    kwargs_t, kwargs_m = {}, {}
    if nets:
        pos_t, pos_m = mlp_pair([3, 16, 12])
        attn_t, attn_m = mlp_pair([12, 16, 12])
        kwargs_t = dict(pos_nn=pos_t, attn_nn=attn_t)
        kwargs_m = dict(pos_nn=pos_m, attn_nn=attn_m)
    ref = pyg_nn.PointTransformerConv(x.shape[1], 12, **kwargs_t)
    layer = PointTransformerConv(x.shape[1], 12, **kwargs_m)
    load_torch(layer, torch_state(ref, SEQ))
    ei, em = torch.tensor(edges), mx.array(edges)
    compare_grads(
        ref,
        layer,
        [leaf(x), leaf(pos)],
        [mx.array(x), mx.array(pos)],
        lambda a, p: ref(a, p, ei),
        lambda a, p: layer(em, a, p),
        SEQ,
    )
    expected = ref(torch.tensor(x), torch.tensor(pos), ei).detach().numpy()
    assert_close(layer(em, mx.array(x), mx.array(pos)), expected)


def test_point_transformer_conv_bipartite_matches_pyg():
    torch.manual_seed(0)
    pos, x, _ = point_cloud(num_nodes=30)
    rng = np.random.default_rng(4)
    x_dst = rng.normal(size=(10, 5)).astype(np.float32)
    edges = np.stack([rng.integers(0, 30, 40), np.repeat(np.arange(10), 4)])
    ref = pyg_nn.PointTransformerConv((x.shape[1], 5), 12)
    layer = PointTransformerConv((x.shape[1], 5), 12)
    load_torch(layer, torch_state(ref))
    expected = ref(
        (torch.tensor(x), torch.tensor(x_dst)),
        (torch.tensor(pos), torch.tensor(pos[:10])),
        torch.tensor(edges),
    )
    out = layer(
        mx.array(edges),
        (mx.array(x), mx.array(x_dst)),
        (mx.array(pos), mx.array(pos[:10])),
    )
    assert_close(out, expected.detach().numpy())


# ------------------------------------------------------------------- PPFConv


@pytest.mark.parametrize("with_x", [True, False])
def test_ppf_conv_matches_pyg(with_x):
    torch.manual_seed(0)
    pos, x, edges = point_cloud()
    normal = np.random.default_rng(5).normal(size=pos.shape).astype(np.float32)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    in_dim = (x.shape[1] if with_x else 0) + 4
    local_t, local_m = mlp_pair([in_dim, 16, 16])
    global_t, global_m = mlp_pair([16, 8])
    ref = pyg_nn.PPFConv(local_t, global_t)
    layer = PPFConv(local_m, global_m)
    load_torch(layer, torch_state(ref, SEQ))
    ei, em = torch.tensor(edges), mx.array(edges)
    pt, nt = torch.tensor(pos), torch.tensor(normal)
    pm, nm = mx.array(pos), mx.array(normal)
    if with_x:
        with exact_device():
            compare_grads(
                ref,
                layer,
                [leaf(x)],
                [mx.array(x)],
                lambda a: ref(a, pt, nt, ei),
                lambda a: layer(em, a, pm, nm),
                SEQ,
            )
    xt = torch.tensor(x) if with_x else None
    expected = ref(xt, pt, nt, ei).detach().numpy()
    out = layer(em, mx.array(x) if with_x else None, pm, nm)
    assert_close(out, expected)


def test_ppf_conv_gradients_are_finite():
    """Self-loops give zero vectors, whose norm and angle have finite
    gradients."""
    pos, x, edges = point_cloud()
    layer = PPFConv(nn.Sequential(nn.Linear(x.shape[1] + 4, 8)))

    def loss(p):
        return layer(mx.array(edges), mx.array(x), p, p).sum()

    grad = mx.grad(loss)(mx.array(pos))
    assert mx.all(mx.isfinite(grad)).item()


# ------------------------------------------------------------------ FeaStConv


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("add_self_loops", [True, False])
def test_feast_conv_matches_pyg(heads, add_self_loops):
    torch.manual_seed(0)
    _, x, edges = point_cloud()
    ref = pyg_nn.FeaStConv(x.shape[1], 12, heads=heads, add_self_loops=add_self_loops)
    layer = FeaStConv(x.shape[1], 12, heads=heads, add_self_loops=add_self_loops)
    load_torch(layer, torch_state(ref))
    ei, em = torch.tensor(edges), mx.array(edges)
    compare_grads(
        ref,
        layer,
        [leaf(x)],
        [mx.array(x)],
        lambda a: ref(a, ei),
        lambda a: layer(em, a),
        [],
    )


# -------------------------------------------------------------------- GMMConv


@pytest.mark.parametrize("separate_gaussians", [False, True])
@pytest.mark.parametrize("root_weight", [True, False])
@pytest.mark.parametrize("aggr", ["mean", "add"])
def test_gmm_conv_matches_pyg(separate_gaussians, root_weight, aggr):
    torch.manual_seed(0)
    _, x, edges = point_cloud()
    pseudo = np.random.default_rng(6).uniform(size=(edges.shape[1], 3))
    pseudo = pseudo.astype(np.float32)
    kwargs = dict(
        separate_gaussians=separate_gaussians, root_weight=root_weight, aggr=aggr
    )
    ref = pyg_nn.GMMConv(x.shape[1], 6, dim=3, kernel_size=4, **kwargs)
    torch.nn.init.normal_(ref.bias)  # zeros in PyG, which hides bias bugs
    if not root_weight:
        ref.root = None  # PyG's forward reads `self.root`, never set then
    layer = GMMConv(x.shape[1], 6, dim=3, kernel_size=4, **kwargs)
    load_torch(layer, torch_state(ref))
    ei, em = torch.tensor(edges), mx.array(edges)
    compare_grads(
        ref,
        layer,
        [leaf(x), leaf(pseudo)],
        [mx.array(x), mx.array(pseudo)],
        lambda a, e: ref(a, ei, e),
        lambda a, e: layer(em, a, e),
        [],
    )


# ----------------------------------------------------------------- SplineConv


def torch_spline_basis(pseudo, kernel_size, is_open_spline, degree):
    """Line-by-line port of torch-spline-conv's CPU ``spline_basis`` (which
    PyG >= 2.7 runs from pyg-lib), differentiable with autograd."""

    def basis_1d(v, k_mod):
        if degree == 1:
            return 1.0 - v - k_mod + 2.0 * v * k_mod
        if degree == 2:
            return [0.5 * v * v - v + 0.5, -v * v + v + 0.5, 0.5 * v * v][k_mod]
        return [
            (1.0 - v) * (1.0 - v) * (1.0 - v) / 6.0,
            (3.0 * v * v * v - 6.0 * v * v + 4.0) / 6.0,
            (-3.0 * v * v * v + 3.0 * v * v + 3.0 * v + 1.0) / 6.0,
            v * v * v / 6.0,
        ][k_mod]

    E, D = pseudo.shape
    S = (degree + 1) ** D
    kernel_size, is_open = kernel_size.tolist(), is_open_spline.tolist()
    basis = [[None] * S for _ in range(E)]
    weight_index = torch.empty((E, S), dtype=torch.long)
    for e in range(E):
        for s in range(S):
            k, wi, wi_offset, b = s, 0, 1, torch.tensor(1.0)
            for d in range(D):
                k_mod = k % (degree + 1)
                k //= degree + 1
                v = pseudo[e, d] * (kernel_size[d] - degree * is_open[d])
                wi += ((int(v.item()) + k_mod) % kernel_size[d]) * wi_offset
                wi_offset *= kernel_size[d]
                v = v - float(np.floor(v.item()))
                b = b * basis_1d(v, k_mod)
            basis[e][s] = b
            weight_index[e, s] = wi
    return torch.stack([torch.stack(row) for row in basis]), weight_index


def torch_spline_weighting(x, weight, basis, weight_index):
    out = 0
    for s in range(basis.shape[1]):
        w = weight[weight_index[:, s]]  # [E, in, out]
        out = out + basis[:, s : s + 1] * torch.bmm(x[:, None, :], w)[:, 0]
    return out


@pytest.fixture
def pyg_spline(monkeypatch):
    """PyG's SplineConv runs pyg-lib's spline ops, ported above."""
    import torch_geometric.nn.conv.spline_conv as module

    monkeypatch.setattr(module, "spline_basis", torch_spline_basis)
    monkeypatch.setattr(module, "spline_weighting", torch_spline_weighting)
    return module.SplineConv


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("is_open_spline", [True, False, [True, False, True]])
def test_spline_basis_matches_reference(degree, is_open_spline):
    rng = np.random.default_rng(degree)
    pseudo = rng.uniform(size=(25, 3)).astype(np.float32)
    pseudo[:3] = [[0, 0, 0], [1, 1, 1], [0.5, 1, 0]]  # boundaries
    kernel = [5, 3, 4]
    opens = is_open_spline if isinstance(is_open_spline, list) else [is_open_spline] * 3
    ref_basis, ref_index = torch_spline_basis(
        torch.tensor(pseudo),
        torch.tensor(kernel),
        torch.tensor(opens, dtype=torch.uint8),
        degree,
    )
    basis, index = spline_basis(mx.array(pseudo), kernel, opens, degree)
    assert_close(basis, ref_basis.numpy(), tol=1e-4)
    # weights with a zero coefficient may differ at the boundaries
    active = ref_basis.numpy() > 1e-6
    np.testing.assert_array_equal(np.array(index)[active], ref_index.numpy()[active])

    try:
        import torch_spline_conv
    except ImportError:
        return
    b, i = torch_spline_conv.spline_basis(
        torch.tensor(pseudo),
        torch.tensor(kernel),
        torch.tensor(opens, dtype=torch.uint8),
        degree,
    )
    np.testing.assert_allclose(b.numpy(), ref_basis.numpy(), atol=1e-6)
    np.testing.assert_array_equal(i.numpy(), ref_index.numpy())


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("root_weight", [True, False])
def test_spline_conv_matches_pyg(pyg_spline, degree, root_weight):
    torch.manual_seed(0)
    _, x, edges = point_cloud(num_nodes=20, k=3)
    pseudo = np.random.default_rng(7).uniform(size=(edges.shape[1], 2))
    pseudo = pseudo.astype(np.float32)
    kwargs = dict(degree=degree, root_weight=root_weight, is_open_spline=[True, False])
    ref = pyg_spline(x.shape[1], 6, dim=2, kernel_size=[3, 4], **kwargs)
    torch.nn.init.normal_(ref.bias)
    layer = SplineConv(x.shape[1], 6, dim=2, kernel_size=[3, 4], **kwargs)
    load_torch(layer, torch_state(ref, drop=["kernel_size", "is_open_spline"]))
    ei, em = torch.tensor(edges), mx.array(edges)
    compare_grads(
        ref,
        layer,
        [leaf(x), leaf(pseudo)],
        [mx.array(x), mx.array(pseudo)],
        lambda a, e: ref(a, ei, e),
        lambda a, e: layer(em, a, e),
        [],
    )


def test_spline_weighting_matches_reference():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(30, 5)).astype(np.float32)
    weight = rng.normal(size=(12, 5, 7)).astype(np.float32)
    pseudo = rng.uniform(size=(30, 2)).astype(np.float32)
    basis, index = spline_basis(mx.array(pseudo), [3, 4], [True, True], 1)
    out = spline_weighting(mx.array(x), mx.array(weight), basis, index)
    expected = torch_spline_weighting(
        torch.tensor(x),
        torch.tensor(weight),
        torch.tensor(np.array(basis)),
        torch.tensor(np.array(index)).long(),
    )
    assert_close(out, expected.numpy())


# ---------------------------------------------------------------------- XConv

XCONV_RENAMES = [
    (r"^mlp1\.(\d+)", r"mlp1.layers.\1"),
    (r"^mlp2\.0\.", "mlp2_lin."),
    (r"^mlp2\.2\.", "mlp2_bn1."),
    (r"^mlp2\.4\.", "mlp2_conv1."),
    (r"^mlp2\.6\.", "mlp2_bn2."),
    (r"^mlp2\.8\.", "mlp2_conv2."),
    (r"^mlp2\.9\.", "mlp2_bn3."),
    (r"^conv\.0\.", "conv."),
    (r"^conv\.2\.", "lin."),
]


def pyg_xconv_forward(module, x, pos, knn_index):
    """PyG's ``XConv.forward``, with the k-NN graph (pyg-lib's knn) given."""
    (N, D), K = pos.size(), module.kernel_size
    edge_index = knn_index
    if module.dilation > 1:
        edge_index = edge_index[:, :: module.dilation]
    row, col = edge_index[0], edge_index[1]
    pos = pos[col] - pos[row]
    x_star = module.mlp1(pos)
    if x is not None:
        x = x[col].view(N, K, module.in_channels)
        x_star = torch.cat([x_star, x], dim=-1)
    x_star = x_star.transpose(1, 2).contiguous()
    transform_matrix = module.mlp2(pos.view(N, K * D))
    return module.conv(torch.matmul(x_star, transform_matrix))


@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("training", [True, False])
def test_x_conv_matches_pyg(monkeypatch, dilation, training):
    import torch_geometric.nn.conv.x_conv as module

    monkeypatch.setattr(module.torch_geometric.typing, "WITH_KNN", True)
    torch.manual_seed(0)
    rng = np.random.default_rng(9)
    batch = np.repeat([0, 1], [20, 15])
    pos = rng.normal(size=(35, 3)).astype(np.float32)
    x = rng.normal(size=(35, 8)).astype(np.float32)
    K = 4
    ref = module.XConv(
        8, 12, dim=3, kernel_size=K, hidden_channels=4, dilation=dilation
    )
    layer = XConv(8, 12, dim=3, kernel_size=K, hidden_features_dim=4, dilation=dilation)
    load_torch(layer, torch_state(ref, XCONV_RENAMES))
    ref.train(training)
    layer.train(training)

    # brute-force k-NN (with the point itself), closest first
    d = ((pos[:, None] - pos[None]) ** 2).sum(-1)
    d[batch[:, None] != batch[None]] = np.inf
    nbr = np.argsort(d, axis=1, kind="stable")[:, : K * dilation]
    knn_index = torch.tensor(
        np.stack([np.repeat(np.arange(35), K * dilation), nbr.reshape(-1)])
    )

    bm = mx.array(batch)
    compare_grads(
        ref,
        layer,
        [leaf(x), leaf(pos)],
        [mx.array(x), mx.array(pos)],
        lambda a, p: pyg_xconv_forward(ref, a, p, knn_index),
        lambda a, p: layer(a, p, bm),
        XCONV_RENAMES,
    )


def test_x_conv_needs_enough_points():
    layer = XConv(4, 8, dim=3, kernel_size=6)
    with pytest.raises(ValueError, match="at least"):
        layer(mx.ones((10, 4)), mx.random.normal((10, 3)), mx.array([0] * 5 + [1] * 5))


def test_point_layers_in_a_model():
    """A small PointNet++ step: fps, radius, PointNetConv, global pooling."""
    from mlx_graphs.nn import global_max_pool
    from mlx_graphs.utils import fps, radius

    pos = mx.random.normal((64, 3))
    batch = mx.array(np.repeat([0, 1], 32))
    idx = fps(pos, batch, ratio=0.5)
    row, col = radius(pos, pos[idx], 1.0, batch, batch[idx], max_num_neighbors=8)
    conv = PointNetConv(nn.Sequential(nn.Linear(3, 8), nn.ReLU()))
    out = conv(mx.stack([col, row]), (None, None), (pos, pos[idx]))
    pooled = global_max_pool(out, batch[idx])
    assert pooled.shape == (2, 8)
    assert list(itertools.chain(np.array(batch[idx]).tolist())) == [0] * 16 + [1] * 16
