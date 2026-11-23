"""PyG-parity tests of the sparse hierarchical pooling operators (EdgePooling,
ASAPooling, MemPooling) and of GraphUNet: outputs, auxiliary losses and
gradients."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

from mlx_graphs.nn import (
    ASAPooling,
    EdgePooling,
    GCNConv,
    GraphConv,
    GraphUNet,
    MemPooling,
)
from mlx_graphs.nn.pooling._sparse import spspmm
from tests.pyg_copy import load_torch, torch_grads, torch_state

DIM = 6
# 3 graphs with 9, 6 and 7 nodes
SIZES = [9, 6, 7]
BATCH = np.repeat(np.arange(len(SIZES)), SIZES).astype(np.int64)
NUM_NODES = BATCH.size
# MLX's GPU matmul is only accurate to ~1e-3 relative
TOL = 5e-3


def assert_close(actual, expected, tol=TOL, name=""):
    """Tolerance relative to the largest expected value."""
    actual, expected = np.array(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    assert np.all(np.isfinite(expected)), name
    scale = max(1.0, float(np.abs(expected).max(initial=0.0)))
    np.testing.assert_allclose(
        actual, expected, rtol=tol, atol=tol * scale, err_msg=name
    )


def make_graphs(seed=0, chords=4, weighted=False):
    """Undirected graphs made of a ring plus random chords (every node has at
    least 2 neighbours), with features and optional positive edge weights."""
    rng = np.random.default_rng(seed)
    edges, start = [], 0
    for size in SIZES:
        nodes = start + np.arange(size)
        ring = np.stack([nodes, np.roll(nodes, -1)])
        extra = rng.choice(nodes, size=(2, chords))
        extra = extra[:, extra[0] != extra[1]]
        e = np.concatenate([ring, extra], axis=1)
        edges.append(np.concatenate([e, e[::-1]], axis=1))
        start += size
    edge_index = np.unique(np.concatenate(edges, axis=1), axis=1).astype(np.int64)
    x = rng.normal(size=(NUM_NODES, DIM)).astype(np.float32)
    weights = None
    if weighted:
        weights = rng.uniform(0.5, 2.0, edge_index.shape[1]).astype(np.float32)
    return x, edge_index, weights


def check_grads(grads, ref, renames=(), convert=None):
    grads = dict(tree_flatten(grads))
    expected = torch_grads(ref, renames)
    if convert is not None:
        expected = {k: convert(k, v) for k, v in expected.items()}
    # torch has no gradient for unused parameters, MLX a zero one
    for name in set(grads) - set(expected):
        assert not np.any(np.array(grads[name])), name
        expected[name] = np.zeros(grads[name].shape)
    assert set(grads) == set(expected), (sorted(grads), sorted(expected))
    for name, grad in grads.items():
        assert_close(grad, expected[name], name=f"grad {name}")


@pytest.fixture
def pyg_coo_matmul(monkeypatch):
    """PyG's ASAPooling and GraphUNet multiply sparse CSR matrices, which
    torch builds without MKL (e.g. the macOS arm64 wheels) cannot do on the
    CPU. Run them with (coalesced) COO matrices instead: same products, same
    autograd support, same row-major order of the resulting edges."""
    from torch_geometric.nn.models import graph_unet
    from torch_geometric.nn.pool import asap
    from torch_geometric.utils import to_torch_coo_tensor

    monkeypatch.setattr(asap, "to_torch_csr_tensor", to_torch_coo_tensor)
    monkeypatch.setattr(graph_unet, "to_torch_csr_tensor", to_torch_coo_tensor)
    to_sparse_csr = torch.Tensor.to_sparse_csr

    def coo_to_sparse_csr(self, *args, **kwargs):
        if self.layout == torch.sparse_coo:
            return self
        return to_sparse_csr(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to_sparse_csr", coo_to_sparse_csr)


# ---------------------------------------------------------------------------
# spspmm
# ---------------------------------------------------------------------------


def test_spspmm_matches_dense():
    rng = np.random.default_rng(0)
    a_index = rng.integers(0, 5, (2, 12))  # with duplicates
    b_index = np.stack([rng.integers(0, 5, 9), rng.integers(0, 4, 9)])
    a_value = rng.normal(size=12).astype(np.float32)
    b_value = rng.normal(size=9).astype(np.float32)
    dense_a = np.zeros((5, 5), np.float32)
    np.add.at(dense_a, tuple(a_index), a_value)
    dense_b = np.zeros((5, 4), np.float32)
    np.add.at(dense_b, tuple(b_index), b_value)

    index, value = spspmm(
        mx.array(a_index), mx.array(a_value), mx.array(b_index), mx.array(b_value), 4
    )
    out = np.zeros((5, 4), np.float32)
    out[tuple(np.array(index))] = np.array(value)
    np.testing.assert_allclose(out, dense_a @ dense_b, rtol=1e-4, atol=1e-5)
    # coalesced: sorted by row then column, no duplicates
    keys = np.array(index[0]) * 4 + np.array(index[1])
    assert np.all(np.diff(keys) > 0)

    index, value = spspmm(mx.array(a_index), None, mx.array(b_index), None, 4)
    ones_product = (dense_a != 0).astype(np.float32)
    np.add.at(ones_product := np.zeros((5, 5), np.float32), tuple(a_index), 1)
    counts_b = np.zeros((5, 4), np.float32)
    np.add.at(counts_b, tuple(b_index), 1)
    out = np.zeros((5, 4), np.float32)
    out[tuple(np.array(index))] = np.array(value)
    np.testing.assert_allclose(out, ones_product @ counts_b)


def test_spspmm_gradients():
    """Gradients of the values of S^T A S (as in ASAPooling) match a dense
    torch computation."""
    rng = np.random.default_rng(1)
    a_index = np.unique(rng.integers(0, 6, (2, 15)), axis=1)
    s_index = np.unique(
        np.stack([rng.integers(0, 6, 10), rng.integers(0, 3, 10)]), axis=1
    )
    a_value = rng.uniform(0.5, 2, a_index.shape[1]).astype(np.float32)
    s_value = rng.uniform(0.1, 1, s_index.shape[1]).astype(np.float32)

    ta = torch.tensor(a_value, requires_grad=True)
    ts = torch.tensor(s_value, requires_grad=True)
    dense_a = torch.zeros(6, 6).index_put(
        (torch.tensor(a_index[0]), torch.tensor(a_index[1])), ta
    )
    dense_s = torch.zeros(6, 3).index_put(
        (torch.tensor(s_index[0]), torch.tensor(s_index[1])), ts
    )
    expected = dense_s.T @ dense_a @ dense_s
    (expected**2).sum().backward()

    def f(a, s):
        si = mx.array(s_index)
        idx, val = spspmm(mx.array(a_index), a, si, s, 3)
        idx, val = spspmm(mx.stack([si[1], si[0]]), s, idx, val, 3)
        return (val**2).sum(), (idx, val)

    (_, (idx, val)), (ga, gs) = mx.value_and_grad(f, argnums=(0, 1))(
        mx.array(a_value), mx.array(s_value)
    )
    out = np.zeros((3, 3), np.float32)
    out[tuple(np.array(idx))] = np.array(val)
    assert_close(out, expected.detach().numpy(), tol=1e-4)
    assert_close(ga, ta.grad.numpy(), tol=1e-4)
    assert_close(gs, ts.grad.numpy(), tol=1e-4)


# ---------------------------------------------------------------------------
# EdgePooling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["softmax", "tanh", "sigmoid"])
@pytest.mark.parametrize("add_to_edge_score", [0.5, 0.0])
def test_edge_pooling(method, add_to_edge_score):
    torch.manual_seed(0)
    x, edge_index, _ = make_graphs(seed=1)
    score_method = getattr(pyg_nn.EdgePooling, f"compute_edge_score_{method}")
    ref = pyg_nn.EdgePooling(
        DIM, edge_score_method=score_method, add_to_edge_score=add_to_edge_score
    )
    module = EdgePooling(
        DIM,
        edge_score_method=getattr(EdgePooling, f"compute_edge_score_{method}"),
        add_to_edge_score=add_to_edge_score,
    )
    load_torch(module, torch_state(ref))

    tx = torch.tensor(x, requires_grad=True)
    expected = ref(tx, torch.tensor(edge_index), torch.tensor(BATCH))
    info = expected[3]
    weights = np.random.default_rng(2).normal(size=tuple(expected[0].shape))
    unpooled = ref.unpool(expected[0], info)
    (
        (expected[0] * torch.tensor(weights, dtype=torch.float32)).sum()
        + (unpooled[0] ** 2).sum()
    ).backward()

    def loss_fn(x):
        out = module(mx.array(edge_index), x, mx.array(BATCH))
        up = module.unpool(out[0], out[3])
        loss = (out[0] * mx.array(weights.astype(np.float32))).sum()
        return loss + (up[0] ** 2).sum(), (out, up)

    (_, (out, up)), gx = mx.value_and_grad(loss_fn)(mx.array(x))
    _, grads = nn.value_and_grad(module, lambda: loss_fn(mx.array(x))[0])()

    new_x, new_edge_index, new_batch, new_info = out
    np.testing.assert_array_equal(np.array(new_info.cluster), info.cluster.numpy())
    np.testing.assert_array_equal(np.array(new_edge_index), expected[1].numpy())
    np.testing.assert_array_equal(np.array(new_batch), expected[2].numpy())
    assert_close(new_info.new_edge_score, info.new_edge_score.detach().numpy())
    assert_close(new_x, expected[0].detach().numpy(), name="x")
    assert_close(up[0], unpooled[0].detach().numpy(), name="unpooled x")
    np.testing.assert_array_equal(np.array(up[1]), edge_index)
    np.testing.assert_array_equal(np.array(up[2]), BATCH)
    assert_close(gx, tx.grad.numpy(), name="grad x")
    check_grads(grads, ref)


def test_edge_pooling_isolated_nodes_and_single_graph():
    torch.manual_seed(1)
    # nodes 5 and 6 have no edges: they are kept as their own clusters
    edge_index = np.array([[0, 1, 1, 2, 3, 4, 2, 3], [1, 0, 2, 1, 4, 3, 3, 2]])
    x = np.random.default_rng(3).normal(size=(7, DIM)).astype(np.float32)
    ref = pyg_nn.EdgePooling(DIM)
    module = EdgePooling(DIM)
    load_torch(module, torch_state(ref))
    expected = ref(
        torch.tensor(x), torch.tensor(edge_index), torch.zeros(7, dtype=torch.long)
    )
    out = module(mx.array(edge_index), mx.array(x))
    np.testing.assert_array_equal(np.array(out[3].cluster), expected[3].cluster)
    np.testing.assert_array_equal(np.array(out[1]), expected[1].numpy())
    assert_close(out[0], expected[0].detach().numpy())
    assert repr(module) == f"EdgePooling({DIM})"


def test_edge_pooling_ties_follow_edge_order():
    module = EdgePooling(2, edge_score_method=EdgePooling.compute_edge_score_tanh)
    module.lin.weight = mx.zeros_like(module.lin.weight)  # all scores equal
    edge_index = mx.array([[2, 0, 1], [3, 1, 2]])
    out = module(edge_index, mx.ones((4, 2)))
    # (2, 3) is contracted first, then (0, 1); (1, 2) touches merged nodes
    assert out[3].cluster.tolist() == [1, 1, 0, 0]


# ---------------------------------------------------------------------------
# ASAPooling
# ---------------------------------------------------------------------------

ASAP_RENAMES = [
    (r"gnn_intra_cluster\.lin_rel", "gnn_intra_cluster.neigh_proj"),
    (r"gnn_intra_cluster\.lin_root", "gnn_intra_cluster.self_proj"),
]


def _sorted_edges(edge_index, values=None):
    order = np.lexsort((edge_index[1], edge_index[0]))
    return edge_index[:, order], None if values is None else values[order]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(ratio=0.5),
        dict(ratio=0.8, add_self_loops=True),
        dict(ratio=3),
        dict(ratio=0.5, GNN="GraphConv"),
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_asap_pooling(kwargs, weighted, pyg_coo_matmul):
    torch.manual_seed(0)
    x, edge_index, weights = make_graphs(seed=4, weighted=weighted)
    pyg_kwargs, mlx_kwargs = dict(kwargs), dict(kwargs)
    if "GNN" in kwargs:
        pyg_kwargs["GNN"] = pyg_nn.GraphConv
        mlx_kwargs["GNN"] = GraphConv
    ref = pyg_nn.ASAPooling(DIM, **pyg_kwargs)
    module = ASAPooling(DIM, **mlx_kwargs)
    load_torch(module, torch_state(ref, ASAP_RENAMES))

    # The loss leaves out the pooled edge weights: PyG cannot backpropagate
    # through them (its sparse COO matrix of scores is built with the
    # non-differentiable `_coalesced_`). See test_spspmm_gradients for them.
    tx = torch.tensor(x, requires_grad=True)
    tw = None if weights is None else torch.tensor(weights)
    expected = ref(tx, torch.tensor(edge_index), tw, torch.tensor(BATCH))
    loss_weights = np.random.default_rng(5).normal(size=tuple(expected[0].shape))
    loss = (expected[0] * torch.tensor(loss_weights, dtype=torch.float32)).sum()
    loss.backward()

    mw = None if weights is None else mx.array(weights)

    def loss_fn(x):
        out = module(mx.array(edge_index), x, mw, mx.array(BATCH))
        return (out[0] * mx.array(loss_weights.astype(np.float32))).sum(), out

    (_, out), gx = mx.value_and_grad(loss_fn)(mx.array(x))
    _, grads = nn.value_and_grad(module, lambda: loss_fn(mx.array(x))[0])()

    new_x, new_edge_index, new_weights, new_batch, perm = out
    np.testing.assert_array_equal(np.array(perm), expected[4].numpy())
    np.testing.assert_array_equal(np.array(new_batch), expected[3].numpy())
    assert_close(new_x, expected[0].detach().numpy(), name="x")
    ref_edges, ref_weights = _sorted_edges(
        expected[1].numpy(),
        None if expected[2] is None else expected[2].detach().numpy(),
    )
    if kwargs.get("add_self_loops"):
        # self-loops are appended after the other edges, in node order
        mlx_edges, mlx_weights = _sorted_edges(
            np.array(new_edge_index),
            None if new_weights is None else np.array(new_weights),
        )
    else:
        mlx_edges = np.array(new_edge_index)
        mlx_weights = None if new_weights is None else np.array(new_weights)
    np.testing.assert_array_equal(mlx_edges, ref_edges)
    if weighted:
        assert_close(mlx_weights, ref_weights, name="edge weights")
    else:
        assert new_weights is None and expected[2] is None

    assert_close(gx, tx.grad.numpy(), name="grad x")
    check_grads(grads, ref, ASAP_RENAMES)


def test_asap_pooling_repr_and_single_graph():
    module = ASAPooling(DIM, ratio=0.5)
    assert repr(module) == f"ASAPooling({DIM}, ratio=0.5)"
    x, edge_index, _ = make_graphs()
    out = module(mx.array(edge_index), mx.array(x))
    assert out[0].shape == (int(np.ceil(0.5 * NUM_NODES)), DIM)
    assert out[2] is None


# ---------------------------------------------------------------------------
# MemPooling
# ---------------------------------------------------------------------------


def _mem_convert(name, value):
    # torch Conv2d weight [out, in, kh, kw] -> MLX [out, kh, kw, in]
    return value.transpose(0, 2, 3, 1) if name == "conv.weight" else value


@pytest.mark.parametrize("tau", [1.0, 0.5])
@pytest.mark.parametrize("dense_input", [False, True])
def test_mem_pooling(tau, dense_input):
    torch.manual_seed(0)
    heads, clusters, out_dim = 3, 4, 5
    ref = pyg_nn.MemPooling(DIM, out_dim, heads, clusters, tau=tau)
    module = MemPooling(DIM, out_dim, heads, clusters, tau=tau)
    load_torch(module, {k: _mem_convert(k, v) for k, v in torch_state(ref).items()})
    x, _, _ = make_graphs()
    tx = torch.tensor(x, requires_grad=True)
    if dense_input:
        from torch_geometric.utils import to_dense_batch

        dense, mask = to_dense_batch(tx, torch.tensor(BATCH))
        expected = ref(dense, mask=mask)
    else:
        expected = ref(tx, torch.tensor(BATCH))
    kl = pyg_nn.MemPooling.kl_loss(expected[1])
    loss_weights = np.random.default_rng(6).normal(size=tuple(expected[0].shape))
    (
        (expected[0] * torch.tensor(loss_weights, dtype=torch.float32)).sum() + kl
    ).backward()

    def loss_fn(x):
        if dense_input:
            from mlx_graphs.utils import to_dense_batch

            dense, mask = to_dense_batch(x, mx.array(BATCH))
            out = module(dense, mask=mask)
        else:
            out = module(x, mx.array(BATCH))
        kl = MemPooling.kl_loss(out[1])
        loss = (out[0] * mx.array(loss_weights.astype(np.float32))).sum() + kl
        return loss, (out, kl)

    (_, (out, mkl)), gx = mx.value_and_grad(loss_fn)(mx.array(x))
    _, grads = nn.value_and_grad(module, lambda: loss_fn(mx.array(x))[0])()

    assert_close(out[0], expected[0].detach().numpy(), name="x")
    assert_close(out[1], expected[1].detach().numpy(), name="S")
    assert_close(mkl, kl.detach().numpy(), name="kl_loss")
    assert_close(gx, tx.grad.numpy(), name="grad x")
    check_grads(grads, ref, convert=_mem_convert)


def test_mem_pooling_batch_size_and_repr():
    module = MemPooling(DIM, 4, heads=2, num_clusters=3)
    x, _, _ = make_graphs()
    out, S = module(mx.array(x), mx.array(BATCH), max_num_nodes=10, batch_size=5)
    assert out.shape == (5, 3, 4) and S.shape == (5, 10, 3)
    assert not np.any(np.array(S[3:]))
    assert repr(module) == f"MemPooling({DIM}, 4, heads=2, num_clusters=3)"


# ---------------------------------------------------------------------------
# GraphUNet
# ---------------------------------------------------------------------------

UNET_RENAMES = [(r"convs\.(\d+)\.lin\.", r"convs.\1.linear.")]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(depth=1),
        dict(depth=2),
        dict(depth=3, pool_ratios=[0.8, 0.5]),
        dict(depth=2, sum_res=False),
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_graph_unet(kwargs, weighted, pyg_coo_matmul):
    torch.manual_seed(0)
    x, edge_index, weights = make_graphs(seed=7, weighted=weighted)
    ref = pyg_nn.GraphUNet(DIM, 8, 3, **kwargs)
    for conv in list(ref.down_convs) + list(ref.up_convs):  # non-zero biases
        torch.nn.init.normal_(conv.bias, std=0.1)
    module = GraphUNet(DIM, 8, 3, **kwargs)
    load_torch(module, torch_state(ref, UNET_RENAMES))

    tx = torch.tensor(x, requires_grad=True)
    tw = None if weights is None else torch.tensor(weights)
    expected = ref(tx, torch.tensor(edge_index), torch.tensor(BATCH), tw)
    loss_weights = np.random.default_rng(8).normal(size=tuple(expected.shape))
    (expected * torch.tensor(loss_weights, dtype=torch.float32)).sum().backward()

    mw = None if weights is None else mx.array(weights)

    def loss_fn(x):
        out = module(mx.array(edge_index), x, mx.array(BATCH), mw)
        return (out * mx.array(loss_weights.astype(np.float32))).sum(), out

    (_, out), gx = mx.value_and_grad(loss_fn)(mx.array(x))
    _, grads = nn.value_and_grad(module, lambda: loss_fn(mx.array(x))[0])()

    assert_close(out, expected.detach().numpy(), name="out")
    assert_close(gx, tx.grad.numpy(), name="grad x")
    check_grads(grads, ref, UNET_RENAMES)


def test_graph_unet_augment_adj_matches_pyg(pyg_coo_matmul):
    x, edge_index, weights = make_graphs(seed=9, weighted=True)
    # with a self-loop, which is replaced by the identity
    edge_index = np.concatenate([edge_index, [[0], [0]]], axis=1)
    weights = np.concatenate([weights, [3.0]]).astype(np.float32)
    ref = pyg_nn.GraphUNet(DIM, 8, 3, depth=1)
    expected = ref.augment_adj(
        torch.tensor(edge_index), torch.tensor(weights), NUM_NODES
    )
    module = GraphUNet(DIM, 8, 3, depth=1)
    out = module.augment_adj(mx.array(edge_index), mx.array(weights), NUM_NODES)
    ref_edges, ref_weights = _sorted_edges(expected[0].numpy(), expected[1].numpy())
    np.testing.assert_array_equal(np.array(out[0]), ref_edges)
    np.testing.assert_allclose(np.array(out[1]), ref_weights, rtol=1e-5)


def test_graph_unet_repr_and_errors():
    module = GraphUNet(DIM, 8, 3, depth=2, pool_ratios=0.7)
    assert repr(module) == f"GraphUNet({DIM}, 8, 3, depth=2, pool_ratios=[0.7, 0.7])"
    with pytest.raises(ValueError):
        GraphUNet(DIM, 8, 3, depth=0)


@pytest.mark.parametrize("improved", [False, True])
def test_gcn_conv_improved_matches_pyg(improved):
    torch.manual_seed(0)
    x, edge_index, weights = make_graphs(seed=10, weighted=True)
    edge_index = np.concatenate([edge_index, [[1], [1]]], axis=1)  # a self-loop
    weights = np.concatenate([weights, [0.7]]).astype(np.float32)
    ref = pyg_nn.GCNConv(DIM, 4, improved=improved)
    torch.nn.init.normal_(ref.bias)
    module = GCNConv(DIM, 4, improved=improved)
    load_torch(module, torch_state(ref, [(r"^lin\.", "linear.")]))
    for w in (None, weights):
        # PyG ignores `improved` without edge weights (the self-loops of
        # `add_remaining_self_loops` only get `fill_value` with edge
        # attributes): compare with explicit unit weights instead.
        expected = ref(
            torch.tensor(x),
            torch.tensor(edge_index),
            torch.ones(edge_index.shape[1]) if w is None else torch.tensor(w),
        )
        out = module(
            mx.array(edge_index), mx.array(x), None if w is None else mx.array(w)
        )
        assert_close(out, expected.detach().numpy())


# ---------------------------------------------------------------------------
# PANConv / PANPooling
# ---------------------------------------------------------------------------


def _pan_reference(edge_index, x, lin, weight, num_nodes):
    """PyG's PANConv with dense matrices (PyG's own needs torch-sparse)."""
    adj_t = torch.zeros(num_nodes, num_nodes)
    adj_t.index_put_(
        (torch.tensor(edge_index[1]), torch.tensor(edge_index[0])),
        torch.ones(edge_index.shape[1]),
        accumulate=True,
    )
    tmp = torch.eye(num_nodes) * weight[0]
    total = tmp
    for i in range(1, weight.shape[0]):
        tmp = (tmp @ adj_t) * weight[i]
        total = total + tmp
    deg = (total.detach() != 0).sum(dim=1).float()
    d = deg.pow(-0.5)
    m = d[:, None] * total * d[None, :]
    return lin(m @ x), m


@pytest.mark.parametrize("filter_size", [1, 2, 3])
def test_pan_conv(filter_size):
    torch.manual_seed(0)
    x, edge_index, _ = make_graphs(seed=11)
    lin = torch.nn.Linear(DIM, 4)
    weight = torch.tensor(
        np.random.default_rng(12).uniform(0.2, 0.9, filter_size + 1),
        dtype=torch.float32,
        requires_grad=True,
    )
    tx = torch.tensor(x, requires_grad=True)
    expected, m = _pan_reference(edge_index, tx, lin, weight, NUM_NODES)
    loss_weights = np.random.default_rng(13).normal(size=tuple(expected.shape))
    loss = (expected * torch.tensor(loss_weights, dtype=torch.float32)).sum()
    loss.backward()

    from mlx_graphs.nn import PANConv

    module = PANConv(DIM, 4, filter_size)
    state = torch_state(lin, [(r"^", "lin.")])
    state["weight"] = weight.detach().numpy()
    load_torch(module, state)

    def loss_fn(x):
        out, m = module(mx.array(edge_index), x)
        return (out * mx.array(loss_weights.astype(np.float32))).sum(), (out, m)

    (_, (out, (m_index, m_weights))), gx = mx.value_and_grad(loss_fn)(mx.array(x))
    _, grads = nn.value_and_grad(module, lambda: loss_fn(mx.array(x))[0])()

    assert_close(out, expected.detach().numpy(), name="out")
    dense_m = np.zeros((NUM_NODES, NUM_NODES), np.float32)
    dense_m[tuple(np.array(m_index))] = np.array(m_weights)
    assert_close(dense_m, m.detach().numpy(), name="M")
    assert np.array_equal(dense_m != 0, m.detach().numpy() != 0)
    assert_close(gx, tx.grad.numpy(), name="grad x")
    grads = dict(tree_flatten(grads))
    assert_close(grads["weight"], weight.grad.numpy(), name="grad weight")
    assert_close(grads["lin.weight"], lin.weight.grad.numpy(), name="grad lin")
    assert_close(grads["lin.bias"], lin.bias.grad.numpy(), name="grad lin bias")
    assert repr(module) == f"PANConv({DIM}, 4, filter_size={filter_size})"


class _COOMatrix:
    """The ``coo()`` of a torch-sparse ``SparseTensor``, all PANPooling
    reads from its ``M`` input."""

    def __init__(self, row, col, value):
        self.row, self.col, self.value = row, col, value

    def coo(self):
        return self.row, self.col, self.value


@pytest.mark.parametrize(
    "kwargs",
    [dict(ratio=0.5), dict(ratio=0.7, multiplier=2.0), dict(min_score=0.1)],
)
def test_pan_pooling(kwargs):
    from mlx_graphs.nn import PANConv, PANPooling

    torch.manual_seed(0)
    x, edge_index, _ = make_graphs(seed=14)
    conv = PANConv(DIM, DIM, filter_size=2)
    conv.weight = mx.array([0.9, 0.6, 0.3])
    _, (m_index, m_weights) = conv(mx.array(edge_index), mx.array(x))
    m_rows, m_cols = (torch.tensor(np.array(i), dtype=torch.long) for i in m_index)

    ref = pyg_nn.PANPooling(DIM, **kwargs)
    rng = np.random.default_rng(15)
    with torch.no_grad():
        ref.p.copy_(torch.tensor(rng.normal(size=DIM)))
        ref.beta.copy_(torch.tensor([0.7, 1.3]))
    module = PANPooling(DIM, **kwargs)
    load_torch(module, torch_state(ref))

    tx = torch.tensor(x, requires_grad=True)
    tm = torch.tensor(np.array(m_weights), requires_grad=True)
    expected = ref(tx, _COOMatrix(m_rows, m_cols, tm), torch.tensor(BATCH))
    loss_weights = rng.normal(size=tuple(expected[0].shape)).astype(np.float32)
    (
        (expected[0] * torch.tensor(loss_weights)).sum() + (expected[2] ** 2).sum()
    ).backward()

    def loss_fn(x, mw):
        out = module(m_index, x, mw, mx.array(BATCH))
        loss = (out[0] * mx.array(loss_weights)).sum() + (out[2] ** 2).sum()
        return loss, out

    (_, out), (gx, gm) = mx.value_and_grad(loss_fn, argnums=(0, 1))(
        mx.array(x), m_weights
    )
    _, grads = nn.value_and_grad(module, lambda: loss_fn(mx.array(x), m_weights)[0])()

    names = ["x", "edge_index", "edge_weights", "batch", "perm", "score"]
    for name, a, b in zip(names, out, expected):
        if name in ("edge_index", "batch", "perm"):
            np.testing.assert_array_equal(np.array(a), b.numpy(), err_msg=name)
        else:
            assert_close(a, b.detach().numpy(), name=name)
    assert_close(gx, tx.grad.numpy(), name="grad x")
    assert_close(gm, tm.grad.numpy(), name="grad M")
    check_grads(grads, ref)
