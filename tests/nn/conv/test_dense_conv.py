"""PyG-parity tests of the dense convolution layers (outputs and gradients)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten

from mlx_graphs.nn import DenseGCNConv, DenseGINConv, DenseGraphConv, DenseSAGEConv
from tests.pyg_copy import load_torch, torch_grads, torch_state

B, N, DIM, OUT = 3, 7, 5, 4
# MLX's GPU matmul is only accurate to ~1e-3 relative; a wrong formula is O(1)
# off with these O(1) inputs.
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


def make_inputs(seed=0):
    """Weighted adjacency with ~40% non-zero entries (some on the diagonal),
    and a mask where the graphs have 7, 5 and 3 real nodes."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(B, N, DIM)).astype(np.float32)
    adj = rng.uniform(0.2, 1.5, size=(B, N, N)).astype(np.float32)
    adj *= rng.uniform(size=(B, N, N)) < 0.4
    mask = np.arange(N)[None, :] < np.array([7, 5, 3])[:, None]
    adj *= mask[:, :, None] & mask[:, None, :]
    weights = rng.normal(size=(B, N, OUT)).astype(np.float32)
    return x, adj, mask, weights


def compare(module, ref, call_kwargs=None, use_mask=True, squeeze=False, seed=0):
    """Compares outputs and the gradients of the parameters and inputs of a
    weighted sum of the outputs."""
    call_kwargs = call_kwargs or {}
    x, adj, mask, weights = make_inputs(seed)
    if squeeze:  # a single graph given as 2D arrays
        x, adj, mask, weights = x[0], adj[0], None, weights[:1]
    elif not use_mask:
        mask = None

    tx = torch.tensor(x, requires_grad=True)
    tadj = torch.tensor(adj, requires_grad=True)
    tmask = None if mask is None else torch.tensor(mask)
    expected = ref(tx, tadj, tmask, **call_kwargs)
    (expected * torch.tensor(weights)).sum().backward()

    mmask = None if mask is None else mx.array(mask)

    def loss_fn(x, adj):
        out = module(adj, x, mmask, **call_kwargs)
        return (out * mx.array(weights)).sum(), out

    def param_loss():
        return loss_fn(mx.array(x), mx.array(adj))[0]

    (_, out), (gx, gadj) = mx.value_and_grad(loss_fn, argnums=(0, 1))(
        mx.array(x), mx.array(adj)
    )
    _, grads = nn.value_and_grad(module, param_loss)()

    assert_close(out, expected.detach().numpy(), name="out")
    assert_close(gx, tx.grad.numpy(), name="grad x")
    # (the max aggregation only uses the sparsity pattern of adj)
    gadj_expected = np.zeros(adj.shape) if tadj.grad is None else tadj.grad.numpy()
    assert_close(gadj, gadj_expected, name="grad adj")
    return dict(tree_flatten(grads))


def check_param_grads(grads, ref, renames=()):
    expected = torch_grads(ref, renames)
    # torch has no gradient for unused parameters, MLX a zero one
    for name in set(grads) - set(expected):
        assert not np.any(np.array(grads[name])), name
        expected[name] = np.zeros(grads[name].shape)
    assert set(grads) == set(expected), (sorted(grads), sorted(expected))
    for name, grad in grads.items():
        assert_close(grad, expected[name], name=name)


@pytest.mark.parametrize("improved", [False, True])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("add_loop", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
def test_dense_gcn_conv(improved, bias, add_loop, use_mask):
    torch.manual_seed(0)
    ref = pyg_nn.DenseGCNConv(DIM, OUT, improved=improved, bias=bias)
    if bias:  # PyG starts with a zero bias
        torch.nn.init.normal_(ref.bias)
    module = DenseGCNConv(DIM, OUT, improved=improved, bias=bias)
    load_torch(module, torch_state(ref))
    grads = compare(module, ref, dict(add_loop=add_loop), use_mask=use_mask)
    check_param_grads(grads, ref)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
def test_dense_sage_conv(normalize, bias, use_mask):
    torch.manual_seed(1)
    ref = pyg_nn.DenseSAGEConv(DIM, OUT, normalize=normalize, bias=bias)
    module = DenseSAGEConv(DIM, OUT, normalize=normalize, bias=bias)
    load_torch(module, torch_state(ref))
    grads = compare(module, ref, use_mask=use_mask)
    check_param_grads(grads, ref)


@pytest.mark.parametrize("learn_eps", [False, True])
@pytest.mark.parametrize("add_loop", [True, False])
def test_dense_gin_conv(learn_eps, add_loop):
    torch.manual_seed(2)
    ref = pyg_nn.DenseGINConv(
        torch.nn.Sequential(
            torch.nn.Linear(DIM, 8), torch.nn.ReLU(), torch.nn.Linear(8, OUT)
        ),
        eps=0.3,
        train_eps=learn_eps,
    )
    mlp = nn.Sequential(nn.Linear(DIM, 8), nn.ReLU(), nn.Linear(8, OUT))
    module = DenseGINConv(mlp, eps=0.3, learn_eps=learn_eps)
    renames = [(r"^nn\.", "mlp.layers.")]
    load_torch(module, torch_state(ref, renames))
    grads = compare(module, ref, dict(add_loop=add_loop))
    check_param_grads(grads, ref, renames)
    assert ("eps" in grads) == learn_eps


@pytest.mark.parametrize("aggr", ["add", "mean", "max"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
def test_dense_graph_conv(aggr, bias, use_mask):
    torch.manual_seed(3)
    ref = pyg_nn.DenseGraphConv(DIM, OUT, aggr=aggr, bias=bias)
    module = DenseGraphConv(DIM, OUT, aggr=aggr, bias=bias)
    load_torch(module, torch_state(ref))
    grads = compare(module, ref, use_mask=use_mask)
    check_param_grads(grads, ref)


@pytest.mark.parametrize(
    "cls, args",
    [
        (DenseGCNConv, (DIM, OUT)),
        (DenseSAGEConv, (DIM, OUT)),
        (DenseGraphConv, (DIM, OUT)),
    ],
)
def test_dense_conv_single_graph(cls, args):
    torch.manual_seed(4)
    ref = getattr(pyg_nn, cls.__name__)(*args)
    if cls is DenseGCNConv:
        torch.nn.init.normal_(ref.bias)
    module = cls(*args)
    load_torch(module, torch_state(ref))
    compare(module, ref, squeeze=True)
    assert repr(module) == f"{cls.__name__}({DIM}, {OUT})"


def test_dense_graph_conv_invalid_aggr():
    with pytest.raises(ValueError):
        DenseGraphConv(DIM, OUT, aggr="min")


def test_dense_gcn_conv_matches_sparse_gcn_conv():
    """On a batch of graphs without self-loops, DenseGCNConv equals GCNConv
    on the same graphs."""
    from mlx_graphs.nn import GCNConv
    from mlx_graphs.utils import to_dense_adj, to_dense_batch

    rng = np.random.default_rng(5)
    batch = np.repeat(np.arange(3), [6, 4, 5])
    edges = []
    for g in range(3):
        nodes = np.nonzero(batch == g)[0]
        e = rng.choice(nodes, size=(2, 12))
        edges.append(e[:, e[0] != e[1]])
    edge_index = np.unique(np.concatenate(edges, axis=1), axis=1)
    x = rng.normal(size=(batch.size, DIM)).astype(np.float32)

    sparse = GCNConv(DIM, OUT)
    dense = DenseGCNConv(DIM, OUT)
    dense.lin.weight = sparse.linear.weight
    dense.bias = mx.random.normal((OUT,))
    sparse.bias = dense.bias

    # GCNConv aggregates over incoming edges: adj[i, j] = edge j -> i
    adj = to_dense_adj(mx.array(edge_index[::-1].copy()), mx.array(batch))
    dense_x, mask = to_dense_batch(mx.array(x), mx.array(batch))
    out = dense(adj, dense_x, mask)
    expected = sparse(mx.array(edge_index), mx.array(x))
    assert_close(np.array(out)[np.array(mask)], np.array(expected))
