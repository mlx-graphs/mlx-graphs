"""SchNet, DimeNet, DimeNet++ and AttentiveFP against PyG, with copied weights.

PyG computes the interaction graphs with ``radius_graph`` (pyg-lib or
torch-cluster) and DimeNet's triplets with ``torch_sparse``, which are not
installed here: the PyG models get pure-torch versions of them, computed by
brute force.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
import torch
import torch_geometric.nn.models.dimenet as pyg_dimenet
from torch_geometric.nn.models import AttentiveFP as PyGAttentiveFP
from torch_geometric.nn.models import DimeNet as PyGDimeNet
from torch_geometric.nn.models import DimeNetPlusPlus as PyGDimeNetPlusPlus
from torch_geometric.nn.models import SchNet as PyGSchNet

from mlx_graphs.nn import AttentiveFP, DimeNet, DimeNetPlusPlus, SchNet
from mlx_graphs.nn.models.dimenet import glorot_orthogonal, triplets
from mlx_graphs.nn.models.schnet import GaussianSmearing, cosine_cutoff
from mlx_graphs.utils import radius_graph
from tests.pyg_copy import assert_grads_close, load_torch, torch_grads, torch_state

# MLX GPU matmuls are only accurate to ~1e-3 relative.
TOL = dict(rtol=2e-2, atol=2e-3)


def molecules(seed=0, sizes=(5, 3, 7, 1), min_dist=0.7, spread=1.3):
    """Random molecules: atomic numbers, positions (no two atoms of a
    molecule closer than ``min_dist``) and batch vector."""
    rng = np.random.default_rng(seed)
    z, pos, batch = [], [], []
    for graph, size in enumerate(sizes):
        points = []
        while len(points) < size:
            p = rng.normal(0, spread, 3)
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        pos.append(np.array(points))
        z.append(rng.integers(1, 10, size))
        batch.append(np.full(size, graph))
    return (
        np.concatenate(z).astype(np.int64),
        np.concatenate(pos).astype(np.float32),
        np.concatenate(batch).astype(np.int64),
    )


def torch_radius_graph(pos, r, batch=None, **kwargs):
    """Brute-force ``radius_graph`` (edges from neighbor to center)."""
    if batch is None:
        batch = torch.zeros(pos.shape[0], dtype=torch.long)
    dist = torch.cdist(pos, pos)
    mask = (dist < r) & (batch[:, None] == batch[None, :])
    mask.fill_diagonal_(False)
    center, neighbor = mask.nonzero(as_tuple=True)
    return torch.stack([neighbor, center])


def brute_force_triplets(edge_index, num_nodes):
    """PyG's ``triplets`` with Python loops instead of ``torch_sparse``."""
    row, col = edge_index.tolist()  # j -> i
    idx_i, idx_j, idx_k, idx_kj, idx_ji = [], [], [], [], []
    for e_ji, (j, i) in enumerate(zip(row, col)):
        incoming = sorted(
            (k, e_kj) for e_kj, (k, t) in enumerate(zip(row, col)) if t == j
        )
        for k, e_kj in incoming:
            if k != i:
                idx_i.append(i)
                idx_j.append(j)
                idx_k.append(k)
                idx_kj.append(e_kj)
                idx_ji.append(e_ji)
    as_tensor = lambda a: torch.tensor(a, dtype=torch.long)  # noqa: E731
    return (
        edge_index[1],
        edge_index[0],
        *(as_tensor(a) for a in (idx_i, idx_j, idx_k, idx_kj, idx_ji)),
    )


@pytest.fixture
def pyg_dimenet_without_extensions(monkeypatch):
    monkeypatch.setattr(pyg_dimenet, "radius_graph", torch_radius_graph)
    monkeypatch.setattr(pyg_dimenet, "triplets", brute_force_triplets)


def compare(model, pyg, z, pos, batch, renames=(), with_batch=True, grad_rtol=2e-2):
    """Compares the outputs, the gradients with respect to the positions and
    to all the parameters of ``model`` and ``pyg`` (loss: sum of squares)."""
    pos_torch = torch.tensor(pos, requires_grad=True)
    args = (torch.tensor(z), pos_torch) + ((torch.tensor(batch),) if with_batch else ())
    expected = pyg(*args)
    (expected**2).sum().backward()

    mx_args = (mx.array(z), mx.array(pos)) + ((mx.array(batch),) if with_batch else ())

    def loss_fn(model, pos):
        out = model(mx_args[0], pos, *mx_args[2:])
        return (out**2).sum(), out

    (_, out), grads = nn.value_and_grad(model, loss_fn)(model, mx_args[1])
    pos_grad = mx.grad(lambda p: loss_fn(model, p)[0])(mx_args[1])

    np.testing.assert_allclose(np.array(out), expected.detach().numpy(), **TOL)
    expected_pos_grad = pos_torch.grad.numpy()
    diff = np.linalg.norm(np.array(pos_grad) - expected_pos_grad)
    assert diff <= 2e-2 * np.linalg.norm(expected_pos_grad)
    assert_grads_close(grads, torch_grads(pyg, renames), rtol=grad_rtol)
    return out


# SchNet ---------------------------------------------------------------------

SCHNET_RENAMES = [
    (r"\.mlp\.(\d)\.", r".conv.nn.layers.\1."),
    (r"\.conv\.nn\.(\d)\.", r".conv.nn.layers.\1."),
]


def build_schnet(atomref=False, **kwargs):
    torch.manual_seed(0)
    kwargs = dict(
        hidden_channels=32,
        num_filters=24,
        num_interactions=2,
        num_gaussians=20,
        cutoff=4.0,
        **kwargs,
    )
    ref = None
    if atomref:
        ref = torch.zeros(100, 1)
        ref[torch.tensor([1, 6, 7, 8, 9])] = torch.randn(5, 1)
    pyg = PyGSchNet(
        interaction_graph=lambda pos, batch: _torch_interaction_graph(
            pos, batch, kwargs["cutoff"]
        ),
        atomref=ref,
        **kwargs,
    )
    model = SchNet(atomref=None if ref is None else mx.array(ref.numpy()), **kwargs)
    load_torch(
        model, torch_state(pyg, renames=SCHNET_RENAMES, drop=[r"^initial_atomref$"])
    )
    return model, pyg


def _torch_interaction_graph(pos, batch, cutoff):
    edge_index = torch_radius_graph(pos, cutoff, batch)
    row, col = edge_index
    return edge_index, (pos[row] - pos[col]).norm(dim=-1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"readout": "mean"},
        {"mean": 1.5, "std": 0.3, "atomref": True},
        {"dipole": True},
    ],
)
def test_schnet_matches_pyg(kwargs):
    if kwargs.get("dipole"):
        # The atomic masses of the dipole readout come from ase, in both libraries.
        pytest.importorskip("ase")
    model, pyg = build_schnet(**kwargs)
    z, pos, batch = molecules()
    out = compare(model, pyg, z, pos, batch, SCHNET_RENAMES)
    assert out.shape == (4, 1)


def test_schnet_single_molecule_and_padding():
    model, pyg = build_schnet()
    z, pos, _ = molecules(1, sizes=(6,))
    z[0] = 0  # padding atom: zero embedding, no gradient
    compare(model, pyg, z, pos, None, SCHNET_RENAMES, with_batch=False)
    grads = nn.value_and_grad(model, lambda m: m(mx.array(z), mx.array(pos)).sum())(
        model
    )[1]
    assert float(mx.abs(grads["embedding"]["weight"][0]).sum()) == 0.0


def test_schnet_blocks():
    distances = mx.array([0.0, 1.0, 2.5, 5.0])
    np.testing.assert_allclose(
        np.array(cosine_cutoff(distances, 5.0)),
        [1.0, 0.5 * (np.cos(0.2 * np.pi) + 1), 0.5, 0.0],
        atol=1e-6,
    )
    smearing = GaussianSmearing(0.0, 5.0, 11)
    out = np.array(smearing(distances))
    assert out.shape == (4, 11)
    np.testing.assert_allclose(
        out[:, 0], np.exp(-0.5 / 0.25 * np.array([0, 1, 6.25, 25])), rtol=1e-5
    )
    # the offsets are a buffer, not a trainable parameter
    assert "offset" not in dict(smearing.trainable_parameters())


def test_schnet_trains():
    mx.random.seed(0)
    model = SchNet(hidden_channels=32, num_filters=32, num_interactions=2, cutoff=4.0)
    z, pos, batch = (mx.array(a) for a in molecules(2))
    target = mx.array([[1.0], [-2.0], [0.5], [3.0]])
    optimizer = optim.Adam(learning_rate=3e-3)
    loss_and_grad = nn.value_and_grad(
        model, lambda m: ((m(z, pos, batch) - target) ** 2).mean()
    )
    losses = []
    for _ in range(60):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())
    assert losses[-1] < 0.01 * losses[0]


# DimeNet / DimeNet++ --------------------------------------------------------

DIMENET_KWARGS = dict(
    hidden_channels=16,
    out_channels=2,
    num_blocks=2,
    num_spherical=4,
    num_radial=5,
    cutoff=3.0,
    num_output_layers=2,
    output_initializer="glorot_orthogonal",
)


def test_triplets_match_brute_force():
    z, pos, batch = molecules(3)
    edge_index = radius_graph(mx.array(pos), 3.0, mx.array(batch))
    expected = brute_force_triplets(torch.tensor(np.array(edge_index)), z.size)
    out = triplets(edge_index, z.size)
    assert int(out[3].size) > 0
    for got, ref in zip(out, expected):
        np.testing.assert_array_equal(np.array(got), ref.numpy())


def test_glorot_orthogonal():
    mx.random.seed(0)
    w = np.array(glorot_orthogonal((8, 20), scale=2.0))
    np.testing.assert_allclose(w.var(ddof=1), 2.0 / 28, rtol=1e-4)
    q = w / np.linalg.norm(w, axis=1, keepdims=True)
    np.testing.assert_allclose(q @ q.T, np.eye(8), atol=1e-5)


@pytest.mark.parametrize("plus_plus", [False, True])
@pytest.mark.parametrize("with_batch", [True, False])
def test_dimenet_matches_pyg(pyg_dimenet_without_extensions, plus_plus, with_batch):
    torch.manual_seed(0)
    if plus_plus:
        kwargs = dict(
            DIMENET_KWARGS, int_emb_size=8, basis_emb_size=4, out_emb_channels=12
        )
        pyg, model = PyGDimeNetPlusPlus(**kwargs), DimeNetPlusPlus(**kwargs)
    else:
        kwargs = dict(DIMENET_KWARGS, num_bilinear=3)
        pyg, model = PyGDimeNet(**kwargs), DimeNet(**kwargs)
    load_torch(model, torch_state(pyg))
    sizes = (5, 3, 7, 1) if with_batch else (8,)
    z, pos, batch = molecules(4, sizes=sizes)
    # the basis functions amplify the float32 rounding errors of the GPU
    # (all the gradients are within 1e-3 on the CPU)
    out = compare(model, pyg, z, pos, batch, with_batch=with_batch, grad_rtol=5e-2)
    assert out.shape == ((4, 2) if with_batch else (2,))


def test_dimenet_zero_output_init_and_errors():
    model = DimeNetPlusPlus(
        **dict(DIMENET_KWARGS, output_initializer="zeros"),
        int_emb_size=8,
        basis_emb_size=4,
        out_emb_channels=12,
    )
    z, pos, batch = (mx.array(a) for a in molecules(5))
    assert float(mx.abs(model(z, pos, batch)).sum()) == 0.0
    with pytest.raises(ValueError, match="num_spherical"):
        DimeNet(**dict(DIMENET_KWARGS, num_spherical=1), num_bilinear=2)
    with pytest.raises(ValueError, match="output_initializer"):
        DimeNet(**dict(DIMENET_KWARGS, output_initializer="ones"), num_bilinear=2)


def test_dimenet_collinear_atoms_have_finite_gradients():
    # angles of 0 or pi: the norm of the cross product is 0
    pos = mx.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    model = DimeNetPlusPlus(
        **DIMENET_KWARGS, int_emb_size=8, basis_emb_size=4, out_emb_channels=12
    )
    grad = mx.grad(lambda p: model(mx.array([6, 6, 8]), p).sum())(pos)
    assert bool(mx.all(mx.isfinite(grad)).item())


# AttentiveFP ----------------------------------------------------------------

ATTENTIVE_FP_RENAMES = [(r"(atom_convs\.\d+|mol_conv)\.lin\.", r"\1.lin_proj.")]


def small_molecules(seed=0, sizes=(5, 3, 7, 1), channels=6, edge_dim=3):
    rng = np.random.default_rng(seed)
    edges, batch, offset = [], [], 0
    for graph, n in enumerate(sizes):
        # a chain plus random bonds, both directions
        src = list(range(n - 1)) + list(rng.integers(0, n, n // 2))
        dst = list(range(1, n)) + list(rng.integers(0, n, n // 2))
        pairs = {(a, b) for a, b in zip(src, dst) if a != b}
        pairs |= {(b, a) for a, b in pairs}
        edges += [(a + offset, b + offset) for a, b in sorted(pairs)]
        batch += [graph] * n
        offset += n
    edge_index = np.array(edges, dtype=np.int64).T.reshape(2, -1)
    x = rng.standard_normal((offset, channels)).astype(np.float32)
    e = rng.standard_normal((edge_index.shape[1], edge_dim)).astype(np.float32)
    return x, edge_index, e, np.array(batch, dtype=np.int64)


@pytest.mark.parametrize("num_layers,num_timesteps", [(1, 1), (3, 2)])
def test_attentive_fp_matches_pyg(num_layers, num_timesteps):
    torch.manual_seed(0)
    kwargs = dict(
        in_channels=6,
        hidden_channels=16,
        out_channels=3,
        edge_dim=3,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
    )
    pyg, model = PyGAttentiveFP(**kwargs), AttentiveFP(**kwargs)
    load_torch(model, torch_state(pyg, renames=ATTENTIVE_FP_RENAMES))
    x, edge_index, e, batch = small_molecules()

    x_t = torch.tensor(x, requires_grad=True)
    e_t = torch.tensor(e, requires_grad=True)
    expected = pyg(x_t, torch.tensor(edge_index), e_t, torch.tensor(batch))
    (expected**2).sum().backward()

    args = [mx.array(edge_index), mx.array(batch)]

    def loss_fn(model, x, e):
        out = model(x, args[0], e, args[1])
        return (out**2).sum(), out

    (_, out), grads = nn.value_and_grad(model, loss_fn)(model, mx.array(x), mx.array(e))
    np.testing.assert_allclose(np.array(out), expected.detach().numpy(), **TOL)
    assert out.shape == (4, 3)
    assert_grads_close(grads, torch_grads(pyg, ATTENTIVE_FP_RENAMES))
    input_grads = mx.grad(lambda x, e: loss_fn(model, x, e)[0], argnums=[0, 1])(
        mx.array(x), mx.array(e)
    )
    for got, ref in zip(input_grads, (x_t.grad, e_t.grad)):
        diff = np.linalg.norm(np.array(got) - ref.numpy())
        assert diff <= 2e-2 * np.linalg.norm(ref.numpy())


def test_attentive_fp_dropout_and_training():
    mx.random.seed(0)
    model = AttentiveFP(6, 16, 1, 3, num_layers=2, num_timesteps=2, dropout=0.2)
    x, edge_index, e, batch = (mx.array(a) for a in small_molecules(1))
    model.eval()
    ref = model(x, edge_index, e, batch)
    np.testing.assert_allclose(np.array(model(x, edge_index, e, batch)), np.array(ref))
    model.train()
    target = mx.array([[1.0], [0.0], [-1.0], [2.0]])
    optimizer = optim.Adam(learning_rate=1e-2)
    loss_and_grad = nn.value_and_grad(
        model, lambda m: ((m(x, edge_index, e, batch) - target) ** 2).mean()
    )
    losses = []
    for _ in range(40):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
    # a single molecule without batch vector
    model.eval()
    first = np.nonzero(np.array(edge_index).max(axis=0) < 5)[0]
    first = mx.array(first)
    single = model(x[:5], edge_index[:, first], e[first])
    assert single.shape == (1, 1)
