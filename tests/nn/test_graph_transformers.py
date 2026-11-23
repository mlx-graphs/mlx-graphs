"""PyG-parity tests of the attention modules, encodings and graph transformer
models (SGFormer, Polynormer)."""

import mlx.core as mx
import numpy as np
import pytest
import torch
import torch_geometric.nn as pyg_nn
from mlx.utils import tree_flatten
from torch_geometric.nn import attention as pyg_attention
from torch_geometric.nn import encoding as pyg_encoding

from mlx_graphs.nn import (
    LapPEEncoder,
    Polynormer,
    PositionalEncoding,
    RWSEEncoder,
    SGFormer,
    TemporalEncoding,
)
from mlx_graphs.nn.attention import (
    MultiheadAttention,
    PerformerAttention,
    PolynormerAttention,
    SGFormerAttention,
)
from tests.pyg_copy import load_torch, torch_state

# MLX GPU matmuls are only accurate to ~1e-3 relative.
TOL = dict(rtol=1e-2, atol=2e-3)
GCN_RENAMES = [(r"convs\.(\d+)\.lin\.weight", r"convs.\1.linear.weight")]
GAT_RENAMES = [(r"convs\.(\d+)\.lin\.weight", r"convs.\1.lin_proj.weight")]


def dense_input(seed=0, channels=8):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((2, 5, channels)).astype(np.float32)
    mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=bool)
    return x, mask


def compare_dense(module, pyg_module, seed=0, channels=8, tol=TOL):
    x, mask = dense_input(seed, channels)
    module.eval()
    pyg_module.eval()
    out = np.array(module(mx.array(x), mx.array(mask)))
    with torch.no_grad():
        expected = pyg_module(torch.tensor(x), torch.tensor(mask)).numpy()
    # padded rows are not meaningful
    np.testing.assert_allclose(out[mask], expected[mask], **tol)
    return out


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_multihead_attention(heads):
    torch.manual_seed(0)
    pyg_module = torch.nn.MultiheadAttention(8, heads, batch_first=True)
    torch.nn.init.normal_(pyg_module.in_proj_bias)
    torch.nn.init.normal_(pyg_module.out_proj.bias)
    module = MultiheadAttention(8, heads)
    load_torch(module, torch_state(pyg_module))
    x, mask = dense_input()
    y = np.random.default_rng(1).standard_normal((2, 3, 8)).astype(np.float32)
    out = module(mx.array(y), mx.array(x), mx.array(x), key_mask=mx.array(mask))
    with torch.no_grad():
        expected, _ = pyg_module(
            torch.tensor(y),
            torch.tensor(x),
            torch.tensor(x),
            key_padding_mask=~torch.tensor(mask),
            need_weights=False,
        )
    np.testing.assert_allclose(np.array(out), expected.numpy(), **TOL)

    # queries without keys: zero attention, as torch's fused path
    no_keys = mx.zeros((2, 5), dtype=mx.bool_)
    out = module(mx.array(y), mx.array(x), mx.array(x), key_mask=no_keys)
    np.testing.assert_allclose(
        np.array(out),
        np.broadcast_to(pyg_module.out_proj.bias.detach().numpy(), out.shape),
        **TOL,
    )


def test_multihead_attention_init_and_dropout():
    module = MultiheadAttention(16, 4, dropout=0.5)
    limit = np.sqrt(6 / (16 + 48))
    assert np.abs(np.array(module.query_proj.weight)).max() <= limit
    assert np.all(np.array(module.query_proj.bias) == 0)
    x = mx.random.normal((2, 6, 16))
    module.eval()
    a, b = module(x, x, x), module(x, x, x)
    np.testing.assert_array_equal(np.array(a), np.array(b))
    module.train()
    assert not np.allclose(np.array(module(x, x, x)), np.array(a))
    with pytest.raises(ValueError):
        MultiheadAttention(10, 3)


def test_attention_exported_by_nn():
    # As torch_geometric.nn, mlx_graphs.nn exports the attention modules.
    import mlx_graphs.nn as gnn
    import mlx_graphs.nn.attention as attention

    for name in (
        "MultiheadAttention",
        "PerformerAttention",
        "PolynormerAttention",
        "SGFormerAttention",
    ):
        assert getattr(gnn, name) is getattr(attention, name)


def test_performer_attention():
    torch.manual_seed(0)
    pyg_module = pyg_attention.PerformerAttention(8, heads=2, head_channels=4)
    module = PerformerAttention(8, heads=2, head_channels=4)
    load_torch(module, torch_state(pyg_module, renames=[(r"^fast_attn\.", "")]))
    compare_dense(module, pyg_module, tol=dict(rtol=1e-2, atol=1e-2))
    # projection matrix rows are orthonormal within each block
    proj = np.array(PerformerAttention(8, heads=1, head_channels=8).projection_matrix)
    assert proj.shape == (int(8 * np.log(8)), 8)
    np.testing.assert_allclose(proj[:8] @ proj[:8].T, np.eye(8), atol=1e-4)


@pytest.mark.parametrize("heads", [1, 2])
def test_sgformer_attention(heads):
    torch.manual_seed(0)
    pyg_module = pyg_attention.SGFormerAttention(8, heads=heads, head_channels=6)
    module = SGFormerAttention(8, heads=heads, head_channels=6)
    load_torch(module, torch_state(pyg_module))
    out = compare_dense(module, pyg_module)
    assert out.shape == (2, 5, 6)


@pytest.mark.parametrize("qk_shared", [True, False])
@pytest.mark.parametrize("heads", [1, 2])
def test_polynormer_attention(qk_shared, heads):
    torch.manual_seed(0)
    kwargs = dict(heads=heads, head_channels=3, qk_shared=qk_shared, beta=0.7)
    pyg_module = pyg_attention.PolynormerAttention(8, **kwargs)
    module = PolynormerAttention(8, **kwargs)
    load_torch(module, torch_state(pyg_module))
    out = compare_dense(module, pyg_module)
    assert out.shape == (2, 5, 3 * heads)


@pytest.mark.parametrize("granularity", [1.0, 0.5])
def test_positional_encoding(granularity):
    module = PositionalEncoding(16, granularity=granularity)
    pyg_module = pyg_encoding.PositionalEncoding(16, granularity=granularity)
    x = np.array([0.0, 1.0, 2.5, 100.0, 1234.5], dtype=np.float32)
    np.testing.assert_allclose(
        np.array(module(mx.array(x))),
        pyg_module(torch.tensor(x)).numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
    assert len(tree_flatten(module.trainable_parameters())) == 0
    with pytest.raises(ValueError):
        PositionalEncoding(15)


def test_temporal_encoding():
    module = TemporalEncoding(10)
    pyg_module = pyg_encoding.TemporalEncoding(10)
    x = np.array([0.0, 1.0, 2.5, 100.0, 1234.5], dtype=np.float32)
    np.testing.assert_allclose(
        np.array(module(mx.array(x))),
        pyg_module(torch.tensor(x)).numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
    assert len(tree_flatten(module.trainable_parameters())) == 0


def test_rwse_encoder_matches_pyg_example():
    # the GPS example of PyG: `pe_lin(pe_norm(pe))`
    torch.manual_seed(0)
    pe_norm, pe_lin = torch.nn.BatchNorm1d(20), torch.nn.Linear(20, 8)
    pe_norm.running_mean.uniform_(0, 1)
    pe_norm.running_var.uniform_(0.5, 2)
    module = RWSEEncoder(20, 8)
    state = torch_state(pe_norm, renames=[(r"^", "norm.")])
    state.update(torch_state(pe_lin, renames=[(r"^", "lin.")]))
    load_torch(module, state)
    pe = np.random.default_rng(0).uniform(size=(10, 20)).astype(np.float32)
    module.eval()
    pe_norm.eval()
    with torch.no_grad():
        expected = pe_lin(pe_norm(torch.tensor(pe))).numpy()
    np.testing.assert_allclose(np.array(module(mx.array(pe))), expected, **TOL)
    assert "norm" not in RWSEEncoder(20, 8, norm=None)


def test_lap_pe_encoder_sign_flip():
    mx.random.seed(0)
    module = LapPEEncoder(4, 6)
    pe = mx.random.normal((10, 4))
    batch = mx.array([0] * 6 + [1] * 4)
    module.eval()
    ref = module(pe, batch)
    np.testing.assert_allclose(np.array(ref), np.array(module.lin(pe)), **TOL)

    module.train()
    seen = set()
    for _ in range(10):
        out = module(pe, batch)
        # undo the linear layer: (out - b) = (pe * s) W^T, so recover s
        signed = np.linalg.lstsq(
            np.array(module.lin.weight),
            (np.array(out) - np.array(module.lin.bias)).T,
            rcond=None,
        )[0].T
        signs = np.sign(signed / np.array(pe))
        # one sign per graph and eigenvector
        assert np.all(signs[:6] == signs[0]) and np.all(signs[6:] == signs[6])
        seen.add(tuple(signs[0]))
    assert len(seen) > 1
    module = LapPEEncoder(4, 6, random_sign_flip=False)
    np.testing.assert_array_equal(np.array(module(pe, batch)), np.array(module.lin(pe)))


def test_lap_pe_encoder_compiles_in_training_mode():
    """The random sign flips need no host sync, so the encoder compiles."""
    module = LapPEEncoder(4, 6)
    module.train()
    pe = mx.random.normal((10, 4))
    batch = mx.array([0] * 6 + [1] * 4)
    state = [mx.random.state]
    compiled = mx.compile(lambda pe, batch: module(pe, batch), inputs=state)
    assert compiled(pe, batch).shape == (10, 6)
    # graph ids larger than the number of nodes (after empty graphs)
    assert module(pe[:2], mx.array([0, 5])).shape == (2, 6)


def graph_batch(seed=0, num_features=6, sizes=(4, 6, 5)):
    rng = np.random.default_rng(seed)
    batch = np.concatenate([np.full(n, i) for i, n in enumerate(sizes)])
    offsets = np.cumsum((0,) + sizes[:-1])
    edges = []
    for n, off in zip(sizes, offsets):
        src = rng.integers(0, n, 2 * n)
        dst = (src + rng.integers(1, n, 2 * n)) % n  # no self-loops
        edges.append(
            np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]) + off
        )
    edge_index = np.concatenate(edges, axis=1).astype(np.int64)
    x = rng.standard_normal((batch.size, num_features)).astype(np.float32)
    return x, edge_index, batch.astype(np.int64)


def randomize_bn(module):
    for name, buf in module.named_buffers():
        if name.endswith("running_mean"):
            buf.uniform_(-0.5, 0.5)
        elif name.endswith("running_var"):
            buf.uniform_(0.5, 2.0)


def compare_model(model, pyg_model, seed=0):
    x, edge_index, batch = graph_batch(seed)
    model.eval()
    pyg_model.eval()
    out = model(mx.array(edge_index), mx.array(x), mx.array(batch))
    with torch.no_grad():
        expected = pyg_model(
            torch.tensor(x), torch.tensor(edge_index), torch.tensor(batch)
        )
    np.testing.assert_allclose(np.array(out), expected.numpy(), **TOL)
    return np.array(out)


@pytest.mark.parametrize("aggregate", ["add", "cat"])
@pytest.mark.parametrize("heads", [1, 2])
def test_sgformer(aggregate, heads):
    torch.manual_seed(0)
    kwargs = dict(
        trans_num_layers=2,
        trans_num_heads=heads,
        gnn_num_layers=2,
        graph_weight=0.3,
        aggregate=aggregate,
    )
    pyg_model = pyg_nn.models.SGFormer(6, 8, 3, **kwargs)
    randomize_bn(pyg_model)
    model = SGFormer(6, 8, 3, **kwargs)
    load_torch(model, torch_state(pyg_model, renames=GCN_RENAMES))
    out = compare_model(model, pyg_model)
    assert out.shape == (15, 3)
    np.testing.assert_allclose(np.exp(out).sum(-1), 1.0, rtol=1e-4)
    with pytest.raises(ValueError):
        SGFormer(6, 8, 3, aggregate="max")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(local_attn=True, pre_ln=True, qk_shared=True),
        dict(post_bn=False, beta=0.5),
    ],
)
@pytest.mark.parametrize("use_global", [False, True])
def test_polynormer(kwargs, use_global):
    torch.manual_seed(0)
    kwargs = dict(local_layers=3, global_layers=2, **kwargs)
    pyg_model = pyg_nn.models.Polynormer(6, 8, 3, **kwargs)
    randomize_bn(pyg_model)
    pyg_model._global = use_global
    model = Polynormer(6, 8, 3, **kwargs)
    model._global = use_global
    renames = GAT_RENAMES if kwargs.get("local_attn") else GCN_RENAMES
    load_torch(model, torch_state(pyg_model, renames=renames))
    out = compare_model(model, pyg_model)
    assert out.shape == (15, 3)


def test_polynormer_multi_head_global():
    model = Polynormer(6, 4, 3, local_layers=2, global_layers=1, heads=2)
    model._global = True
    x, edge_index, batch = graph_batch()
    out = model(mx.array(edge_index), mx.array(x), mx.array(batch))
    assert out.shape == (15, 3)


@pytest.mark.parametrize("name", ["sgformer", "polynormer"])
def test_models_precomputed_layout_and_compile(name):
    """With a device-side layout of the exact sizes, the models give the
    default output without a host sync, so they compile."""
    from mlx_graphs.nn.aggr import DenseLayout

    mx.random.seed(0)
    if name == "sgformer":
        model = SGFormer(6, 8, 3)
    else:
        model = Polynormer(6, 8, 3, local_layers=2, global_layers=1)
        model._global = True
    model.eval()
    x, edge_index, batch = (mx.array(a) for a in graph_batch(2))
    ref = model(edge_index, x, batch)

    def forward(edge_index, x, batch):
        layout = DenseLayout.from_sizes(batch, x.shape[0], 3, 6)  # sizes 4, 6, 5
        return model(edge_index, x, batch, layout=layout)

    np.testing.assert_allclose(np.array(forward(edge_index, x, batch)), ref, **TOL)
    out = mx.compile(forward)(edge_index, x, batch)
    np.testing.assert_allclose(np.array(out), np.array(ref), **TOL)


@pytest.mark.parametrize("name", ["sgformer", "polynormer"])
def test_models_train(name):
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(0)
    if name == "sgformer":
        model = SGFormer(6, 16, 3, trans_dropout=0.1, gnn_dropout=0.1)
    else:
        model = Polynormer(6, 16, 3, local_layers=2, global_layers=1)
        model._global = True
    x, edge_index, batch = (mx.array(a) for a in graph_batch(1))
    y = mx.array(np.random.default_rng(0).integers(0, 3, x.shape[0]))
    optimizer = optim.Adam(learning_rate=0.01)

    def loss_fn(model):
        out = model(edge_index, x, batch)
        return -mx.take_along_axis(out, y[:, None], axis=1).mean()

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    model.train()
    losses = []
    for _ in range(30):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())
    assert losses[-1] < losses[0]
