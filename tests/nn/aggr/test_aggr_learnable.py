"""PyG-parity tests of the learnable aggregations (RNN, set transformers, MLP,
LCM, equilibrium) and of the variance-preserving aggregation."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import aggr as pyg_aggr

from mlx_graphs.nn import aggr
from mlx_graphs.nn.aggr.utils import DenseLayout
from tests.pyg_copy import gru_state, load_torch, lstm_state, torch_state

DIM = 4
# PyG's dense aggregations need a sorted index; set 2 is empty.
INDEX = np.array([0, 0, 0, 1, 1, 3, 3, 3, 3, 4], dtype=np.int64)
NUM_SETS = 5
# MLX GPU matmuls are only accurate to ~1e-3 relative; wrong formulas give
# O(1) errors with these O(1) inputs.
TOL = dict(rtol=1e-2, atol=2e-3)


def make_x(seed=0, dim=DIM, n=INDEX.size):
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def compare(module, pyg_module, x=None, index=INDEX, dim_size=NUM_SETS, **kwargs):
    x = make_x() if x is None else x
    module.eval()
    pyg_module.eval()
    out = module(mx.array(x), mx.array(index), dim_size, **kwargs)
    with torch.no_grad():
        expected = pyg_module(
            torch.tensor(x), torch.tensor(index), dim_size=dim_size, **kwargs
        )
    np.testing.assert_allclose(np.array(out), expected.numpy(), **TOL)
    return np.array(out)


def test_dense_layout_from_sizes():
    """The device-side layout equals the host one for the exact sizes, pads
    larger sizes, and drops (zero rows) elements beyond the bound or with an
    invalid index."""
    index = mx.array([2, 0, 2, 1, 0, 2])
    x = mx.arange(1, 7, dtype=mx.float32)[:, None]
    host = DenseLayout(index, 6, dim_size=4)
    device = DenseLayout.from_sizes(index, 6, num_sets=4, max_num_elements=3)
    assert mx.array_equal(device.mask, host.mask)
    assert mx.array_equal(device.to_dense(x, -1.0), host.to_dense(x, -1.0))
    assert mx.array_equal(device.from_dense(device.to_dense(x)), x)

    padded = DenseLayout.from_sizes(index, 6, num_sets=4, max_num_elements=5)
    assert padded.to_dense(x).shape == (4, 5, 1)
    assert mx.array_equal(padded.to_dense(x)[:, :3], host.to_dense(x))
    assert mx.array_equal(padded.from_dense(padded.to_dense(x)), x)

    truncated = DenseLayout.from_sizes(index, 6, num_sets=4, max_num_elements=2)
    assert truncated.from_dense(truncated.to_dense(x))[:, 0].tolist() == [
        1, 2, 3, 4, 5, 0  # the third element of set 2 is dropped
    ]  # fmt: skip
    invalid = DenseLayout.from_sizes(
        mx.array([-1, 0, 5, 0]), 4, num_sets=2, max_num_elements=2
    )
    assert invalid.mask.tolist() == [[True, True], [False, False]]
    out = invalid.from_dense(invalid.to_dense(x[:4]))
    assert out[:, 0].tolist() == [0, 2, 0, 4]

    single = DenseLayout.from_sizes(None, 3, num_sets=1, max_num_elements=3)
    assert single.mask.tolist() == [[True, True, True]]


def test_dense_layout_unsorted():
    index = mx.array([2, 0, 2, 1, 0, 2])
    x = mx.arange(6, dtype=mx.float32)[:, None]
    layout = DenseLayout(index, 6, dim_size=4)
    dense = layout.to_dense(x, fill_value=-1.0)
    assert dense.shape == (4, 3, 1)
    expected = [[1, 4, -1], [3, -1, -1], [0, 2, 5], [-1, -1, -1]]
    np.testing.assert_array_equal(np.array(dense)[..., 0], expected)
    np.testing.assert_array_equal(np.array(layout.from_dense(dense)), np.array(x))
    np.testing.assert_array_equal(np.array(layout.mask), np.array(expected) >= 0)
    truncated = DenseLayout(index, 6, dim_size=4, max_num_elements=2)
    assert truncated.to_dense(x).shape == (4, 2, 1)
    with pytest.raises(ValueError):
        truncated.from_dense(truncated.to_dense(x))


@pytest.mark.parametrize("max_num_elements", [None, 2])
@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_rnn_aggregations(kind, max_num_elements):
    torch.manual_seed(0)
    if kind == "lstm":
        module, pyg_module = (
            aggr.LSTMAggregation(DIM, 6),
            pyg_aggr.LSTMAggregation(DIM, 6),
        )
        load_torch(module, lstm_state("lstm", pyg_module.lstm))
    else:
        module, pyg_module = (
            aggr.GRUAggregation(DIM, 6),
            pyg_aggr.GRUAggregation(DIM, 6),
        )
        load_torch(module, gru_state("gru", pyg_module.gru))
    out = compare(module, pyg_module, max_num_elements=max_num_elements)
    assert out.shape == (NUM_SETS, 6)

    # unsorted index: same result as PyG on the stably sorted input
    perm = np.random.default_rng(1).permutation(INDEX.size)
    x = make_x()
    order = np.argsort(INDEX[perm], kind="stable")
    shuffled = module(mx.array(x[perm]), mx.array(INDEX[perm]), NUM_SETS)
    with torch.no_grad():
        expected = pyg_module(
            torch.tensor(x[perm][order]),
            torch.tensor(INDEX[perm][order]),
            dim_size=NUM_SETS,
        )
    np.testing.assert_allclose(np.array(shuffled), expected.numpy(), **TOL)


@pytest.mark.parametrize("layer_norm", [False, True])
@pytest.mark.parametrize("concat", [True, False])
def test_set_transformer_aggregation(layer_norm, concat):
    torch.manual_seed(0)
    kwargs = dict(
        num_seed_points=2,
        num_encoder_blocks=2,
        num_decoder_blocks=1,
        heads=2,
        concat=concat,
        layer_norm=layer_norm,
    )
    module = aggr.SetTransformerAggregation(DIM, **kwargs)
    pyg_module = pyg_aggr.SetTransformerAggregation(DIM, **kwargs)
    load_torch(module, torch_state(pyg_module))
    out = compare(module, pyg_module)
    assert out.shape == (NUM_SETS, 2 * DIM if concat else DIM)


@pytest.mark.parametrize("layer_norm", [False, True])
def test_graph_multiset_transformer(layer_norm):
    torch.manual_seed(0)
    index = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3], dtype=np.int64)
    module = aggr.GraphMultisetTransformer(
        DIM, k=3, num_encoder_blocks=2, heads=2, layer_norm=layer_norm
    )
    pyg_module = pyg_aggr.GraphMultisetTransformer(
        DIM, k=3, num_encoder_blocks=2, heads=2, layer_norm=layer_norm
    )
    load_torch(module, torch_state(pyg_module))
    out = compare(module, pyg_module, index=index, dim_size=4)
    assert out.shape == (4, DIM)


def test_set_transformer_blocks():
    torch.manual_seed(0)
    from torch_geometric.nn.aggr.utils import InducedSetAttentionBlock

    module = aggr.InducedSetAttentionBlock(DIM, 3, heads=2)
    pyg_module = InducedSetAttentionBlock(DIM, 3, heads=2)
    load_torch(module, torch_state(pyg_module))
    x = make_x().reshape(2, 5, DIM)
    mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=bool)
    out = module(mx.array(x), mx.array(mask))
    with torch.no_grad():
        expected = pyg_module(torch.tensor(x), torch.tensor(mask))
    np.testing.assert_allclose(np.array(out), expected.numpy(), **TOL)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_mlp_aggregation(num_layers):
    torch.manual_seed(0)
    kwargs = dict(num_layers=num_layers)
    if num_layers > 1:
        kwargs["hidden_channels"] = 8
    module = aggr.MLPAggregation(
        DIM,
        6,
        max_num_elements=3,
        hidden_features_dim=kwargs.get("hidden_channels"),
        num_layers=num_layers,
    )
    pyg_module = pyg_aggr.MLPAggregation(DIM, 6, max_num_elements=3, **kwargs)
    # give the batch norm non-trivial running statistics
    for bn in pyg_module.mlp.norms:
        if hasattr(bn, "module"):
            bn.module.running_mean.uniform_(-1, 1)
            bn.module.running_var.uniform_(0.5, 2)
    state = torch_state(
        pyg_module,
        renames=[(r"^mlp\.norms\.(\d+)\.module\.", r"mlp.norms.\1.")],
    )
    load_torch(module, state)
    out = compare(module, pyg_module)
    assert out.shape == (NUM_SETS, 6)


@pytest.mark.parametrize("project", [True, False])
def test_lcm_aggregation(project):
    torch.manual_seed(0)
    out_dim = 6 if project else DIM
    module = aggr.LCMAggregation(DIM, out_dim, project=project)
    pyg_module = pyg_aggr.LCMAggregation(DIM, out_dim, project=project)
    state = gru_state("gru_cell", pyg_module.gru_cell, suffix="")
    if project:
        state.update(torch_state(pyg_module.lin, renames=[(r"^", "lin.")]))
    load_torch(module, state)
    # sets of sizes 3, 2, 0, 4, 1 exercise odd and even tree levels
    compare(module, pyg_module)
    with pytest.raises(ValueError):
        aggr.LCMAggregation(DIM, 6, project=False)


def test_variance_preserving_aggregation():
    out = compare(
        aggr.VariancePreservingAggregation(),
        pyg_aggr.VariancePreservingAggregation(),
    )
    assert np.all(out[2] == 0)


def test_equilibrium_aggregation():
    torch.manual_seed(0)
    module = aggr.EquilibriumAggregation(DIM, 3, num_layers=[8, 8], grad_iter=3)
    pyg_module = pyg_aggr.EquilibriumAggregation(DIM, 3, num_layers=[8, 8], grad_iter=3)
    state = torch_state(
        pyg_module,
        renames=[
            (r"^potential\.layers\.(\d+)\.(\d+)\.", r"potential.layers.\1.layers.\2."),
            (r"^optimizer\._lr$", "optimizer.lr"),
            (r"^optimizer\._mom$", "optimizer.mom"),
        ],
    )
    load_torch(module, state)

    x = make_x()
    out = module(mx.array(x), mx.array(INDEX), NUM_SETS)
    expected = pyg_module(torch.tensor(x), torch.tensor(INDEX), dim_size=NUM_SETS)
    np.testing.assert_allclose(np.array(out), expected.detach().numpy(), **TOL)

    # the output is differentiable w.r.t. the parameters, through the inner
    # gradient steps
    def loss(params):
        module.update(params)
        return module(mx.array(x), mx.array(INDEX), NUM_SETS).sum()

    grads = mx.grad(loss)(module.trainable_parameters())
    expected_out = pyg_module(torch.tensor(x), torch.tensor(INDEX), dim_size=NUM_SETS)
    expected_out.sum().backward()
    np.testing.assert_allclose(
        np.array(grads["lamb"]), pyg_module.lamb.grad.numpy(), rtol=1e-2, atol=1e-4
    )
    np.testing.assert_allclose(
        np.array(grads["optimizer"]["lr"]),
        pyg_module.optimizer._lr.grad.numpy(),
        rtol=1e-2,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "module",
    [
        aggr.LSTMAggregation(DIM, 3),
        aggr.GRUAggregation(DIM, 3),
        aggr.SetTransformerAggregation(DIM, heads=2, dropout=0.1),
        aggr.GraphMultisetTransformer(DIM, k=2, heads=2),
        aggr.MLPAggregation(DIM, 3, max_num_elements=4),
        aggr.LCMAggregation(DIM, 3),
    ],
    ids=lambda m: type(m).__name__,
)
def test_gradients(module):
    index = mx.array([1, 0, 1, 0, 2, 2, 1])
    x = mx.array(make_x(n=7))

    def loss(params):
        module.update(params)
        return module(x, index, 3).sum()

    module.train()
    grads = tree_flatten(mx.grad(loss)(module.trainable_parameters()))
    assert all(np.isfinite(np.array(g)).all() for _, g in grads)
    assert sum(float(mx.abs(g).sum()) for _, g in grads) > 0


def test_exports():
    import mlx_graphs.nn as gnn

    for name in [
        "LSTMAggregation",
        "GRUAggregation",
        "SetTransformerAggregation",
        "GraphMultisetTransformer",
        "MLPAggregation",
        "LCMAggregation",
        "EquilibriumAggregation",
        "VariancePreservingAggregation",
    ]:
        assert getattr(gnn, name) is getattr(aggr, name)
