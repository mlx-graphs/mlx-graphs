import os
import tempfile

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_graphs.nn import (
    GAT,
    GCN,
    GIN,
    GATConv,
    GCNConv,
    GraphConv,
    GraphSAGE,
    HeteroConv,
    Linear,
    SAGEConv,
    ToHeteroModel,
    to_hetero,
    to_hetero_with_bases,
)
from mlx_graphs.nn.models.basic_gnn import MLP
from mlx_graphs.nn.to_hetero import _BasisModules, _type_key, _TypeDispatch

DIMS = {"a": 5, "b": 3, "c": 4}
NUM_NODES = {"a": 6, "b": 7, "c": 4}
EDGE_TYPES = [
    ("a", "to", "b"),
    ("b", "to", "a"),
    ("b", "cites", "b"),
    ("c", "to", "b"),
    ("b", "to", "c"),
]
METADATA = (list(DIMS), EDGE_TYPES)


def random_graph(seed=0, dims=DIMS, edge_types=EDGE_TYPES, num_edges=12):
    rng = np.random.default_rng(seed)
    x_dict = {
        t: rng.standard_normal((NUM_NODES[t], d)).astype(np.float32)
        for t, d in dims.items()
    }
    edge_index_dict = {}
    for src, rel, dst in edge_types:
        edge_index_dict[(src, rel, dst)] = np.stack(
            [
                rng.integers(0, NUM_NODES[src], num_edges),
                rng.integers(0, NUM_NODES[dst], num_edges),
            ]
        ).astype(np.int64)
    return x_dict, edge_index_dict


def to_mx(x_dict, edge_index_dict):
    return (
        {k: mx.array(v) for k, v in x_dict.items()},
        {k: mx.array(v).astype(mx.int32) for k, v in edge_index_dict.items()},
    )


# --------------------------------------------------------------------------- #
# Copying weights from PyG's to_hetero models
# --------------------------------------------------------------------------- #


def _t(tensor) -> mx.array:
    return mx.array(tensor.detach().cpu().numpy())


def _copy_linear(lin, torch_lin):
    lin.weight = _t(torch_lin.weight)
    if "bias" in lin:
        lin.bias = _t(torch_lin.bias)


def copy_module(module, torch_module):
    """Copies the weights of a PyG module into the equivalent mlx module."""
    import torch

    if not tree_flatten(module.parameters()):
        return  # e.g. JumpingKnowledge("cat")
    if isinstance(module, SAGEConv):
        module.neigh_proj.weight = _t(torch_module.lin_l.weight)
        module.self_proj.weight = _t(torch_module.lin_r.weight)
        module.self_proj.bias = _t(torch_module.lin_l.bias)
    elif isinstance(module, GraphConv):
        _copy_linear(module.neigh_proj, torch_module.lin_rel)
        _copy_linear(module.self_proj, torch_module.lin_root)
    elif isinstance(module, GATConv):
        if "lin_proj" in module:
            module.lin_proj.weight = _t(torch_module.lin.weight)
        else:
            module.lin_src_proj.weight = _t(torch_module.lin_src.weight)
            module.lin_dst_proj.weight = _t(torch_module.lin_dst.weight)
        module.att_src = _t(torch_module.att_src)
        module.att_dst = _t(torch_module.att_dst)
        module.bias = _t(torch_module.bias)
    elif isinstance(module, GCNConv):
        module.linear.weight = _t(torch_module.lin.weight)
        module.bias = _t(torch_module.bias)
    elif type(module).__name__ == "GINConv":
        copy_module(module.mlp, torch_module.nn)
    elif isinstance(module, MLP):
        for lin, torch_lin in zip(module.lins, torch_module.lins):
            _copy_linear(lin, torch_lin)
    elif isinstance(module, nn.Linear):
        _copy_linear(module, torch_module)
    elif isinstance(module, nn.BatchNorm):
        bn = torch_module.module
        module.weight, module.bias = _t(bn.weight), _t(bn.bias)
        module.running_mean = _t(bn.running_mean)
        module.running_var = _t(bn.running_var)
    elif isinstance(module, torch.nn.Module):
        raise TypeError(type(module))
    else:
        raise TypeError(f"no weight copy for {type(module).__name__}")


def copy_from_pyg(model: ToHeteroModel, torch_model):
    """Copies the per-type weights of PyG's to_hetero model into ours."""
    for name, sub in model.module.named_modules():
        if not isinstance(sub, _TypeDispatch):
            continue
        torch_container = torch_model.get_submodule(name) if name else torch_model
        for key in sub._keys:
            copy_module(sub[_type_key(key)], torch_container[_type_key(key)])


def assert_outputs_close(out, torch_out, tol=2e-2, rel_tol=5e-3):
    """MLX float32 matmuls on the GPU have a reduced precision (relative errors
    ~1e-3), hence the tolerance used by the other PyG-parity tests, plus a
    tighter bound on the relative error of the whole output, which any wrong
    weight or wrong aggregation would exceed by orders of magnitude."""
    assert set(out) == set(torch_out)
    for node_type, value in out.items():
        expected = torch_out[node_type].detach().cpu().numpy()
        value = np.array(value)
        assert value.shape == expected.shape, node_type
        np.testing.assert_allclose(value, expected, rtol=tol, atol=tol)
        error = np.linalg.norm(value - expected) / max(np.linalg.norm(expected), 1e-6)
        assert error < rel_tol, (node_type, error)


@pytest.fixture
def torch_env():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    yield torch
    torch.set_num_threads(threads)


def torch_inputs(torch, x_dict, edge_index_dict):
    return (
        {k: torch.from_numpy(v) for k, v in x_dict.items()},
        {k: torch.from_numpy(v) for k, v in edge_index_dict.items()},
    )


# --------------------------------------------------------------------------- #
# PyG parity: user models in the PyG style
# --------------------------------------------------------------------------- #


class SkipGNN(nn.Module):
    """PyG's hetero example model: convs with linear skip connections."""

    def __init__(self, hidden, out, dropout=0.0):
        super().__init__()
        self.conv1 = SAGEConv(hidden, hidden)
        self.lin1 = Linear(hidden, hidden)
        self.conv2 = SAGEConv(hidden, out)
        self.lin2 = Linear(hidden, out)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, edge_index, x):
        x = self.dropout(x)
        x = nn.relu(self.conv1(edge_index, x) + self.lin1(x))
        x = self.dropout(x)
        return self.conv2(edge_index, x) + self.lin2(x)


def torch_skip_gnn(torch, hidden, out):
    from torch_geometric.nn import Linear as PygLinear
    from torch_geometric.nn import SAGEConv as PygSAGEConv

    class TorchSkipGNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = PygSAGEConv((-1, -1), hidden)
            self.lin1 = PygLinear(-1, hidden)
            self.conv2 = PygSAGEConv((-1, -1), out)
            self.lin2 = PygLinear(-1, out)

        def forward(self, x, edge_index):
            x = (self.conv1(x, edge_index) + self.lin1(x)).relu()
            return self.conv2(x, edge_index) + self.lin2(x)

    return TorchSkipGNN()


@pytest.mark.parametrize("aggr", ["sum", "mean", "max", "min", "mul"])
def test_pyg_parity_skip_connections(torch_env, aggr):
    from torch_geometric.nn import to_hetero as pyg_to_hetero

    torch = torch_env
    x_np, ei_np = random_graph(seed=1)
    x_t, ei_t = torch_inputs(torch, x_np, ei_np)
    torch_model = pyg_to_hetero(torch_skip_gnn(torch, 8, 3), METADATA, aggr=aggr)
    with torch.no_grad():
        torch_model(x_t, ei_t)  # initializes the lazy layers
        expected = torch_model(x_t, ei_t)

    model = to_hetero(SkipGNN(8, 3, dropout=0.5), METADATA, aggr=aggr, input_dims=DIMS)
    copy_from_pyg(model, torch_model)
    model.eval()
    x_dict, edge_index_dict = to_mx(x_np, ei_np)
    assert_outputs_close(model(edge_index_dict, x_dict), expected)


def _pyg_basic_gnn(torch, name, **kwargs):
    from torch_geometric import nn as pyg_nn

    return getattr(pyg_nn, name)(**kwargs)


@pytest.mark.parametrize(
    "name,kwargs,torch_kwargs",
    [
        ("GraphSAGE", {}, {}),
        ("GraphSAGE", {"jk": "cat", "norm": "batch"}, {"jk": "cat", "norm": "batch"}),
        ("GraphSAGE", {"jk": "max"}, {"jk": "max"}),
        ("GraphSAGE", {"aggr": "max"}, {"aggr": "max"}),
        (
            "GAT",
            {"heads": 2, "add_self_loops": False},
            {"heads": 2, "add_self_loops": False},
        ),
        (
            "GAT",
            {"heads": 2, "jk": "cat", "add_self_loops": False},
            {"heads": 2, "jk": "cat", "add_self_loops": False},
        ),
    ],
)
def test_pyg_parity_basic_gnn(torch_env, name, kwargs, torch_kwargs):
    from torch_geometric.nn import to_hetero as pyg_to_hetero

    torch = torch_env
    x_np, ei_np = random_graph(seed=2)
    x_t, ei_t = torch_inputs(torch, x_np, ei_np)
    torch_model = _pyg_basic_gnn(
        torch,
        name,
        in_channels=(-1, -1),
        hidden_channels=8,
        num_layers=3,
        out_channels=3,
        **torch_kwargs,
    )
    torch_model = pyg_to_hetero(torch_model, METADATA, aggr="sum")
    with torch.no_grad():
        torch_model(x_t, ei_t)
        if "norm" in torch_kwargs:
            # non-trivial running statistics
            torch_model.train()
            torch_model(x_t, ei_t)
        torch_model.eval()
        expected = torch_model(x_t, ei_t)

    cls = {"GraphSAGE": GraphSAGE, "GAT": GAT}[name]
    model = to_hetero(
        cls((8, 8), 8, num_layers=3, out_features_dim=3, **kwargs),
        METADATA,
        input_dims=DIMS,
    )
    copy_from_pyg(model, torch_model)
    model.eval()
    x_dict, edge_index_dict = to_mx(x_np, ei_np)
    assert_outputs_close(model(edge_index_dict, x_dict), expected)


def test_pyg_parity_gin(torch_env):
    """GIN needs the same input size for both ends of an edge type."""
    from torch_geometric.nn import GIN as PygGIN
    from torch_geometric.nn import to_hetero as pyg_to_hetero

    torch = torch_env
    dims = {t: 4 for t in DIMS}
    x_np, ei_np = random_graph(seed=3, dims=dims)
    x_t, ei_t = torch_inputs(torch, x_np, ei_np)
    torch_model = pyg_to_hetero(PygGIN(4, 8, 2, out_channels=3), METADATA)
    with torch.no_grad():
        expected = torch_model(x_t, ei_t)

    model = to_hetero(GIN(4, 8, num_layers=2, out_features_dim=3), METADATA)
    copy_from_pyg(model, torch_model)
    x_dict, edge_index_dict = to_mx(x_np, ei_np)
    assert_outputs_close(model(edge_index_dict, x_dict), expected)


def test_pyg_parity_gcn_with_edge_weights(torch_env):
    """GCN only works without bipartite edge types (as in PyG)."""
    from torch_geometric.nn import GCN as PygGCN
    from torch_geometric.nn import to_hetero as pyg_to_hetero

    torch = torch_env
    dims = {"a": 5, "b": 3}
    edge_types = [("a", "to", "a"), ("b", "to", "b"), ("a", "near", "a")]
    metadata = (list(dims), edge_types)
    x_np, ei_np = random_graph(seed=4, dims=dims, edge_types=edge_types)
    rng = np.random.default_rng(0)
    ew_np = {et: rng.uniform(0.5, 1.5, 12).astype(np.float32) for et in edge_types}
    x_t, ei_t = torch_inputs(torch, x_np, ei_np)
    ew_t = {k: torch.from_numpy(v) for k, v in ew_np.items()}
    torch_model = pyg_to_hetero(PygGCN(-1, 8, 2, out_channels=3), metadata)
    with torch.no_grad():
        torch_model(x_t, ei_t, ew_t)
        expected = torch_model(x_t, ei_t, ew_t)

    model = to_hetero(
        GCN(16, 8, num_layers=2, out_features_dim=3), metadata, input_dims=dims
    )
    copy_from_pyg(model, torch_model)
    x_dict, edge_index_dict = to_mx(x_np, ei_np)
    ew = {k: mx.array(v) for k, v in ew_np.items()}
    assert_outputs_close(model(edge_index_dict, x_dict, edge_weights=ew), expected)


# --------------------------------------------------------------------------- #
# Behaviour without PyG
# --------------------------------------------------------------------------- #


def test_single_layer_matches_hetero_conv():
    """A converted SAGEConv is HeteroConv with one SAGEConv per edge type."""
    dims = {t: 4 for t in DIMS}
    x_dict, edge_index_dict = to_mx(*random_graph(seed=5, dims=dims))
    model = to_hetero(SAGEConv(4, 6), METADATA, aggr="mean")
    hetero = HeteroConv(
        {et: model.module.module_for(et) for et in EDGE_TYPES}, aggr="mean"
    )
    out = model(edge_index_dict, x_dict)
    expected = hetero(x_dict, edge_index_dict)
    for node_type in DIMS:
        assert mx.allclose(out[node_type], expected[node_type], atol=1e-5)


def test_copies_are_independent_and_reinitialized():
    model = to_hetero(GraphSAGE(4, 8, num_layers=2), METADATA)
    conv = model.module.convs[0]
    weights = [conv.module_for(et).neigh_proj.weight for et in EDGE_TYPES]
    assert all(w.shape == (8, 4) for w in weights)
    assert not mx.array_equal(weights[0], weights[1])
    # parameters for every type, visible to MLX
    names = [k for k, _ in tree_flatten(model.trainable_parameters())]
    assert "module.convs.0.a__to__b.neigh_proj.weight" in names
    assert "module.convs.1.b__cites__b.self_proj.bias" in names


def test_input_dims_rebuilds_first_layers():
    model = to_hetero(SkipGNN(8, 3), METADATA, input_dims=DIMS)
    conv1, lin1 = model.module.conv1, model.module.lin1
    assert conv1.module_for(("a", "to", "b")).neigh_proj.weight.shape == (8, 5)
    assert conv1.module_for(("a", "to", "b")).self_proj.weight.shape == (8, 3)
    assert conv1.module_for(("c", "to", "b")).neigh_proj.weight.shape == (8, 4)
    assert lin1.module_for("a").weight.shape == (8, 5)
    assert lin1.module_for("c").weight.shape == (8, 4)
    # later layers keep the hidden size
    assert model.module.conv2.module_for(("a", "to", "b")).neigh_proj.weight.shape == (
        3,
        8,
    )


def test_dropout_on_input_features():
    """Row-wise ops on the (padded) input features before the first layers."""

    class DropFirst(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = SAGEConv(8, 3)

        def __call__(self, edge_index, x):
            return self.conv(edge_index, nn.relu(x) * 2.0)

    x_dict, edge_index_dict = to_mx(*random_graph(seed=6))
    model = to_hetero(DropFirst(), METADATA, input_dims=DIMS)
    out = model(edge_index_dict, x_dict)
    # the same as passing the transformed features directly
    direct = to_hetero(SAGEConv(8, 3), METADATA, input_dims=DIMS)
    direct.module.update(model.module.conv.parameters())
    expected = direct(edge_index_dict, {k: nn.relu(v) * 2.0 for k, v in x_dict.items()})
    for node_type in DIMS:
        assert mx.allclose(out[node_type], expected[node_type], atol=1e-5)


def test_jk_and_norm_models_run():
    x_dict, edge_index_dict = to_mx(*random_graph(seed=7))
    for model in [
        GraphSAGE(16, 8, num_layers=3, out_features_dim=2, jk="lstm"),
        GAT(
            16,
            8,
            num_layers=2,
            heads=2,
            norm="layer",
            dropout=0.2,
            add_self_loops=False,
        ),
        GraphSAGE(16, 8, num_layers=2, norm="batch", jk="max"),
    ]:
        out = to_hetero(model, METADATA, input_dims=DIMS)(edge_index_dict, x_dict)
        assert set(out) == set(DIMS)
        assert all(out[t].shape[0] == NUM_NODES[t] for t in DIMS)


def test_nested_model_and_tuple_outputs():
    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.gnn = GraphSAGE(4, 8, num_layers=2)
            self.head = Linear(8, 2)

        def __call__(self, edge_index, x):
            h = self.gnn(edge_index, x)
            return self.head(h), h

    dims = {t: 4 for t in DIMS}
    x_dict, edge_index_dict = to_mx(*random_graph(seed=8, dims=dims))
    logits, h = to_hetero(Outer(), METADATA)(edge_index_dict, x_dict)
    assert {k: v.shape for k, v in logits.items()} == {
        t: (NUM_NODES[t], 2) for t in DIMS
    }
    assert h["a"].shape == (NUM_NODES["a"], 8)


def test_missing_edge_types_are_skipped():
    dims = {t: 4 for t in DIMS}
    x_dict, edge_index_dict = to_mx(*random_graph(seed=9, dims=dims))
    model = to_hetero(SAGEConv(4, 6), METADATA)
    partial = dict(edge_index_dict)
    del partial[("c", "to", "b")]
    out = model(partial, x_dict)
    expected = HeteroConv(
        {et: model.module.module_for(et) for et in EDGE_TYPES}, aggr="sum"
    )(x_dict, partial)
    assert mx.allclose(out["b"], expected["b"], atol=1e-5)


def test_empty_edge_types():
    """Edge types without edges (e.g. in a sampled mini-batch) still run."""
    dims = {t: 4 for t in DIMS}
    x_dict, edge_index_dict = to_mx(*random_graph(seed=10, dims=dims))
    edge_index_dict[("c", "to", "b")] = mx.zeros((2, 0), dtype=mx.int32)
    out = to_hetero(GraphSAGE(4, 8, num_layers=2), METADATA)(edge_index_dict, x_dict)
    assert out["c"].shape == (NUM_NODES["c"], 8)


def test_training_step_and_weights_roundtrip():
    x_dict, edge_index_dict = to_mx(*random_graph(seed=11))
    labels = mx.array(np.arange(NUM_NODES["a"]) % 3)
    model = to_hetero(
        GraphSAGE(16, 8, num_layers=2, out_features_dim=3, dropout=0.2),
        METADATA,
        input_dims=DIMS,
    )
    mx.eval(model.parameters())

    def loss_fn(model):
        out = model(edge_index_dict, x_dict)
        return nn.losses.cross_entropy(out["a"], labels, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=0.01)
    losses = []
    for _ in range(30):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    flat_grads = dict(tree_flatten(grads))
    assert "module.convs.0.b__to__a.neigh_proj.weight" in flat_grads

    model.eval()
    out = model(edge_index_dict, x_dict)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "weights.safetensors")
        model.save_weights(path)
        other = to_hetero(
            GraphSAGE(16, 8, num_layers=2, out_features_dim=3),
            METADATA,
            input_dims=DIMS,
        )
        other.load_weights(path)
    other.eval()
    out2 = other(edge_index_dict, x_dict)
    for t in DIMS:
        assert mx.allclose(out[t], out2[t])


def test_compile():
    dims = {t: 4 for t in DIMS}
    x_dict, edge_index_dict = to_mx(*random_graph(seed=12, dims=dims))
    model = to_hetero(GraphSAGE(4, 8, num_layers=2), METADATA)
    expected = model(edge_index_dict, x_dict)
    compiled = mx.compile(lambda ei, x: model(ei, x))
    out = compiled(edge_index_dict, x_dict)
    for t in DIMS:
        assert mx.allclose(out[t], expected[t], atol=1e-5)


def test_original_model_is_unchanged():
    gnn = GraphSAGE(4, 8, num_layers=2)
    to_hetero(gnn, METADATA)
    assert isinstance(gnn.convs[0], SAGEConv)
    out = gnn(mx.array([[0, 1], [1, 0]]), mx.ones((2, 4)))
    assert out.shape == (2, 8)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_error_gcn_on_bipartite_edge_types():
    with pytest.raises(ValueError, match="bipartite"):
        to_hetero(GCN(4, 8, num_layers=2), METADATA)


def test_error_node_type_without_incoming_edges():
    metadata = (["a", "b"], [("a", "to", "b")])
    with pytest.raises(ValueError, match="not the destination"):
        to_hetero(GraphSAGE(4, 8, num_layers=2), metadata)


def test_error_different_dims_without_input_dims():
    x_dict, edge_index_dict = to_mx(*random_graph(seed=13))
    model = to_hetero(GraphSAGE(4, 8, num_layers=2), METADATA)
    with pytest.raises(ValueError, match="input_dims"):
        model(edge_index_dict, x_dict)


def test_error_wrong_dims_for_input_dims():
    x_dict, edge_index_dict = to_mx(*random_graph(seed=13))
    model = to_hetero(GraphSAGE(4, 8, num_layers=2), METADATA, input_dims=DIMS)
    x_dict["a"] = mx.ones((NUM_NODES["a"], 9))
    with pytest.raises(ValueError, match="input_dims"):
        model(edge_index_dict, x_dict)


def test_error_unsupported_first_layer_with_input_dims():
    from mlx_graphs.nn import TransformerConv

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = TransformerConv(4, 8)

        def __call__(self, edge_index, x):
            return self.conv(edge_index, x)

    with pytest.raises(RuntimeError, match="HeteroDictLinear"):
        to_hetero(Model(), METADATA, input_dims=DIMS)


def test_error_graph_level_module():
    class Pooling(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = SAGEConv(4, 8)
            self.lin = Linear(8, 2)

        def __call__(self, edge_index, x):
            return self.lin(self.conv(edge_index, x).mean(axis=0, keepdims=True))

    dims = {t: 4 for t in DIMS}
    x_dict, edge_index_dict = to_mx(*random_graph(seed=14, dims=dims))
    model = to_hetero(Pooling(), METADATA)
    with pytest.raises(NotImplementedError, match="node-level"):
        model(edge_index_dict, x_dict)


def test_error_missing_node_type_and_bad_metadata():
    model = to_hetero(GraphSAGE(4, 8, num_layers=2), METADATA)
    x_dict, edge_index_dict = to_mx(*random_graph(seed=15, dims={t: 4 for t in DIMS}))
    del x_dict["c"]
    with pytest.raises(ValueError, match="missing node types"):
        model(edge_index_dict, x_dict)
    with pytest.raises(ValueError, match="aggregation"):
        to_hetero(GraphSAGE(4, 8, num_layers=2), METADATA, aggr="cat")
    with pytest.raises(ValueError, match="not in the node types"):
        to_hetero(GraphSAGE(4, 8, num_layers=2), (["a"], [("a", "to", "z")]))


def test_module_shared_under_two_names_stays_shared():
    """As in PyG, a module stored under two names keeps a single set of
    per-type copies, instead of being untied."""

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = SAGEConv(4, 4)
            self.lin = Linear(4, 4)
            self.lin_alias = self.lin

        def __call__(self, edge_index, x):
            h = self.conv(edge_index, x)
            return self.lin(h) - self.lin_alias(h)

    dims = {t: 4 for t in DIMS}
    x_dict, edge_index_dict = to_mx(*random_graph(seed=12, dims=dims))
    model = to_hetero(Tied(), METADATA)
    assert model.module.lin is model.module.lin_alias
    for value in model(edge_index_dict, x_dict).values():
        assert mx.array_equal(value, mx.zeros_like(value))


def test_reinitialization_keeps_frozen_buffers():
    """Re-randomizing the copies must not scramble buffers such as the
    running statistics of a batch norm (means and variances of a feature
    must stay paired)."""
    model = GraphSAGE(4, 8, num_layers=2, out_features_dim=3, norm="batch")
    model.norms[0].running_mean = mx.arange(8).astype(mx.float32)
    model.norms[0].running_var = mx.arange(1, 9).astype(mx.float32)
    model = to_hetero(model, METADATA)
    for node_type in DIMS:
        norm = model.module.norms[0][node_type]
        assert norm.running_mean.tolist() == list(range(8))
        assert norm.running_var.tolist() == list(range(1, 9))
    # the weights are still re-randomized per type
    lins = [model.module.convs[0][_type_key(et)].neigh_proj for et in EDGE_TYPES]
    assert not mx.array_equal(lins[0].weight, lins[1].weight)


def test_reinitialization_uses_reset_parameters():
    """A module's own reset_parameters() gives the copies a fresh
    initialization (here a non-iid one, which a permutation would break)."""

    class Ramp(nn.Module):
        def __init__(self):
            super().__init__()
            self.reset_parameters()

        def reset_parameters(self):
            self.scale = mx.arange(1, 5).astype(mx.float32)
            self.lin = nn.Linear(4, 4)

        def __call__(self, x):
            return self.lin(x) * self.scale

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = SAGEConv(4, 4)
            self.ramp = Ramp()

        def __call__(self, edge_index, x):
            return self.ramp(self.conv(edge_index, x))

    model = to_hetero(Model(), METADATA)
    copies = [model.module.ramp[t] for t in DIMS]
    for copy_ in copies:
        assert copy_.scale.tolist() == [1, 2, 3, 4]
    assert not mx.array_equal(copies[0].lin.weight, copies[1].lin.weight)
    dims = {t: 4 for t in DIMS}
    x_dict, edge_index_dict = to_mx(*random_graph(seed=14, dims=dims))
    assert model(edge_index_dict, x_dict)["a"].shape == (NUM_NODES["a"], 4)


@pytest.mark.parametrize("bases", [False, True])
def test_error_edge_index_changed_in_model(bases):
    """Edges removed inside the model cannot be split per edge type: raise
    instead of passing the whole packed edge index to every copy."""

    class DropEdges(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = SAGEConv(4, 4)

        def __call__(self, edge_index, x):
            return self.conv(edge_index[:, :-3], x)

    dims = {t: 4 for t in DIMS}
    x_dict, edge_index_dict = to_mx(*random_graph(seed=13, dims=dims))
    if bases:
        model = to_hetero_with_bases(DropEdges(), METADATA, num_bases=2)
    else:
        model = to_hetero(DropEdges(), METADATA)
    with pytest.raises(ValueError, match="edge index"):
        model(edge_index_dict, x_dict)


def test_layer_used_outside_model_raises():
    model = to_hetero(GraphSAGE(4, 8, num_layers=2), METADATA)
    with pytest.raises(RuntimeError, match="to_hetero"):
        model.module.convs[0](mx.array([[0], [0]]), mx.ones((1, 4)))


# --------------------------------------------------------------------------- #
# Real datasets (slow: downloads DBLP and IMDB)
# --------------------------------------------------------------------------- #


def _add_constant_features(x_dict, num_nodes):
    for node_type, n in num_nodes.items():
        if node_type not in x_dict:
            x_dict[node_type] = np.ones((n, 1), dtype=np.float32)
    return x_dict


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["DBLP", "IMDB"])
def test_pyg_parity_on_dataset(torch_env, tmp_path, dataset):
    from torch_geometric import datasets as pyg_datasets
    from torch_geometric.nn import GraphSAGE as PygGraphSAGE
    from torch_geometric.nn import to_hetero as pyg_to_hetero

    from mlx_graphs import datasets

    torch = torch_env
    data = getattr(datasets, dataset)(base_dir=str(tmp_path / "mlx"))[0]
    torch_data = getattr(pyg_datasets, dataset)(str(tmp_path / "pyg"))[0]

    x_np = {t: np.array(v) for t, v in data.node_features_dict.items()}
    x_np = _add_constant_features(x_np, data.num_nodes)
    ei_np = {et: np.array(v) for et, v in data.edge_index_dict.items()}
    metadata = data.metadata()
    dims = {t: v.shape[1] for t, v in x_np.items()}

    x_t, ei_t = torch_inputs(torch, x_np, ei_np)
    torch_model = pyg_to_hetero(
        PygGraphSAGE((-1, -1), 16, 2, out_channels=4), torch_data.metadata()
    )
    with torch.no_grad():
        torch_model(x_t, ei_t)
        expected = torch_model(x_t, ei_t)

    model = to_hetero(
        GraphSAGE((16, 16), 16, num_layers=2, out_features_dim=4),
        metadata,
        input_dims=dims,
    )
    copy_from_pyg(model, torch_model)
    x_dict, edge_index_dict = to_mx(x_np, ei_np)
    assert_outputs_close(model(edge_index_dict, x_dict), expected)


# --------------------------------------------------------------------------- #
# to_hetero_with_bases
# --------------------------------------------------------------------------- #


class BasesGNN(nn.Module):
    def __init__(self, in_dim, hidden, out):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.lin = Linear(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out)

    def __call__(self, edge_index, x):
        x = nn.relu(self.conv1(edge_index, x) + self.lin(x))
        return self.conv2(edge_index, x)


def test_pyg_parity_with_bases(torch_env):
    from torch_geometric.nn import Linear as PygLinear
    from torch_geometric.nn import SAGEConv as PygSAGEConv
    from torch_geometric.nn import to_hetero_with_bases as pyg_with_bases

    torch = torch_env

    class TorchBasesGNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = PygSAGEConv(6, 8)
            self.lin = PygLinear(6, 8)
            self.conv2 = PygSAGEConv(8, 3)

        def forward(self, x, edge_index):
            x = (self.conv1(x, edge_index) + self.lin(x)).relu()
            return self.conv2(x, edge_index)

    x_np, ei_np = random_graph(seed=16)
    x_t, ei_t = torch_inputs(torch, x_np, ei_np)
    torch_model = pyg_with_bases(
        TorchBasesGNN(), METADATA, num_bases=2, in_channels={"x": 6}
    )
    with torch.no_grad():
        torch_model(x_t, ei_t)
        expected = torch_model(x_t, ei_t)

    model = to_hetero_with_bases(
        BasesGNN(6, 8, 3),
        METADATA,
        num_bases=2,
        in_channels={"x": 6},
        input_dims=DIMS,
    )
    for name, sub in model.module.named_modules():
        if isinstance(sub, _BasisModules):
            torch_convs = torch_model.get_submodule(name).convs
            for conv, torch_conv in zip(sub.convs, torch_convs):
                copy_module(conv, torch_conv)
                conv.edge_type_weight = _t(torch_conv.edge_type_weight)
    _copy_linear(model.module.lin, torch_model.lin)
    for t in DIMS:
        model.align["x"][t].weight = _t(torch_model.align_lin__x.lins[t].weight)

    x_dict, edge_index_dict = to_mx(x_np, ei_np)
    assert_outputs_close(model(edge_index_dict, x_dict), expected)


def test_with_bases_parameters_and_errors():
    dims = {t: 4 for t in DIMS}
    x_dict, edge_index_dict = to_mx(*random_graph(seed=17, dims=dims))
    model = to_hetero_with_bases(GraphSAGE(4, 8, num_layers=2), METADATA, num_bases=3)
    names = [k for k, _ in tree_flatten(model.trainable_parameters())]
    assert "module.convs.0.convs.2.edge_type_weight" in names
    # the number of weights does not grow with the number of edge types
    assert sum("neigh_proj.weight" in n for n in names) == 2 * 3
    out = model(edge_index_dict, x_dict)
    assert {k: v.shape for k, v in out.items()} == {t: (NUM_NODES[t], 8) for t in DIMS}

    def loss_fn(model):
        return model(edge_index_dict, x_dict)["a"].sum()

    _, grads = nn.value_and_grad(model, loss_fn)(model)
    grad = dict(tree_flatten(grads))["module.convs.0.convs.0.edge_type_weight"]
    assert mx.abs(grad).sum().item() > 0

    with pytest.raises(ValueError, match="self-loops"):
        to_hetero_with_bases(GCN(4, 8, num_layers=2), METADATA, num_bases=2)
    model = to_hetero_with_bases(GraphSAGE(4, 8, num_layers=2), METADATA, num_bases=2)
    x_dict, edge_index_dict = to_mx(*random_graph(seed=18))
    with pytest.raises(ValueError, match="in_channels"):
        model(edge_index_dict, x_dict)
