import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from mlx_graphs.nn.models import GAT, GCN, GIN, MLP, GraphSAGE

NUM_NODES, NUM_EDGES, DIM = 30, 120, 12


@pytest.fixture
def graph():
    rng = np.random.default_rng(0)
    edge_index = mx.array(rng.integers(0, NUM_NODES, (2, NUM_EDGES)))
    x = mx.array(rng.standard_normal((NUM_NODES, DIM)).astype(np.float32))
    return edge_index, x


MODELS = {
    "gcn": lambda **kw: GCN(DIM, 16, **kw),
    "sage": lambda **kw: GraphSAGE(DIM, 16, **kw),
    "gin": lambda **kw: GIN(DIM, 16, **kw),
    "gat": lambda **kw: GAT(DIM, 16, heads=4, **kw),
    "gatv2": lambda **kw: GAT(DIM, 16, heads=4, v2=True, **kw),
}


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("out_dim", [None, 5])
@pytest.mark.parametrize("jk", [None, "cat", "max", "lstm"])
def test_output_shape(graph, name, num_layers, out_dim, jk):
    edge_index, x = graph
    model = MODELS[name](num_layers=num_layers, out_features_dim=out_dim, jk=jk)
    out = model(edge_index, x)
    assert out.shape == (NUM_NODES, out_dim if out_dim is not None else 16)
    assert len(model.convs) == num_layers
    assert bool(mx.all(mx.isfinite(out)).item())


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("norm", ["batch", "layer"])
def test_norm_and_dropout(graph, name, norm):
    edge_index, x = graph
    model = MODELS[name](num_layers=3, out_features_dim=4, norm=norm, dropout=0.5)
    assert len(model.norms) == 2  # not after the output layer
    model.train()
    assert model(edge_index, x).shape == (NUM_NODES, 4)
    model.eval()
    out1, out2 = model(edge_index, x), model(edge_index, x)
    # no dropout in eval mode (the fused COO kernel sums in a varying order,
    # so repeated runs agree to float rounding, not bit for bit)
    np.testing.assert_allclose(np.array(out1), np.array(out2), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("name", MODELS)
def test_training_reduces_loss(graph, name):
    edge_index, x = graph
    labels = mx.array(np.random.default_rng(1).integers(0, 3, NUM_NODES))
    mx.random.seed(0)
    model = MODELS[name](num_layers=2, out_features_dim=3)
    # bias_correction=True: the update rule of torch.optim.Adam
    optimizer = optim.Adam(learning_rate=0.01, bias_correction=True)

    def loss_fn(model):
        return nn.losses.cross_entropy(model(edge_index, x), labels, reduction="mean")

    step = nn.value_and_grad(model, loss_fn)
    losses = []
    for _ in range(80):
        loss, grads = step(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss.item())
    assert losses[-1] < 0.7 * losses[0]


def test_edge_arguments(graph):
    edge_index, x = graph
    weights = mx.random.uniform(shape=(NUM_EDGES,))
    edge_features = mx.random.normal((NUM_EDGES, 3))

    for name in ("gcn", "sage", "gin"):
        model = MODELS[name](num_layers=2, out_features_dim=2)
        weighted = model(edge_index, x, edge_weights=weights)
        assert weighted.shape == (NUM_NODES, 2)
        assert not np.allclose(np.array(weighted), np.array(model(edge_index, x)))
        with pytest.raises(ValueError):
            model(edge_index, x, edge_features=edge_features)

    model = GAT(DIM, 16, num_layers=2, out_features_dim=2, heads=2, edge_features_dim=3)
    assert model(edge_index, x, edge_features=edge_features).shape == (NUM_NODES, 2)
    with pytest.raises(ValueError):
        model(edge_index, x, edge_weights=weights)


def test_gcn_adds_self_loops_by_default(graph):
    edge_index, x = graph
    assert GCN(DIM, 8, num_layers=1).convs[0]._add_self_loops is True
    assert (
        GCN(DIM, 8, num_layers=1, add_self_loops=False).convs[0]._add_self_loops
        is False
    )


def test_invalid_arguments():
    with pytest.raises(ValueError):
        GCN(DIM, 16, num_layers=0)
    with pytest.raises(ValueError):
        GAT(DIM, 10, num_layers=2, heads=4)  # 10 not divisible by 4
    with pytest.raises(ValueError):
        GCN(DIM, 16, num_layers=2, act="nope")
    with pytest.raises(ValueError):
        MLP([4])


@pytest.mark.parametrize("plain_last", [True, False])
@pytest.mark.parametrize("norm", [None, "batch", "layer"])
def test_mlp(plain_last, norm):
    mlp = MLP([DIM, 32, 32, 3], dropout=0.2, norm=norm, plain_last=plain_last)
    out = mlp(mx.random.normal((7, DIM)))
    assert out.shape == (7, 3)
    assert len(mlp.lins) == 3
    if norm is not None:
        assert len(mlp.norms) == (2 if plain_last else 3)
    if not plain_last:
        mlp.eval()
        assert bool(mx.all(mlp(mx.random.normal((7, DIM))) >= 0).item())  # relu last


@pytest.mark.parametrize("norm", [None, "batch"])
def test_gin_matches_pyg_with_act_and_norm(graph, norm):
    """As PyG's GIN, the MLP of every GINConv uses the model's activation and
    normalization (compared in training mode: batch statistics)."""
    torch = pytest.importorskip("torch")
    pyg_nn = pytest.importorskip("torch_geometric.nn")
    edge_index, x = graph
    model = GIN(DIM, 16, num_layers=3, out_features_dim=4, act="elu", norm=norm)
    pyg_model = pyg_nn.GIN(
        DIM,
        16,
        num_layers=3,
        out_channels=4,
        act="elu",
        norm=None if norm is None else "batch_norm",
    )
    with torch.no_grad():
        for conv, pyg_conv in zip(model.convs, pyg_model.convs):
            assert len(conv.mlp.lins) == len(pyg_conv.nn.lins) == 2
            pyg_norms = [
                n for n in pyg_conv.nn.norms if not isinstance(n, torch.nn.Identity)
            ]
            assert len(conv.mlp.get("norms", [])) == len(pyg_norms)
            for lin, pyg_lin in zip(conv.mlp.lins, pyg_conv.nn.lins):
                pyg_lin.weight.copy_(torch.tensor(np.array(lin.weight)))
                pyg_lin.bias.copy_(torch.tensor(np.array(lin.bias)))
    model.train()
    pyg_model.train()
    out = model(edge_index, x)
    with torch.no_grad():
        expected = pyg_model(
            torch.tensor(np.array(x)), torch.tensor(np.array(edge_index)).long()
        )
    np.testing.assert_allclose(np.array(out), expected.numpy(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("v2", [False, True])
def test_gat_applies_its_dropout_to_attention_like_pyg(v2):
    """Regression: PyG's GAT passes its dropout to every GATConv (attention
    dropout); mlx-graphs' only applied it between layers."""
    model = GAT(8, 16, num_layers=2, out_features_dim=3, heads=2, v2=v2, dropout=0.5)
    assert all("dropout" in conv for conv in model.convs)
    assert all("dropout" not in conv for conv in GAT(8, 16, num_layers=2).convs)

    edge_index = mx.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    x = mx.random.normal((4, 8))
    model = GAT(8, 16, num_layers=2, out_features_dim=3, heads=2, v2=v2, dropout=0.5)
    model.eval()
    assert mx.array_equal(model(edge_index, x), model(edge_index, x))
