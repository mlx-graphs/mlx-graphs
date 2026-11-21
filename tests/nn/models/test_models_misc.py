"""Tests of EdgeCNN, LINKX, DeepGCNLayer, MaskLabel, RECT_L and DeepGraphInfomax."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from mlx_graphs.nn import GCN, GCNConv
from mlx_graphs.nn.models import (
    LINKX,
    RECT_L,
    DeepGCNLayer,
    DeepGraphInfomax,
    EdgeCNN,
    MaskLabel,
)

NUM_NODES, NUM_EDGES, DIM = 20, 60, 6
# GPU matmuls run in reduced precision (see tests/nn/conv/test_gcn_conv.py)
TOL = 2e-2


@pytest.fixture
def graph():
    rng = np.random.default_rng(0)
    edge_index = rng.integers(0, NUM_NODES, (2, NUM_EDGES))
    x = rng.standard_normal((NUM_NODES, DIM)).astype(np.float32)
    weights = rng.random(NUM_EDGES).astype(np.float32)
    return edge_index, x, weights


def _copy_linear(torch, dst, src):
    """Copies an MLX linear layer into a torch one (both store [out, in])."""
    with torch.no_grad():
        dst.weight.copy_(torch.tensor(np.array(src.weight)))
        if "bias" in src:
            dst.bias.copy_(torch.tensor(np.array(src.bias)))


def _train(model, loss_fn, steps=50, lr=0.01):
    optimizer = optim.Adam(learning_rate=lr)
    step = nn.value_and_grad(model, loss_fn)
    first = loss_fn(model).item()
    for _ in range(steps):
        _, grads = step(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    return first, loss_fn(model).item()


# EdgeCNN


@pytest.mark.parametrize("out_dim", [None, 3])
@pytest.mark.parametrize("jk", [None, "cat"])
def test_edge_cnn(graph, out_dim, jk):
    edge_index, x, _ = graph
    model = EdgeCNN(DIM, 16, num_layers=2, out_features_dim=out_dim, jk=jk)
    out = model(mx.array(edge_index), mx.array(x))
    assert out.shape == (NUM_NODES, out_dim or 16)
    assert model.convs[0].mlp.lins[0].weight.shape == (16, 2 * DIM)
    assert model.convs[0].aggr == "max"
    assert EdgeCNN(DIM, 8, 1, aggr="mean").convs[0].aggr == "mean"


def test_edge_cnn_parity_act_and_norm(graph):
    """The inner MLPs use the model's activation and normalization, as PyG."""
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric.nn")
    edge_index, x, _ = graph
    model = EdgeCNN(DIM, 8, num_layers=2, out_features_dim=3, act="elu", norm="batch")
    torch.manual_seed(0)
    pyg_model = pyg.EdgeCNN(DIM, 8, 2, out_channels=3, act="elu", norm="batch_norm")
    for conv, pyg_conv in zip(model.convs, pyg_model.convs):
        for lin, pyg_lin in zip(conv.mlp.lins, pyg_conv.nn.lins):
            _copy_linear(torch, pyg_lin, lin)
        assert len(conv.mlp.norms) == len(pyg_conv.nn.norms) == 1
    # batch statistics in training mode (no dropout)
    model.train()
    pyg_model.train()
    out = model(mx.array(edge_index), mx.array(x))
    with torch.no_grad():
        expected = pyg_model(torch.tensor(x), torch.tensor(edge_index))
    np.testing.assert_allclose(np.array(out), expected.numpy(), rtol=TOL, atol=TOL)


def test_edge_cnn_training(graph):
    edge_index, x, _ = graph
    mx.random.seed(0)
    labels = mx.array(np.random.default_rng(1).integers(0, 3, NUM_NODES))
    model = EdgeCNN(DIM, 16, num_layers=2, out_features_dim=3, aggr="mean")
    ei, xx = mx.array(edge_index), mx.array(x)

    def loss_fn(model):
        return nn.losses.cross_entropy(model(ei, xx), labels, reduction="mean")

    first, last = _train(model, loss_fn)
    assert last < 0.8 * first


# LINKX


@pytest.mark.parametrize("num_edge_layers", [1, 2])
@pytest.mark.parametrize("use_features", [True, False])
@pytest.mark.parametrize("use_weights", [True, False])
def test_linkx_parity(graph, num_edge_layers, use_features, use_weights):
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric.nn")
    edge_index, x, weights = graph
    kwargs = dict(num_edge_layers=num_edge_layers, num_node_layers=2)
    model = LINKX(NUM_NODES, DIM, 16, 3, num_layers=2, **kwargs)
    pyg_model = pyg.LINKX(NUM_NODES, DIM, 16, 3, num_layers=2, **kwargs)
    model.eval()
    pyg_model.eval()

    with torch.no_grad():
        pyg_model.edge_lin.weight.copy_(torch.tensor(np.array(model.edge_weight)))
        pyg_model.edge_lin.bias.copy_(torch.tensor(np.array(model.edge_bias)))
        for name in ("node_mlp", "final_mlp", "edge_mlp"):
            if name in model:
                for lin, pyg_lin in zip(
                    getattr(model, name).lins, getattr(pyg_model, name).lins
                ):
                    _copy_linear(torch, pyg_lin, lin)
        _copy_linear(torch, pyg_model.cat_lin1, model.cat_lin1)
        _copy_linear(torch, pyg_model.cat_lin2, model.cat_lin2)

        out = model(
            mx.array(edge_index),
            mx.array(x) if use_features else None,
            mx.array(weights) if use_weights else None,
        )
        expected = pyg_model(
            torch.tensor(x) if use_features else None,
            torch.tensor(edge_index),
            torch.tensor(weights) if use_weights else None,
        )
    assert out.shape == (NUM_NODES, 3)
    np.testing.assert_allclose(np.array(out), expected.numpy(), rtol=TOL, atol=TOL)


def test_linkx_training(graph):
    edge_index, x, _ = graph
    mx.random.seed(0)
    labels = mx.array(np.random.default_rng(1).integers(0, 3, NUM_NODES))
    model = LINKX(NUM_NODES, DIM, 16, 3, num_layers=2, num_edge_layers=2, dropout=0.1)
    ei, xx = mx.array(edge_index), mx.array(x)

    def loss_fn(model):
        return nn.losses.cross_entropy(model(ei, xx), labels, reduction="mean")

    first, last = _train(model, loss_fn)
    assert last < 0.5 * first


# DeepGCNLayer


@pytest.mark.parametrize("block", ["res+", "res", "dense", "plain"])
def test_deepgcn_layer_parity(graph, block):
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric.nn")
    edge_index, x, _ = graph

    conv = GCNConv(DIM, DIM, add_self_loops=True)
    conv.bias = mx.random.normal((DIM,))
    norm = nn.LayerNorm(DIM)
    layer = DeepGCNLayer(conv, norm, nn.relu, block=block)

    pyg_conv = pyg.GCNConv(DIM, DIM)
    with torch.no_grad():
        pyg_conv.lin.weight.copy_(torch.tensor(np.array(conv.linear.weight)))
        pyg_conv.bias.copy_(torch.tensor(np.array(conv.bias)))
    pyg_layer = pyg.DeepGCNLayer(
        pyg_conv, torch.nn.LayerNorm(DIM), torch.nn.ReLU(), block=block
    )

    out = layer(mx.array(edge_index), mx.array(x))
    with torch.no_grad():
        expected = pyg_layer(torch.tensor(x), torch.tensor(edge_index))
    assert out.shape == ((NUM_NODES, 2 * DIM) if block == "dense" else (NUM_NODES, DIM))
    np.testing.assert_allclose(np.array(out), expected.numpy(), rtol=TOL, atol=TOL)


def test_deepgcn_layer_dropout_and_checkpoint(graph):
    edge_index, x, weights = graph
    ei, xx, w = mx.array(edge_index), mx.array(x), mx.array(weights)
    layer = DeepGCNLayer(GCNConv(DIM, DIM), nn.LayerNorm(DIM), nn.relu, dropout=0.5)
    layer.eval()
    out1 = layer(ei, xx, w)  # extra args are passed to the conv
    np.testing.assert_allclose(
        np.array(out1), np.array(layer(ei, xx, w)), rtol=1e-4, atol=1e-5
    )
    assert not np.allclose(np.array(out1), np.array(layer(ei, xx)))

    # gradients are the same with and without checkpointing
    def loss_fn(layer):
        return layer(ei, xx).sum()

    plain = DeepGCNLayer(GCNConv(DIM, DIM), nn.LayerNorm(DIM), nn.relu)
    ckpt = DeepGCNLayer(plain.conv, plain.norm, nn.relu, ckpt_grad=True)
    g1 = nn.utils.tree_flatten(nn.value_and_grad(plain, loss_fn)(plain)[1])
    g2 = nn.utils.tree_flatten(nn.value_and_grad(ckpt, loss_fn)(ckpt)[1])
    for (k1, a), (k2, b) in zip(g1, g2):
        assert k1 == k2
        np.testing.assert_allclose(np.array(a), np.array(b), rtol=1e-4, atol=1e-5)

    with pytest.raises(ValueError):
        DeepGCNLayer(GCNConv(DIM, DIM), block="nope")


# MaskLabel


@pytest.mark.parametrize("method", ["add", "concat"])
def test_mask_label_parity(graph, method):
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric.nn")
    _, x, _ = graph
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, NUM_NODES)
    mask = rng.random(NUM_NODES) < 0.5

    layer = MaskLabel(3, DIM, method=method)
    pyg_layer = pyg.MaskLabel(3, DIM, method=method)
    with torch.no_grad():
        pyg_layer.emb.weight.copy_(torch.tensor(np.array(layer.emb.weight)))

    expected = pyg_layer(torch.tensor(x), torch.tensor(y), torch.tensor(mask))
    for m in (mask, np.nonzero(mask)[0]):
        out = layer(mx.array(x), mx.array(y), mx.array(m))
        np.testing.assert_allclose(np.array(out), expected.detach().numpy(), rtol=1e-5)

    # labels of unmasked nodes are ignored, even if invalid
    y_invalid = np.where(mask, y, -1)
    out = layer(mx.array(x), mx.array(y_invalid), mx.array(mask))
    np.testing.assert_allclose(np.array(out), expected.detach().numpy(), rtol=1e-5)

    with pytest.raises(ValueError):
        MaskLabel(3, DIM, method="nope")


def test_mask_label_ratio_mask():
    mx.random.seed(0)
    mask = mx.array(np.arange(1000) % 2 == 0)
    out = MaskLabel.ratio_mask(mask, 0.3)
    assert not bool(mx.any(out & ~mask).item())
    kept = out.sum().item() / mask.sum().item()
    assert 0.2 < kept < 0.4


# RECT_L


@pytest.mark.parametrize("normalize", [True, False])
def test_rect_parity(graph, normalize):
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric.nn")
    edge_index, x, weights = graph
    model = RECT_L(DIM, 8, normalize=normalize)
    model.conv.bias = mx.random.normal((8,))
    pyg_model = pyg.models.RECT_L(DIM, 8, normalize=normalize)
    with torch.no_grad():
        pyg_model.conv.lin.weight.copy_(
            torch.tensor(np.array(model.conv.linear.weight))
        )
        pyg_model.conv.bias.copy_(torch.tensor(np.array(model.conv.bias)))
    _copy_linear(torch, pyg_model.lin, model.lin)

    for w in (None, weights):
        ei, tei = mx.array(edge_index), torch.tensor(edge_index)
        mw = None if w is None else mx.array(w)
        tw = None if w is None else torch.tensor(w)
        with torch.no_grad():
            np.testing.assert_allclose(
                np.array(model(ei, mx.array(x), mw)),
                pyg_model(torch.tensor(x), tei, tw).numpy(),
                rtol=TOL,
                atol=TOL,
            )
            np.testing.assert_allclose(
                np.array(model.embed(ei, mx.array(x), mw)),
                pyg_model.embed(torch.tensor(x), tei, tw).numpy(),
                rtol=TOL,
                atol=TOL,
            )

    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, NUM_NODES)
    mask = rng.random(NUM_NODES) < 0.6
    expected = pyg_model.get_semantic_labels(
        torch.tensor(x), torch.tensor(y), torch.tensor(mask)
    ).numpy()
    for m in (mask, np.nonzero(mask)[0]):
        out = model.get_semantic_labels(mx.array(x), mx.array(y), mx.array(m))
        np.testing.assert_allclose(np.array(out), expected, rtol=1e-5, atol=1e-6)


# DeepGraphInfomax


def _corruption(edge_index, node_features):
    perm = mx.random.permutation(node_features.shape[0])
    return edge_index, node_features[perm]


def _summary(z, *args, **kwargs):
    return mx.sigmoid(z.mean(axis=0))


def test_dgi_parity():
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric.nn")
    rng = np.random.default_rng(0)
    pos_z = rng.standard_normal((10, 8)).astype(np.float32)
    neg_z = rng.standard_normal((10, 8)).astype(np.float32)
    summary = rng.random(8).astype(np.float32)

    model = DeepGraphInfomax(8, nn.Identity(), _summary, _corruption)
    pyg_model = pyg.DeepGraphInfomax(8, torch.nn.Identity(), None, None)
    with torch.no_grad():
        pyg_model.weight.copy_(torch.tensor(np.array(model.weight)))
        np.testing.assert_allclose(
            np.array(model.discriminate(mx.array(pos_z), mx.array(summary))),
            pyg_model.discriminate(torch.tensor(pos_z), torch.tensor(summary)).numpy(),
            rtol=1e-4,
            atol=1e-5,
        )
        loss = model.loss(mx.array(pos_z), mx.array(neg_z), mx.array(summary))
        pyg_loss = pyg_model.loss(
            torch.tensor(pos_z), torch.tensor(neg_z), torch.tensor(summary)
        )
    assert abs(loss.item() - pyg_loss.item()) < 1e-4


def test_dgi_training_and_test():
    mx.random.seed(0)
    rng = np.random.default_rng(0)
    # two communities with different features
    labels = np.repeat([0, 1], 15)
    x = rng.standard_normal((30, 8)).astype(np.float32) + 2.0 * labels[:, None]
    src, dst = [], []
    for i in range(30):
        for j in range(30):
            if i != j and labels[i] == labels[j] and rng.random() < 0.3:
                src.append(i)
                dst.append(j)
    ei, xx = mx.array(np.array([src, dst])), mx.array(x)

    model = DeepGraphInfomax(
        16, GCN(8, 16, num_layers=1, act=None), _summary, _corruption
    )
    pos_z, neg_z, summary = model(ei, xx)
    assert pos_z.shape == neg_z.shape == (30, 16)
    assert summary.shape == (16,)

    # keyword arguments are corrupted too
    out = model(ei, node_features=xx)
    assert out[1].shape == (30, 16)

    def loss_fn(model):
        mx.random.seed(1)  # same corruption every time
        return model.loss(*model(ei, xx))

    first, last = _train(model, loss_fn, steps=30)
    assert last < first

    z = model.encoder(ei, xx)
    y = mx.array(labels)
    train = mx.array(np.r_[0:10, 15:25])
    test = mx.array(np.r_[10:15, 25:30])
    acc = model.test(z[train], y[train], z[test], y[test])
    assert acc >= 0.9
