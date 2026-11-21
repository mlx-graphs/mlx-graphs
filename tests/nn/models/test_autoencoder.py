import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from mlx_graphs.nn import GCN, GCNConv
from mlx_graphs.nn.models import ARGA, ARGVA, GAE, VGAE, InnerProductDecoder
from mlx_graphs.nn.models.autoencoder import average_precision_score, roc_auc_score

NUM_NODES, DIM = 12, 6


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((NUM_NODES, DIM)).astype(np.float32)
    pos = rng.integers(0, NUM_NODES, (2, 20))
    neg = rng.integers(0, NUM_NODES, (2, 15))
    return z, pos, neg


def test_inner_product_decoder(data):
    z, pos, _ = data
    decoder = InnerProductDecoder()
    out = decoder(mx.array(pos), mx.array(z))
    expected = 1 / (1 + np.exp(-(z[pos[0]] * z[pos[1]]).sum(1)))
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-5)
    logits = decoder(mx.array(pos), mx.array(z), sigmoid=False)
    np.testing.assert_allclose(np.array(logits), (z[pos[0]] * z[pos[1]]).sum(1), 1e-5)
    adj = decoder.forward_all(mx.array(z), sigmoid=False)
    # GPU matmuls run in reduced precision
    np.testing.assert_allclose(np.array(adj), z @ z.T, rtol=1e-2, atol=1e-2)


def test_parity_with_pyg(data):
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric.nn")
    z, pos, neg = data

    pyg_gae = pyg.GAE(torch.nn.Identity())
    gae = GAE(nn.Identity())
    tz, tpos, tneg = torch.tensor(z), torch.tensor(pos), torch.tensor(neg)
    np.testing.assert_allclose(
        np.array(gae.decoder.forward_all(mx.array(z))),
        pyg_gae.decoder.forward_all(tz).numpy(),
        rtol=1e-2,
    )
    loss = gae.recon_loss(mx.array(z), mx.array(pos), mx.array(neg))
    expected = pyg_gae.recon_loss(tz, tpos, tneg)
    assert abs(loss.item() - expected.item()) < 1e-5

    auc, ap = gae.test(mx.array(z), mx.array(pos), mx.array(neg))
    pyg_auc, pyg_ap = pyg_gae.test(tz, tpos, tneg)
    assert abs(auc - pyg_auc) < 1e-6
    assert abs(ap - pyg_ap) < 1e-6

    mu, logstd = z, np.random.default_rng(1).standard_normal(z.shape) * 0.5
    vgae = VGAE(nn.Identity())
    pyg_vgae = pyg.VGAE(torch.nn.Identity())
    kl = vgae.kl_loss(mx.array(mu), mx.array(logstd.astype(np.float32)))
    pyg_kl = pyg_vgae.kl_loss(tz, torch.tensor(logstd, dtype=torch.float32))
    assert abs(kl.item() - pyg_kl.item()) < 1e-4


def test_metrics_match_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = rng.integers(0, 2, 50)
        y[:2] = [0, 1]
        # rounded scores to create ties
        scores = np.round(rng.random(50), 1)
        assert abs(roc_auc_score(y, scores) - metrics.roc_auc_score(y, scores)) < 1e-9
        assert (
            abs(
                average_precision_score(y, scores)
                - metrics.average_precision_score(y, scores)
            )
            < 1e-9
        )
    assert roc_auc_score([0, 1, 1], [0.1, 0.5, 0.9]) == 1.0


def _graph():
    rng = np.random.default_rng(0)
    # two dense clusters
    src, dst = [], []
    for offset in (0, 10):
        for i in range(10):
            for j in range(10):
                if i != j and rng.random() < 0.6:
                    src.append(offset + i)
                    dst.append(offset + j)
    edge_index = mx.array(np.array([src, dst]))
    x = mx.array(rng.standard_normal((20, 8)).astype(np.float32))
    return edge_index, x


def test_gae_training_reduces_loss():
    mx.random.seed(0)
    edge_index, x = _graph()
    model = GAE(GCN(8, 16, num_layers=2, out_features_dim=8))
    optimizer = optim.Adam(learning_rate=0.01)
    from mlx_graphs.utils import negative_sampling

    neg = negative_sampling(edge_index, 20)

    def loss_fn(model):
        z = model.encode(edge_index, x)
        return model.recon_loss(z, edge_index, neg)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    first = loss_fn(model).item()
    for _ in range(30):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    assert loss_fn(model).item() < first
    # negative sampling when no negatives are given
    assert np.isfinite(model.recon_loss(model.encode(edge_index, x), edge_index).item())
    auc, ap = model.test(model.encode(edge_index, x), edge_index, neg)
    assert 0.5 < auc <= 1.0 and 0.5 < ap <= 1.0


class VEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_mu = GCNConv(in_dim, out_dim)
        self.conv_logstd = GCNConv(in_dim, out_dim)

    def __call__(self, edge_index, x):
        return self.conv_mu(edge_index, x), self.conv_logstd(edge_index, x)


@pytest.mark.parametrize("cls", [VGAE, ARGVA])
def test_variational_training(cls):
    mx.random.seed(0)
    edge_index, x = _graph()
    if cls is VGAE:
        model = VGAE(VEncoder(8, 4))
    else:
        model = ARGVA(VEncoder(8, 4), discriminator=nn.Linear(4, 1))
    optimizer = optim.Adam(learning_rate=0.01)

    def loss_fn(model):
        z = model.encode(edge_index, x)
        loss = model.recon_loss(z, edge_index) + model.kl_loss() / 20
        if cls is ARGVA:
            loss = loss + model.reg_loss(z)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for _ in range(5):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        assert np.isfinite(loss.item())

    model.train()
    z1, z2 = model.encode(edge_index, x), model.encode(edge_index, x)
    assert not np.allclose(np.array(z1), np.array(z2))  # sampled
    model.eval()
    z1, z2 = model.encode(edge_index, x), model.encode(edge_index, x)
    np.testing.assert_allclose(np.array(z1), np.array(z2), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.array(z1), np.array(model._mu), rtol=1e-5)


def test_arga_losses():
    mx.random.seed(0)
    edge_index, x = _graph()
    model = ARGA(GCN(8, 8, num_layers=1), discriminator=nn.Linear(8, 1))
    z = model.encode(edge_index, x)
    assert model.reg_loss(z).shape == ()
    assert model.discriminator_loss(z).shape == ()

    # the discriminator loss does not backpropagate into the encoder
    def loss_fn(model):
        return model.discriminator_loss(model.encode(edge_index, x))

    _, grads = nn.value_and_grad(model, loss_fn)(model)
    enc = nn.utils.tree_flatten(grads["encoder"])
    assert all(float(mx.abs(g).sum().item()) == 0.0 for _, g in enc)
    disc = nn.utils.tree_flatten(grads["discriminator"])
    assert any(float(mx.abs(g).sum().item()) > 0.0 for _, g in disc)


def test_metrics_are_the_ones_of_mlx_graphs_metrics():
    """No duplicated implementation; the old import path still works."""
    from mlx_graphs import metrics

    assert roc_auc_score is metrics.roc_auc_score
    assert average_precision_score is metrics.average_precision_score
