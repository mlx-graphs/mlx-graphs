import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from mlx_graphs.nn.models import (
    ComplEx,
    DistMult,
    KGTripletLoader,
    RotatE,
    TransE,
)

NUM_NODES, NUM_RELATIONS, DIM = 20, 4, 8

MODELS = {
    "TransE": (TransE, {"margin": 0.5, "p_norm": 1.0}),
    "TransE-p2": (TransE, {"margin": 0.5, "p_norm": 2.0}),
    "DistMult": (DistMult, {"margin": 0.5}),
    "ComplEx": (ComplEx, {}),
    "RotatE": (RotatE, {"margin": 2.0}),
}
EMBEDDINGS = ["node_emb", "rel_emb", "node_emb_im", "rel_emb_im"]


@pytest.fixture
def triplets():
    rng = np.random.default_rng(0)
    head = rng.integers(0, NUM_NODES, 30)
    rel = rng.integers(0, NUM_RELATIONS, 30)
    tail = rng.integers(0, NUM_NODES, 30)
    return head, rel, tail


def _pair(name):
    torch = pytest.importorskip("torch")
    kge = pytest.importorskip("torch_geometric.nn.kge")
    cls, kwargs = MODELS[name]
    model = cls(NUM_NODES, NUM_RELATIONS, DIM, **kwargs)
    pyg_model = getattr(kge, cls.__name__)(NUM_NODES, NUM_RELATIONS, DIM, **kwargs)
    with torch.no_grad():
        for emb in EMBEDDINGS:
            if emb in model:
                getattr(pyg_model, emb).weight.copy_(
                    torch.tensor(np.array(getattr(model, emb).weight))
                )
    return torch, model, pyg_model


@pytest.mark.parametrize("name", MODELS)
def test_score_and_loss_parity(triplets, name):
    torch, model, pyg_model = _pair(name)
    head, rel, tail = triplets
    h, r, t = mx.array(head), mx.array(rel), mx.array(tail)
    th, tr, tt = torch.tensor(head), torch.tensor(rel), torch.tensor(tail)
    with torch.no_grad():
        np.testing.assert_allclose(
            np.array(model(h, r, t)),
            pyg_model(th, tr, tt).numpy(),
            rtol=1e-4,
            atol=1e-5,
        )

        # same negatives on both sides
        neg_h, neg_r, neg_t = model.random_sample(h, r, t)
        model.random_sample = lambda *args: (neg_h, neg_r, neg_t)
        pyg_model.random_sample = lambda *args: (
            torch.tensor(np.array(neg_h)),
            torch.tensor(np.array(neg_r)),
            torch.tensor(np.array(neg_t)),
        )
        loss = model.loss(h, r, t)
        pyg_loss = pyg_model.loss(th, tr, tt)
        assert abs(loss.item() - pyg_loss.item()) < 1e-4


@pytest.mark.parametrize("name", MODELS)
def test_ranking_metrics_parity(triplets, name):
    torch, model, pyg_model = _pair(name)
    head, rel, tail = triplets
    # small batch_size: several chunks of triplets
    out = model.test(
        mx.array(head), mx.array(rel), mx.array(tail), batch_size=50, k=5, log=False
    )
    expected = pyg_model.test(
        torch.tensor(head),
        torch.tensor(rel),
        torch.tensor(tail),
        batch_size=7,
        k=5,
        log=False,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_ranking_ties_follow_a_stable_argsort(triplets):
    """Tied scores: the rank is the position in a stable descending argsort
    (PyG's argsort position, with ties ordered by index)."""
    head, rel, tail = triplets
    model = DistMult(NUM_NODES, NUM_RELATIONS, DIM)
    model.node_emb.weight = mx.zeros_like(model.node_emb.weight)  # all scores 0
    mean_rank, mrr, hits = model.test(
        mx.array(head), mx.array(rel), mx.array(tail), batch_size=50, k=5, log=False
    )
    np.testing.assert_allclose(mean_rank, tail.mean(), rtol=1e-6)
    np.testing.assert_allclose(mrr, (1.0 / (tail + 1)).mean(), rtol=1e-6)
    np.testing.assert_allclose(hits, (tail < 5).mean(), rtol=1e-6)


def test_random_sample(triplets):
    head, rel, tail = (mx.array(a) for a in triplets)
    model = DistMult(NUM_NODES, NUM_RELATIONS, DIM)
    neg_h, neg_r, neg_t = model.random_sample(head, rel, tail)
    half = head.size // 2
    np.testing.assert_array_equal(np.array(neg_t[:half]), np.array(tail[:half]))
    np.testing.assert_array_equal(np.array(neg_h[half:]), np.array(head[half:]))
    np.testing.assert_array_equal(np.array(neg_r), np.array(rel))
    assert int(neg_h.max().item()) < NUM_NODES


def test_loader(triplets):
    head, rel, tail = (mx.array(a) for a in triplets)
    model = TransE(NUM_NODES, NUM_RELATIONS, DIM)
    loader = model.loader(head, rel, tail, batch_size=8)
    assert isinstance(loader, KGTripletLoader)
    batches = list(loader)
    assert len(batches) == len(loader) == 4
    np.testing.assert_array_equal(
        np.concatenate([np.array(b[0]) for b in batches]), np.array(head)
    )
    assert len(list(model.loader(head, rel, tail, batch_size=8, drop_last=True))) == 3

    mx.random.seed(0)
    shuffled = list(model.loader(head, rel, tail, batch_size=8, shuffle=True))
    h = np.concatenate([np.array(b[0]) for b in shuffled])
    t = np.concatenate([np.array(b[2]) for b in shuffled])
    assert sorted(zip(h, t)) == sorted(zip(np.array(head), np.array(tail)))


@pytest.mark.parametrize("name", MODELS)
def test_training_reduces_loss(triplets, name):
    mx.random.seed(0)
    cls, kwargs = MODELS[name]
    model = cls(NUM_NODES, NUM_RELATIONS, DIM, **kwargs)
    head, rel, tail = (mx.array(a) for a in triplets)
    optimizer = optim.Adam(learning_rate=0.01)
    loss_and_grad = nn.value_and_grad(model, lambda m, h, r, t: m.loss(h, r, t))

    def eval_loss():
        mx.random.seed(1)  # same negatives every time
        return model.loss(head, rel, tail).item()

    first = eval_loss()
    for _ in range(20):
        for h, r, t in model.loader(head, rel, tail, batch_size=10, shuffle=True):
            _, grads = loss_and_grad(model, h, r, t)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
    assert eval_loss() < first
