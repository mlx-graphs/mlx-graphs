import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from mlx_graphs.nn.models import LightGCN

NUM_NODES, DIM, NUM_LAYERS = 12, 8, 3


@pytest.fixture
def graph():
    rng = np.random.default_rng(0)
    edge_index = rng.integers(0, NUM_NODES, (2, 40))
    label_index = rng.integers(0, NUM_NODES, (2, 10))
    weights = rng.random(40).astype(np.float32)
    return edge_index, label_index, weights


def _pair(alpha=None):
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric.nn")
    model = LightGCN(NUM_NODES, DIM, NUM_LAYERS, alpha=alpha)
    pyg_alpha = None if alpha is None else torch.tensor(alpha, dtype=torch.float32)
    pyg_model = pyg.LightGCN(NUM_NODES, DIM, NUM_LAYERS, alpha=pyg_alpha)
    with torch.no_grad():
        pyg_model.embedding.weight.copy_(torch.tensor(np.array(model.embedding.weight)))
    return torch, model, pyg_model


@pytest.mark.parametrize("alpha", [None, [0.1, 0.2, 0.3, 0.4]])
@pytest.mark.parametrize("use_weights", [False, True])
def test_parity_with_pyg(graph, alpha, use_weights):
    torch, model, pyg_model = _pair(alpha)
    edge_index, label_index, weights = graph
    w = mx.array(weights) if use_weights else None
    tw = torch.tensor(weights) if use_weights else None
    ei, tei = mx.array(edge_index), torch.tensor(edge_index)
    li, tli = mx.array(label_index), torch.tensor(label_index)

    with torch.no_grad():
        np.testing.assert_allclose(
            np.array(model.get_embedding(ei, w)),
            pyg_model.get_embedding(tei, tw).numpy(),
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(model(ei, li, w)),
            pyg_model(tei, tli, tw).numpy(),
            rtol=1e-4,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(model.predict_link(ei, li, w, prob=True)),
            pyg_model.predict_link(tei, tli, tw, prob=True).numpy(),
            rtol=1e-4,
            atol=1e-5,
        )

        pos, neg = model(ei, li, w), model(ei, li[::-1], w)
        node_id = mx.array([0, 3, 5])
        loss = model.recommendation_loss(pos, neg, node_id=node_id, lambda_reg=0.1)
        pyg_pos, pyg_neg = pyg_model(tei, tli, tw), pyg_model(tei, tli.flip(0), tw)
        pyg_loss = pyg_model.recommendation_loss(
            pyg_pos, pyg_neg, node_id=torch.tensor([0, 3, 5]), lambda_reg=0.1
        )
        assert abs(loss.item() - pyg_loss.item()) < 1e-4

        label = mx.array([1, 0] * 5)
        lp_loss = model.link_pred_loss(pos, label)
        pyg_lp_loss = pyg_model.link_pred_loss(pyg_pos, torch.tensor([1, 0] * 5))
        assert abs(lp_loss.item() - pyg_lp_loss.item()) < 1e-4


def test_recommend(graph):
    torch, model, pyg_model = _pair()
    edge_index, _, _ = graph
    ei, tei = mx.array(edge_index), torch.tensor(edge_index)
    src, dst = [0, 1, 2], [3, 5, 7, 9, 11]
    with torch.no_grad():
        top = model.recommend(ei, src_index=mx.array(src), dst_index=mx.array(dst), k=3)
        pyg_top = pyg_model.recommend(
            tei, src_index=torch.tensor(src), dst_index=torch.tensor(dst), k=3
        )
        np.testing.assert_array_equal(np.array(top), pyg_top.numpy())
        top_all = model.recommend(ei, k=2)
        assert top_all.shape == (NUM_NODES, 2)
        unsorted = model.recommend(ei, k=2, sorted=False)
        np.testing.assert_array_equal(
            np.sort(np.array(unsorted), 1), np.sort(np.array(top_all), 1)
        )


def test_alpha_is_not_trained():
    model = LightGCN(NUM_NODES, DIM, NUM_LAYERS)
    names = [k for k, _ in nn.utils.tree_flatten(model.trainable_parameters())]
    assert names == ["embedding.weight"]
    with pytest.raises(ValueError):
        LightGCN(NUM_NODES, DIM, NUM_LAYERS, alpha=[0.5, 0.5])


def test_training_reduces_loss(graph):
    mx.random.seed(0)
    edge_index, _, _ = graph
    ei = mx.array(edge_index)
    model = LightGCN(NUM_NODES, DIM, 2)
    optimizer = optim.Adam(learning_rate=0.05)
    pos = ei
    neg = mx.array(np.random.default_rng(1).integers(0, NUM_NODES, (2, 40)))

    def loss_fn(model):
        rank = model(ei, mx.concatenate([pos, neg], axis=1))
        pos_rank, neg_rank = mx.split(rank, 2)
        return model.recommendation_loss(pos_rank, neg_rank)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    first = loss_fn(model).item()
    for _ in range(20):
        _, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    assert loss_fn(model).item() < first
