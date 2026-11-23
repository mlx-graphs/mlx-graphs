import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten
from torch_geometric.nn import JumpingKnowledge as PyGJumpingKnowledge

from mlx_graphs.nn.pooling.jumping_knowledge import JumpingKnowledge

# MLX's GPU matmul is only accurate to ~1e-3 relative on some machines, while a
# wrong formula gives O(1) errors with these O(1) inputs.
TOL = dict(rtol=2e-2, atol=2e-2)

NUM_NODES, DIM = 10, 6


def _layer_outputs(rng, num_layers):
    return [
        rng.normal(size=(NUM_NODES, DIM)).astype(np.float32) for _ in range(num_layers)
    ]


def _copy_lstm(mlx_lstm, torch_lstm, suffix):
    """PyTorch and MLX LSTMs use the same gate order (i, f, g, o). PyTorch has
    two biases (input and hidden), MLX a single one."""
    params = dict(torch_lstm.named_parameters())
    mlx_lstm.Wx = mx.array(params[f"weight_ih_l0{suffix}"].detach().numpy())
    mlx_lstm.Wh = mx.array(params[f"weight_hh_l0{suffix}"].detach().numpy())
    mlx_lstm.bias = mx.array(
        (params[f"bias_ih_l0{suffix}"] + params[f"bias_hh_l0{suffix}"]).detach().numpy()
    )


def _copy_weights(jk, pyg_jk):
    _copy_lstm(jk.lstm_forward, pyg_jk.lstm, "")
    _copy_lstm(jk.lstm_backward, pyg_jk.lstm, "_reverse")
    jk.att.weight = mx.array(pyg_jk.att.weight.detach().numpy())
    jk.att.bias = mx.array(pyg_jk.att.bias.detach().numpy())


def test_jumping_knowledge_shapes():
    rng = np.random.default_rng(0)
    xs = [mx.array(x) for x in _layer_outputs(rng, 4)]

    assert JumpingKnowledge("cat")(xs).shape == (NUM_NODES, 4 * DIM)
    assert JumpingKnowledge("max")(xs).shape == (NUM_NODES, DIM)
    jk = JumpingKnowledge("lstm", node_features_dim=DIM, num_layers=4)
    assert jk(xs).shape == (NUM_NODES, DIM)

    # "cat" also accepts layers of different sizes
    xs_mixed = [mx.zeros((NUM_NODES, 3)), mx.zeros((NUM_NODES, 5))]
    assert JumpingKnowledge("cat")(xs_mixed).shape == (NUM_NODES, 8)


def test_jumping_knowledge_parameters():
    assert tree_flatten(JumpingKnowledge("cat").parameters()) == []
    assert tree_flatten(JumpingKnowledge("max").parameters()) == []
    assert JumpingKnowledge("MAX").mode == "max"

    # an odd `num_layers * node_features_dim` rounds the hidden size down
    jk = JumpingKnowledge("lstm", node_features_dim=5, num_layers=3)
    assert jk.lstm_forward.Wx.shape == (4 * 7, 5)
    assert jk.lstm_backward.Wh.shape == (4 * 7, 7)
    assert jk.att.weight.shape == (1, 14)

    with pytest.raises(ValueError):
        JumpingKnowledge("mean")
    with pytest.raises(ValueError):
        JumpingKnowledge("lstm")
    with pytest.raises(ValueError):
        JumpingKnowledge("lstm", node_features_dim=DIM)


@pytest.mark.parametrize("mode", ["cat", "max"])
def test_jumping_knowledge_cat_max_match_pyg(mode):
    rng = np.random.default_rng(1)
    xs = _layer_outputs(rng, 3)

    expected = PyGJumpingKnowledge(mode)([torch.from_numpy(x) for x in xs]).numpy()
    out = JumpingKnowledge(mode)([mx.array(x) for x in xs])

    assert out.shape == expected.shape
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("num_layers", [1, 2, 3, 5])
def test_jumping_knowledge_lstm_matches_pyg(num_layers):
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    xs = _layer_outputs(rng, num_layers)

    pyg_jk = PyGJumpingKnowledge("lstm", channels=DIM, num_layers=num_layers)
    with torch.no_grad():
        # default LSTM weights are small, which gives near-uniform attention:
        # scale them up so that a wrong gate or direction mapping is visible
        for param in pyg_jk.parameters():
            param.mul_(4.0)
        expected = pyg_jk([torch.from_numpy(x) for x in xs]).numpy()

    jk = JumpingKnowledge("lstm", node_features_dim=DIM, num_layers=num_layers)
    _copy_weights(jk, pyg_jk)
    out = jk([mx.array(x) for x in xs])

    assert out.shape == expected.shape
    np.testing.assert_allclose(np.array(out), expected, **TOL)

    if num_layers > 1:
        # sanity check: the attention is far from uniform, so the comparison
        # is sensitive to the LSTM weights
        uniform = np.mean(xs, axis=0)
        assert np.abs(expected - uniform).max() > 0.2


def test_jumping_knowledge_lstm_gradients():
    mx.random.seed(0)
    rng = np.random.default_rng(3)
    xs = [mx.array(x) for x in _layer_outputs(rng, 3)]
    jk = JumpingKnowledge("lstm", node_features_dim=DIM, num_layers=3)

    def loss_fn():
        return (jk(xs) ** 2).sum()

    _, grads = nn.value_and_grad(jk, loss_fn)()
    params = dict(tree_flatten(jk.trainable_parameters()))
    grads = dict(tree_flatten(grads))

    assert set(grads) == set(params)
    assert set(params) == {
        "lstm_forward.Wx",
        "lstm_forward.Wh",
        "lstm_forward.bias",
        "lstm_backward.Wx",
        "lstm_backward.Wh",
        "lstm_backward.bias",
        "att.weight",
        "att.bias",
    }
    for name, grad in grads.items():
        assert grad.shape == params[name].shape, name
        assert mx.all(mx.isfinite(grad)).item(), f"non-finite gradient for {name}"
        # the attention is a softmax over layers, so it does not depend on the
        # bias of `att`: its gradient is zero
        if name != "att.bias":
            assert mx.any(grad != 0).item(), f"zero gradient for {name}"
