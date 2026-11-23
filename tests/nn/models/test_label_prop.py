import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.nn.models import CorrectAndSmooth, LabelPropagation

NUM_NODES, NUM_CLASSES = 15, 3


@pytest.fixture
def graph():
    rng = np.random.default_rng(0)
    edge_index = rng.integers(0, NUM_NODES, (2, 50))
    y = rng.integers(0, NUM_CLASSES, NUM_NODES)
    mask = rng.random(NUM_NODES) < 0.5
    y_soft = rng.random((NUM_NODES, NUM_CLASSES)).astype(np.float32)
    y_soft /= y_soft.sum(1, keepdims=True)
    weights = rng.random(50).astype(np.float32)
    return edge_index, y, mask, y_soft, weights


@pytest.mark.parametrize("use_mask", [False, True, "index"])
@pytest.mark.parametrize("use_weights", [False, True])
def test_label_propagation_parity(graph, use_mask, use_weights):
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric.nn")
    edge_index, y, mask, _, weights = graph

    mask_np = None
    if use_mask is True:
        mask_np = mask
    elif use_mask == "index":
        mask_np = np.nonzero(mask)[0]
    w = weights if use_weights else None

    out = LabelPropagation(num_layers=3, alpha=0.8)(
        mx.array(edge_index),
        mx.array(y),
        mask=None if mask_np is None else mx.array(mask_np),
        edge_weights=None if w is None else mx.array(w),
    )
    expected = pyg.LabelPropagation(num_layers=3, alpha=0.8)(
        torch.tensor(y),
        torch.tensor(edge_index),
        mask=None if mask_np is None else torch.tensor(mask_np),
        edge_weight=None if w is None else torch.tensor(w),
    )
    np.testing.assert_allclose(np.array(out), expected.numpy(), rtol=1e-4, atol=1e-5)


def test_label_propagation_soft_labels_and_post_step(graph):
    edge_index, _, _, y_soft, _ = graph
    lp = LabelPropagation(num_layers=2, alpha=0.5)
    out = lp(mx.array(edge_index), mx.array(y_soft), post_step=lambda x: x * 2)
    assert out.shape == (NUM_NODES, NUM_CLASSES)
    assert float(out.max().item()) > 1.0  # no clamping with a custom post step


@pytest.mark.parametrize("autoscale", [True, False])
@pytest.mark.parametrize("mask_kind", ["bool", "index"])
def test_correct_and_smooth_parity(graph, autoscale, mask_kind):
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric.nn")
    edge_index, y, mask, y_soft, _ = graph
    mask_np = mask if mask_kind == "bool" else np.nonzero(mask)[0]
    y_train = y[mask]

    kwargs = dict(
        num_correction_layers=3,
        correction_alpha=0.9,
        num_smoothing_layers=4,
        smoothing_alpha=0.7,
        autoscale=autoscale,
        scale=2.0,
    )
    cs = CorrectAndSmooth(**kwargs)
    pyg_cs = pyg.CorrectAndSmooth(**kwargs)

    ei, tei = mx.array(edge_index), torch.tensor(edge_index)
    corrected = cs.correct(ei, mx.array(y_soft), mx.array(y_train), mx.array(mask_np))
    pyg_corrected = pyg_cs.correct(
        torch.tensor(y_soft), torch.tensor(y_train), torch.tensor(mask_np), tei
    )
    np.testing.assert_allclose(
        np.array(corrected), pyg_corrected.numpy(), rtol=1e-4, atol=1e-5
    )

    smoothed = cs.smooth(ei, corrected, mx.array(y_train), mx.array(mask_np))
    pyg_smoothed = pyg_cs.smooth(
        pyg_corrected, torch.tensor(y_train), torch.tensor(mask_np), tei
    )
    np.testing.assert_allclose(
        np.array(smoothed), pyg_smoothed.numpy(), rtol=1e-4, atol=1e-5
    )

    both = cs(ei, mx.array(y_soft), mx.array(y_train), mx.array(mask_np))
    np.testing.assert_allclose(np.array(both), np.array(smoothed), rtol=1e-5)


def test_correct_and_smooth_wrong_size(graph):
    edge_index, y, mask, y_soft, _ = graph
    cs = CorrectAndSmooth(1, 0.5, 1, 0.5)
    with pytest.raises(ValueError):
        cs.correct(mx.array(edge_index), mx.array(y_soft), mx.array(y), mx.array(mask))
