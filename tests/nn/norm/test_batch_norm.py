import mlx.core as mx
import pytest

from mlx_graphs.nn import BatchNormalization

mx.random.seed(42)


def test_batch_norm():
    batch_norm = BatchNormalization(num_features=8)
    batch_norm.train()
    node_features = mx.random.uniform(0, 1, [6, 8])
    normalized = batch_norm(node_features)
    assert normalized.shape == (6, 8)
    mean = normalized.mean(axis=0)
    normalized_std = mx.sqrt(normalized.var(axis=0))
    assert mx.allclose(mean, mx.zeros(8), atol=1e-3)
    assert mx.allclose(
        normalized_std,
        mx.ones(8),
        atol=1e-3,
    )


def test_batch_norm_single_element():
    x = mx.random.uniform(16, 20, [1, 16])

    with pytest.raises(ValueError, match="requires 'track_running_stats'"):
        norm = BatchNormalization(
            16, track_running_stats=False, allow_single_element=True
        )

    norm = BatchNormalization(16, track_running_stats=True, allow_single_element=True)
    out = norm(x)
    assert mx.allclose(out, x, atol=1e-3)
