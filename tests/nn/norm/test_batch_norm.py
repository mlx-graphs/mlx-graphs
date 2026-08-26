import mlx.core as mx
import pytest

from mlx_graphs.nn import BatchNormalization, HeteroBatchNormalization

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


def test_hetero_batch_norm():
    batch_norm = HeteroBatchNormalization(in_channels=8, num_types=2)
    batch_norm.train()

    node_features = mx.random.uniform(0, 1, [30, 8])
    type_vec = mx.concatenate(
        [mx.zeros(15, dtype=mx.int32), mx.ones(15, dtype=mx.int32)]
    )
    normalized = batch_norm(node_features, type_vec)

    assert normalized.shape == (30, 8)

    for group in (normalized[:15], normalized[15:]):
        mean = group.mean(axis=0)
        std = mx.sqrt(group.var(axis=0))
        assert mx.allclose(mean, mx.zeros(8), atol=1e-3)
        assert mx.allclose(std, mx.ones(8), atol=1e-3)


def test_hetero_batch_norm_single_element_type():
    batch_norm = HeteroBatchNormalization(in_channels=16, num_types=2)
    batch_norm.train()

    x = mx.concatenate(
        [
            mx.random.uniform(0, 1, [5, 16]),
            mx.random.uniform(16, 20, [1, 16]),
        ]
    )
    type_vec = mx.concatenate([mx.zeros(5, dtype=mx.int32), mx.ones(1, dtype=mx.int32)])

    out = batch_norm(x, type_vec)

    assert out.shape == (6, 16)
    # a lone item in a type has zero variance; eps should keep this finite
    # rather than dividing by zero
    assert mx.all(mx.isfinite(out[5]))
    assert mx.allclose(out[5], mx.zeros(16), atol=1e-2)
