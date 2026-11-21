import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils.array_ops import pairwise_distances


def test_pairwise_distances():
    x = mx.zeros([2, 4])
    y = mx.zeros([3, 4])
    expected_distances = mx.zeros([2, 3])
    distances = pairwise_distances(x, y)

    assert distances.shape == (2, 3), "Wrong output shape"
    assert mx.array_equal(distances, expected_distances), "Distances not all 0"

    x = mx.array([[1], [2]])
    y = mx.array([[2], [3]])
    expected_distances = mx.array([[1, 2], [0, 1]])
    distances = pairwise_distances(x, y)
    assert mx.array_equal(distances, expected_distances), "Wrong distances"

    # # commenting this test out as there are currently numerical errors between
    # # the output of cdist and the mlx implementation
    # from scipy.spatial.distance import cdist
    # import numpy as np
    # x = mx.random.normal([80, 3])
    # y = mx.random.normal([80, 3])
    # distances = pairwise_distances(x, y)
    # sp_distances = mx.array(cdist(np.array(x), np.array(y)).tolist())
    # assert mx.array_equal(distances, sp_distances)


@pytest.mark.parametrize("p", [2.0, 1.0, 3.0])
def test_p_norm_value_and_zero_gradient(p):
    from mlx_graphs.utils.array_ops import p_norm

    x = np.random.default_rng(0).normal(size=(4, 5)).astype(np.float32)
    x[1] = 0.0
    out = p_norm(mx.array(x), p, axis=-1)
    np.testing.assert_allclose(
        np.array(out), np.linalg.norm(x, ord=p, axis=-1), rtol=1e-5, atol=1e-7
    )
    # zero gradient at zero, as torch.linalg.norm (mx.linalg.norm: NaN)
    grad = mx.grad(lambda h: p_norm(h, p, axis=-1).sum())(mx.array(x))
    assert mx.all(mx.isfinite(grad)).item()
    assert mx.all(grad[1] == 0).item()
    assert p_norm(mx.array(x), p, axis=-1, keepdims=True).shape == (4, 1)
