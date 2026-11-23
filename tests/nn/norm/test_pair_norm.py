import mlx.core as mx
import numpy as np
import pytest
import torch
from torch_geometric.nn import PairNorm as PyGPairNorm

from mlx_graphs.nn import PairNormalization

TOL = dict(rtol=1e-4, atol=1e-4)

BATCH = np.array([0, 0, 1, 1, 1, 0, 2, 1], dtype=np.int64)


@pytest.mark.parametrize("scale_individually", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_pair_norm_matches_pyg(scale_individually, batched):
    rng = np.random.default_rng(0)
    x = rng.normal(loc=1.0, size=(len(BATCH), 4)).astype(np.float32)

    kwargs = dict(scale=2.5, scale_individually=scale_individually)
    batch = torch.from_numpy(BATCH) if batched else None
    expected = PyGPairNorm(**kwargs)(torch.from_numpy(x), batch).numpy()

    norm = PairNormalization(**kwargs)
    out = norm(mx.array(x), mx.array(BATCH) if batched else None)
    np.testing.assert_allclose(np.array(out), expected, **TOL)


def test_pair_norm_num_graphs():
    x = mx.random.normal((len(BATCH), 4))
    norm = PairNormalization()
    out = norm(x, mx.array(BATCH))
    padded = norm(x, mx.array(BATCH), num_graphs=4)
    np.testing.assert_allclose(np.array(padded), np.array(out), atol=1e-6)
    assert repr(norm) == "PairNormalization()"
