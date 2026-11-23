import mlx.core as mx
import numpy as np
import pytest
import torch
from torch_geometric.nn import MeanSubtractionNorm as PyGMeanSubtractionNorm

from mlx_graphs.nn import MeanSubtractionNormalization

BATCH = np.array([0, 0, 1, 1, 1, 0, 2, 1, 0], dtype=np.int64)


@pytest.mark.parametrize("batched", [False, True])
def test_mean_subtraction_norm_matches_pyg(batched):
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, size=(len(BATCH), 3)).astype(np.float32)

    batch = torch.from_numpy(BATCH) if batched else None
    expected = PyGMeanSubtractionNorm()(torch.from_numpy(x), batch).numpy()

    norm = MeanSubtractionNormalization()
    out = norm(mx.array(x), mx.array(BATCH) if batched else None)
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-5, atol=1e-5)

    if batched:
        padded = norm(mx.array(x), mx.array(BATCH), num_graphs=4)
        np.testing.assert_allclose(np.array(padded), expected, rtol=1e-5, atol=1e-5)
