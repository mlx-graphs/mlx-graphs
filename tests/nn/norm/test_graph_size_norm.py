import mlx.core as mx
import numpy as np
import pytest
import torch
from torch_geometric.nn import GraphSizeNorm as PyGGraphSizeNorm

from mlx_graphs.nn import GraphSizeNormalization

BATCH = np.array([0, 0, 1, 1, 1, 0, 2, 1, 0], dtype=np.int64)


@pytest.mark.parametrize("batched", [False, True])
def test_graph_size_norm_matches_pyg(batched):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(len(BATCH), 3)).astype(np.float32)

    batch = torch.from_numpy(BATCH) if batched else None
    expected = PyGGraphSizeNorm()(torch.from_numpy(x), batch).numpy()

    out = GraphSizeNormalization()(mx.array(x), mx.array(BATCH) if batched else None)
    np.testing.assert_allclose(np.array(out), expected, rtol=1e-5, atol=1e-6)


def test_graph_size_norm_values():
    x = mx.ones((5, 2))
    out = GraphSizeNormalization()(x, mx.array([0, 0, 0, 0, 1]), num_graphs=3)
    np.testing.assert_allclose(np.array(out[:, 0]), [0.5, 0.5, 0.5, 0.5, 1.0])
