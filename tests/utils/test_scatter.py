import mlx.core as mx
import pytest

from mlx_graphs.utils import scatter_softmax


def test_scatter_softmax():
    src = mx.array([1., 1., 1., 1.])
    index = mx.array([0, 0, 1, 2])
    num_nodes = src.shape[0]
    
    y_hat = scatter_softmax(src, index, num_nodes)
    y = mx.array([0.5, 0.5, 1, 1])

    assert mx.all(y_hat == y), "Simple scatter Softmax failed"
