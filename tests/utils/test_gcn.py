import mlx.core as mx
import pytest

from mlx_graphs.nn.conv.gcn_conv import GCNConv


def test_inv_degree():
    deg = mx.array([1, 2, 3, 0])
    expected = mx.array([1, 0.707107, 0.57735, 0])
    out = GCNConv._deg_inv_sqrt(deg)
    assert mx.array_equal(out, expected), "Incorrect degree inversion"