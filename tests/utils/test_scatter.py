import mlx.core as mx

from mlx_graphs.utils import scatter


def test_scatter():
    src = mx.array([1.0, 1.0, 1.0, 1.0])
    index = mx.array([0, 0, 1, 2])
    num_nodes = src.shape[0]

    y_hat1 = scatter(src, index, aggr="softmax")

    num_nodes = index.max().item() + 1
    y_hat2 = scatter(src, index, num_nodes, aggr="add")
    y_hat3 = scatter(src, index, num_nodes, aggr="max")

    assert mx.all(y_hat1 == mx.array([0.5, 0.5, 1, 1])), "Simple scatter softmax failed"
    assert mx.all(y_hat2 == mx.array([2, 1, 1])), "Simple scatter add failed"
    assert mx.all(y_hat3 == mx.array([1, 1, 1])), "Simple scatter max failed"
