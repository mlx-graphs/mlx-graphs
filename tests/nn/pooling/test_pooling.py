import mlx.core as mx

from mlx_graphs.nn import global_add_pool, global_max_pool, global_mean_pool


def test_global_add_pooling():
    node_features = mx.array([[1, 2], [3, 4], [5, 6]])
    batch = mx.array([0, 0, 1])

    # Add pooling
    y_hat1 = global_add_pool(node_features)
    y_hat2 = global_add_pool(node_features, batch)

    assert y_hat1.shape == [1, 2], "Simple global_add_pool failed"
    assert mx.array_equal(
        y_hat1, mx.array([[1 + 3 + 5, 2 + 4 + 6]])
    ), "Simple global_add_pool failed"

    assert y_hat2.shape == [2, 2], "global_add_pool with batch failed"
    assert mx.array_equal(
        y_hat2, mx.array([[1 + 3, 2 + 4], [5, 6]])
    ), "global_add_pool with batch failed"


def test_global_max_pooling():
    node_features = mx.array([[1, 2], [3, 4], [5, 6]])
    batch = mx.array([0, 0, 1])

    y_hat1 = global_max_pool(node_features)
    y_hat2 = global_max_pool(node_features, batch)

    assert y_hat1.shape == [1, 2], "Simple global_max_pool failed"
    assert mx.array_equal(
        y_hat1, mx.array([[node_features[:, 0].max(), node_features[:, 1].max()]])
    ), "Simple global_max_pool failed"

    assert y_hat2.shape == [2, 2], "global_max_pool with batch failed"
    assert mx.array_equal(
        y_hat2, mx.array([[3, 4], [5, 6]])
    ), "global_max_pool with batch failed"


def test_global_mean_pooling():
    node_features = mx.array([[1, 2], [3, 4], [5, 6]])
    batch = mx.array([0, 0, 1])

    y_hat1 = global_mean_pool(node_features)
    y_hat2 = global_mean_pool(node_features, batch)

    assert y_hat1.shape == [1, 2], "Simple global_mean_pool failed"
    assert mx.array_equal(
        y_hat1, mx.array([[(1 + 3 + 5) / 3, (2 + 4 + 6) / 3]])
    ), "Simple global_mean_pool failed"

    assert y_hat2.shape == [2, 2], "global_mean_pool with batch failed"
    assert mx.array_equal(
        y_hat2, mx.array([[(1 + 3) / 2, (2 + 4) / 2], [5, 6]])
    ), "global_mean_pool with batch failed"
