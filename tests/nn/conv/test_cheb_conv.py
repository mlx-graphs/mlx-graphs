import mlx.core as mx

from mlx_graphs.nn.conv import ChebConv


class TestChebConv:
    def test_output_shape(self):
        conv = ChebConv(16, 32, K=3)
        edge_index = mx.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        x = mx.random.uniform(shape=(4, 16))
        out = conv(edge_index, x)
        assert out.shape == (4, 32)

    def test_k_equals_1(self):
        """K=1 means only T_0, which is just a linear transform."""
        conv = ChebConv(8, 4, K=1)
        edge_index = mx.array([[0, 1], [1, 0]])
        x = mx.random.uniform(shape=(2, 8))
        out = conv(edge_index, x)
        assert out.shape == (2, 4)

    def test_k_equals_2(self):
        conv = ChebConv(8, 4, K=2)
        edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
        x = mx.random.uniform(shape=(3, 8))
        out = conv(edge_index, x)
        assert out.shape == (3, 4)

    def test_no_bias(self):
        conv = ChebConv(8, 4, K=3, bias=False)
        assert "bias" not in conv
        edge_index = mx.array([[0, 1], [1, 0]])
        x = mx.random.uniform(shape=(2, 8))
        out = conv(edge_index, x)
        assert out.shape == (2, 4)

    def test_with_edge_weights(self):
        conv = ChebConv(8, 4, K=2)
        edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
        x = mx.random.uniform(shape=(3, 8))
        weights = mx.array([0.5, 1.0, 0.8])
        out = conv(edge_index, x, edge_weights=weights)
        assert out.shape == (3, 4)

    def test_no_normalization(self):
        conv = ChebConv(8, 4, K=2, normalization=None)
        edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
        x = mx.random.uniform(shape=(3, 8))
        out = conv(edge_index, x)
        assert out.shape == (3, 4)

    def test_single_node(self):
        conv = ChebConv(4, 2, K=2)
        edge_index = mx.array([[0], [0]])  # self-loop only
        x = mx.random.uniform(shape=(1, 4))
        out = conv(edge_index, x)
        assert out.shape == (1, 2)
