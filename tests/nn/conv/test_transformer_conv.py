import mlx.core as mx

from mlx_graphs.nn.conv import TransformerConv


class TestTransformerConv:
    def test_output_shape_concat(self):
        conv = TransformerConv(16, 32, heads=2, concat=True)
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        x = mx.random.uniform(shape=(5, 16))
        out = conv(edge_index, x)
        assert out.shape == (5, 64)  # 2 heads * 32

    def test_output_shape_mean(self):
        conv = TransformerConv(16, 32, heads=2, concat=False)
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        x = mx.random.uniform(shape=(5, 16))
        out = conv(edge_index, x)
        assert out.shape == (5, 32)

    def test_single_head(self):
        conv = TransformerConv(8, 4, heads=1)
        edge_index = mx.array([[0, 1], [1, 0]])
        x = mx.random.uniform(shape=(2, 8))
        out = conv(edge_index, x)
        assert out.shape == (2, 4)

    def test_with_edge_features(self):
        conv = TransformerConv(16, 8, heads=2, edge_dim=4)
        edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
        x = mx.random.uniform(shape=(3, 16))
        e = mx.random.uniform(shape=(3, 4))
        out = conv(edge_index, x, edge_features=e)
        assert out.shape == (3, 16)  # 2 * 8

    def test_with_beta(self):
        conv = TransformerConv(8, 4, heads=1, beta=True)
        edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
        x = mx.random.uniform(shape=(3, 8))
        out = conv(edge_index, x)
        assert out.shape == (3, 4)

    def test_no_root_weight(self):
        conv = TransformerConv(8, 4, heads=1, root_weight=False)
        edge_index = mx.array([[0, 1], [1, 0]])
        x = mx.random.uniform(shape=(2, 8))
        out = conv(edge_index, x)
        assert out.shape == (2, 4)

    def test_no_bias(self):
        conv = TransformerConv(8, 4, heads=1, bias=False)
        assert "bias" not in conv
        edge_index = mx.array([[0, 1], [1, 0]])
        x = mx.random.uniform(shape=(2, 8))
        out = conv(edge_index, x)
        assert out.shape == (2, 4)

    def test_with_dropout(self):
        conv = TransformerConv(8, 4, heads=2, dropout=0.5)
        edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
        x = mx.random.uniform(shape=(3, 8))
        # Eval mode (no dropout applied)
        conv.eval()
        out = conv(edge_index, x)
        assert out.shape == (3, 8)
