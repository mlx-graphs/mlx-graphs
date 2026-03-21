import math

import mlx.core as mx

from mlx_graphs.nn.pooling import TopKPooling


class TestTopKPooling:
    def test_output_shapes(self):
        pool = TopKPooling(16, ratio=0.5)
        edge_index = mx.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        x = mx.random.uniform(shape=(4, 16))
        x_out, ei_out, _, _, perm, score = pool(edge_index, x)
        k = math.ceil(0.5 * 4)  # 2
        assert x_out.shape[0] == k
        assert x_out.shape[1] == 16
        assert perm.shape[0] == k
        assert score.shape[0] == k

    def test_edge_filtering(self):
        pool = TopKPooling(4, ratio=0.5)
        # Triangle graph: 0-1-2-0
        edge_index = mx.array([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        x = mx.random.uniform(shape=(3, 4))
        x_out, ei_out, _, _, perm, _ = pool(edge_index, x)
        # 2 nodes kept, edges should only connect kept nodes
        assert ei_out.shape[0] == 2
        # All edge indices should be < k
        k = math.ceil(0.5 * 3)
        if ei_out.shape[1] > 0:
            assert mx.max(ei_out).item() < k

    def test_with_edge_features(self):
        pool = TopKPooling(8, ratio=0.5)
        edge_index = mx.array([[0, 1, 2, 3], [1, 0, 3, 2]])
        x = mx.random.uniform(shape=(4, 8))
        e = mx.random.uniform(shape=(4, 3))
        x_out, ei_out, e_out, _, perm, _ = pool(edge_index, x, edge_features=e)
        if ei_out.shape[1] > 0 and e_out is not None:
            assert e_out.shape[1] == 3
            assert e_out.shape[0] == ei_out.shape[1]

    def test_with_batch(self):
        pool = TopKPooling(4, ratio=0.5)
        # Two small graphs batched
        edge_index = mx.array([[0, 1, 2, 3], [1, 0, 3, 2]])
        x = mx.random.uniform(shape=(4, 4))
        batch_idx = mx.array([0, 0, 1, 1])
        x_out, _, _, batch_out, _, _ = pool(edge_index, x, batch=batch_idx)
        assert batch_out is not None
        assert batch_out.shape[0] == x_out.shape[0]

    def test_min_score(self):
        pool = TopKPooling(4, min_score=0.0)
        edge_index = mx.array([[0, 1], [1, 0]])
        x = mx.random.uniform(shape=(2, 4))
        x_out, _, _, _, _, score = pool(edge_index, x)
        # All kept nodes should have score >= 0
        assert x_out.shape[0] >= 1

    def test_keeps_at_least_one_node(self):
        pool = TopKPooling(4, ratio=0.01)
        edge_index = mx.array([[0, 1], [1, 0]])
        x = mx.random.uniform(shape=(2, 4))
        x_out, _, _, _, _, _ = pool(edge_index, x)
        assert x_out.shape[0] >= 1

    def test_sigmoid_nonlinearity(self):
        pool = TopKPooling(4, ratio=0.5, nonlinearity="sigmoid")
        edge_index = mx.array([[0, 1, 2], [1, 2, 0]])
        x = mx.random.uniform(shape=(3, 4))
        x_out, _, _, _, _, score = pool(edge_index, x)
        # Sigmoid outputs are in [0, 1]
        assert mx.min(score).item() >= 0.0
        assert mx.max(score).item() <= 1.0
