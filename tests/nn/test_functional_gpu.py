"""
Functional GPU tests for mlx-graphs conv layers, pooling, and batching.

Exercises the Metal GPU with real graph data at scale. Tests forward passes,
shape correctness, numerical sanity, and batched operations.

Note: nn.value_and_grad is broken on macOS Tahoe + MLX (see ml-explore/mlx#3256).
These tests use forward-only passes which fully exercise the GPU compute path.
"""

import math

import mlx.core as mx
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.data.batch import batch
from mlx_graphs.nn.conv import ChebConv, GCNConv, TransformerConv
from mlx_graphs.nn.pooling import TopKPooling


@pytest.fixture
def large_graph():
    """A 2000-node random graph for GPU stress testing."""
    num_nodes, num_edges, feat_dim = 2000, 10000, 64
    edge_index = mx.stack(
        [
            mx.random.randint(0, num_nodes, shape=(num_edges,)),
            mx.random.randint(0, num_nodes, shape=(num_edges,)),
        ]
    )
    node_features = mx.random.normal(shape=(num_nodes, feat_dim))
    mx.eval(edge_index, node_features)
    return edge_index, node_features, num_nodes, feat_dim


class TestGCNConvGPU:
    def test_forward_shape(self, large_graph):
        edge_index, x, num_nodes, feat_dim = large_graph
        conv = GCNConv(feat_dim, 32)
        mx.eval(conv.parameters())
        out = conv(edge_index, x)
        mx.eval(out)
        assert out.shape == (num_nodes, 32)

    def test_output_finite(self, large_graph):
        edge_index, x, _, feat_dim = large_graph
        conv = GCNConv(feat_dim, 32)
        mx.eval(conv.parameters())
        out = conv(edge_index, x)
        mx.eval(out)
        assert mx.isfinite(out).all().item()


class TestChebConvGPU:
    def test_forward_shape(self, large_graph):
        edge_index, x, num_nodes, feat_dim = large_graph
        conv = ChebConv(feat_dim, 32, K=3)
        mx.eval(conv.parameters())
        out = conv(edge_index, x)
        mx.eval(out)
        assert out.shape == (num_nodes, 32)

    def test_different_k_values(self, large_graph):
        edge_index, x, num_nodes, feat_dim = large_graph
        for k in [1, 2, 3, 5]:
            conv = ChebConv(feat_dim, 16, K=k)
            mx.eval(conv.parameters())
            out = conv(edge_index, x)
            mx.eval(out)
            assert out.shape == (num_nodes, 16), f"Failed for K={k}"

    def test_output_finite(self, large_graph):
        edge_index, x, _, feat_dim = large_graph
        conv = ChebConv(feat_dim, 32, K=3)
        mx.eval(conv.parameters())
        out = conv(edge_index, x)
        mx.eval(out)
        assert mx.isfinite(out).all().item()


class TestTransformerConvGPU:
    def test_forward_concat(self, large_graph):
        edge_index, x, num_nodes, feat_dim = large_graph
        conv = TransformerConv(feat_dim, 8, heads=4, concat=True)
        mx.eval(conv.parameters())
        out = conv(edge_index, x)
        mx.eval(out)
        assert out.shape == (num_nodes, 32)  # 8 * 4 heads

    def test_forward_mean(self, large_graph):
        edge_index, x, num_nodes, feat_dim = large_graph
        conv = TransformerConv(feat_dim, 16, heads=4, concat=False)
        mx.eval(conv.parameters())
        out = conv(edge_index, x)
        mx.eval(out)
        assert out.shape == (num_nodes, 16)

    def test_output_finite(self, large_graph):
        edge_index, x, _, feat_dim = large_graph
        conv = TransformerConv(feat_dim, 8, heads=4, concat=True)
        mx.eval(conv.parameters())
        out = conv(edge_index, x)
        mx.eval(out)
        assert mx.isfinite(out).all().item()


class TestTopKPoolingGPU:
    def test_pooling_ratio(self, large_graph):
        edge_index, x, num_nodes, feat_dim = large_graph
        gcn = GCNConv(feat_dim, 32)
        pool = TopKPooling(32, ratio=0.5)
        mx.eval(gcn.parameters(), pool.parameters())
        h = gcn(edge_index, x)
        mx.eval(h)
        h_pooled, ei_pooled, _, _, perm, score = pool(edge_index, h)
        mx.eval(h_pooled, ei_pooled)
        expected_k = math.ceil(0.5 * num_nodes)
        assert h_pooled.shape == (expected_k, 32)
        assert perm.shape[0] == expected_k
        assert score.shape[0] == expected_k

    def test_edge_reindexing(self, large_graph):
        edge_index, x, num_nodes, feat_dim = large_graph
        gcn = GCNConv(feat_dim, 16)
        pool = TopKPooling(16, ratio=0.3)
        mx.eval(gcn.parameters(), pool.parameters())
        h = gcn(edge_index, x)
        mx.eval(h)
        _, ei_out, _, _, _, _ = pool(edge_index, h)
        mx.eval(ei_out)
        k = math.ceil(0.3 * num_nodes)
        if ei_out.shape[1] > 0:
            assert mx.max(ei_out).item() < k


class TestBatchPadGPU:
    def test_batch_200_graphs(self):
        graphs = []
        for _ in range(200):
            n = int(mx.random.randint(5, 30, shape=()).item())
            e = int(mx.random.randint(10, 60, shape=()).item())
            g = GraphData(
                edge_index=mx.stack(
                    [
                        mx.random.randint(0, n, shape=(e,)),
                        mx.random.randint(0, n, shape=(e,)),
                    ]
                ),
                node_features=mx.random.normal(shape=(n, 16)),
            )
            graphs.append(g)
        batched = batch(graphs, pad=True)
        mx.eval(batched.edge_index, batched.node_features)
        assert batched.node_features is not None
        assert batched.node_features.shape[1] == 16
        assert batched.node_features.shape[0] > 0

    def test_batched_conv_forward(self):
        """Run a conv layer on a batched graph."""
        graphs = [
            GraphData(
                edge_index=mx.stack(
                    [
                        mx.random.randint(0, 10, shape=(20,)),
                        mx.random.randint(0, 10, shape=(20,)),
                    ]
                ),
                node_features=mx.random.normal(shape=(10, 32)),
            )
            for _ in range(50)
        ]
        batched = batch(graphs)
        mx.eval(batched.edge_index, batched.node_features)
        assert batched.edge_index is not None
        assert batched.node_features is not None

        conv = GCNConv(32, 16)
        mx.eval(conv.parameters())
        out = conv(batched.edge_index, batched.node_features)
        mx.eval(out)
        assert out.shape[1] == 16


class TestGPUStress:
    def test_repeated_forward_passes(self, large_graph):
        """50 iterations of all 3 conv layers on a 2K-node graph."""
        edge_index, x, _, feat_dim = large_graph
        gcn = GCNConv(feat_dim, 32)
        cheb = ChebConv(feat_dim, 32, K=3)
        trans = TransformerConv(feat_dim, 8, heads=4, concat=True)
        mx.eval(gcn.parameters(), cheb.parameters(), trans.parameters())

        for _ in range(50):
            o1 = gcn(edge_index, x)
            o2 = cheb(edge_index, x)
            o3 = trans(edge_index, x)
            mx.eval(o1, o2, o3)

    def test_large_graph_5k_nodes(self):
        """Forward pass on a 5000-node graph."""
        ei = mx.stack(
            [
                mx.random.randint(0, 5000, shape=(50000,)),
                mx.random.randint(0, 5000, shape=(50000,)),
            ]
        )
        x = mx.random.normal(shape=(5000, 128))
        mx.eval(ei, x)

        gcn = GCNConv(128, 64)
        cheb = ChebConv(128, 64, K=3)
        mx.eval(gcn.parameters(), cheb.parameters())

        for _ in range(10):
            o1 = gcn(ei, x)
            o2 = cheb(ei, x)
            mx.eval(o1, o2)
