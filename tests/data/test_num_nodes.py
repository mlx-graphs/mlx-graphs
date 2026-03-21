"""
Tests for GraphData.num_nodes computation.

Verifies that num_nodes correctly returns max(edge_index) + 1 when node_features
is not set, and node_features.shape[0] when it is. Ensures isolated nodes (not in
edge_index) are counted properly during batching.
"""

import mlx.core as mx

from mlx_graphs.data.batch import batch, unbatch
from mlx_graphs.data.data import GraphData

# ── Unit Tests ──────────────────────────────────────────────────────


def test_num_nodes_with_isolated_nodes():
    """Nodes 1, 3, 4 are isolated but should still be counted."""
    edge_index = mx.array([[0, 2, 5], [2, 5, 0]])
    graph = GraphData(edge_index=edge_index)
    # max node index is 5, so num_nodes should be 6
    assert graph.num_nodes == 6


def test_num_nodes_from_node_features():
    """When node_features is set, num_nodes comes from its shape."""
    edge_index = mx.array([[0, 1], [1, 0]])
    node_features = mx.ones((10, 4))  # 10 nodes, only 2 in edges
    graph = GraphData(edge_index=edge_index, node_features=node_features)
    assert graph.num_nodes == 10


def test_num_nodes_single_edge():
    """Single edge between nodes 0 and 99 → 100 nodes."""
    edge_index = mx.array([[0], [99]])
    graph = GraphData(edge_index=edge_index)
    assert graph.num_nodes == 100


# ── E2E Test ────────────────────────────────────────────────────────


def test_batch_unbatch_roundtrip_preserves_num_nodes():
    """End-to-end: batch then unbatch must preserve num_nodes for each graph."""
    graphs = [
        GraphData(
            edge_index=mx.array([[0, 2, 5], [2, 5, 0]]),
            node_features=mx.random.normal((6, 4)),
        ),
        GraphData(
            edge_index=mx.array([[0, 1], [1, 0]]),
            node_features=mx.random.normal((3, 4)),
        ),
        GraphData(
            edge_index=mx.array([[0], [9]]),
            node_features=mx.random.normal((10, 4)),
        ),
    ]

    batched = batch(graphs)
    restored = unbatch(batched)

    for orig, rest in zip(graphs, restored):
        assert orig.num_nodes == rest.num_nodes, (
            f"num_nodes mismatch: {orig.num_nodes} vs {rest.num_nodes}"
        )
        assert mx.array_equal(orig.node_features, rest.node_features)
