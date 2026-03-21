"""
Tests for GraphData.forward_dict().

Verifies that forward_dict() returns a dict of core attributes (edge_index,
node_features, edge_features) that can be unpacked directly into conv layer calls.
"""

import mlx.core as mx

from mlx_graphs.data.data import GraphData
from mlx_graphs.nn import GCNConv, SAGEConv

# ── Unit Tests ──────────────────────────────────────────────────────


def test_forward_dict_keys():
    """forward_dict should return edge_index and non-None features."""
    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 0]]),
        node_features=mx.ones((2, 8)),
    )
    fd = graph.forward_dict()
    assert "edge_index" in fd
    assert "node_features" in fd
    assert "edge_features" not in fd  # None → excluded


def test_forward_dict_includes_edge_features():
    """forward_dict includes edge_features when present."""
    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 0]]),
        node_features=mx.ones((2, 8)),
        edge_features=mx.ones((2, 4)),
    )
    fd = graph.forward_dict()
    assert "edge_features" in fd
    assert fd["edge_features"].shape == (2, 4)


def test_forward_dict_excludes_labels():
    """forward_dict should NOT include labels or graph features."""
    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 0]]),
        node_features=mx.ones((2, 8)),
        node_labels=mx.array([0, 1]),
        graph_labels=mx.array([1]),
    )
    fd = graph.forward_dict()
    assert "node_labels" not in fd
    assert "graph_labels" not in fd


# ── E2E Test ────────────────────────────────────────────────────────


def test_forward_dict_with_conv_layers():
    """End-to-end: use forward_dict() to run multiple conv layers."""
    graph = GraphData(
        edge_index=mx.array([[0, 1, 2, 3], [1, 0, 3, 2]]),
        node_features=mx.random.normal((4, 16)),
    )

    # GCNConv
    gcn = GCNConv(16, 32)
    out_gcn = gcn(**graph.forward_dict())
    mx.eval(out_gcn)
    assert out_gcn.shape == (4, 32)

    # SAGEConv
    sage = SAGEConv(16, 32)
    out_sage = sage(**graph.forward_dict())
    mx.eval(out_sage)
    assert out_sage.shape == (4, 32)
