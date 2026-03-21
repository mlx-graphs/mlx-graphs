"""
Tests for GraphData attribute caching.

Verifies that computed properties like num_nodes are cached in GraphData._cache
and that the cache is automatically invalidated when attributes are modified.
"""

import mlx.core as mx

from mlx_graphs.data.data import GraphData

# ── Unit Tests ──────────────────────────────────────────────────────


def test_cache_exists():
    """GraphData should have a _cache attribute."""
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 0]]))
    assert hasattr(graph, "_cache")
    assert isinstance(getattr(graph, "_cache"), dict)


def test_num_nodes_is_cached():
    """Second access to num_nodes should hit the cache."""
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 0]]))
    _ = graph.num_nodes  # first access computes
    assert "num_nodes" in getattr(graph, "_cache")
    assert getattr(graph, "_cache")["num_nodes"] == 2


def test_cache_invalidated_on_attribute_change():
    """Modifying any public attribute must clear the cache."""
    graph = GraphData(
        edge_index=mx.array([[0, 1], [1, 0]]),
        node_features=mx.ones((2, 4)),
    )
    _ = graph.num_nodes  # populate cache
    assert "num_nodes" in getattr(graph, "_cache")

    # Change node_features → cache should be cleared
    graph.node_features = mx.ones((5, 4))
    assert len(getattr(graph, "_cache")) == 0

    # num_nodes should now reflect the new node_features
    assert graph.num_nodes == 5


def test_cache_not_invalidated_by_private_attrs():
    """Private attributes (starting with _) should not clear cache."""
    graph = GraphData(edge_index=mx.array([[0, 1], [1, 0]]))
    _ = graph.num_nodes
    assert "num_nodes" in getattr(graph, "_cache")

    graph._some_private = "test"
    assert "num_nodes" in getattr(graph, "_cache")  # cache still valid


# ── E2E Test ────────────────────────────────────────────────────────


def test_cache_correctness_through_mutations():
    """End-to-end: cache stays correct through multiple attribute changes."""
    graph = GraphData(edge_index=mx.array([[0], [99]]))
    assert graph.num_nodes == 100

    graph.node_features = mx.ones((50, 8))
    assert graph.num_nodes == 50  # now from node_features

    graph.node_features = mx.ones((200, 8))
    assert graph.num_nodes == 200  # updated

    graph.edge_index = mx.array([[0, 1], [1, 0]])
    assert graph.num_nodes == 200  # still from node_features
