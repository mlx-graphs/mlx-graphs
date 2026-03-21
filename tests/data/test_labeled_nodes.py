"""
Tests for from_networkx() with non-integer node labels.

Verifies that networkx graphs with string or other non-integer node labels
are correctly remapped to contiguous integer indices during conversion.
"""

import mlx.core as mx
import networkx as nx

from mlx_graphs.utils.convert import from_networkx, to_networkx

# ── Unit Tests ──────────────────────────────────────────────────────


def test_string_labeled_nodes():
    """Graphs with string node labels should convert without error."""
    G = nx.Graph()
    G.add_node("alice", features=[1.0, 0.0])
    G.add_node("bob", features=[0.0, 1.0])
    G.add_edge("alice", "bob")

    graph = from_networkx(G)

    assert graph.num_nodes == 2
    assert graph.num_edges == 1
    assert graph.node_features is not None
    assert graph.node_features.shape == (2, 2)


def test_integer_labeled_nodes_still_work():
    """Standard integer-labeled graphs should still work correctly."""
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    graph = from_networkx(G)

    assert graph.num_nodes == 3
    assert graph.num_edges == 2


def test_non_contiguous_integer_labels():
    """Graphs with non-contiguous integer labels (e.g. 10, 20, 30)."""
    G = nx.Graph()
    G.add_node(10, features=[1.0])
    G.add_node(20, features=[2.0])
    G.add_node(30, features=[3.0])
    G.add_edge(10, 20)
    G.add_edge(20, 30)

    graph = from_networkx(G)

    assert graph.num_nodes == 3
    assert graph.edge_index.shape == (2, 2)
    # All indices should be in [0, 3)
    assert int(graph.edge_index.max().item()) < 3
    assert int(graph.edge_index.min().item()) >= 0


def test_empty_graph():
    """Graph with nodes but no edges."""
    G = nx.Graph()
    G.add_node("x")
    G.add_node("y")

    graph = from_networkx(G)

    assert graph.edge_index.shape == (2, 0)


def test_feature_order_preserved():
    """Node features should match the order of nodes in the networkx graph."""
    G = nx.Graph()
    G.add_node("first", features=[1.0, 0.0, 0.0])
    G.add_node("second", features=[0.0, 1.0, 0.0])
    G.add_node("third", features=[0.0, 0.0, 1.0])
    G.add_edge("first", "second")
    G.add_edge("second", "third")

    graph = from_networkx(G)

    # Features should be in order: first→[1,0,0], second→[0,1,0], third→[0,0,1]
    assert graph.node_features is not None
    assert mx.allclose(graph.node_features[0], mx.array([1.0, 0.0, 0.0]))
    assert mx.allclose(graph.node_features[1], mx.array([0.0, 1.0, 0.0]))
    assert mx.allclose(graph.node_features[2], mx.array([0.0, 0.0, 1.0]))


# ── E2E Test ────────────────────────────────────────────────────────


def test_roundtrip_string_labels():
    """End-to-end: create a string-labeled nx graph, convert to GraphData,
    convert back to nx, verify structure is preserved."""
    G = nx.DiGraph()
    G.add_node("alice", features=[1.0, 2.0])
    G.add_node("bob", features=[3.0, 4.0])
    G.add_node("carol", features=[5.0, 6.0])
    G.add_edge("alice", "bob")
    G.add_edge("bob", "carol")
    G.add_edge("carol", "alice")

    # nx → GraphData
    graph = from_networkx(G)
    assert graph.num_nodes == 3
    assert graph.num_edges == 3

    # GraphData → nx (roundtrip)
    G2 = to_networkx(graph)
    assert G2.number_of_nodes() == 3
    assert G2.number_of_edges() == 3

    # Verify features survived roundtrip
    for node in G2.nodes():
        assert "features" in G2.nodes[node]
