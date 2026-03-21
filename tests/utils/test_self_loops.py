"""
Tests for add_self_loops with only_existing_nodes parameter.

Verifies that when only_existing_nodes=True, self-loops are only added for
nodes actually present in the edge_index, skipping isolated nodes.
"""

import mlx.core as mx

from mlx_graphs.utils.transformations import add_self_loops

# ── Unit Tests ──────────────────────────────────────────────────────


def test_default_adds_all_self_loops():
    """Default behavior: self-loops for nodes 0..max(edge_index)."""
    edge_index = mx.array([[0, 2], [2, 0]])
    # Nodes 0, 1, 2 exist (0 through max=2), so 3 self-loops
    new_edge_index = add_self_loops(edge_index)
    # Original 2 edges + 3 self-loops = 5
    assert new_edge_index.shape[1] == 5


def test_only_existing_nodes():
    """only_existing_nodes=True: skip self-loops for absent nodes."""
    edge_index = mx.array([[0, 5], [5, 0]])
    # Only nodes 0 and 5 appear in edge_index

    new_edge_index = add_self_loops(edge_index, only_existing_nodes=True)
    # Original 2 edges + 2 self-loops (for nodes 0, 5) = 4
    assert new_edge_index.shape[1] == 4

    # Without the flag: would add self-loops for 0,1,2,3,4,5 = 6 self-loops
    new_edge_index_all = add_self_loops(edge_index, only_existing_nodes=False)
    assert new_edge_index_all.shape[1] == 8  # 2 + 6


def test_only_existing_no_unnecessary_loops():
    """Verify no self-loops are created for nodes NOT in edge_index."""
    edge_index = mx.array([[0, 100], [100, 0]])
    new_ei = add_self_loops(edge_index, only_existing_nodes=True)

    # Extract self-loop targets using mx.where (boolean indexing not supported)
    is_self_loop = mx.equal(new_ei[0], new_ei[1])
    # Count self-loops — should be exactly 2 (for nodes 0 and 100)
    num_self_loops = int(is_self_loop.sum().item())
    assert num_self_loops == 2
    # Total edges = 2 original + 2 self-loops
    assert new_ei.shape[1] == 4


# ── E2E Test ────────────────────────────────────────────────────────


def test_speedup_on_sparse_graph():
    """End-to-end: on a sparse graph with large node indices,
    only_existing_nodes dramatically reduces unnecessary self-loops."""
    # Graph with 3 edges among a few nodes, but max index is 999
    edge_index = mx.array([[0, 500, 999], [500, 999, 0]])

    result_all = add_self_loops(edge_index, only_existing_nodes=False)
    result_existing = add_self_loops(edge_index, only_existing_nodes=True)

    # Without: 3 edges + 1000 self-loops = 1003
    assert result_all.shape[1] == 1003
    # With: 3 edges + 3 self-loops = 6
    assert result_existing.shape[1] == 6

    # Verify the graph is still valid (all edges present)
    for i in range(3):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        # Find this edge in result_existing
        found = False
        for j in range(result_existing.shape[1]):
            s = result_existing[0, j].item()
            d = result_existing[1, j].item()
            if s == src and d == dst:
                found = True
                break
        assert found, f"Edge ({src}, {dst}) not found in result"
