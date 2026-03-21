"""
Tests for batch padding (batch(graphs, pad=True)).

Verifies that graphs are padded to uniform node/edge counts with boolean masks
(_node_mask and _edge_mask) to support static tensor shapes for mx.compile.
"""

import mlx.core as mx

from mlx_graphs.data.batch import batch
from mlx_graphs.data.data import GraphData

# ── Unit Tests ──────────────────────────────────────────────────────


def test_padded_batch_has_masks():
    """Padded batch should contain _node_mask and _edge_mask."""
    graphs = [
        GraphData(
            edge_index=mx.array([[0, 1], [1, 0]]),
            node_features=mx.ones((2, 4)),
        ),
        GraphData(
            edge_index=mx.array([[0, 1, 2], [1, 2, 0]]),
            node_features=mx.ones((3, 4)),
        ),
    ]

    padded = batch(graphs, pad=True)

    assert hasattr(padded, "_node_mask")
    assert hasattr(padded, "_edge_mask")


def test_padded_batch_uniform_sizes():
    """Each graph in the padded batch should have the same node/edge count."""
    g1 = GraphData(
        edge_index=mx.array([[0], [1]]),
        node_features=mx.ones((2, 4)),
    )
    g2 = GraphData(
        edge_index=mx.array([[0, 1, 2, 3], [1, 2, 3, 0]]),
        node_features=mx.ones((4, 4)),
    )

    padded = batch([g1, g2], pad=True)

    # Total nodes should be 2 * max(2,4) = 8
    assert padded.node_features is not None
    assert padded.node_features.shape[0] == 8
    # Total edges should be 2 * max(1,4) = 8
    assert padded.edge_index.shape[1] == 8


def test_node_mask_correctness():
    """Mask should be True for real nodes and False for padded ones."""
    g1 = GraphData(
        edge_index=mx.array([[0], [1]]),
        node_features=mx.ones((2, 4)),
    )
    g2 = GraphData(
        edge_index=mx.array([[0, 1, 2], [1, 2, 0]]),
        node_features=mx.ones((3, 4)),
    )

    padded = batch([g1, g2], pad=True)

    mask = getattr(padded, "_node_mask")
    # g1 has 2 real + 1 pad, g2 has 3 real + 0 pad → mask = [T,T,F,T,T,T]
    real_count = int(mask.sum().item())
    assert real_count == 5  # 2 + 3 real nodes


# ── E2E Test ────────────────────────────────────────────────────────


def test_padded_batch_compile_compatible():
    """End-to-end: padded batch should work with mx.compile (static shapes)."""
    graphs = [
        GraphData(
            edge_index=mx.array([[0, 1], [1, 0]]),
            node_features=mx.random.normal((3, 8)),
        )
        for _ in range(5)
    ]

    padded = batch(graphs, pad=True)

    # All per-graph sizes should be equal (already equal here)
    # Verify node_features has the expected total shape
    assert padded.node_features is not None
    assert padded.node_features.shape == (15, 8)

    # Verify we can run a simple compiled function on the padded batch
    import mlx.nn as nn

    @mx.compile
    def simple_gnn_step(x):
        return nn.relu(x @ mx.ones((8, 4)))

    out = simple_gnn_step(padded.node_features)
    mx.eval(out)
    assert out.shape == (15, 4)
