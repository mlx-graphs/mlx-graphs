"""
Tests for Dataloader batch_size=-1 support.

Verifies that batch_size=-1 loads the entire dataset as a single batch.
"""

import mlx.core as mx

from mlx_graphs.data.data import GraphData
from mlx_graphs.loaders import Dataloader

# ── Unit Tests ──────────────────────────────────────────────────────


def _make_graphs(n: int) -> list[GraphData]:
    return [
        GraphData(
            edge_index=mx.array([[0, 1], [1, 0]]),
            node_features=mx.ones((2, 4)) * i,
        )
        for i in range(n)
    ]


def test_batch_size_neg1_loads_all():
    """batch_size=-1 should yield a single batch containing all graphs."""
    graphs = _make_graphs(10)
    loader = Dataloader(graphs, batch_size=-1, shuffle=False)

    batches = list(loader)
    assert len(batches) == 1
    assert batches[0].num_graphs == 10


def test_batch_size_neg1_vs_explicit():
    """batch_size=-1 should be equivalent to batch_size=len(dataset)."""
    graphs = _make_graphs(7)

    loader_neg1 = Dataloader(graphs, batch_size=-1, shuffle=False)
    loader_full = Dataloader(graphs, batch_size=7, shuffle=False)

    batch_neg1 = next(iter(loader_neg1))
    batch_full = next(iter(loader_full))

    assert batch_neg1.num_graphs == batch_full.num_graphs
    assert mx.array_equal(batch_neg1.node_features, batch_full.node_features)


# ── E2E Test ────────────────────────────────────────────────────────


def test_full_batch_training_step():
    """End-to-end: load all graphs, run a forward pass, compute loss."""
    graphs = _make_graphs(20)
    loader = Dataloader(graphs, batch_size=-1, shuffle=False)

    full_batch = next(iter(loader))
    assert full_batch.num_graphs == 20

    # Simple "model": linear projection
    W = mx.ones((4, 2))
    out = full_batch.node_features @ W
    mx.eval(out)

    # Should have 20 graphs * 2 nodes each = 40 node embeddings
    assert out.shape == (40, 2)
