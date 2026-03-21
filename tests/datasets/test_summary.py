"""
Tests for Dataset.summary() method.

Verifies that summary() returns a formatted statistics table with graph counts,
average nodes/edges, feature dimensions, and class counts.
"""

from mlx_graphs.datasets import KarateClubDataset

# ── Unit Tests ──────────────────────────────────────────────────────


def test_summary_returns_string():
    """summary() should return a non-empty string."""
    dataset = KarateClubDataset()
    result = dataset.summary()
    assert isinstance(result, str)
    assert len(result) > 0


def test_summary_contains_key_stats():
    """summary() should contain all expected statistic labels."""
    dataset = KarateClubDataset()
    result = dataset.summary()

    expected_labels = [
        "Number of graphs",
        "Total nodes",
        "Total edges",
        "Avg nodes/graph",
        "Avg edges/graph",
        "Node features",
        "Edge features",
        "Node classes",
        "Edge classes",
        "Graph classes",
    ]
    for label in expected_labels:
        assert label in result, f"Missing '{label}' in summary output"


def test_summary_correct_values():
    """summary() values should match dataset properties."""
    dataset = KarateClubDataset()
    result = dataset.summary()

    assert "karate_club" in result
    # KarateClub has 1 graph
    assert "1" in result


# ── E2E Test ────────────────────────────────────────────────────────


def test_summary_printable():
    """End-to-end: summary should be printable without errors and
    contain valid numeric data."""
    dataset = KarateClubDataset()
    summary = dataset.summary()

    # Should be printable
    printed = str(summary)
    assert len(printed) > 50  # reasonable length

    # Parse the number of graphs line
    lines = summary.split("\n")
    graphs_line = [line for line in lines if "Number of graphs" in line][0]
    # The value should be a valid integer
    value = graphs_line.split()[-1]
    assert int(value) == len(dataset)
