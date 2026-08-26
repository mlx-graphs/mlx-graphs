"""
Compute and display statistics for mlx-graphs datasets.

This script generates useful statistics about any dataset in mlx-graphs,
including number of graphs, nodes, edges, features, and labels.

Usage:
    python compute_dataset_stats.py                    # Stats for all datasets
    python compute_dataset_stats.py --dataset karate_club  # Stats for one dataset
    python compute_dataset_stats.py --format markdown  # Output as markdown table
    python compute_dataset_stats.py --format rst       # Output as RST table

Designed for issue #158: Dataset statistics in the docs
https://github.com/mlx-graphs/mlx-graphs/issues/158
"""

import argparse
import sys

import mlx.core as mx

from mlx_graphs.data import GraphData


def compute_graph_stats(data: GraphData) -> dict:
    """Compute statistics for a single graph."""
    stats = {}

    # Number of nodes
    stats["num_nodes"] = data.num_nodes

    # Number of edges
    stats["num_edges"] = data.num_edges

    # Average node degree
    if data.num_nodes > 0:
        stats["avg_node_degree"] = round(2.0 * data.num_edges / data.num_nodes, 2)
    else:
        stats["avg_node_degree"] = 0.0

    # Features
    feature_types = []
    if data.node_features is not None:
        feature_types.append("node")
    if data.edge_features is not None:
        feature_types.append("edge")
    if data.graph_features is not None:
        feature_types.append("graph")
    stats["features"] = ", ".join(feature_types) if feature_types else "none"

    # Feature dimensions
    if data.node_features is not None:
        stats["node_feature_dim"] = data.node_features.shape[-1]
    if data.edge_features is not None:
        stats["edge_feature_dim"] = data.edge_features.shape[-1]

    # Labels
    label_types = []
    if data.node_labels is not None:
        label_types.append("node")
    if data.edge_labels is not None:
        label_types.append("edge")
    if data.graph_labels is not None:
        label_types.append("graph")
    stats["labels"] = ", ".join(label_types) if label_types else "none"

    # Number of classes
    for label_attr in ["node_labels", "edge_labels", "graph_labels"]:
        labels = getattr(data, label_attr, None)
        if labels is not None:
            num_classes = int(mx.max(labels).item()) + 1
            stats[f"num_{label_attr.replace('_labels', '')}_classes"] = num_classes

    return stats


def compute_dataset_stats(
    dataset, dataset_name: str, num_samples: int = 1
) -> list[dict]:
    """Compute statistics for all graphs in a dataset."""
    all_stats = []
    num_graphs = len(dataset)

    for i in range(min(num_samples, num_graphs)):
        data = dataset[i]
        stats = compute_graph_stats(data)
        stats["dataset"] = dataset_name
        stats["graph_index"] = i
        all_stats.append(stats)

    # Aggregate stats
    if all_stats:
        agg = {
            "dataset": dataset_name,
            "num_graphs": num_graphs,
            "num_nodes": (
                int(mx.mean(mx.array([s["num_nodes"] for s in all_stats])).item())
            ),
            "num_edges": (
                int(mx.mean(mx.array([s["num_edges"] for s in all_stats])).item())
            ),
            "avg_node_degree": round(
                float(mx.mean(mx.array([s["avg_node_degree"] for s in all_stats]))), 2
            ),
            "features": all_stats[0]["features"],
            "labels": all_stats[0]["labels"],
        }
        # Add class counts from first sample
        for key in all_stats[0]:
            if key.startswith("num_") and key.endswith("_classes"):
                agg[key] = all_stats[0][key]
        return [agg]
    return []


def format_markdown_table(stats_list: list[dict]) -> str:
    """Format statistics as a markdown table."""
    columns = [
        "dataset",
        "num_graphs",
        "num_nodes",
        "num_edges",
        "avg_node_degree",
        "features",
        "labels",
    ]
    # Add class columns if present
    for key in stats_list[0]:
        if key.endswith("_classes") and key not in columns:
            columns.append(key)

    header = "| Dataset | Graphs | Nodes | Edges | Avg Degree | Features | Labels |"
    separator = "|" + "|".join(["------"] * (len(columns) - 1)) + "|"

    rows = []
    for stats in stats_list:
        name = stats["dataset"].replace("_", " ").title()
        vals = [
            name,
            str(stats.get("num_graphs", 1)),
            str(stats.get("num_nodes", "-")),
            str(stats.get("num_edges", "-")),
            str(stats.get("avg_node_degree", "-")),
            stats.get("features", "-"),
            stats.get("labels", "-"),
        ]
        # Add class counts
        for key in columns[7:]:
            vals.append(str(stats.get(key, "-")))
        rows.append("| " + " | ".join(vals) + " |")

    return "\n".join([header, separator] + rows)


def format_rst_table(stats_list: list[dict]) -> str:
    """Format statistics as an RST table."""
    columns = [
        "Dataset",
        "Graphs",
        "Nodes",
        "Edges",
        "Avg Degree",
        "Features",
        "Labels",
    ]

    # Build rows
    rows = []
    for stats in stats_list:
        name = stats["dataset"].replace("_", " ").title()
        rows.append(
            [
                name,
                str(stats.get("num_graphs", 1)),
                str(stats.get("num_nodes", "-")),
                str(stats.get("num_edges", "-")),
                str(stats.get("avg_node_degree", "-")),
                stats.get("features", "-"),
                stats.get("labels", "-"),
            ]
        )

    # Calculate column widths
    widths = [len(c) for c in columns]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def make_row(cells, sep="|"):
        parts = [f" {c.ljust(w)} " for c, w in zip(cells, widths)]
        return sep.join([""] + parts + [""])

    def make_separator():
        parts = [f"{'=' * (w + 2)}" for w in widths]
        return " ".join(parts)

    lines = [make_separator(), make_row(columns, sep=" "), make_separator()]
    for row in rows:
        lines.append(make_row(row, sep=" "))
    lines.append(make_separator())

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics for mlx-graphs"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset name (e.g., karate_club). Default: all available.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "rst"],
        default="text",
        help="Output format",
    )
    args = parser.parse_args()

    # Import datasets lazily to avoid unnecessary downloads
    from mlx_graphs.datasets import (
        EllipticBitcoinDataset,
        KarateClubDataset,
        MovieLens100K,
        PlanetoidDataset,
        QM7bDataset,
    )

    # Dataset registry
    dataset_registry = {
        "karate_club": ("KarateClubDataset", KarateClubDataset, {}),
        "qm7b": ("QM7bDataset", QM7bDataset, {}),
        "cora": ("PlanetoidDataset", PlanetoidDataset, {"name": "cora"}),
        "citeseer": ("PlanetoidDataset", PlanetoidDataset, {"name": "citeseer"}),
        "pubmed": ("PlanetoidDataset", PlanetoidDataset, {"name": "pubmed"}),
        "elliptic": ("EllipticBitcoinDataset", EllipticBitcoinDataset, {}),
        "movie_lens_100k": ("MovieLens100K", MovieLens100K, {}),
    }

    if args.dataset:
        if args.dataset not in dataset_registry:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {', '.join(dataset_registry.keys())}")
            sys.exit(1)
        targets = {args.dataset: dataset_registry[args.dataset]}
    else:
        targets = dataset_registry

    all_stats = []
    for name, (cls_name, cls, kwargs) in targets.items():
        try:
            print(f"Loading {name}...", file=sys.stderr)
            ds = cls(**kwargs)
            stats = compute_dataset_stats(ds, name)
            all_stats.extend(stats)
            print(
                f"  Done: {stats[0]['num_nodes']} nodes, "
                f"{stats[0]['num_edges']} edges",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"  Error loading {name}: {e}", file=sys.stderr)

    if not all_stats:
        print("No dataset statistics computed.")
        sys.exit(1)

    if args.format == "markdown":
        print(format_markdown_table(all_stats))
    elif args.format == "rst":
        print(format_rst_table(all_stats))
    else:
        for stats in all_stats:
            name = stats["dataset"].replace("_", " ").title()
            print(f"\n{name}")
            print(f"  Graphs:       {stats.get('num_graphs', 1)}")
            print(f"  Nodes:        {stats.get('num_nodes', '-')}")
            print(f"  Edges:        {stats.get('num_edges', '-')}")
            print(f"  Avg Degree:   {stats.get('avg_node_degree', '-')}")
            print(f"  Features:     {stats.get('features', '-')}")
            print(f"  Labels:       {stats.get('labels', '-')}")


if __name__ == "__main__":
    main()
