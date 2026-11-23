"""
Compute and display statistics for mlx-graphs datasets.

This script generates useful statistics about any dataset in mlx-graphs,
including number of graphs, nodes, edges, features, and labels.

Usage:
    python compute_dataset_stats.py                    # Stats for all datasets
    python compute_dataset_stats.py --dataset karate_club  # Stats for one dataset
    python compute_dataset_stats.py --format markdown  # Output as markdown table
    python compute_dataset_stats.py --format rst       # Output as RST table
    python compute_dataset_stats.py --base-dir DIR     # Datasets stored in DIR

Heterogeneous graphs count the nodes and edges of all types, temporal graphs
count each event as an edge, and datasets of several graphs give the averages
per graph ("~"). docs/datasets.yml holds the numbers shown in the docs.

Designed for issue #158: Dataset statistics in the docs
https://github.com/mlx-graphs/mlx-graphs/issues/158
"""

import argparse
import inspect
import sys
from typing import Optional

import mlx.core as mx

from mlx_graphs.data import GraphData, HeteroGraphData, TemporalData


def _present(*values) -> bool:
    return any(v is not None and (not isinstance(v, dict) or v) for v in values)


def compute_hetero_graph_stats(data: HeteroGraphData) -> dict:
    """Statistics of a heterogeneous graph: totals over all node and edge types."""
    num_nodes = sum(data.num_nodes.values())
    num_edges = sum(data.num_edges.values())
    kinds = {
        "features": [
            ("node", data.node_features_dict),
            ("edge", data.edge_features_dict),
            ("graph", data.graph_features),
        ],
        "labels": [
            ("node", data.node_labels_dict),
            ("edge", data.edge_labels_dict),
            ("graph", data.graph_labels),
        ],
    }
    stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_node_degree": round(2.0 * num_edges / num_nodes, 2) if num_nodes else 0.0,
        "node_types": len(data.node_types),
        "edge_types": len(data.edge_types),
    }
    for key, items in kinds.items():
        found = [kind for kind, value in items if _present(value)]
        stats[key] = ", ".join(found) if found else "none"
    return stats


def compute_temporal_graph_stats(data: TemporalData) -> dict:
    """Statistics of a temporal graph: each event is an edge."""
    num_nodes, num_edges = data.num_nodes, data.num_events
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_node_degree": round(2.0 * num_edges / num_nodes, 2) if num_nodes else 0.0,
        "features": "edge" if "msg" in data else "none",
        "labels": "edge" if "y" in data else "none",
    }


def compute_graph_stats(data: GraphData) -> dict:
    """Compute statistics for a single graph."""
    if isinstance(data, HeteroGraphData):
        return compute_hetero_graph_stats(data)
    if isinstance(data, TemporalData):
        return compute_temporal_graph_stats(data)
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
    dataset, dataset_name: str, num_samples: Optional[int] = None
) -> list[dict]:
    """Compute statistics for the graphs of a dataset (all of them by default).

    For a dataset of several graphs, nodes and edges are averages per graph,
    prefixed with "~", and the average degree is 2 * edges / nodes over all
    the graphs.
    """
    all_stats = []
    num_graphs = len(dataset)
    if num_samples is None:
        num_samples = num_graphs

    for i in range(min(num_samples, num_graphs)):
        data = dataset[i]
        stats = compute_graph_stats(data)
        stats["dataset"] = dataset_name
        stats["graph_index"] = i
        all_stats.append(stats)

    # Aggregate stats
    if all_stats:
        total_nodes = sum(s["num_nodes"] for s in all_stats)
        total_edges = sum(s["num_edges"] for s in all_stats)
        if num_graphs == 1:
            nodes, edges = f"{total_nodes:,}", f"{total_edges:,}"
            degree = f"{2.0 * total_edges / total_nodes:.2f}"
        else:
            n = len(all_stats)
            nodes = f"~{total_nodes / n:,.1f}"
            edges = f"~{total_edges / n:,.1f}"
            degree = f"~{2.0 * total_edges / total_nodes:.2f}"
        agg = {
            "dataset": dataset_name,
            "num_graphs": f"{num_graphs:,}",
            "num_nodes": nodes,
            "num_edges": edges,
            "avg_node_degree": degree,
            "features": all_stats[0]["features"],
            "labels": all_stats[0]["labels"],
        }
        for key in ("node_types", "edge_types"):
            if key in all_stats[0]:
                agg[key] = all_stats[0][key]
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
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Directory of the downloaded datasets. Default: ./.mlx_graphs_data",
    )
    args = parser.parse_args()

    # Import datasets lazily to avoid unnecessary downloads
    from mlx_graphs.datasets import (
        AmazonDataset,
        CoauthorDataset,
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
        "amazon_computers": ("AmazonDataset", AmazonDataset, {"name": "computers"}),
        "amazon_photo": ("AmazonDataset", AmazonDataset, {"name": "photo"}),
        "coauthor_cs": ("CoauthorDataset", CoauthorDataset, {"name": "cs"}),
        "coauthor_physics": ("CoauthorDataset", CoauthorDataset, {"name": "physics"}),
        "elliptic": ("EllipticBitcoinDataset", EllipticBitcoinDataset, {}),
        "movie_lens_100k": ("MovieLens100K", MovieLens100K, {}),
    }

    # Citation, heterophily, Wikipedia and knowledge graph datasets
    from mlx_graphs.datasets import (
        Actor,
        CitationFull,
        FB15k_237,
        HeterophilousGraphDataset,
        WebKB,
        WikiCS,
        WikipediaNetwork,
        WordNet18RR,
    )

    for name in ["cora", "cora_ml", "citeseer", "dblp", "pubmed"]:
        dataset_registry[f"{name}_full"] = (
            "CitationFull",
            CitationFull,
            {"name": name},
        )
    for name in [
        "roman_empire",
        "amazon_ratings",
        "minesweeper",
        "tolokers",
        "questions",
    ]:
        dataset_registry[name] = (
            "HeterophilousGraphDataset",
            HeterophilousGraphDataset,
            {"name": name},
        )
    for name in ["cornell", "texas", "wisconsin"]:
        dataset_registry[name] = ("WebKB", WebKB, {"name": name})
    for name in ["chameleon", "squirrel"]:
        dataset_registry[name] = ("WikipediaNetwork", WikipediaNetwork, {"name": name})
    dataset_registry["actor"] = ("Actor", Actor, {})
    dataset_registry["wikics"] = ("WikiCS", WikiCS, {})
    dataset_registry["fb15k_237"] = ("FB15k_237", FB15k_237, {})
    dataset_registry["wordnet18rr"] = ("WordNet18RR", WordNet18RR, {})

    # Collections (one example each; the split is the one the docs table loads),
    # heterogeneous and temporal graphs
    from mlx_graphs.datasets import (
        DBLP,
        IMDB,
        OGBMAG,
        GNNBenchmarkDataset,
        JODIEDataset,
        LRGBDataset,
        OGBDataset,
        SuperPixelDataset,
        TGBLinkPropPredDataset,
        TGBNodePropPredDataset,
        TUDataset,
        WordNet18,
    )

    dataset_registry.update(
        {
            "mutag": ("TUDataset", TUDataset, {"name": "MUTAG"}),
            "ogbn_arxiv": ("OGBDataset", OGBDataset, {"name": "ogbn-arxiv"}),
            "ogbn_mag": ("OGBMAG", OGBMAG, {}),
            "imdb": ("IMDB", IMDB, {}),
            "dblp_hetero": ("DBLP", DBLP, {}),
            "peptides_func": (
                "LRGBDataset",
                LRGBDataset,
                {"name": "peptides-func", "split": "train"},
            ),
            "pattern": (
                "GNNBenchmarkDataset",
                GNNBenchmarkDataset,
                {"name": "PATTERN", "split": "train"},
            ),
            "mnist_superpixels": (
                "SuperPixelDataset",
                SuperPixelDataset,
                {"name": "MNIST", "split": "train"},
            ),
            "wordnet18": ("WordNet18", WordNet18, {}),
            "jodie_wikipedia": ("JODIEDataset", JODIEDataset, {"name": "wikipedia"}),
            "jodie_mooc": ("JODIEDataset", JODIEDataset, {"name": "mooc"}),
            "tgbl_wiki": (
                "TGBLinkPropPredDataset",
                TGBLinkPropPredDataset,
                {"name": "tgbl-wiki"},
            ),
            "tgbn_trade": (
                "TGBNodePropPredDataset",
                TGBNodePropPredDataset,
                {"name": "tgbn-trade"},
            ),
        }
    )

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
            if args.base_dir and "base_dir" in inspect.signature(cls).parameters:
                kwargs = {**kwargs, "base_dir": args.base_dir}
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
