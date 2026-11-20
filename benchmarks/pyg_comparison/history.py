"""Compares the benchmark results of the versions of mlx-graphs measured on a
machine, to track performance across versions.

    python benchmarks/pyg_comparison/history.py [results/<chip>]

Reads every ``results/<chip>/<label>/`` written by ``run.py`` (one per
measured version), orders them by the date of their library commit, and
writes ``results/<chip>/history.md`` with ``figures/history_*_{light,dark}.png``:
mlx-graphs' time, memory and accuracy in every version, and the speedup of
the newest version over the oldest one. Only configurations measured in at
least two versions are compared.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import plot
from plot import (
    DATASET_NAMES,
    DPI,
    LAYER_NAMES,
    PX,
    THEMES,
    failure_reason,
    index,
    log_ticks,
    ordered_datasets,
    setup_style,
    style_axis,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MLX = "mlx-graphs"

# Ordinal blue ramp (oldest -> newest). Newest is the most prominent: darkest
# on the light surface, lightest on the dark one.
RAMPS = {
    "light": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
    "dark": ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_versions(chip_dir):
    """``[(label, system, records)]`` for every version, oldest first."""
    versions = []
    for path in glob.glob(os.path.join(chip_dir, "*", "results.jsonl")):
        version_dir = os.path.dirname(path)
        system_path = os.path.join(version_dir, "system.json")
        if not os.path.exists(system_path):
            continue
        with open(system_path) as f:
            system = json.load(f)
        library = system.get("library") or {}
        label = library.get("label") or os.path.basename(version_dir)
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        versions.append((label, system, records))
    versions.sort(key=lambda v: (v[1].get("library") or {}).get("commit_date") or "")
    return versions


def version_name(label, system):
    branch = (system.get("library") or {}).get("branch")
    return f"{label} ({branch})" if branch and branch != "HEAD" else label


def ramp(theme_name, n):
    colors = RAMPS[theme_name]
    if n == 1:
        return [colors[-1]]
    return [colors[round(i * (len(colors) - 1) / (n - 1))] for i in range(n)]


# ---------------------------------------------------------------------------
# Data per task
# ---------------------------------------------------------------------------


def task_rows(versions, task):
    """Rows to compare for one task: ``[(panel, row label, key)]`` and, per
    version, ``{key: {framework: record}}``."""
    keys = (
        ("num_nodes", "num_edges", "layer") if task == "layer" else ("dataset", "model")
    )
    per_version = [index(records, task, *keys) for _, _, records in versions]
    measured = defaultdict(int)
    for data in per_version:
        for key, per_fw in data.items():
            if MLX in per_fw:
                measured[key] += 1
    common = {key for key, count in measured.items() if count >= 2}
    if not common:
        return [], per_version

    rows = []
    if task == "layer":
        for n, e in sorted({(k[0], k[1]) for k in common}, key=lambda g: g[1]):
            panel = f"{n:,} nodes · {e:,} edges"
            for layer in LAYER_NAMES:
                if (n, e, layer) in common:
                    rows.append((panel, LAYER_NAMES[layer], (n, e, layer)))
    else:
        merged = {k: v for data in per_version for k, v in data.items()}
        for model in LAYER_NAMES:
            for d in ordered_datasets(merged, task):
                if (d, model) in common:
                    rows.append(
                        (LAYER_NAMES[model], DATASET_NAMES.get(d, d), (d, model))
                    )
    return rows, per_version


def mlx_record(data, key):
    return data.get(key, {}).get(MLX)


def value(record, metric):
    if record is None or record.get("status") != "ok":
        return None
    return record.get(metric)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def fig_history(versions, task, theme_name, out):
    theme = THEMES[theme_name]
    rows, per_version = task_rows(versions, task)
    if not rows:
        return False
    metric = "forward_backward_ms" if task == "layer" else "epoch_ms"
    colors = ramp(theme_name, len(versions))
    panels = list(dict.fromkeys(panel for panel, _, _ in rows))
    labels = list(dict.fromkeys(label for _, label, _ in rows))

    times = [
        v
        for data in per_version
        for _, _, key in rows
        if (v := value(mlx_record(data, key), metric)) is not None
    ]
    lo = 10 ** math.floor(math.log10(min(times)))
    hi = 10 ** math.ceil(math.log10(max(times)))

    height = 0.36 * len(labels) + 2.5
    fig, axes = plt_subplots(len(panels), height)
    unit = (
        "forward + backward pass of one layer" if task == "layer" else "training epoch"
    )
    title = {
        "layer": "mlx-graphs across versions: one layer",
        "node": "mlx-graphs across versions: node classification",
        "graph": "mlx-graphs across versions: graph classification",
    }[task]
    fig.text(0.015, 1 - 0.12 / height, title, fontsize=12, weight="semibold", va="top")
    fig.text(
        0.015,
        1 - 0.42 / height,
        f"Time of mlx-graphs per {unit} in each measured version (log scale). "
        "Lower is better.\nThe label on the right is the speedup of the newest "
        "version over the oldest.",
        fontsize=8.5,
        color=theme["secondary"],
        va="top",
        linespacing=1.5,
    )
    handles = [
        plot.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=8 * PX,
            markerfacecolor=color,
            markeredgecolor=theme["surface"],
            label=version_name(label, system),
        )
        for (label, system, _), color in zip(versions, colors)
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.004, 1 - 0.95 / height),
        ncol=min(len(handles), 3),
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.4,
        labelcolor=theme["secondary"],
    )
    fig.subplots_adjust(
        left=0.12, right=0.97, top=1 - 1.65 / height, bottom=0.55 / height,
        wspace=0.12,
    )  # fmt: skip

    for ax, panel in zip(axes, panels):
        ax.set_xscale("log")
        ax.set_xlim(lo / 1.35, hi * 2.2)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        log_ticks(ax, lo, hi)
        style_axis(ax, theme)
        ax.set_title(panel, fontsize=9)
        ax.set_xlabel(f"ms per {unit.split(' of ')[0]} (log scale)")
        for i, label in enumerate(labels):
            key = next((k for p, lb, k in rows if p == panel and lb == label), None)
            if key is None:
                continue
            points = [
                (j, value(mlx_record(data, key), metric))
                for j, data in enumerate(per_version)
            ]
            present = [(j, v) for j, v in points if v is not None]
            if len(present) > 1:
                ax.plot(
                    [v for _, v in present],
                    [i] * len(present),
                    color=theme["axis"],
                    linewidth=1 * PX,
                    zorder=1,
                )
            for j, v in present:
                ax.plot(
                    v,
                    i,
                    marker="o",
                    markersize=9 * PX,
                    markerfacecolor=colors[j],
                    markeredgecolor=theme["surface"],
                    markeredgewidth=2 * PX,
                    linestyle="none",
                    zorder=3,
                )
            first, last = points[0][1], points[-1][1]
            if first is not None and last is not None:
                note = f"{first / last:.1f}×"
            else:
                missing = [j for j, v in points if v is None]
                note = failure_reason(
                    mlx_record(per_version[missing[0]], key) or {}
                ) if missing else ""  # fmt: skip
            ax.text(
                hi * 2.0,
                i,
                note,
                ha="right",
                va="center",
                fontsize=7,
                color=theme["secondary"],
            )
    fig.savefig(out, dpi=DPI)
    plot.plt.close(fig)
    return True


def plt_subplots(n_panels, height):
    fig, axes = plot.plt.subplots(
        1, n_panels, figsize=(3.4 * n_panels, height), sharey=True, squeeze=False
    )
    return fig, axes[0]


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def fmt(record, metric, digits):
    if record is None:
        return "–"
    if record.get("status") != "ok":
        return failure_reason(record)
    v = record.get(metric)
    if v is None:
        return "–"
    std = record.get(f"{metric}_std")
    out = f"{v:,.{digits}f}"
    return out + (f" ± {std:.{digits}f}" if std else "")


def ratio(old, new, metric):
    a, b = value(old, metric), value(new, metric)
    if a is None or b is None or b == 0:
        return "–"
    return f"{a / b:.1f}×"


def history_markdown(versions, chip_dir, figures):
    names = [version_name(label, system) for label, system, _ in versions]
    lines = [
        "# mlx-graphs across versions",
        "",
        f"Benchmark results of every version of mlx-graphs measured on this "
        f"machine ({versions[-1][1].get('chip')}, "
        f"{versions[-1][1].get('memory_gb')} GB), oldest first. Each version "
        "is a directory next to this file, with its own `results.md`. "
        "Regenerate with `python benchmarks/pyg_comparison/history.py`.",
        "",
        "| version | branch | library commit | date | mlx | torch / PyG |",
        "|---|---|---|---|---|---|",
    ]
    for label, system, _ in versions:
        lib = system.get("library") or {}
        commit = (lib.get("commit") or "")[:7]
        subject = lib.get("commit_subject") or ""
        lines.append(
            f"| [{label}]({label}/results.md) | {lib.get('branch') or '–'} | "
            f"`{commit}` {subject} | {(lib.get('commit_date') or '')[:10]} | "
            f"{system.get('mlx')} | {system.get('torch')} / "
            f"{system.get('torch_geometric')} |"
        )
    lines += [
        "",
        f"Speedup columns compare the newest version ({names[-1]}) with the "
        f"oldest ({names[0]}). Times are medians, accuracies are mean ± "
        "standard deviation over seeds.",
        "",
    ]

    sections = (
        ("layer", "One layer, forward + backward", "forward_backward_ms"),
        ("node", "Full-batch node classification", "epoch_ms"),
        ("graph", "Graph classification", "epoch_ms"),
    )
    for task, title, metric in sections:
        rows, per_version = task_rows(versions, task)
        if not rows:
            continue
        lines += [f"## {title}", ""]
        if task in figures:
            lines += plot.figure(
                f"history_{task}",
                f"{title}: mlx-graphs time per version for each configuration",
            )
        unit = "ms per pass" if task == "layer" else "ms per epoch"
        header = " | ".join(names)
        lines += [
            f"mlx-graphs time ({unit}):",
            "",
            f"| | | {header} | speedup |",
            "|---|---|" + "---:|" * len(names) + "---:|",
        ]
        for panel, label, key in rows:
            recs = [mlx_record(data, key) for data in per_version]
            lines.append(
                f"| {panel} | {label} | "
                + " | ".join(fmt(r, metric, 2) for r in recs)
                + f" | {ratio(recs[0], recs[-1], metric)} |"
            )
        lines.append("")

        second, second_name = (
            ("peak_memory_gb", "Peak memory (GB)")
            if task == "layer"
            else ("test_acc", "Test accuracy")
        )
        lines += [
            f"{second_name} of mlx-graphs:",
            "",
            f"| | | {header} |",
            "|---|---|" + "---:|" * len(names),
        ]
        for panel, label, key in rows:
            recs = [mlx_record(data, key) for data in per_version]
            digits = 2 if task == "layer" else 3
            lines.append(
                f"| {panel} | {label} | "
                + " | ".join(fmt(r, second, digits) for r in recs)
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("chip_dir", nargs="?", default=None)
    args = parser.parse_args()
    chip_dir = (
        args.chip_dir
        or sorted(
            (
                d
                for d in glob.glob(os.path.join(HERE, "results", "*"))
                if os.path.isdir(d)
            ),
            key=os.path.getmtime,
        )[-1]
    )
    versions = load_versions(chip_dir)
    if len(versions) < 2:
        raise SystemExit(
            f"{chip_dir} has {len(versions)} measured version(s); at least 2 "
            "are needed to compare. Run run.py on another version first."
        )
    fig_dir = os.path.join(chip_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    figures = set()
    for theme_name, theme in THEMES.items():
        setup_style(theme)
        for task in ("layer", "node", "graph"):
            path = os.path.join(fig_dir, f"history_{task}_{theme_name}.png")
            if fig_history(versions, task, theme_name, path):
                figures.add(task)
    out = os.path.join(chip_dir, "history.md")
    with open(out, "w") as f:
        f.write(history_markdown(versions, chip_dir, figures))
    print(f"compared {len(versions)} versions -> {out}")


if __name__ == "__main__":
    main()
