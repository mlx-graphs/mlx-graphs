"""Draws the figures and tables of the PyG comparison from ``run.py``'s results.

    python benchmarks/pyg_comparison/plot.py [results/<chip>]

Writes ``figures/<name>_{light,dark}.png``, ``results.md`` (every number in
the figures, as tables) and ``results.rst`` (the section of the docs'
benchmarks page, included by ``docs/source/benchmarks.rst``) into the results
directory.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

FRAMEWORKS = ["mlx-graphs", "pyg-mps", "pyg-cpu"]
LABELS = {
    "mlx-graphs": "mlx-graphs (GPU)",
    "pyg-mps": "PyG (MPS GPU)",
    "pyg-cpu": "PyG (CPU)",
}
LAYER_NAMES = {
    "gcn": "GCN",
    "sage": "GraphSAGE",
    "gin": "GIN",
    "gat": "GAT",
    "gatv2": "GATv2",
}
DATASET_NAMES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
    "amazon-photo": "Amazon Photo",
    "amazon-computers": "Amazon Computers",
    "coauthor-cs": "Coauthor CS",
    "coauthor-physics": "Coauthor Physics",
    "ogbn-arxiv": "ogbn-arxiv",
}

# Reference palette (categorical slots 1-3, validated all-pairs in both modes)
# and chart chrome, light and dark.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": {"mlx-graphs": "#2a78d6", "pyg-mps": "#eb6834", "pyg-cpu": "#1baf7a"},
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": {"mlx-graphs": "#3987e5", "pyg-mps": "#d95926", "pyg-cpu": "#199e70"},
    },
}

DPI = 200
PX = 72 / 96  # one CSS pixel in points (figures are sized for ~96 px/inch)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load(results_dir):
    records = []
    with open(os.path.join(results_dir, "results.jsonl")) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    with open(os.path.join(results_dir, "system.json")) as f:
        system = json.load(f)
    return records, system


def index(records, task, *keys):
    """``{(key values...): {framework: record}}`` for one task.

    Runs of the same configuration with different seeds are merged into one
    record (see :func:`merge_seeds`).
    """
    groups = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("task") != task:
            continue
        key = tuple(
            int(r[k]) if k in ("num_nodes", "num_edges") else r[k] for k in keys
        )
        groups[key][r["framework"]].append(r)
    return {
        key: {fw: merge_seeds(runs) for fw, runs in per_fw.items()}
        for key, per_fw in groups.items()
    }


TIME_KEYS = ("forward_ms", "forward_backward_ms", "epoch_ms", "peak_memory_gb")
METRIC_KEYS = ("train_acc", "val_acc", "test_acc", "final_train_loss")


def merge_seeds(runs):
    """One record for the runs of a configuration with different seeds.

    Times and memory are the median over seeds; accuracies and the final
    loss are the mean, with ``<key>_std``, ``<key>_min`` and ``<key>_max``
    over seeds. ``seeds`` keeps the individual runs (for the curves). A
    configuration that failed for any seed counts as failed.
    """
    failed = [r for r in runs if r.get("status") != "ok"]
    if failed:
        return failed[0]
    out = dict(runs[0])
    out["seeds"] = runs
    for k in TIME_KEYS:
        if k in out:
            out[k] = float(np.median([r[k] for r in runs]))
    for k in METRIC_KEYS:
        if k in out:
            values = np.array([r[k] for r in runs], dtype=float)
            out[k] = float(values.mean())
            out[f"{k}_std"] = float(values.std())
            out[f"{k}_min"] = float(values.min())
            out[f"{k}_max"] = float(values.max())
    return out


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def setup_style(theme):
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.facecolor": theme["surface"],
            "figure.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "text.color": theme["primary"],
            "axes.labelcolor": theme["secondary"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["secondary"],
            "xtick.labelcolor": theme["muted"],
            "ytick.labelcolor": theme["secondary"],
            "axes.edgecolor": theme["axis"],
            "axes.linewidth": 1 * PX,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": theme["primary"],
            "axes.titlepad": 8,
        }
    )


def style_axis(ax, theme, grid_axis="x"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.grid(True, axis=grid_axis, color=theme["grid"], linewidth=1 * PX)
    ax.set_axisbelow(True)


def log_ticks(ax, lo, hi):
    """Clean 1-3-10 ticks on a log x axis, labeled in full (no 10^x)."""
    ticks = []
    decade = 10 ** int(__import__("math").floor(__import__("math").log10(lo)))
    while decade <= hi * 10:
        for m in (1, 3):  # evenly spaced on a log axis
            t = m * decade
            if lo <= t <= hi:
                ticks.append(t)
        decade *= 10
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:,.0f}" if t >= 1 else f"{t:g}" for t in ticks])


def rounded_hbar(ax, y, value, height_px, color, radius_px=4):
    """Horizontal bar from 0 to ``value`` with a rounded data-end and a square
    baseline. Axis limits and figure layout must be final."""
    to_px = ax.transData.transform
    (x0, y0), (x1, y1) = to_px((0, 0)), to_px((1, 1))
    px_per_x, px_per_y = abs(x1 - x0), abs(y1 - y0)
    h = height_px / px_per_y / 2
    rx = min(radius_px / px_per_x, value / 2)
    ry = min(radius_px / px_per_y, h)
    verts = [
        (0, y - h),
        (value - rx, y - h),
        (value, y - h),
        (value, y - h + ry),
        (value, y + h - ry),
        (value, y + h),
        (value - rx, y + h),
        (0, y + h),
        (0, y - h),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none"))


def legend(fig, theme, frameworks, top_in):
    """One row of legend entries, left-aligned, ``top_in`` inches from the top."""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=8 * PX,
            markerfacecolor=theme["series"][fw],
            markeredgecolor=theme["surface"],
            label=LABELS[fw],
        )
        for fw in frameworks
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.004, 1 - top_in / fig.get_figheight()),
        ncol=len(handles),
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.4,
        labelcolor=theme["secondary"],
    )


def header(fig, theme, title, subtitle, system, frameworks=FRAMEWORKS):
    """Title, subtitle, machine line and legend. Returns the fraction of the
    figure height left for the plots (the ``top`` of ``subplots_adjust``)."""
    height = fig.get_figheight()
    fig.text(0.015, 1 - 0.12 / height, title, fontsize=12, weight="semibold", va="top")
    machine = (
        f"{system['chip']}, {system['memory_gb']} GB · mlx {system['mlx']} · "
        f"torch {system['torch']} · PyG {system['torch_geometric']}"
    )
    text = f"{subtitle}\n{machine}"
    fig.text(
        0.015,
        1 - 0.42 / height,
        text,
        fontsize=8.5,
        color=theme["secondary"],
        va="top",
        linespacing=1.5,
    )
    legend_in = 0.42 + 0.19 * (text.count("\n") + 1) + 0.08
    legend(fig, theme, frameworks, legend_in)
    return 1 - (legend_in + 0.6) / height


def dot_rows(ax, theme, rows, values, failed, lo, hi):
    """Dot plot on a log axis: one row per label, one dot per framework.

    ``values[row][fw]`` is a time in ms (or None), ``failed[row][fw]`` a short
    reason to print instead of a dot.
    """
    ax.set_xscale("log")
    # pad the limits so the end tick labels of neighbouring panels don't touch
    ax.set_xlim(lo / 1.35, hi * 1.35)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    log_ticks(ax, lo, hi)
    style_axis(ax, theme)
    for i, row in enumerate(rows):
        present = [v for v in values[row].values() if v is not None]
        if len(present) > 1:  # hairline connector from fastest to slowest
            ax.plot(
                [min(present), max(present)],
                [i, i],
                color=theme["axis"],
                linewidth=1 * PX,
                zorder=1,
            )
        for j, fw in enumerate(FRAMEWORKS):
            v = values[row].get(fw)
            if v is not None:
                ax.plot(
                    v,
                    # small vertical offset per framework, so that two
                    # frameworks with the same time don't hide each other
                    i + (j - 1) * 0.13,
                    marker="o",
                    markersize=9 * PX,
                    markerfacecolor=theme["series"][fw],
                    markeredgecolor=theme["surface"],
                    markeredgewidth=2 * PX,
                    linestyle="none",
                    zorder=3,
                )
        reasons = [f"{LABELS[fw]}: {failed[row][fw]}" for fw in failed[row]]
        if reasons:
            ax.text(
                hi,
                i + 0.32,
                "; ".join(reasons),
                ha="right",
                va="center",
                fontsize=7,
                color=theme["muted"],
            )


def failure_reason(record):
    if not record:
        return "not run"
    status = record.get("status", "ok")
    if status.startswith("oom"):
        return "out of memory"
    if status.startswith("timeout"):
        return "timed out"
    return "failed"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_layers(records, system, theme, out):
    data = index(records, "layer", "num_nodes", "num_edges", "layer")
    graphs = sorted({(n, e) for n, e, _ in data}, key=lambda g: g[1])
    layers = [layer for layer in LAYER_NAMES if any(k[2] == layer for k in data)]
    rows = [LAYER_NAMES[layer] for layer in layers]

    all_times = [
        r["forward_backward_ms"]
        for per_fw in data.values()
        for r in per_fw.values()
        if r.get("status") == "ok"
    ]
    lo = 10 ** __import__("math").floor(__import__("math").log10(min(all_times)))
    hi = 10 ** __import__("math").ceil(__import__("math").log10(max(all_times)))

    fig, axes = plt.subplots(
        1, len(graphs), figsize=(3.4 * len(graphs), 4.2), sharey=True
    )
    top = header(
        fig,
        theme,
        "One layer, forward + backward, random graphs",
        "128 features in and out, 4 heads for GAT/GATv2. Lower is better.",
        system,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=top, bottom=0.55 / 4.2, wspace=0.08)
    for ax, (n, e) in zip(axes, graphs):
        values, failed = defaultdict(dict), defaultdict(dict)
        for layer, row in zip(layers, rows):
            for fw, r in data.get((n, e, layer), {}).items():
                if r.get("status") == "ok":
                    values[row][fw] = r["forward_backward_ms"]
                else:
                    values[row][fw] = None
                    failed[row][fw] = failure_reason(r)
        dot_rows(ax, theme, rows, values, failed, lo, hi)
        ax.set_title(f"{n:,} nodes · {e:,} edges", fontsize=9)
        ax.set_xlabel("ms per forward + backward pass (log scale)")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def fig_layer_speedup(records, system, theme, out):
    """Time of every framework relative to mlx-graphs (mlx-graphs = 1×)."""
    data = index(records, "layer", "num_nodes", "num_edges", "layer")
    graphs = sorted({(n, e) for n, e, _ in data}, key=lambda g: g[1])
    layers = [layer for layer in LAYER_NAMES if any(k[2] == layer for k in data)]

    ratios = {}
    for key, per_fw in data.items():
        mlx = per_fw.get("mlx-graphs")
        if not mlx or mlx.get("status") != "ok":
            continue
        for fw in FRAMEWORKS:
            r = per_fw.get(fw)
            if r and r.get("status") == "ok":
                ratios[key + (fw,)] = (
                    r["forward_backward_ms"] / mlx["forward_backward_ms"]
                )
    hi = max(ratios.values()) * 1.18

    height = 5.0
    fig, axes = plt.subplots(
        1, len(graphs), figsize=(3.4 * len(graphs), height), sharey=True
    )
    top = header(
        fig,
        theme,
        "Time per layer, relative to mlx-graphs",
        "Forward + backward time of each framework divided by mlx-graphs' time "
        "(1×). Lower is better.",
        system,
    )
    fig.subplots_adjust(
        left=0.08, right=0.985, top=top, bottom=0.55 / height, wspace=0.08
    )
    bar_px, gap_px = 7, 2
    for ax, (n, e) in zip(axes, graphs):
        ax.set_xlim(0, hi)
        ax.set_ylim(len(layers) - 0.5, -0.5)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([LAYER_NAMES[layer] for layer in layers])
        style_axis(ax, theme)
        ax.set_title(f"{n:,} nodes · {e:,} edges", fontsize=9)
        ax.set_xlabel("time relative to mlx-graphs (×)")
        fig.canvas.draw()  # layout is final: pixel sizes can be computed
        px_per_y = abs(
            ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1]
        )
        step = (bar_px + gap_px) / px_per_y
        for i, layer in enumerate(layers):
            for j, fw in enumerate(FRAMEWORKS):
                y = i + (j - 1) * step
                ratio = ratios.get((n, e, layer, fw))
                if ratio is None:
                    reason = failure_reason(data[(n, e, layer)].get(fw, {}))
                    ax.text(
                        0.02 * hi,
                        y,
                        f"{LABELS[fw]} {reason}",
                        va="center",
                        fontsize=7,
                        color=theme["muted"],
                    )
                    continue
                rounded_hbar(ax, y, ratio, bar_px, theme["series"][fw])
                if fw == "mlx-graphs":
                    continue  # always 1×, as the subtitle says
                ax.text(
                    ratio + 0.012 * hi,
                    y,
                    f"{ratio:.0f}×" if ratio >= 10 else f"{ratio:.1f}×",
                    va="center",
                    fontsize=6.5,
                    color=theme["secondary"],
                )
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def headline_numbers(records):
    """The numbers of the headline figure.

    ``speed``: ``[(row label, {framework: time relative to mlx-graphs})]``,
    medians over the configurations of each task (one layer: the largest
    graph). ``accuracy``: ``[(row label, {framework: mean test accuracy})]``
    over the dataset × model pairs every framework finished.
    """
    speed, accuracy = [], []
    layers = index(records, "layer", "num_nodes", "num_edges", "layer")
    if layers:
        n, e = max({(n, e) for n, e, _ in layers}, key=lambda g: g[1])
        largest = {k: v for k, v in layers.items() if k[:2] == (n, e)}
        speed.append(
            (
                f"One layer\n({e / 1e6:g}M edges)",
                {
                    fw: float(np.median(list(r.values()))) if r else None
                    for fw in FRAMEWORKS
                    for r in [speedups(largest, "forward_backward_ms", fw)]
                },
            )
        )
    for task, label in (
        ("node", "Node\nclassification"),
        ("graph", "Graph\nclassification"),
    ):
        data = index(records, task, "dataset", "model")
        if not data:
            continue
        speed.append(
            (
                label,
                {
                    fw: float(np.median(list(r.values()))) if r else None
                    for fw in FRAMEWORKS
                    for r in [speedups(data, "epoch_ms", fw)]
                },
            )
        )
        complete = [
            per_fw for per_fw in data.values()
            if all(per_fw.get(fw, {}).get("status") == "ok" for fw in FRAMEWORKS)
        ]  # fmt: skip
        if complete and "test_acc" in complete[0]["mlx-graphs"]:
            accuracy.append(
                (
                    label,
                    {
                        fw: float(np.mean([p[fw]["test_acc"] for p in complete]))
                        for fw in FRAMEWORKS
                    },
                )
            )
    return speed, accuracy


def fig_headline(records, system, theme, out):
    """Two panels: how much longer PyG takes than mlx-graphs (median per
    task), and the test accuracy every framework reaches."""
    speed, accuracy = headline_numbers(records)
    panels = [p for p in (speed, accuracy) if p]
    height = 4.3
    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.2 * len(panels), height), squeeze=False
    )
    axes = axes[0]
    top = header(
        fig,
        theme,
        "mlx-graphs vs PyTorch Geometric: faster, same accuracy",
        "Left: time of each framework divided by mlx-graphs' time, median over "
        "the configurations of each task (lower is better).\nRight: test "
        "accuracy, mean over datasets, models and seeds (higher is better).",
        system,
    )
    fig.subplots_adjust(
        left=0.135, right=0.975, top=top, bottom=0.55 / height, wspace=0.4
    )
    rows = max(len(p) for p in panels)
    bar_px, gap_px = 9, 2
    for ax, panel in zip(axes, panels):
        is_speed = panel is speed
        values = [v for _, per_fw in panel for v in per_fw.values() if v]
        hi = max(values) * 1.16 if is_speed else 1.0
        ax.set_xlim(0, hi)
        ax.set_ylim(rows - 0.5, -0.5)
        offset = (rows - len(panel)) / 2  # center a panel with fewer rows
        ax.set_yticks([i + offset for i in range(len(panel))])
        ax.set_yticklabels([label for label, _ in panel])
        style_axis(ax, theme)
        if is_speed:
            ax.set_title("How long each framework takes", fontsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}×")
            ax.set_xlabel("time relative to mlx-graphs")
        else:
            ax.set_title("Test accuracy reached", fontsize=9)
            ticks = np.arange(0, 1.01, 0.2)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.set_xticklabels([f"{t:.0%}" for t in ticks])
            ax.set_xlabel("test accuracy")
        fig.canvas.draw()  # layout is final: pixel sizes can be computed
        px_per_y = abs(
            ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1]
        )
        step = (bar_px + gap_px) / px_per_y
        for i, (_, per_fw) in enumerate(panel):
            for j, fw in enumerate(FRAMEWORKS):
                v = per_fw.get(fw)
                if v is None:
                    continue
                y = i + offset + (j - 1) * step
                rounded_hbar(ax, y, v, bar_px, theme["series"][fw])
                if not is_speed:
                    label = f"{v:.1%}"
                elif fw == "mlx-graphs":
                    label = "1×"  # the reference
                else:
                    label = times_x(v)
                ax.text(
                    v + 0.012 * hi,
                    y,
                    label,
                    va="center",
                    fontsize=7.5,
                    color=theme["primary" if fw == "mlx-graphs" else "secondary"],
                )
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def fig_memory(records, system, theme, out):
    data = index(records, "layer", "num_nodes", "num_edges", "layer")
    n, e = max({(n, e) for n, e, _ in data}, key=lambda g: g[1])
    layers = [layer for layer in LAYER_NAMES if (n, e, layer) in data]
    peaks = {
        (layer, fw): r["peak_memory_gb"]
        for layer in layers
        for fw, r in data[(n, e, layer)].items()
        if r.get("status") == "ok"
    }
    hi = max(peaks.values()) * 1.15

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    top = header(
        fig,
        theme,
        "Peak memory, one layer forward + backward",
        f"{n:,} nodes · {e:,} edges · 128 features. Lower is better. PyG on MPS\n"
        "includes ~1.4 GB of PyTorch MPS runtime overhead.",
        system,
    )
    fig.subplots_adjust(left=0.15, right=0.97, top=top, bottom=0.55 / 4.8)
    ax.set_xlim(0, hi)
    ax.set_ylim(len(layers) - 0.5, -0.5)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([LAYER_NAMES[layer] for layer in layers])
    style_axis(ax, theme)
    ax.set_xlabel("peak memory of the process, GB (CPU + GPU)")
    fig.canvas.draw()
    px_per_y = abs(
        ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1]
    )
    bar_px, gap_px = 7, 2
    step = (bar_px + gap_px) / px_per_y
    for i, layer in enumerate(layers):
        for j, fw in enumerate(FRAMEWORKS):
            y = i + (j - 1) * step
            v = peaks.get((layer, fw))
            if v is None:
                reason = failure_reason(data[(n, e, layer)].get(fw, {}))
                ax.text(
                    0.01 * hi,
                    y,
                    f"{LABELS[fw]} {reason}",
                    va="center",
                    fontsize=7,
                    color=theme["muted"],
                )
                continue
            rounded_hbar(ax, y, v, bar_px, theme["series"][fw])
            ax.text(
                v + 0.01 * hi,
                y,
                f"{v:.1f}",
                va="center",
                fontsize=6.5,
                color=theme["secondary"],
            )
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def fig_training(records, system, theme, out, task):
    data = index(records, task, "dataset", "model")
    models = [m for m in LAYER_NAMES if any(k[1] == m for k in data)]
    datasets = list(dict.fromkeys(k[0] for k in data))
    if task == "node":
        order = list(DATASET_NAMES)
        datasets.sort(key=lambda d: order.index(d) if d in order else len(order))
    names = {d: DATASET_NAMES.get(d, d) for d in datasets}
    rows = [names[d] for d in datasets]

    times = [
        r["epoch_ms"]
        for per_fw in data.values()
        for r in per_fw.values()
        if r.get("status") == "ok"
    ]
    math = __import__("math")
    lo = 10 ** math.floor(math.log10(min(times)))
    hi = 10 ** math.ceil(math.log10(max(times)))

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(3.4 * len(models), 0.36 * len(rows) + 2.6),
        sharey=True,
    )
    if task == "node":
        title = "Full-batch node classification: time per training epoch"
        subtitle = (
            "2-layer models, 64 hidden units, Adam, the whole graph each step. "
            "Lower is better."
        )
    else:
        title = "Graph classification: time per training epoch"
        subtitle = (
            "3 layers + mean pooling, batches of 64 graphs; includes batching "
            "the graphs. Lower is better."
        )
    top = header(fig, theme, title, subtitle, system)
    fig.subplots_adjust(
        left=0.135, right=0.985, top=top, bottom=0.55 / fig.get_figheight(), wspace=0.08
    )
    for ax, model in zip(axes, models):
        values, failed = defaultdict(dict), defaultdict(dict)
        for d in datasets:
            for fw, r in data.get((d, model), {}).items():
                if r.get("status") == "ok":
                    values[names[d]][fw] = r["epoch_ms"]
                else:
                    values[names[d]][fw] = None
                    failed[names[d]][fw] = failure_reason(r)
        dot_rows(ax, theme, rows, values, failed, lo, hi)
        ax.set_title(LAYER_NAMES[model], fontsize=9)
        ax.set_xlabel("ms per training epoch (log scale)")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def ordered_datasets(data, task):
    datasets = list(dict.fromkeys(k[0] for k in data))
    if task == "node":
        order = list(DATASET_NAMES)
        datasets.sort(key=lambda d: order.index(d) if d in order else len(order))
    return datasets


def fig_convergence(records, system, theme, out, task):
    """Training loss per epoch: one panel per dataset and model, the three
    frameworks overlaid (mean over seeds, with the min-max range shaded)."""
    data = index(records, task, "dataset", "model")
    models = [m for m in LAYER_NAMES if any(k[1] == m for k in data)]
    datasets = ordered_datasets(data, task)
    seeds = max(
        (len(r.get("seeds", [])) for per_fw in data.values() for r in per_fw.values()),
        default=1,
    )

    height = 1.3 * len(datasets) + 2.2
    fig, axes = plt.subplots(
        len(datasets), len(models), figsize=(3.4 * len(models), height), squeeze=False
    )
    noun = "node" if task == "node" else "graph"
    top = header(
        fig,
        theme,
        f"Training loss, {noun} classification",
        "Cross-entropy on the training set at every epoch (log scale). "
        f"Lines: mean over {plural(seeds, 'seed')}; shaded: min to max.",
        system,
    )
    fig.subplots_adjust(
        left=0.13, right=0.985, top=top, bottom=0.5 / height, wspace=0.18,
        hspace=0.35,
    )  # fmt: skip
    for i, d in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i][j]
            style_axis(ax, theme, grid_axis="y")
            ax.tick_params(labelsize=7)
            per_fw = data.get((d, model), {})
            failed, plotted = [], False
            # mlx-graphs last, so that its line is drawn on top of PyG's
            for fw in reversed(FRAMEWORKS):
                r = per_fw.get(fw)
                if r is None or r.get("status") != "ok":
                    failed.insert(0, f"{LABELS[fw]}: {failure_reason(r or {})}")
                    continue
                plotted = True
                curves = np.array([run["loss_curve"] for run in r["seeds"]])
                epochs = np.arange(1, curves.shape[1] + 1)
                color = theme["series"][fw]
                if len(curves) > 1:
                    ax.fill_between(
                        epochs,
                        curves.min(axis=0),
                        curves.max(axis=0),
                        color=color,
                        alpha=0.10,
                        linewidth=0,
                    )
                ax.plot(epochs, curves.mean(axis=0), color=color, linewidth=1.5 * PX)
            if failed:
                ax.text(
                    0.98,
                    0.92,
                    "\n".join(failed),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.5,
                    color=theme["muted"],
                )
            if plotted:
                # log scale: the late, small losses are where frameworks differ
                ax.set_yscale("log")
                ax.set_xlim(left=1)
            if i == 0:
                ax.set_title(LAYER_NAMES[model], fontsize=9)
            if j == 0:
                ax.set_ylabel(
                    DATASET_NAMES.get(d, d),
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=8.5,
                    color=theme["secondary"],
                )
            if i == len(datasets) - 1:
                ax.set_xlabel("epoch", fontsize=8)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def fig_accuracy(records, system, theme, out, task):
    """Test accuracy per dataset and model: mean over seeds (dot) and the
    range over seeds (line), for each framework."""
    data = index(records, task, "dataset", "model")
    models = [m for m in LAYER_NAMES if any(k[1] == m for k in data)]
    datasets = ordered_datasets(data, task)
    rows = [DATASET_NAMES.get(d, d) for d in datasets]
    ok = [
        r for per_fw in data.values() for r in per_fw.values()
        if r.get("status") == "ok"
    ]  # fmt: skip
    seeds = max((len(r.get("seeds", [])) for r in ok), default=1)
    lo = max(0.0, np.floor(min(r["test_acc_min"] for r in ok) * 20) / 20 - 0.02)
    hi = min(1.0, np.ceil(max(r["test_acc_max"] for r in ok) * 20) / 20 + 0.01)

    height = 0.36 * len(rows) + 2.6
    fig, axes = plt.subplots(
        1, len(models), figsize=(3.4 * len(models), height), sharey=True
    )
    noun = "node" if task == "node" else "graph"
    selection = (
        "at the epoch with the best validation accuracy"
        if task == "node"
        else "at the epoch with the best validation accuracy (80/10/10 split)"
    )
    top = header(
        fig,
        theme,
        f"Test accuracy, {noun} classification",
        f"Dots: mean over {plural(seeds, 'seed')}; lines: min to max. "
        f"Measured {selection}.",
        system,
    )
    fig.subplots_adjust(
        left=0.135, right=0.985, top=top, bottom=0.55 / height, wspace=0.08
    )
    for ax, model in zip(axes, models):
        ax.set_xlim(lo, hi)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)
        style_axis(ax, theme)
        ticks = np.arange(np.ceil(lo * 10) / 10, hi + 1e-9, 0.1)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([f"{t:.0%}" for t in ticks])
        ax.set_title(LAYER_NAMES[model], fontsize=9)
        ax.set_xlabel("test accuracy")
        for i, d in enumerate(datasets):
            per_fw = data.get((d, model), {})
            failed = []
            for j, fw in enumerate(FRAMEWORKS):
                r = per_fw.get(fw)
                y = i + (j - 1) * 0.2
                if r is None or r.get("status") != "ok":
                    failed.append(f"{LABELS[fw]}: {failure_reason(r or {})}")
                    continue
                color = theme["series"][fw]
                ax.plot(
                    [r["test_acc_min"], r["test_acc_max"]],
                    [y, y],
                    color=color,
                    linewidth=1.5 * PX,
                    solid_capstyle="round",
                    zorder=2,
                )
                ax.plot(
                    r["test_acc"],
                    y,
                    marker="o",
                    markersize=8 * PX,
                    markerfacecolor=color,
                    markeredgecolor=theme["surface"],
                    markeredgewidth=2 * PX,
                    linestyle="none",
                    zorder=3,
                )
            if failed:
                ax.text(
                    hi,
                    i + 0.38,
                    "; ".join(failed),
                    ha="right",
                    va="center",
                    fontsize=6.5,
                    color=theme["muted"],
                )
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def fmt(r, key, digits=1):
    if r is None:
        return "–"
    if r.get("status") != "ok":
        return failure_reason(r)
    return f"{r[key]:,.{digits}f}"


def figure(name, alt):
    """Markdown for a figure of ``figures/``: the dark version in dark mode
    (GitHub, VS Code), the light version otherwise."""
    return [
        "<picture>",
        '  <source media="(prefers-color-scheme: dark)" '
        f'srcset="figures/{name}_dark.png">',
        f'  <img alt="{alt}" src="figures/{name}_light.png">',
        "</picture>",
        "",
    ]


def tables(records, system):
    lines = [
        "# mlx-graphs vs PyTorch Geometric",
        "",
        f"Machine: {system['chip']}, {system['memory_gb']} GB, {system['os']}. "
        f"mlx {system['mlx']}, mlx-graphs "
        # the measured checkout's version (the installed package may differ)
        f"{system.get('library', {}).get('version') or system['mlx-graphs']}, "
        f"mlx-cluster {system['mlx-cluster']}, torch {system['torch']}, "
        f"torch_geometric {system['torch_geometric']}, Python {system['python']}.",
        "",
        "Times are medians in milliseconds; lower is better. Speedup is the "
        "PyG time divided by the mlx-graphs time.",
        "",
    ]
    sentence = headline_sentence(records)
    if sentence:
        lines += [
            sentence,
            "",
            *figure(
                "headline",
                "Time of PyG on MPS and on CPU relative to mlx-graphs for one "
                "layer, node classification and graph classification, and the "
                "test accuracy of each framework",
            ),
        ]

    data = index(records, "layer", "num_nodes", "num_edges", "layer")
    if data:
        lines += [
            "## One layer, forward + backward",
            "",
            "Random graphs, 128 input and output features, 4 heads for GAT/GATv2.",
            "",
            *figure(
                "layers_speedup",
                "Forward and backward time per layer and graph size, relative "
                "to mlx-graphs, for mlx-graphs, PyG on MPS and PyG on CPU",
            ),
            *figure(
                "layers_time",
                "Time of one forward and backward pass per layer and graph size, "
                "for mlx-graphs, PyG on MPS and PyG on CPU",
            ),
            *figure(
                "layers_memory",
                "Peak memory per layer on the largest graph, for mlx-graphs, "
                "PyG on MPS and PyG on CPU",
            ),
            "| graph | layer | mlx-graphs | PyG MPS | PyG CPU | speedup vs MPS "
            "| speedup vs CPU | peak GB mlx / MPS / CPU |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for (n, e, layer), per_fw in sorted(
            data.items(), key=lambda kv: (kv[0][1], list(LAYER_NAMES).index(kv[0][2]))
        ):
            mlx = per_fw.get("mlx-graphs")

            def speed(fw):
                r = per_fw.get(fw)
                if not (mlx and r and mlx.get("status") == r.get("status") == "ok"):
                    return "–"
                return f"{r['forward_backward_ms'] / mlx['forward_backward_ms']:.1f}×"

            mem = " / ".join(fmt(per_fw.get(fw), "peak_memory_gb") for fw in FRAMEWORKS)
            lines.append(
                f"| {n:,} nodes, {e:,} edges | {LAYER_NAMES[layer]} | "
                + " | ".join(
                    fmt(per_fw.get(fw), "forward_backward_ms", 2) for fw in FRAMEWORKS
                )
                + f" | {speed('pyg-mps')} | {speed('pyg-cpu')} | {mem} |"
            )
        lines.append("")

    for task, title in (
        ("node", "Full-batch node classification"),
        ("graph", "Graph classification"),
    ):
        data = index(records, task, "dataset", "model")
        if not data:
            continue
        seeds = max(
            (len(r.get("seeds", [])) for v in data.values() for r in v.values()),
            default=1,
        )
        over = f" over {seeds} seeds" if seeds > 1 else ""
        lines += [
            f"## {title}",
            "",
            f"Median time per training epoch{over}.",
            "",
            *figure(
                f"{task}_classification",
                f"{title}: time per training epoch per dataset and model, for "
                "mlx-graphs, PyG on MPS and PyG on CPU",
            ),
            "| dataset | model | mlx-graphs | PyG MPS | PyG CPU | speedup vs MPS "
            "| speedup vs CPU |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        rows = [
            (d, model, data[(d, model)])
            for d in ordered_datasets(data, task)
            for model in LAYER_NAMES
            if (d, model) in data
        ]
        for d, model, per_fw in rows:
            mlx = per_fw.get("mlx-graphs")

            def speed(fw):
                r = per_fw.get(fw)
                if not (mlx and r and mlx.get("status") == r.get("status") == "ok"):
                    return "–"
                return f"{r['epoch_ms'] / mlx['epoch_ms']:.1f}×"

            lines.append(
                f"| {DATASET_NAMES.get(d, d)} | {LAYER_NAMES[model]} | "
                + " | ".join(fmt(per_fw.get(fw), "epoch_ms", 2) for fw in FRAMEWORKS)
                + f" | {speed('pyg-mps')} | {speed('pyg-cpu')} |"
            )
        lines.append("")

        if not any("final_train_loss" in r for _, _, v in rows for r in v.values()):
            continue
        split = (
            "the dataset's validation split"
            if task == "node"
            else "a random 80/10/10 train/validation/test split"
        )
        lines += [
            f"### {title}: training quality",
            "",
            "Do the models train the same in both frameworks? Training loss at "
            "every epoch, and test accuracy at the epoch with the best "
            f"validation accuracy on {split}, mean ± standard deviation{over}. "
            "Final loss is the mean training loss over the last 10 epochs.",
            "",
            *figure(
                f"{task}_loss",
                f"{title}: training loss per epoch for each dataset and model, "
                "mlx-graphs, PyG on MPS and PyG on CPU overlaid",
            ),
            *figure(
                f"{task}_accuracy",
                f"{title}: test accuracy per dataset and model, mean and range "
                "over seeds, for mlx-graphs, PyG on MPS and PyG on CPU",
            ),
            "| dataset | model | test acc. mlx-graphs | PyG MPS | PyG CPU "
            "| final loss mlx-graphs | PyG MPS | PyG CPU |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for d, model, per_fw in rows:

            def acc(fw):
                r = per_fw.get(fw)
                if r is None or r.get("status") != "ok":
                    return fmt(r, "test_acc")
                std = r.get("test_acc_std")
                return f"{r['test_acc']:.3f}" + (f" ± {std:.3f}" if std else "")

            lines.append(
                f"| {DATASET_NAMES.get(d, d)} | {LAYER_NAMES[model]} | "
                + " | ".join(acc(fw) for fw in FRAMEWORKS)
                + " | "
                + " | ".join(
                    fmt(per_fw.get(fw), "final_train_loss", 3) for fw in FRAMEWORKS
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Docs page (reStructuredText, included by docs/source/benchmarks.rst)
# ---------------------------------------------------------------------------

DOCS_SOURCE = os.path.join(HERE, "..", "..", "docs", "source")


def times_x(ratio):
    return f"{ratio:.0f}×" if ratio >= 10 else f"{ratio:.1f}×"


def speedups(data, time_key, fw):
    """``{key: PyG time / mlx-graphs time}`` where both ran."""
    out = {}
    for key, per_fw in data.items():
        mlx, other = per_fw.get("mlx-graphs"), per_fw.get(fw)
        if mlx and other and mlx.get("status") == other.get("status") == "ok":
            out[key] = other[time_key] / mlx[time_key]
    return out


def rst_figure(fig_dir, name, alt):
    """Light and dark versions of a figure; the theme shows the right one.
    With a width, Sphinx links each to the full-size image."""
    lines = []
    for mode in ("light", "dark"):
        path = os.path.relpath(
            os.path.join(fig_dir, f"{name}_{mode}.png"), DOCS_SOURCE
        ).replace(os.sep, "/")
        lines += [
            f".. image:: /{path}",  # leading '/': relative to docs/source
            f"   :alt: {alt}",
            f"   :class: only-{mode}",
            "   :width: 100%",
            "",
        ]
    return lines


def heading(text, char):
    return [text, char * len(text), ""]


def headline_sentence(records):
    """'trains GNNs 5.6× faster than PyG on MPS and 5.8× faster than PyG
    on CPU (median over 35 benchmarks, up to 58×)' from the results."""
    ratios = {fw: [] for fw in ("pyg-mps", "pyg-cpu")}
    for task in ("node", "graph"):
        data = index(records, task, "dataset", "model")
        for fw in ratios:
            ratios[fw] += list(speedups(data, "epoch_ms", fw).values())
    if not ratios["pyg-mps"]:
        return None
    med = {fw: times_x(float(np.median(v))) for fw, v in ratios.items() if v}
    top = times_x(max(ratios["pyg-mps"] + ratios["pyg-cpu"]))
    return (
        f"mlx-graphs trains GNNs **{med['pyg-mps']} faster than PyG on the "
        f"Mac's GPU (MPS)** and **{med.get('pyg-cpu', '–')} faster than PyG "
        f"on the CPU** (median over {len(ratios['pyg-mps'])} training "
        f"benchmarks, up to {top}), and reaches the same accuracy."
    )


def docs_rst(records, system, fig_dir):
    """The docs section of one machine: the headline figure, then the
    figures of each task. Figures only; the tables are in ``results.md``."""
    library = system.get("library", {})
    version = library.get("version") or system.get("mlx-graphs")
    commit = (library.get("commit") or "")[:7]
    tasks = {r.get("task") for r in records}
    lines = [
        *heading(system["chip"], "-"),
        f"*{system['chip']}, {system['memory_gb']} GB, {system['os']} · "
        f"mlx-graphs {version}"
        + (f" ({commit})" if commit else "")
        + f" · mlx {system['mlx']} · PyTorch {system['torch']} · "
        f"PyG {system['torch_geometric']}*",
        "",
    ]
    sentence = headline_sentence(records)
    if sentence:
        lines += [sentence, ""]
    if tasks & {"node", "graph"}:
        lines += rst_figure(
            fig_dir,
            "headline",
            "Time of PyG on MPS and on CPU relative to mlx-graphs for one "
            "layer, node classification and graph classification, and the "
            "test accuracy of each framework",
        )

    if "layer" in tasks:
        lines += [
            *heading("One layer", "~"),
            "One convolution layer, forward + backward, on random graphs "
            "with 128 input and output features (4 heads for GAT and GATv2).",
            "",
            *rst_figure(
                fig_dir,
                "layers_speedup",
                "Forward and backward time per layer and graph size, relative "
                "to mlx-graphs, for mlx-graphs, PyG on MPS and PyG on CPU",
            ),
            *rst_figure(
                fig_dir,
                "layers_memory",
                "Peak memory per layer on the largest graph, for mlx-graphs, "
                "PyG on MPS and PyG on CPU",
            ),
        ]

    for task, title, setup in (
        (
            "node",
            "Node classification",
            "Full-batch training of 2-layer models (64 hidden units, Adam, "
            "lr 0.01), the whole graph at every step.",
        ),
        (
            "graph",
            "Graph classification",
            "3 layers + mean pooling, batches of 64 graphs, random 80/10/10 "
            "split. The time per epoch includes batching the graphs.",
        ),
    ):
        if task not in tasks:
            continue
        lines += [
            *heading(title, "~"),
            setup,
            "",
            *rst_figure(
                fig_dir,
                f"{task}_classification",
                f"{title}: time per training epoch per dataset and model, for "
                "mlx-graphs, PyG on MPS and PyG on CPU",
            ),
        ]
        if not any("loss_curve" in r for r in records if r.get("task") == task):
            continue
        lines += [
            "**Same training, same results.** The training loss of every "
            "framework at every epoch, overlaid, and the test accuracy "
            "reached at the epoch with the best validation accuracy.",
            "",
            *rst_figure(
                fig_dir,
                f"{task}_loss",
                f"{title}: training loss per epoch for each dataset and model, "
                "mlx-graphs, PyG on MPS and PyG on CPU overlaid",
            ),
            *rst_figure(
                fig_dir,
                f"{task}_accuracy",
                f"{title}: test accuracy per dataset and model, mean and range "
                "over seeds, for mlx-graphs, PyG on MPS and PyG on CPU",
            ),
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", nargs="?", default=None)
    args = parser.parse_args()
    results_dir = (
        args.results_dir
        or sorted(glob.glob(os.path.join(HERE, "results", "*")), key=os.path.getmtime)[
            -1
        ]
    )
    records, system = load(results_dir)
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    tasks = {r.get("task") for r in records}
    for mode, theme in THEMES.items():
        setup_style(theme)
        if "layer" in tasks:
            fig_layers(
                records, system, theme, os.path.join(fig_dir, f"layers_time_{mode}.png")
            )
            fig_layer_speedup(
                records,
                system,
                theme,
                os.path.join(fig_dir, f"layers_speedup_{mode}.png"),
            )
            fig_memory(
                records,
                system,
                theme,
                os.path.join(fig_dir, f"layers_memory_{mode}.png"),
            )
        if tasks & {"node", "graph"}:
            fig_headline(
                records, system, theme, os.path.join(fig_dir, f"headline_{mode}.png")
            )
        for task in ("node", "graph"):
            if task in tasks:
                fig_training(
                    records,
                    system,
                    theme,
                    os.path.join(fig_dir, f"{task}_classification_{mode}.png"),
                    task,
                )
                has_curves = any(
                    "loss_curve" in r for r in records if r.get("task") == task
                )
                if has_curves:
                    for name, draw in (
                        ("loss", fig_convergence),
                        ("accuracy", fig_accuracy),
                    ):
                        path = os.path.join(fig_dir, f"{task}_{name}_{mode}.png")
                        draw(records, system, theme, path, task)

    with open(os.path.join(results_dir, "results.md"), "w") as f:
        f.write(tables(records, system))
    with open(os.path.join(results_dir, "results.rst"), "w") as f:
        f.write(docs_rst(records, system, fig_dir))
    print(f"wrote {fig_dir}, results.md and results.rst into {results_dir}")


if __name__ == "__main__":
    main()
