"""Plotting helpers, mirroring ``torch_geometric.visualization``.

Requires the optional ``viz`` extra: ``pip install "mlx-graphs[viz]"``
(matplotlib and networkx). The graphviz backend additionally needs the
``graphviz`` Python package and the Graphviz program (``brew install graphviz``).
"""

import inspect
from collections.abc import Sequence
from typing import Any, Optional, Union

import mlx.core as mx
import numpy as np

BACKENDS = {"graphviz", "networkx"}
_HIGHLIGHT = "#d62728"


def _require(module: str):
    try:
        return __import__(module)
    except ImportError as e:
        raise ImportError(
            f"Visualization requires `{module}`. Install the optional extra with "
            '`pip install "mlx-graphs[viz]"`.'
        ) from e


def has_graphviz() -> bool:
    """Whether the graphviz Python package and Graphviz program are available."""
    try:
        import graphviz
    except ImportError:
        return False
    try:
        graphviz.Digraph().pipe()
    except graphviz.backend.ExecutableNotFound:
        return False
    return True


def _normalize(w: np.ndarray) -> np.ndarray:
    w = w.astype(np.float64)
    w = w - w.min()
    return w / w.max() if w.max() > 0 else np.ones_like(w)


def _as_list(index) -> list[int]:
    if index is None:
        return []
    return np.asarray(index).reshape(-1).astype(int).tolist()


def visualize_graph(
    edge_index: mx.array,
    edge_weight: Optional[mx.array] = None,
    path: Optional[str] = None,
    backend: Optional[str] = None,
    node_labels: Optional[Sequence[str]] = None,
    node_weight: Optional[mx.array] = None,
    highlight: Optional[Union[int, Sequence[int]]] = None,
    edge_labels: bool = False,
) -> Any:
    """Draws a graph with edge opacity given by ``edge_weight``.

    Edge weights are rescaled to ``[0, 1]`` and edges with a weight of 0 are
    not drawn, as in PyG.

    Args:
        edge_index: graph connectivity, shape ``[2, E]``.
        edge_weight: optional importance of each edge, shape ``[E]``.
        path: where to save the figure (e.g. ``"explanation.png"``). If
            ``None``, the figure is shown (networkx) or the graphviz object is
            returned (displayed inline in notebooks).
        backend: ``"networkx"`` or ``"graphviz"``. Default: graphviz if
            available, else networkx.
        node_labels: optional label for every node of the graph.
        node_weight: optional importance of each node, shape ``[N]``, used to
            shade nodes.
        highlight: node index (or indices) to draw in red, e.g. the explained
            node.
        edge_labels: write each drawn edge's (original, unscaled) weight on
            the edge, e.g. ``0.86``.

    Returns:
        The matplotlib ``Axes`` (networkx) or the ``graphviz.Digraph``.
    """
    ei = np.asarray(edge_index).astype(int)
    raw = (
        np.ones(ei.shape[1])
        if edge_weight is None
        else np.asarray(edge_weight).reshape(-1).astype(np.float64)
    )
    w = _normalize(raw) if edge_weight is not None else raw
    keep = w > 1e-7
    ei, w, raw = ei[:, keep], w[keep], raw[keep]
    labels = [f"{v:.2f}" for v in raw] if edge_labels else None

    highlight = _as_list(highlight)
    nodes = sorted(set(ei.reshape(-1).tolist()) | set(highlight))
    nw = None
    if node_weight is not None:
        nw = _normalize(np.asarray(node_weight).reshape(-1))

    if backend is None:
        backend = "graphviz" if has_graphviz() else "networkx"
    backend = backend.lower()
    if backend not in BACKENDS:
        raise ValueError(
            f"`backend` must be one of {sorted(BACKENDS)} (got {backend!r})"
        )

    draw = _draw_graphviz if backend == "graphviz" else _draw_networkx
    return draw(ei, w, nodes, path, node_labels, nw, set(highlight), labels)


def _label(node: int, node_labels) -> str:
    return str(node_labels[node]) if node_labels is not None else str(node)


def _draw_networkx(ei, w, nodes, path, node_labels, nw, highlight, edge_labels):
    _require("matplotlib")
    nx = _require("networkx")
    import matplotlib.pyplot as plt

    edges = list(zip(ei[0].tolist(), ei[1].tolist(), strict=True))
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(g) if len(nodes) > 1 else {nodes[0]: (0, 0)}

    fig, ax = plt.subplots(figsize=(7, 7))
    edge_colors = [(0.0, 0.0, 0.0, float(a)) for a in w]
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=edges,  # same order as the weights
        edge_color=edge_colors,
        arrows=True,
        arrowsize=12,
        node_size=1100,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    fill = []
    for n in nodes:
        shade = 1.0 - 0.75 * (nw[n] if nw is not None and n < len(nw) else 0.0)
        fill.append((shade, shade, 1.0))
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=nodes,
        node_color=fill,
        edgecolors=[_HIGHLIGHT if n in highlight else "black" for n in nodes],
        linewidths=[3.0 if n in highlight else 1.0 for n in nodes],
        node_size=1100,
        ax=ax,
    )
    nx.draw_networkx_labels(
        g, pos, labels={n: _label(n, node_labels) for n in nodes}, font_size=8, ax=ax
    )
    if edge_labels is not None:
        kwargs = {}
        # networkx >= 3.3 can place labels on curved edges
        if (
            "connectionstyle"
            in inspect.signature(nx.draw_networkx_edge_labels).parameters
        ):
            kwargs = {"connectionstyle": "arc3,rad=0.1", "node_size": 1100}
        nx.draw_networkx_edge_labels(
            g,
            pos,
            edge_labels=dict(zip(edges, edge_labels, strict=True)),
            font_size=7,
            label_pos=0.4,
            rotate=False,  # keep values horizontal and readable
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.8,
            },
            ax=ax,
            **kwargs,
        )
    ax.set_axis_off()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return ax


def _draw_graphviz(ei, w, nodes, path, node_labels, nw, highlight, edge_labels):
    graphviz = _require("graphviz")

    g = graphviz.Digraph("explanation", graph_attr={"overlap": "false"})
    g.attr("node", shape="circle", style="filled", fontsize="11")
    for n in nodes:
        level = nw[n] if nw is not None and n < len(nw) else 0.0
        shade = int(255 * (1.0 - 0.75 * level))
        attrs = {"fillcolor": f"#{shade:02x}{shade:02x}ff"}
        if n in highlight:
            attrs.update(color=_HIGHLIGHT, penwidth="3")
        g.node(str(n), _label(n, node_labels), **attrs)
    for i, ((src, dst), a) in enumerate(zip(ei.T.tolist(), w.tolist(), strict=True)):
        attrs = {"color": f"#000000{int(255 * a):02x}"}
        if edge_labels is not None:
            attrs.update(label=edge_labels[i], fontsize="9")
        g.edge(str(src), str(dst), **attrs)

    if path is not None:
        stem, _, fmt = path.rpartition(".")
        if not stem:
            stem, fmt = path, "pdf"
        g.render(stem, format=fmt, cleanup=True)
    return g


def visualize_feature_importance(
    node_mask: mx.array,
    path: Optional[str] = None,
    feat_labels: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
) -> Any:
    """Horizontal bar plot of feature importance, summed over all nodes.

    Args:
        node_mask: a ``[N, F]`` (``"attributes"``) or ``[1, F]``
            (``"common_attributes"``) node mask.
        path: where to save the figure; shown if ``None``.
        feat_labels: optional name of each feature.
        top_k: only plot the ``top_k`` most important features.

    Returns:
        The matplotlib ``Axes``.
    """
    _require("matplotlib")
    import matplotlib.pyplot as plt

    mask = np.asarray(node_mask)
    if mask.ndim != 2 or mask.shape[1] <= 1:
        raise ValueError(
            "Feature importance needs a node mask with one value per feature "
            f"('attributes' or 'common_attributes'), got shape {mask.shape}"
        )
    score = mask.sum(axis=0)
    labels = list(feat_labels) if feat_labels is not None else list(range(len(score)))
    if len(labels) != len(score):
        raise ValueError(f"Got {len(labels)} feature labels for {len(score)} features")

    order = np.argsort(-score)
    if top_k is not None:
        order = order[:top_k]
    title = (
        f"Feature importance for top {len(order)} features"
        if top_k is not None
        else f"Feature importance for {len(order)} features"
    )

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(order) + 1)))
    ax.barh([str(labels[i]) for i in order][::-1], score[order][::-1], color="#4c72b0")
    ax.set_title(title)
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature label")
    shown = score[order]
    # signed attributions (e.g. IntegratedGradients) can be negative
    ax.set_xlim(
        min(0.0, float(shown.min()) * 1.1), max(float(shown.max()) * 1.1, 1e-12)
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return ax
