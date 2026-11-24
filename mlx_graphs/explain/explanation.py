"""The :class:`Explanation` object returned by every explainer."""

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.explain.config import ThresholdConfig


def _apply_threshold(mask: Optional[mx.array], config: ThresholdConfig):
    if mask is None:
        return None
    if config.threshold_type == "hard":
        return mx.where(mask > config.value, mask, mx.zeros_like(mask))

    flat = mask.reshape(-1)
    k = min(config.value, flat.size)
    # indices of the k largest values (argsort is ascending)
    top = mx.argsort(flat)[flat.size - k :]
    keep = mx.zeros(flat.shape, dtype=mx.bool_).at[top].add(True)
    if config.threshold_type == "topk":
        out = mx.where(keep, flat, mx.zeros_like(flat))
    else:  # topk_hard
        out = keep.astype(flat.dtype)
    return out.reshape(mask.shape)


@dataclass
class Explanation:
    """Holds the masks that explain one prediction.

    Attributes:
        node_mask: importance of nodes or node features, shape ``[N, 1]``
            (``"object"``), ``[N, F]`` (``"attributes"``) or ``[1, F]``
            (``"common_attributes"``).
        edge_mask: importance of each edge, shape ``[E]``.
        edge_index: the explained graph's connectivity.
        node_features: the explained graph's node features.
        target: the target that was explained.
        index: the node index (or indices) explained, for node-level tasks.
        prediction: the model output on the full graph.
        extra: any additional algorithm-specific output.
    """

    node_mask: Optional[mx.array] = None
    edge_mask: Optional[mx.array] = None
    edge_index: Optional[mx.array] = None
    node_features: Optional[mx.array] = None
    target: Optional[mx.array] = None
    index: Optional[Union[int, mx.array]] = None
    prediction: Optional[mx.array] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def threshold(
        self, threshold_type: Union[str, ThresholdConfig], value: float = None
    ) -> "Explanation":
        """Returns a copy of this explanation with thresholded masks."""
        config = (
            threshold_type
            if isinstance(threshold_type, ThresholdConfig)
            else ThresholdConfig(threshold_type, value)
        )
        return replace(
            self,
            node_mask=_apply_threshold(self.node_mask, config),
            edge_mask=_apply_threshold(self.edge_mask, config),
        )

    def get_explanation_subgraph(self) -> mx.array:
        """Returns the ``edge_index`` of the edges with a non-zero mask."""
        return self._edges_where(lambda m: m > 0)

    def get_complement_subgraph(self) -> mx.array:
        """Returns the ``edge_index`` of the edges with a zero mask."""
        return self._edges_where(lambda m: m <= 0)

    def _edges_where(self, cond) -> mx.array:
        if self.edge_mask is None or self.edge_index is None:
            raise ValueError("This explanation has no edge mask")
        keep = np.nonzero(np.asarray(cond(self.edge_mask)))[0]
        return self.edge_index[:, mx.array(keep)]

    def visualize_feature_importance(
        self,
        path: Optional[str] = None,
        feat_labels: Optional[list] = None,
        top_k: Optional[int] = None,
    ):
        """Bar plot of node feature importance, summed over all nodes.

        Needs a ``node_mask_type`` of ``"attributes"`` or ``"common_attributes"``.
        Requires ``pip install "mlx-graphs[viz]"``.

        Args:
            path: where to save the figure; shown if ``None``.
            feat_labels: optional name of each feature.
            top_k: only plot the ``top_k`` most important features.
        """
        from mlx_graphs.explain.visualization import visualize_feature_importance

        if self.node_mask is None:
            raise ValueError("This explanation has no node mask")
        return visualize_feature_importance(self.node_mask, path, feat_labels, top_k)

    def visualize_graph(
        self,
        path: Optional[str] = None,
        backend: Optional[str] = None,
        node_labels: Optional[list] = None,
        *,
        highlight_index: bool = True,
        show_node_mask: bool = True,
        edge_labels: bool = False,
    ):
        """Draws the explanation subgraph, with edge opacity given by the edge mask.

        Edges with a mask of 0 are not drawn. Requires
        ``pip install "mlx-graphs[viz]"``.

        Args:
            path: where to save the figure (e.g. ``"explanation.png"``).
            backend: ``"networkx"`` or ``"graphviz"`` (default: graphviz if
                available, else networkx).
            node_labels: optional label for every node.
            highlight_index: draw the explained node(s) (``index``) in red.
            show_node_mask: shade nodes by importance when the node mask has
                one value per node (``"object"``).
            edge_labels: write each edge's mask value on the edge.
        """
        from mlx_graphs.explain.visualization import visualize_graph

        if self.edge_mask is None or self.edge_index is None:
            raise ValueError("This explanation has no edge mask")
        node_weight = None
        if show_node_mask and self.node_mask is not None:
            if self.node_mask.ndim == 2 and self.node_mask.shape[1] == 1:
                node_weight = self.node_mask.reshape(-1)
        return visualize_graph(
            self.edge_index,
            self.edge_mask,
            path=path,
            backend=backend,
            node_labels=node_labels,
            node_weight=node_weight,
            highlight=self.index if highlight_index else None,
            edge_labels=edge_labels,
        )

    def __repr__(self) -> str:
        parts = []
        for name in ("node_mask", "edge_mask", "target", "prediction"):
            value = getattr(self, name)
            if value is not None:
                parts.append(f"{name}={list(value.shape)}")
        if self.index is not None:
            parts.append(f"index={self.index}")
        return f"Explanation({', '.join(parts)})"
