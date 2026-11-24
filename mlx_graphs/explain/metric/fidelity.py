"""Fidelity metrics, mirroring ``torch_geometric.explain.metric``.

From `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods
for Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_.
"""

from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

from mlx_graphs.explain.explanation import Explanation


def _masked_inputs(explanation: Explanation, complement: bool):
    x = explanation.node_features
    node_mask, edge_mask = explanation.node_mask, explanation.edge_mask
    if node_mask is not None:
        x = x * ((1 - node_mask) if complement else node_mask)
    ew = None
    if edge_mask is not None:
        ew = (1 - edge_mask) if complement else edge_mask
    return x, ew


def _rows(y: mx.array, index) -> mx.array:
    if index is None:
        return y
    return y[mx.array(index).reshape(-1)]


def fidelity(explainer, explanation: Explanation) -> Tuple[float, float]:
    """Returns ``(fidelity+, fidelity-)`` of an explanation.

    For ``"model"`` explanations, with :math:`\\hat{y}` the original prediction,
    :math:`\\hat{y}^{\\bar{S}}` the prediction with the explanation :math:`S`
    removed and :math:`\\hat{y}^{S}` the prediction with only :math:`S` kept:

    * fidelity+ =
      :math:`1 - \\frac{1}{N}\\sum \\mathbb{1}(\\hat{y} = \\hat{y}^{\\bar{S}})`
      (how much the prediction changes when the explanation is removed;
      higher is better),
    * fidelity- =
      :math:`1 - \\frac{1}{N}\\sum \\mathbb{1}(\\hat{y} = \\hat{y}^{S})`
      (how much it changes when only the explanation is kept; lower is better).

    For ``"phenomenon"`` explanations the ground-truth target is used instead,
    as in PyG. Masks are applied as they are, so threshold them first
    (e.g. ``explanation.threshold("topk_hard", 10)``) for hard subgraphs.
    """
    if explainer.model_config.mode == "regression":
        raise ValueError("Fidelity is only defined for classification")

    edge_index, index = explanation.edge_index, explanation.index
    y = explanation.target
    x_s, ew_s = _masked_inputs(explanation, complement=False)
    x_c, ew_c = _masked_inputs(explanation, complement=True)

    y_s = explainer.get_target(explainer.get_prediction(edge_index, x_s, ew_s))
    y_c = explainer.get_target(explainer.get_prediction(edge_index, x_c, ew_c))

    is_model = explainer.explainer_config.explanation_type == "model"

    if is_model:
        y_hat = explainer.get_target(explanation.prediction)
    else:
        y_hat = y

    if explainer.model_config.task_level == "node":
        y, y_hat, y_s, y_c = (_rows(a, index) for a in (y, y_hat, y_s, y_c))

    if is_model:
        pos = 1.0 - (y_hat == y_c).astype(mx.float32).mean()
        neg = 1.0 - (y_hat == y_s).astype(mx.float32).mean()
    else:
        pos = (y_hat == y).astype(mx.float32) - (y_c == y).astype(mx.float32)
        neg = (y_hat == y).astype(mx.float32) - (y_s == y).astype(mx.float32)
        pos, neg = mx.abs(pos).mean(), mx.abs(neg).mean()
    return float(pos), float(neg)


def characterization_score(
    pos_fidelity: float,
    neg_fidelity: float,
    pos_weight: float = 0.5,
    neg_weight: float = 0.5,
) -> float:
    """Weighted harmonic mean of fidelity+ and ``1 - fidelity-``."""
    if abs(pos_weight + neg_weight - 1.0) > 1e-6:
        raise ValueError("`pos_weight` and `neg_weight` must sum to 1")
    denom = pos_weight / max(pos_fidelity, 1e-12) + neg_weight / max(
        1.0 - neg_fidelity, 1e-12
    )
    return 1.0 / denom


def groundtruth_auc(mask: mx.array, groundtruth: mx.array) -> Optional[float]:
    """ROC AUC of a (soft) mask against a binary ground-truth mask.

    Useful on synthetic benchmarks such as BA-Shapes. Returns ``None`` when
    the ground truth has a single class.
    """
    scores = np.asarray(mask.reshape(-1).astype(mx.float32))
    labels = np.asarray(groundtruth.reshape(-1)).astype(bool)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # rank-based AUC (Mann-Whitney U); tied scores share their average rank
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    rank_sums = np.bincount(inverse, weights=ranks)
    ranks = (rank_sums / counts)[inverse]
    pos_ranks = ranks[labels].sum()
    return float((pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
