from typing import Union

import mlx.core as mx
import numpy as np

ArrayLike = Union[mx.array, np.ndarray, list]


def _host(array: ArrayLike, dtype) -> np.ndarray:
    out = np.array(array) if isinstance(array, mx.array) else np.asarray(array)
    return out.reshape(-1).astype(dtype)


def roc_auc_score(labels: ArrayLike, scores: ArrayLike) -> float:
    """Area under the ROC curve of binary ``labels`` ranked by ``scores``.

    Computed with the Mann-Whitney U statistic, tied scores getting their
    average rank, which gives the same value as scikit-learn's
    ``roc_auc_score``.

    Args:
        labels: Binary ground truth (``1`` for positives), shape `[n]`
        scores: Scores of the samples (higher means more likely positive),
            shape `[n]`

    Returns:
        The AUC, between 0 and 1.

    Example:

    .. code-block:: python

        from mlx_graphs.metrics import roc_auc_score

        roc_auc_score(mx.array([1, 0, 1, 0]), mx.array([0.9, 0.1, 0.4, 0.5]))
        >>> 0.75
    """
    labels = _host(labels, bool)
    scores = _host(scores, np.float64)
    num_pos = int(labels.sum())
    num_neg = labels.size - num_pos
    if num_pos == 0 or num_neg == 0:
        raise ValueError("roc_auc_score needs both positive and negative samples")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    # average (1-based) rank of each group of tied scores
    _, start, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    avg_rank = start + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg_rank, counts)
    sum_pos_ranks = ranks[labels].sum()
    return float((sum_pos_ranks - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg))


def average_precision_score(labels: ArrayLike, scores: ArrayLike) -> float:
    r"""Average precision of binary ``labels`` ranked by ``scores``.

    .. math::
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    where :math:`P_n` and :math:`R_n` are the precision and recall at the
    :math:`n`-th distinct score threshold, as in scikit-learn's
    ``average_precision_score``.

    Args:
        labels: Binary ground truth (``1`` for positives), shape `[n]`
        scores: Scores of the samples (higher means more likely positive),
            shape `[n]`

    Returns:
        The average precision, between 0 and 1.

    Example:

    .. code-block:: python

        from mlx_graphs.metrics import average_precision_score

        average_precision_score(mx.array([1, 0, 1, 0]),
                                mx.array([0.9, 0.1, 0.4, 0.5]))
        >>> 0.8333
    """
    labels = _host(labels, np.float64)
    scores = _host(scores, np.float64)
    num_pos = labels.sum()
    if num_pos == 0:
        raise ValueError("average_precision_score needs positive samples")

    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    # last position of each distinct threshold
    distinct = np.where(np.diff(scores))[0]
    threshold_idx = np.r_[distinct, labels.size - 1]
    tps = np.cumsum(labels)[threshold_idx]
    precision = tps / (threshold_idx + 1)
    recall = tps / num_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))
