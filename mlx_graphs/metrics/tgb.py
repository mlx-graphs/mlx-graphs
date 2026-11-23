"""Evaluation of the Temporal Graph Benchmark (TGB), as the official ``py-tgb``
package (``tgb.linkproppred.evaluate.Evaluator`` and
``tgb.nodeproppred.evaluate.Evaluator``)."""

import math
from typing import Any, Union

import mlx.core as mx
import numpy as np

ArrayLike = Union[mx.array, np.ndarray]

#: Official evaluation metric of every TGB dataset (``tgb.utils.info``).
TGB_EVAL_METRICS = {
    "tgbl-wiki": "mrr",
    "tgbl-review": "mrr",
    "tgbl-coin": "mrr",
    "tgbl-comment": "mrr",
    "tgbl-flight": "mrr",
    "tgbl-enron": "mrr",
    "tgbl-uci": "mrr",
    "tgbl-subreddit": "mrr",
    "tgbl-lastfm": "mrr",
    "tkgl-polecat": "mrr",
    "tkgl-icews": "mrr",
    "tkgl-yago": "mrr",
    "tkgl-wikidata": "mrr",
    "tkgl-smallpedia": "mrr",
    "thgl-myket": "mrr",
    "thgl-github": "mrr",
    "thgl-forum": "mrr",
    "thgl-software": "mrr",
    "tgbn-trade": "ndcg",
    "tgbn-genre": "ndcg",
    "tgbn-reddit": "ndcg",
    "tgbn-token": "ndcg",
}


def _host(array: Any) -> np.ndarray:
    if isinstance(array, mx.array):
        return np.array(array)
    if isinstance(array, np.ndarray):
        return array
    # torch tensors and anything else array-like
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def tgb_mrr_hits(
    y_pred_pos: ArrayLike, y_pred_neg: ArrayLike, k: int = 10
) -> tuple[float, float]:
    r"""MRR and Hits@k of positive scores ranked against their negative scores,
    as TGB.

    The rank of a positive is ``1 + (#negatives scoring higher + #negatives
    scoring at least as high) / 2``: a positive tied with ``m`` negatives gets
    the average of its best (first) and worst (last) possible ranks among them,
    as TGB (and OGB). The reciprocal ranks and the hits are float32 and
    averaged in float32, as TGB with NumPy inputs.

    Args:
        y_pred_pos: Score of each positive edge, shape `[num_pos]`
        y_pred_neg: Scores of the negative edges of each positive, shape
            `[num_pos, num_neg]` (or `[num_neg]` for a single positive).
            Positives with fewer negatives can be padded with ``-inf``, which
            never ranks above a (finite) positive score.
        k: The ``k`` of Hits@k. Default: ``10``

    Returns:
        The mean reciprocal rank and the Hits@k.
    """
    y_pred_pos = _host(y_pred_pos).reshape(-1, 1)
    y_pred_neg = _host(y_pred_neg)
    optimistic_rank = (y_pred_neg > y_pred_pos).sum(axis=1)
    pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(axis=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    hits = (ranking_list <= k).astype(np.float32)
    mrr = 1.0 / ranking_list.astype(np.float32)
    return float(mrr.mean()), float(hits.mean())


def _tie_averaged_dcg(
    y_true: np.ndarray, y_score: np.ndarray, discount_cumsum: np.ndarray
) -> float:
    # Every item of a group of tied scores gets the average gain of the group
    # (McSherry & Najork, 2008), as sklearn's `_tie_averaged_dcg`.
    _, inv, counts = np.unique(-y_score, return_inverse=True, return_counts=True)
    ranked = np.zeros(len(counts))
    np.add.at(ranked, inv, y_true)
    ranked /= counts
    groups = np.cumsum(counts) - 1
    discount_sums = np.empty(len(counts))
    discount_sums[0] = discount_cumsum[groups[0]]
    discount_sums[1:] = np.diff(discount_cumsum[groups])
    return (ranked * discount_sums).sum()


def tgb_ndcg(y_true: ArrayLike, y_pred: ArrayLike, k: int = 10) -> float:
    r"""NDCG@k of the predicted scores of each node over the classes, averaged
    over the nodes, as TGB (``sklearn.metrics.ndcg_score(y_true, y_pred,
    k=k)``).

    The gains are the true label weights and the discount of position ``i``
    (from 0) is ``1 / log2(i + 2)`` for the first ``k`` positions and 0
    after. Tied predicted scores share the average gain of their group, and
    nodes whose true labels are all zero score 0.

    Args:
        y_true: Non-negative true label weights, shape `[num_nodes, num_classes]`
        y_pred: Predicted scores, shape `[num_nodes, num_classes]`
        k: Only the first ``k`` ranked classes count. Default: ``10``

    Returns:
        The mean NDCG@k.
    """
    y_true = _host(y_true)
    y_pred = _host(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape (got {y_true.shape} "
            f"and {y_pred.shape})"
        )
    if y_true.ndim != 2 or y_true.shape[1] <= 1:
        raise ValueError(
            "Computing NDCG is only meaningful with a 2D array of more than one "
            f"class per node (got shape {y_true.shape})"
        )
    if y_true.size and y_true.min() < 0:
        raise ValueError("ndcg only supports non-negative y_true values")

    discount = 1 / (np.log(np.arange(y_true.shape[1]) + 2) / np.log(2))
    discount[k:] = 0

    discount_cumsum = np.cumsum(discount)
    gain = np.asarray(
        [
            _tie_averaged_dcg(y_t, y_s, discount_cumsum)
            for y_t, y_s in zip(y_true, y_pred)
        ]
    )
    ranking = np.argsort(y_true)[:, ::-1]
    ranked = y_true[np.arange(ranking.shape[0])[:, np.newaxis], ranking]
    normalizing_gain = discount.dot(ranked.T)

    all_irrelevant = normalizing_gain == 0
    gain[all_irrelevant] = 0
    gain[~all_irrelevant] /= normalizing_gain[~all_irrelevant]
    return float(np.average(gain))


class TGBEvaluator:
    r"""The evaluator of a Temporal Graph Benchmark dataset, with the same
    interface and results as ``py-tgb``'s ``Evaluator`` classes.

    - Link property prediction (``tgbl-*``, ``thgl-*``, ``tkgl-*``): each
      positive edge is ranked against its negative edges (one-vs-many) and
      the metrics are the mean reciprocal rank ``"mrr"`` and ``"hits@k"``
      (see :func:`tgb_mrr_hits`). Inputs: ``"y_pred_pos"`` of shape
      `[num_pos]` and ``"y_pred_neg"`` of shape `[num_pos, num_neg]`.
    - Node property prediction (``tgbn-*``): ``"ndcg"`` is the NDCG@10 of the
      predicted class scores (see :func:`tgb_ndcg`), ``"mse"`` also returns
      ``"rmse"``. Inputs: ``"y_true"`` and ``"y_pred"`` of shape
      `[num_nodes, num_classes]`.

    Inputs can be MLX or NumPy arrays (or torch tensors).

    Args:
        name: Name of the TGB dataset, e.g. ``"tgbl-wiki"``
        k_value: The ``k`` of ``"hits@k"`` for link prediction. Default: ``10``

    Example:

    .. code-block:: python

        from mlx_graphs.metrics import TGBEvaluator

        evaluator = TGBEvaluator("tgbl-wiki")
        evaluator.eval(
            {"y_pred_pos": pos_scores, "y_pred_neg": neg_scores,
             "eval_metric": ["mrr"]}
        )
        >>> {"hits@10": 0.83, "mrr": 0.61}

    References:
        - Huang et al., `Temporal Graph Benchmark for Machine Learning on
          Temporal Graphs <https://arxiv.org/abs/2307.01026>`_
        - `py-tgb <https://github.com/shenyangHuang/TGB>`_
    """

    def __init__(self, name: str, k_value: int = 10):
        if name not in TGB_EVAL_METRICS:
            raise NotImplementedError(
                f"Unknown TGB dataset '{name}', available: {list(TGB_EVAL_METRICS)}"
            )
        self.name = name
        self.k_value = k_value
        self.metric = TGB_EVAL_METRICS[name]
        if self.metric == "mrr":
            self.valid_metric_list = ["hits@", "mrr"]
        else:
            self.valid_metric_list = ["mse", "rmse", "ndcg"]

    def eval(self, input_dict: dict, verbose: bool = False) -> dict:
        """Computes the metrics.

        Args:
            input_dict: The predictions (see the class description) and
                ``"eval_metric"``, a list of metric names
            verbose: Whether to print the metrics

        Returns:
            A dictionary of metric values (floats).
        """
        if "eval_metric" not in input_dict:
            raise RuntimeError("Missing key of eval_metric!")
        for metric in input_dict["eval_metric"]:
            if metric not in self.valid_metric_list:
                raise ValueError(
                    f"Unsupported eval metric: {metric}, not found in "
                    f"{self.valid_metric_list}"
                )

        if self.metric == "mrr":
            for key in ("y_pred_pos", "y_pred_neg"):
                if key not in input_dict:
                    raise RuntimeError(f"Missing key of {key}")
            mrr, hits = tgb_mrr_hits(
                input_dict["y_pred_pos"], input_dict["y_pred_neg"], self.k_value
            )
            perf = {f"hits@{self.k_value}": hits, "mrr": mrr}
        else:
            for key in ("y_true", "y_pred"):
                if key not in input_dict:
                    raise RuntimeError(f"Missing key of {key}")
            y_true = _host(input_dict["y_true"])
            y_pred = _host(input_dict["y_pred"])
            if y_true.shape != y_pred.shape:
                raise RuntimeError("Shape of y_true and y_pred must be the same!")
            perf = {}
            for metric in input_dict["eval_metric"]:
                if metric in ("mse", "rmse"):
                    mse = float(np.mean((y_true - y_pred) ** 2))
                    perf.update(mse=mse, rmse=math.sqrt(mse))
                elif metric == "ndcg":
                    perf["ndcg"] = tgb_ndcg(y_true, y_pred, k=10)

        if verbose:
            for key, value in perf.items():
                print(f"\t>>> {key}: {value:.4f}")
        return perf

    def __repr__(self) -> str:
        return f"TGBEvaluator(name={self.name}, metric={self.metric})"
