"""Link prediction (retrieval) metrics, mirroring ``torch_geometric.metrics``.

All metrics are computed on the host with NumPy (they are not
differentiable) from MLX or NumPy inputs, and :meth:`compute` returns a
scalar ``mx.array``.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import numpy as np

ArrayLike = Union[mx.array, np.ndarray, Sequence]
EdgeLabelIndex = Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]


def _host(array: ArrayLike) -> np.ndarray:
    return np.array(array) if isinstance(array, mx.array) else np.asarray(array)


def _exclusive_cumsum(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(x)])


class LinkPredMetricData:
    """The predictions and ground truth of a mini-batch, with the quantities
    shared by the link prediction metrics.

    Ground-truth entries with a non-positive ``edge_label_weight`` are removed.

    Args:
        pred_index_mat: Top-:math:`k` predictions of every example, shape
            `[batch_size, k]`
        edge_label_index: Ground-truth ``(example, item)`` pairs, shape
            `[2, num_labels]` (or a tuple of two arrays)
        edge_label_weight: Relevance of every ground-truth pair, shape
            `[num_labels]`. Default: ``None``
    """

    def __init__(
        self,
        pred_index_mat: ArrayLike,
        edge_label_index: EdgeLabelIndex,
        edge_label_weight: Optional[ArrayLike] = None,
    ):
        self.pred_index_mat = _host(pred_index_mat).astype(np.int64)
        if self.pred_index_mat.ndim == 1:
            self.pred_index_mat = self.pred_index_mat.reshape(-1, 1)
        if isinstance(edge_label_index, (tuple, list)):
            row, col = edge_label_index
        else:
            row, col = _host(edge_label_index)
        self.row = _host(row).reshape(-1).astype(np.int64)
        self.col = _host(col).reshape(-1).astype(np.int64)
        self.edge_label_weight = None
        if edge_label_weight is not None:
            weight = _host(edge_label_weight).reshape(-1).astype(np.float64)
            pos_mask = weight > 0
            self.edge_label_weight = weight[pos_mask]
            self.row, self.col = self.row[pos_mask], self.col[pos_mask]

    @property
    def pred_rel_mat(self) -> np.ndarray:
        """Relevance of the :math:`k`-th prediction of every example: a boolean
        matrix, or the edge label weights if given."""
        if hasattr(self, "_pred_rel_mat"):
            return self._pred_rel_mat
        pred = self.pred_index_mat
        weighted = self.edge_label_weight is not None
        if self.col.size == 0:
            dtype = np.float64 if weighted else np.bool_
            self._pred_rel_mat = np.zeros(pred.shape, dtype=dtype)
            return self._pred_rel_mat

        # Flatten predictions and ground truth into single ids per example.
        max_index = max(pred.max() if pred.size else 0, self.col.max()) + 1
        offset = (np.arange(pred.shape[0], dtype=np.int64) * max_index)[:, None]
        flat_pred = (pred + offset).reshape(-1)
        flat_label = max_index * self.row + self.col
        perm = np.argsort(flat_label, kind="stable")
        flat_label = flat_label[perm]
        pos = np.searchsorted(flat_label, flat_pred)
        pos = np.minimum(pos, flat_label.size - 1)
        rel = flat_label[pos] == flat_pred
        if weighted:
            rel = np.where(rel, self.edge_label_weight[perm][pos], 0.0)
        self._pred_rel_mat = rel.reshape(pred.shape)
        return self._pred_rel_mat

    @property
    def label_count(self) -> np.ndarray:
        """Number of ground-truth labels of every example."""
        return np.bincount(self.row, minlength=self.pred_index_mat.shape[0])

    @property
    def label_weight_sum(self) -> np.ndarray:
        """Sum of the edge label weights of every example."""
        if self.edge_label_weight is None:
            return self.label_count
        return np.bincount(
            self.row,
            weights=self.edge_label_weight,
            minlength=self.pred_index_mat.shape[0],
        )

    @property
    def edge_label_weight_pos(self) -> Optional[np.ndarray]:
        """Position of every ground-truth label within its example, in
        decreasing order of weight."""
        if self.edge_label_weight is None:
            return None
        perm1 = np.argsort(-self.edge_label_weight, kind="stable")
        perm2 = np.argsort(self.row[perm1], kind="stable")
        perm = perm1[perm2]
        pos = np.empty_like(perm)
        pos[perm] = np.arange(perm.size)
        return pos - _exclusive_cumsum(self.label_count)[self.row]


class _LinkPredMetric:
    """Base class of the link prediction retrieval metrics.

    Args:
        k: The number of top-:math:`k` predictions to evaluate against.
    """

    higher_is_better: Optional[bool] = None
    weighted: bool = False

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError(
                f"'k' needs to be a positive integer in "
                f"'{self.__class__.__name__}' (got {k})"
            )
        self.k = k
        self.reset()

    def update(
        self,
        pred_index_mat: ArrayLike,
        edge_label_index: EdgeLabelIndex,
        edge_label_weight: Optional[ArrayLike] = None,
    ):
        """Accumulates the results of a mini-batch of predictions.

        Args:
            pred_index_mat: Top-:math:`k` predictions of every example, shape
                `[batch_size, k]`
            edge_label_index: Ground-truth ``(example, item)`` pairs, shape
                `[2, num_labels]`
            edge_label_weight: Relevance of every ground-truth pair (positive
                values), shape `[num_labels]`. Required by weighted metrics,
                ignored otherwise. Default: ``None``
        """
        raise NotImplementedError

    def compute(self) -> mx.array:
        """Returns the metric accumulated since the last :meth:`reset`."""
        raise NotImplementedError

    def reset(self):
        """Resets the accumulated state."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"


class LinkPredMetric(_LinkPredMetric):
    """Base class of the link prediction metrics averaged over the examples
    with at least one ground-truth label.

    Args:
        k: The number of top-:math:`k` predictions to evaluate against.
    """

    def update(
        self,
        pred_index_mat: ArrayLike,
        edge_label_index: EdgeLabelIndex,
        edge_label_weight: Optional[ArrayLike] = None,
    ):
        if self.weighted and edge_label_weight is None:
            raise ValueError(
                f"'edge_label_weight' is a required argument for weighted "
                f"'{self.__class__.__name__}' metrics"
            )
        if not self.weighted:
            edge_label_weight = None
        data = LinkPredMetricData(pred_index_mat, edge_label_index, edge_label_weight)
        self._update(data)

    def _update(self, data: LinkPredMetricData):
        metric = self._compute(data)
        self.accum += float(metric.sum())
        self.total += int((data.label_count > 0).sum())

    def compute(self) -> mx.array:
        if self.total == 0:
            return mx.array(0.0)
        return mx.array(self.accum / self.total)

    def reset(self):
        self.accum = 0.0
        self.total = 0

    def _compute(self, data: LinkPredMetricData) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        weighted_repr = ", weighted=True" if self.weighted else ""
        return f"{self.__class__.__name__}(k={self.k}{weighted_repr})"


class LinkPredPrecision(LinkPredMetric):
    """Precision @ :math:`k`: the proportion of the top-:math:`k`
    recommendations that are relevant.

    Args:
        k: The number of top-:math:`k` predictions to evaluate against.

    Example:

    .. code-block:: python

        from mlx_graphs.metrics import LinkPredPrecision

        metric = LinkPredPrecision(k=2)
        pred_index_mat = mx.array([[1, 0], [1, 2]])
        edge_label_index = mx.array([[0, 0, 1], [0, 3, 2]])
        metric.update(pred_index_mat, edge_label_index)
        metric.compute()
        >>> array(0.5, dtype=float32)
    """

    higher_is_better = True

    def _compute(self, data: LinkPredMetricData) -> np.ndarray:
        return data.pred_rel_mat[:, : self.k].sum(axis=-1) / self.k


class LinkPredRecall(LinkPredMetric):
    """Recall @ :math:`k`: the proportion of the relevant items that appear
    in the top-:math:`k` recommendations.

    Args:
        k: The number of top-:math:`k` predictions to evaluate against.
        weighted: Whether to weight the items by ``edge_label_weight``.
            Default: ``False``
    """

    higher_is_better = True

    def __init__(self, k: int, weighted: bool = False):
        super().__init__(k)
        self.weighted = weighted

    def _compute(self, data: LinkPredMetricData) -> np.ndarray:
        rel = data.pred_rel_mat[:, : self.k].sum(axis=-1)
        return rel / np.maximum(data.label_weight_sum, 1e-7)


class LinkPredF1(LinkPredMetric):
    """F1 @ :math:`k`, the harmonic mean of precision and recall @ :math:`k`.

    Args:
        k: The number of top-:math:`k` predictions to evaluate against.
    """

    higher_is_better = True

    def _compute(self, data: LinkPredMetricData) -> np.ndarray:
        isin_count = data.pred_rel_mat[:, : self.k].sum(axis=-1)
        precision = isin_count / self.k
        recall = isin_count / np.maximum(data.label_count, 1e-7)
        return 2 * precision * recall / np.maximum(precision + recall, 1e-7)


class LinkPredMAP(LinkPredMetric):
    """MAP @ :math:`k` (Mean Average Precision), which accounts for the order
    of the relevant items in the top-:math:`k`.

    Args:
        k: The number of top-:math:`k` predictions to evaluate against.
    """

    higher_is_better = True

    def _compute(self, data: LinkPredMetricData) -> np.ndarray:
        rel = data.pred_rel_mat[:, : self.k].astype(np.float64)
        arange = np.arange(1, rel.shape[1] + 1)
        cum_precision = np.cumsum(rel, axis=1) / arange
        count = np.clip(data.label_count, 1e-7, self.k)
        return (cum_precision * rel).sum(axis=-1) / count


class LinkPredNDCG(LinkPredMetric):
    """NDCG @ :math:`k` (Normalized Discounted Cumulative Gain), which gives
    more weight to relevant items ranked higher.

    Args:
        k: The number of top-:math:`k` predictions to evaluate against.
        weighted: Whether the ground-truth items have graded relevances given
            by ``edge_label_weight``. Default: ``False``
    """

    higher_is_better = True

    def __init__(self, k: int, weighted: bool = False):
        super().__init__(k)
        self.weighted = weighted
        self.discount = np.log2(np.arange(2, k + 2, dtype=np.float64))
        self.idcg = None if weighted else _exclusive_cumsum(1.0 / self.discount)

    def _compute(self, data: LinkPredMetricData) -> np.ndarray:
        rel = data.pred_rel_mat[:, : self.k].astype(np.float64)
        dcg = (rel / self.discount[: rel.shape[1]].reshape(1, -1)).sum(axis=-1)
        if not self.weighted:
            idcg = self.idcg[np.minimum(data.label_count, self.k)]
        else:
            discount = np.concatenate([self.discount, [np.inf]])
            discount = discount[np.minimum(data.edge_label_weight_pos, self.k)]
            idcg = np.bincount(
                data.row,
                weights=data.edge_label_weight / discount,
                minlength=data.pred_index_mat.shape[0],
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            out = dcg / idcg
        out[~np.isfinite(out)] = 0.0
        return out


class LinkPredMRR(LinkPredMetric):
    """MRR @ :math:`k` (Mean Reciprocal Rank): the reciprocal rank of the first
    relevant item of the top-:math:`k`, or zero.

    Args:
        k: The number of top-:math:`k` predictions to evaluate against.
    """

    higher_is_better = True

    def _compute(self, data: LinkPredMetricData) -> np.ndarray:
        rel = data.pred_rel_mat[:, : self.k].astype(np.float64)
        if rel.shape[1] == 0:
            return np.zeros(rel.shape[0])
        return (rel / np.arange(1, rel.shape[1] + 1)).max(axis=-1)


class LinkPredHitRatio(LinkPredMetric):
    """Hit ratio @ :math:`k`: the proportion of examples with at least one
    relevant item in the top-:math:`k`.

    Args:
        k: The number of top-:math:`k` predictions to evaluate against.
    """

    higher_is_better = True

    def _compute(self, data: LinkPredMetricData) -> np.ndarray:
        rel = data.pred_rel_mat[:, : self.k]
        if rel.shape[1] == 0:
            return np.zeros(rel.shape[0])
        return rel.max(axis=-1).astype(np.float64)


class LinkPredCoverage(_LinkPredMetric):
    """Coverage @ :math:`k`: the proportion of the destination nodes that
    appear in at least one top-:math:`k` recommendation.

    Args:
        k: The number of top-:math:`k` predictions to evaluate against.
        num_dst_nodes: The total number of destination nodes.
    """

    higher_is_better = True

    def __init__(self, k: int, num_dst_nodes: int):
        self.num_dst_nodes = num_dst_nodes
        super().__init__(k)

    def update(self, pred_index_mat, edge_label_index=None, edge_label_weight=None):
        pred = _host(pred_index_mat).astype(np.int64)
        self.mask[pred[:, : self.k].reshape(-1)] = True

    def compute(self) -> mx.array:
        return mx.array(float(self.mask.mean()))

    def reset(self):
        self.mask = np.zeros(self.num_dst_nodes, dtype=bool)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(k={self.k}, "
            f"num_dst_nodes={self.num_dst_nodes})"
        )


class LinkPredDiversity(_LinkPredMetric):
    r"""Diversity @ :math:`k` of the recommendations according to item
    categories,

    .. math::
        div_{u@k} = 1 - \left( \frac{1}{k \cdot (k-1)} \right) \sum_{i \neq j}
        sim(i, j)

    where :math:`sim(i, j)` is 1 if items :math:`i` and :math:`j` share their
    category and 0 otherwise.

    Args:
        k: The number of top-:math:`k` predictions to evaluate against.
        category: The category of every destination node.
    """

    higher_is_better = True

    def __init__(self, k: int, category: ArrayLike):
        self.category = _host(category)
        super().__init__(k)

    def update(self, pred_index_mat, edge_label_index=None, edge_label_weight=None):
        pred = _host(pred_index_mat).astype(np.int64)
        category = self.category[pred[:, : self.k]]
        sim = (category[:, None, :] == category[:, :, None]).sum(axis=-1)
        div = 1 - 1 / (self.k * (self.k - 1)) * (sim - 1).sum(axis=-1)
        self.accum += float(div.sum())
        self.total += pred.shape[0]

    def compute(self) -> mx.array:
        if self.total == 0:
            return mx.array(0.0)
        return mx.array(self.accum / self.total)

    def reset(self):
        self.accum = 0.0
        self.total = 0


class LinkPredPersonalization(_LinkPredMetric):
    """Personalization @ :math:`k`: the dissimilarity of the recommendations
    of different examples (the average of ``1 - overlap / k`` over all pairs
    of examples).

    Args:
        k: The number of top-:math:`k` predictions to evaluate against.
        max_src_nodes: Maximum number of examples kept to compute the pairwise
            dissimilarity (``None`` for all). Default: ``2**12``
        batch_size: Number of pairs processed at once. Default: ``2**16``
    """

    higher_is_better = True

    def __init__(
        self, k: int, max_src_nodes: Optional[int] = 2**12, batch_size: int = 2**16
    ):
        self.max_src_nodes = max_src_nodes
        self.batch_size = batch_size
        super().__init__(k)

    def update(self, pred_index_mat, edge_label_index=None, edge_label_weight=None):
        pred = _host(pred_index_mat).astype(np.int64)[:, : self.k]
        if self.max_src_nodes is not None:
            pred = pred[: max(self.max_src_nodes - self.total, 0)]
        self.preds.append(pred)
        self.total += pred.shape[0]

    def compute(self) -> mx.array:
        if not self.preds:
            return mx.array(0.0)
        pred = np.concatenate(self.preds, axis=0)
        n = pred.shape[0]
        if n < 2:
            return mx.array(0.0)
        row, col = np.triu_indices(n, k=1)
        score, total = 0.0, 0
        for start in range(0, row.size, self.batch_size):
            left = pred[row[start : start + self.batch_size]]
            right = pred[col[start : start + self.batch_size]]
            # Offsets to apply `isin` row by row.
            i = int(max(left.max(), right.max())) + 1
            offset = (np.arange(left.shape[0], dtype=np.int64) * i)[:, None]
            isin = np.isin(left + offset, right + offset)
            cos = isin.sum(axis=-1) / pred.shape[1]
            score += float((1 - cos).sum())
            total += cos.size
        return mx.array(score / total)

    def reset(self):
        self.preds: List[np.ndarray] = []
        self.total = 0


class LinkPredAveragePopularity(_LinkPredMetric):
    """Average Recommendation Popularity @ :math:`k`: the average popularity
    of the recommended items.

    Args:
        k: The number of top-:math:`k` predictions to evaluate against.
        popularity: The popularity of every item, e.g. its number of ratings in
            the training set.
    """

    higher_is_better = False

    def __init__(self, k: int, popularity: ArrayLike):
        self.popularity = _host(popularity)
        super().__init__(k)

    def update(self, pred_index_mat, edge_label_index=None, edge_label_weight=None):
        pred = _host(pred_index_mat).astype(np.int64)[:, : self.k]
        popularity = self.popularity[pred].astype(np.float64).mean(axis=-1)
        self.accum += float(popularity.sum())
        self.total += popularity.size

    def compute(self) -> mx.array:
        if self.total == 0:
            return mx.array(0.0)
        return mx.array(self.accum / self.total)

    def reset(self):
        self.accum = 0.0
        self.total = 0


class LinkPredMetricCollection:
    """A collection of link prediction metrics updated together.

    Args:
        metrics: A list of metrics (named ``"<Name>@<k>"``, with a
            ``"Weighted"`` prefix for weighted metrics) or a dictionary of
            named metrics.

    Example:

    .. code-block:: python

        from mlx_graphs.metrics import (
            LinkPredMAP,
            LinkPredMetricCollection,
            LinkPredPrecision,
            LinkPredRecall,
        )

        metrics = LinkPredMetricCollection(
            [
                LinkPredMAP(k=10),
                LinkPredPrecision(k=100),
                LinkPredRecall(k=50),
            ]
        )
        metrics.update(pred_index_mat, edge_label_index)
        out = metrics.compute()  # {"LinkPredMAP@10": ..., ...}
        metrics.reset()
    """

    def __init__(
        self, metrics: Union[List[_LinkPredMetric], Dict[str, _LinkPredMetric]]
    ):
        if isinstance(metrics, (list, tuple)):
            metrics = {
                f"{'Weighted' if getattr(m, 'weighted', False) else ''}"
                f"{m.__class__.__name__}@{m.k}": m
                for m in metrics
            }
        if not isinstance(metrics, dict) or len(metrics) == 0:
            raise ValueError("Expected a non-empty list or dict of metrics")
        for metric in metrics.values():
            if not isinstance(metric, _LinkPredMetric):
                raise TypeError(f"'{metric}' is not a link prediction metric")
        self.metrics: Dict[str, _LinkPredMetric] = dict(metrics)

    def __getitem__(self, name: str) -> _LinkPredMetric:
        return self.metrics[name]

    def __len__(self) -> int:
        return len(self.metrics)

    def keys(self):
        return self.metrics.keys()

    def values(self):
        return self.metrics.values()

    def items(self):
        return self.metrics.items()

    @property
    def max_k(self) -> int:
        """The largest :math:`k` of the metrics."""
        return max(metric.k for metric in self.metrics.values())

    @property
    def weighted(self) -> bool:
        """Whether the collection has a weighted metric."""
        return any(getattr(m, "weighted", False) for m in self.metrics.values())

    def update(
        self,
        pred_index_mat: ArrayLike,
        edge_label_index: EdgeLabelIndex,
        edge_label_weight: Optional[ArrayLike] = None,
    ):
        """Updates all metrics with a mini-batch of predictions (see
        :meth:`LinkPredMetric.update`). Ground-truth pairs with non-positive
        weights are dropped for all metrics, as in PyG."""
        if self.weighted and edge_label_weight is None:
            raise ValueError(
                f"'edge_label_weight' is a required argument for weighted "
                f"'{self.__class__.__name__}' metrics"
            )
        data = LinkPredMetricData(pred_index_mat, edge_label_index, edge_label_weight)
        unweighted = LinkPredMetricData(data.pred_index_mat, (data.row, data.col))
        for metric in self.metrics.values():
            if isinstance(metric, LinkPredMetric):
                metric._update(data if metric.weighted else unweighted)
            else:
                metric.update(pred_index_mat, edge_label_index, edge_label_weight)

    def compute(self) -> Dict[str, mx.array]:
        """Returns the value of every metric."""
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset(self):
        """Resets all metrics."""
        for metric in self.metrics.values():
            metric.reset()

    def __repr__(self) -> str:
        names = [f"  {name}: {metric},\n" for name, metric in self.metrics.items()]
        return f"{self.__class__.__name__}([\n{''.join(names)}])"
