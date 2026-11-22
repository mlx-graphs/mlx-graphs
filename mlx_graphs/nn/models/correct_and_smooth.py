from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.models.label_prop import (
    LabelPropagation,
    _labels_to_one_hot,
    _to_bool_mask,
)


def _scatter_rows(
    base: mx.array, bool_mask: mx.array, rows: mx.array, mask: mx.array
) -> mx.array:
    """Returns ``base`` with ``base[mask] = rows``, where ``rows`` has one row
    per selected node (in node order for a boolean mask)."""
    if mask.dtype == mx.bool_:
        # position of each node among the selected ones
        position = mx.cumsum(bool_mask.astype(mx.int32)) - 1
        position = mx.maximum(position, 0)
        return mx.where(bool_mask[:, None], rows[position], base)
    return base.at[mask.reshape(-1)].add(rows - base[mask.reshape(-1)])


class CorrectAndSmooth(nn.Module):
    r"""The correct and smooth (C&S) post-processing model from the
    `"Combining Label Propagation And Simple Models Out-performs Graph Neural
    Networks" <https://arxiv.org/abs/2010.13993>`_ paper.

    The soft predictions :math:`\mathbf{Z}` of a base predictor are first
    corrected by propagating their residual error on the training nodes
    (:meth:`correct`), then smoothed by propagating the corrected predictions,
    with the ground truth on the training nodes (:meth:`smooth`). Both steps
    use :class:`LabelPropagation`.

    Args:
        num_correction_layers: The number of propagations :math:`L_1`.
        correction_alpha: The :math:`\alpha_1` coefficient.
        num_smoothing_layers: The number of propagations :math:`L_2`.
        smoothing_alpha: The :math:`\alpha_2` coefficient.
        autoscale: If ``True``, determines the scaling factor :math:`\gamma`
            of the correction automatically. Default: ``True``
        scale: The scaling factor :math:`\gamma`, used when
            ``autoscale=False``. Default: ``1.0``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import CorrectAndSmooth

        post = CorrectAndSmooth(
            num_correction_layers=50,
            correction_alpha=1.0,
            num_smoothing_layers=50,
            smoothing_alpha=0.8,
            autoscale=False,
            scale=20.0,
        )
        y_soft = mx.softmax(logits, axis=-1)
        y_soft = post.correct(edge_index, y_soft, y[train_idx], train_idx)
        y_soft = post.smooth(edge_index, y_soft, y[train_idx], train_idx)

    References:
        [1] `Huang et al. Combining Label Propagation And Simple Models
        Out-performs Graph Neural Networks <https://arxiv.org/abs/2010.13993>`_
    """

    def __init__(
        self,
        num_correction_layers: int,
        correction_alpha: float,
        num_smoothing_layers: int,
        smoothing_alpha: float,
        autoscale: bool = True,
        scale: float = 1.0,
    ):
        super().__init__()
        self.autoscale = autoscale
        self.scale = scale
        self.prop1 = LabelPropagation(num_correction_layers, correction_alpha)
        self.prop2 = LabelPropagation(num_smoothing_layers, smoothing_alpha)

    def __call__(
        self,
        edge_index: mx.array,
        y_soft: mx.array,
        y_true: mx.array,
        mask: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Applies :meth:`correct` then :meth:`smooth`."""
        y_soft = self.correct(edge_index, y_soft, y_true, mask, edge_weights)
        return self.smooth(edge_index, y_soft, y_true, mask, edge_weights)

    def _prepare(self, y_soft: mx.array, y_true: mx.array, mask: mx.array):
        bool_mask = _to_bool_mask(mask, y_soft.shape[0])
        numel = int(bool_mask.sum().item()) if mask.dtype == mx.bool_ else mask.size
        if y_true.shape[0] != numel:
            raise ValueError(
                f"`y_true` must have one row per masked node ({numel}), "
                f"got {y_true.shape[0]}"
            )
        y_true = _labels_to_one_hot(y_true, y_soft.shape[-1], y_soft.dtype)
        return bool_mask, numel, y_true

    def correct(
        self,
        edge_index: mx.array,
        y_soft: mx.array,
        y_true: mx.array,
        mask: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Corrects the soft predictions with the propagated residual error.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            y_soft: The soft predictions :math:`\\mathbf{Z}` of a base
                predictor, shape `[num_nodes, num_classes]`
            y_true: The ground truth of the training nodes: integer labels of
                shape `[num_train]` or soft labels of shape
                `[num_train, num_classes]`
            mask: Boolean mask or index array of the training nodes
            edge_weights: Optional weight of each edge. Default: ``None``

        Returns:
            The corrected predictions, shape `[num_nodes, num_classes]`.
        """
        bool_mask, numel, y_true = self._prepare(y_soft, y_true, mask)
        zeros = mx.zeros_like(y_soft)
        # error[mask] = y_true - y_soft[mask], 0 elsewhere
        error = mx.where(
            bool_mask[:, None],
            _scatter_rows(zeros, bool_mask, y_true, mask) - y_soft,
            zeros,
        )

        if self.autoscale:
            smoothed_error = self.prop1(
                edge_index,
                error,
                edge_weights=edge_weights,
                post_step=lambda x: mx.clip(x, -1.0, 1.0),
            )
            sigma = mx.abs(error).sum() / numel
            scale = sigma / mx.abs(smoothed_error).sum(axis=1, keepdims=True)
            scale = mx.where(mx.isinf(scale) | (scale > 1000), 1.0, scale)
            return y_soft + scale * smoothed_error

        def fix_input(x: mx.array) -> mx.array:
            return mx.where(bool_mask[:, None], error, x)

        smoothed_error = self.prop1(
            edge_index, error, edge_weights=edge_weights, post_step=fix_input
        )
        return y_soft + self.scale * smoothed_error

    def smooth(
        self,
        edge_index: mx.array,
        y_soft: mx.array,
        y_true: mx.array,
        mask: mx.array,
        edge_weights: Optional[mx.array] = None,
    ) -> mx.array:
        """Smooths the (corrected) predictions by propagating them, with the
        ground truth on the training nodes.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            y_soft: The corrected predictions from :meth:`correct`, shape
                `[num_nodes, num_classes]`
            y_true: The ground truth of the training nodes: integer labels of
                shape `[num_train]` or soft labels of shape
                `[num_train, num_classes]`
            mask: Boolean mask or index array of the training nodes
            edge_weights: Optional weight of each edge. Default: ``None``

        Returns:
            The final predictions, shape `[num_nodes, num_classes]`.
        """
        bool_mask, _, y_true = self._prepare(y_soft, y_true, mask)
        y_soft = _scatter_rows(y_soft, bool_mask, y_true, mask)
        return self.prop2(edge_index, y_soft, edge_weights=edge_weights)
