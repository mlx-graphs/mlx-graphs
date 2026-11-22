from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv._norm import cached_gcn_norm
from mlx_graphs.nn.conv.simple_conv import SimpleConv


def _to_bool_mask(mask: mx.array, num_nodes: int) -> mx.array:
    """Turns a boolean mask or an index array into a boolean mask."""
    if mask.dtype == mx.bool_:
        return mask.reshape(-1)
    bool_mask = mx.zeros((num_nodes,), dtype=mx.bool_)
    bool_mask[mask.reshape(-1)] = True
    return bool_mask


def _labels_to_one_hot(
    y: mx.array, num_classes: Optional[int] = None, dtype: mx.Dtype = mx.float32
) -> mx.array:
    """One-hot encodes integer labels of shape `[n]` (or `[n, 1]`); returns
    other inputs unchanged."""
    if mx.issubdtype(y.dtype, mx.integer) and y.size == y.shape[0]:
        y = y.reshape(-1)
        if num_classes is None:
            num_classes = int(y.max().item()) + 1
        return (y[:, None] == mx.arange(num_classes)[None, :]).astype(dtype)
    return y


class LabelPropagation(nn.Module):
    r"""The label propagation operator from the `"Learning from Labeled and
    Unlabeled Data with Label Propagation"
    <http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf>`_ paper,
    in the form used by `"Combining Label Propagation And Simple Models
    Out-performs Graph Neural Networks" <https://arxiv.org/abs/2010.13993>`_.

    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y}

    The layer has no parameters and no gradient flows through it.

    Args:
        num_layers: The number of propagations.
        alpha: The :math:`\alpha` coefficient.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import LabelPropagation

        edge_index = mx.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        labels = mx.array([0, 0, 1, 1])
        train_mask = mx.array([True, False, False, True])

        model = LabelPropagation(num_layers=3, alpha=0.9)
        out = model(edge_index, labels, mask=train_mask)
        >>> out.shape
        (4, 2)

    References:
        [1] `Huang et al. Combining Label Propagation And Simple Models
        Out-performs Graph Neural Networks <https://arxiv.org/abs/2010.13993>`_
    """

    def __init__(self, num_layers: int, alpha: float):
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.conv = SimpleConv(aggr="add")

    def __call__(
        self,
        edge_index: mx.array,
        y: mx.array,
        mask: Optional[mx.array] = None,
        edge_weights: Optional[mx.array] = None,
        post_step: Optional[Callable[[mx.array], mx.array]] = None,
    ) -> mx.array:
        """Propagates the labels.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            y: The label information :math:`\\mathbf{Y}`: integer class labels
                of shape `[num_nodes]` (one-hot encoded) or soft labels of
                shape `[num_nodes, num_classes]`
            mask: Boolean mask or index array of the nodes whose labels are
                propagated; the other nodes start from zero. Default: ``None``
                (all nodes)
            edge_weights: Optional weight of each edge. If given, it is used
                as is, without the symmetric normalization. Default: ``None``
            post_step: Function applied after each propagation. Default:
                ``None`` (clamp to ``[0, 1]``)

        Returns:
            The propagated labels, shape `[num_nodes, num_classes]`.
        """
        y = _labels_to_one_hot(y)
        num_nodes = y.shape[0]

        out = y
        if mask is not None:
            bool_mask = _to_bool_mask(mask, num_nodes)
            out = mx.where(bool_mask[:, None], y, mx.zeros_like(y))

        if edge_weights is None:
            edge_index, edge_weights = cached_gcn_norm(
                edge_index, num_nodes, add_self_loops=False
            )

        res = (1 - self.alpha) * out
        for _ in range(self.num_layers):
            out = self.conv(edge_index, out, edge_weights)
            out = self.alpha * out + res
            if post_step is not None:
                out = post_step(out)
            else:
                out = mx.clip(out, 0.0, 1.0)
        return mx.stop_gradient(out)
