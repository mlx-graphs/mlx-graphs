from typing import Literal

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.models.label_prop import _to_bool_mask


class MaskLabel(nn.Module):
    r"""The label embedding and masking layer from the `"Masked Label
    Prediction: Unified Message Passing Model for Semi-Supervised
    Classification" <https://arxiv.org/abs/2009.03509>`_ paper.

    The labels ``y`` of a subset of the nodes, given by ``mask``, are embedded
    and merged into the node features.

    Args:
        num_classes: The number of classes.
        out_features_dim: Size of the label embeddings. With ``method="add"``,
            it must be the size of the node features.
        method: ``"add"`` adds the label embeddings to the node features,
            ``"concat"`` concatenates them (zeros for the unmasked nodes).
            Default: ``"add"``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import MaskLabel

        mask_label = MaskLabel(num_classes=3, out_features_dim=16)
        x = mx.random.normal((5, 16))
        y = mx.array([0, 1, 2, 1, 0])
        train_mask = mx.array([True, True, False, True, False])

        # hide a random half of the training labels at each epoch
        label_mask = MaskLabel.ratio_mask(train_mask, ratio=0.5)
        >>> mask_label(x, y, label_mask).shape
        (5, 16)

    References:
        [1] `Shi et al. Masked Label Prediction: Unified Message Passing Model
        for Semi-Supervised Classification <https://arxiv.org/abs/2009.03509>`_
    """

    def __init__(
        self,
        num_classes: int,
        out_features_dim: int,
        method: Literal["add", "concat"] = "add",
    ):
        super().__init__()
        if method not in ("add", "concat"):
            raise ValueError(
                f"'method' must be either 'add' or 'concat' (got '{method}')"
            )
        self.method = method
        self.emb = nn.Embedding(num_classes, out_features_dim)
        # same initialization as torch.nn.Embedding
        self.emb.weight = mx.random.normal(self.emb.weight.shape)

    def __call__(self, x: mx.array, y: mx.array, mask: mx.array) -> mx.array:
        """Merges the label embeddings of the masked nodes into ``x``.

        Args:
            x: Node features, shape `[num_nodes, dim]`
            y: Integer labels of all the nodes, shape `[num_nodes]`. Labels
                of unmasked nodes are ignored.
            mask: Boolean mask or index array of the nodes whose label is used

        Returns:
            Node features of shape `[num_nodes, dim]` (``"add"``) or
            `[num_nodes, dim + out_features_dim]` (``"concat"``).
        """
        bool_mask = _to_bool_mask(mask, x.shape[0])[:, None]
        y = mx.where(bool_mask[:, 0], y.reshape(-1), 0)
        emb = mx.where(bool_mask, self.emb(y), 0.0).astype(x.dtype)
        if self.method == "concat":
            return mx.concatenate([x, emb], axis=-1)
        return x + emb

    @staticmethod
    def ratio_mask(mask: mx.array, ratio: float) -> mx.array:
        """Randomly keeps a ratio of the ``True`` entries of a boolean mask.

        Each ``True`` entry stays ``True`` with probability ``ratio``. Does not
        modify ``mask``.

        Args:
            mask: The boolean mask to re-mask
            ratio: The probability of keeping each entry
        """
        return mask & (mx.random.uniform(shape=mask.shape) < ratio)
