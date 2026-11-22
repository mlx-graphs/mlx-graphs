import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils.array_ops import p_norm


def _p_norm(x: mx.array, p: float) -> mx.array:
    """p-norm of each row of ``x``, keeping the last dimension, with a
    zero gradient at zero (a row of zeros would otherwise give NaN)."""
    return p_norm(x, p, axis=-1, keepdims=True)


class MessageNormalization(nn.Module):
    r"""Applies message normalization over the aggregated messages as described
    in the `"DeeperGCNs: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper.

    .. math::
        \mathbf{x}_i^{\prime} = \mathrm{MLP} \left( \mathbf{x}_{i} + s \cdot
        {\| \mathbf{x}_i \|}_2 \cdot
        \frac{\mathbf{m}_{i}}{{\|\mathbf{m}_i\|}_2} \right)

    The layer computes the normalized messages
    :math:`s \cdot {\| \mathbf{x}_i \|}_2 \cdot \mathbf{m}_{i} /
    {\|\mathbf{m}_i\|}_2`; adding them to :math:`\mathbf{x}_i` and applying the
    MLP is left to the caller. Also available as :class:`MessageNorm`, its
    PyG name.

    Args:
        learn_scale: If set to ``True``, the scaling factor :math:`s` is
            learnable. Default: ``False``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import MessageNormalization

        norm = MessageNormalization(learn_scale=True)
        node_features = mx.random.normal((5, 16))
        messages = mx.random.normal((5, 16))

        >>> norm(node_features, messages).shape
        (5, 16)
    """

    def __init__(self, learn_scale: bool = False):
        super().__init__()
        self.learn_scale = learn_scale
        self.scale = mx.ones((1,))
        if not learn_scale:
            self.freeze(keys=["scale"], recurse=False)

    def __call__(
        self, node_features: mx.array, messages: mx.array, p: float = 2.0
    ) -> mx.array:
        """Normalizes the messages and rescales them to the norm of the node
        features.

        Args:
            node_features: Node features :math:`\\mathbf{x}` of shape
                ``[num_nodes, num_features]``.
            messages: Aggregated messages :math:`\\mathbf{m}` of shape
                ``[num_nodes, num_features]``.
            p: The norm :math:`p` used for the normalization. Default: ``2.0``

        Returns:
            The normalized messages.
        """
        # Same epsilon as `torch.nn.functional.normalize`.
        messages = messages / mx.maximum(_p_norm(messages, p), 1e-12)
        return messages * _p_norm(node_features, p) * self.scale

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(learn_scale={self.learn_scale})"
