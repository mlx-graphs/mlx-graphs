from typing import Any, Callable, Literal, Optional

import mlx.core as mx
import mlx.nn as nn


class DeepGCNLayer(nn.Module):
    r"""The skip connection operations from the `"DeepGCNs: Can GCNs Go as
    Deep as CNNs?" <https://arxiv.org/abs/1904.03751>`_ and `"All You Need to
    Train Deeper GCNs" <https://arxiv.org/abs/2006.07739>`_ papers.

    * **Res+** (``"res+"``): Normalization :math:`\to` Activation :math:`\to`
      Dropout :math:`\to` GraphConv :math:`\to` Res
    * **Res** (``"res"``) / **Dense** (``"dense"``) / **Plain** (``"plain"``):
      GraphConv :math:`\to` Normalization :math:`\to` Activation :math:`\to`
      Res/Dense/Plain :math:`\to` Dropout

    Args:
        conv: The graph convolution, called as ``conv(edge_index,
            node_features, *args, **kwargs)``.
        norm: The normalization layer. Default: ``None``
        act: The activation function. Default: ``None``
        block: The skip connection: ``"res+"``, ``"res"``, ``"dense"`` or
            ``"plain"``. Default: ``"res+"``
        dropout: Dropout probability. Default: ``0.0``
        ckpt_grad: If ``True``, checkpoints the convolution, recomputing its
            activations in the backward pass instead of storing them, which
            saves memory on deep models. Default: ``False``

    Example:

    .. code-block:: python

        import mlx.nn as nn
        from mlx_graphs.nn import DeepGCNLayer, GCNConv

        layers = [
            DeepGCNLayer(
                GCNConv(64, 64),
                norm=nn.LayerNorm(64),
                act=nn.relu,
                block="res+",
                dropout=0.1,
            )
            for _ in range(28)
        ]
        x = layers[0].conv(edge_index, x)
        for layer in layers[1:]:
            x = layer(edge_index, x)

    References:
        [1] `Li et al. DeepGCNs: Can GCNs Go as Deep as CNNs?
        <https://arxiv.org/abs/1904.03751>`_

        [2] `Li et al. DeeperGCN: All You Need to Train Deeper GCNs
        <https://arxiv.org/abs/2006.07739>`_
    """

    def __init__(
        self,
        conv: Optional[nn.Module] = None,
        norm: Optional[nn.Module] = None,
        act: Optional[Callable[[mx.array], mx.array]] = None,
        block: Literal["res+", "res", "dense", "plain"] = "res+",
        dropout: float = 0.0,
        ckpt_grad: bool = False,
    ):
        super().__init__()
        self.block = block.lower()
        if self.block not in ("res+", "res", "dense", "plain"):
            raise ValueError(
                f"`block` must be 'res+', 'res', 'dense' or 'plain', got {block!r}"
            )
        if conv is not None:
            self.conv = conv
        if norm is not None:
            self.norm = norm
        self.act = act
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.ckpt_grad = ckpt_grad

    def _conv(self, edge_index: mx.array, x: mx.array, *args: Any, **kwargs: Any):
        if self.ckpt_grad and self.training:
            # only the node features are inputs of the checkpointed function:
            # no gradient can be computed w.r.t. the integer edge_index
            def fn(h: mx.array) -> mx.array:
                return self.conv(edge_index, h, *args, **kwargs)

            return nn.utils.checkpoint(self.conv, fn)(x)
        return self.conv(edge_index, x, *args, **kwargs)

    def _dropout(self, x: mx.array) -> mx.array:
        return x if self.dropout is None else self.dropout(x)

    def __call__(
        self, edge_index: mx.array, node_features: mx.array, *args: Any, **kwargs: Any
    ) -> mx.array:
        """Computes the forward pass of the block.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features
            *args: Additional arguments of the convolution
            **kwargs: Additional keyword arguments of the convolution

        Returns:
            The output node embeddings.
        """
        x = node_features
        if self.block == "res+":
            h = x
            if "norm" in self:
                h = self.norm(h)
            if self.act is not None:
                h = self.act(h)
            h = self._dropout(h)
            h = self._conv(edge_index, h, *args, **kwargs)
            return x + h

        h = self._conv(edge_index, x, *args, **kwargs)
        if "norm" in self:
            h = self.norm(h)
        if self.act is not None:
            h = self.act(h)
        if self.block == "res":
            h = x + h
        elif self.block == "dense":
            h = mx.concatenate([x, h], axis=-1)
        return self._dropout(h)
