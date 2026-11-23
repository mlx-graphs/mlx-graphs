import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class PositionalEncoding(nn.Module):
    r"""The sinusoidal positional encoding from the `"Attention Is All You
    Need" <https://arxiv.org/abs/1706.03762>`_ paper, as PyG's
    ``PositionalEncoding``.

    .. math::
        PE(x)_{2 \cdot i} &= \sin(x / 10000^{2 \cdot i / d}) \\
        PE(x)_{2 \cdot i + 1} &= \cos(x / 10000^{2 \cdot i / d})

    where :math:`x` is the position and :math:`i` the dimension. As in PyG,
    the output is ordered as all the sines followed by all the cosines.

    Args:
        out_features_dim: Size :math:`d` of each output sample (even).
        base_freq: Base frequency of the sinusoids. Default: ``1e-4``
        granularity: Granularity of the positions: smaller values capture
            finer changes in positions. Default: ``1.0``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import PositionalEncoding

        encoder = PositionalEncoding(16)
        >>> encoder(mx.array([0.0, 1.0, 2.5])).shape
        (3, 16)

    References:
        - `Vaswani et al., "Attention Is All You Need", NeurIPS 2017
          <https://arxiv.org/abs/1706.03762>`_
    """

    def __init__(
        self,
        out_features_dim: int,
        base_freq: float = 1e-4,
        granularity: float = 1.0,
    ):
        super().__init__()
        if out_features_dim % 2 != 0:
            raise ValueError(
                "Cannot use sinusoidal positional encoding with odd "
                f"'out_features_dim' (got {out_features_dim})."
            )
        self.out_features_dim = out_features_dim
        self.base_freq = base_freq
        self.granularity = granularity
        self.frequency = mx.power(
            base_freq, mx.linspace(0, 1, out_features_dim // 2)
        ).astype(mx.float32)
        self.freeze(keys=["frequency"], recurse=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Encodes the positions ``x`` (any shape, flattened) into an array
        of shape ``[x.size, out_features_dim]``."""
        if self.granularity != 1.0:
            x = x / self.granularity
        out = x.reshape(-1, 1) * self.frequency.reshape(1, -1)
        return mx.concatenate([mx.sin(out), mx.cos(out)], axis=-1)


class TemporalEncoding(nn.Module):
    r"""The time encoding from the `"Do We Really Need Complicated Model
    Architectures for Temporal Networks?"
    <https://openreview.net/forum?id=ayPPc0SyLv1>`_ paper, as PyG's
    ``TemporalEncoding``.

    Each entry is mapped to a vector with exponentially decreasing
    frequencies, projected to :math:`[-1, 1]` by a cosine:

    .. math::
        y_{i} = \cos \left(x \cdot \sqrt{d}^{-(i - 1)/\sqrt{d}} \right)

    for :math:`1 \leq i \leq d`.

    Args:
        out_features_dim: Size :math:`d` of each output sample.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import TemporalEncoding

        encoder = TemporalEncoding(16)
        >>> encoder(mx.array([0.0, 10.0, 200.0])).shape
        (3, 16)

    References:
        - `Cong et al., "Do We Really Need Complicated Model Architectures
          for Temporal Networks?", ICLR 2023
          <https://openreview.net/forum?id=ayPPc0SyLv1>`_
    """

    def __init__(self, out_features_dim: int):
        super().__init__()
        self.out_features_dim = out_features_dim
        sqrt = math.sqrt(out_features_dim)
        self.weight = (
            1.0 / mx.power(sqrt, mx.linspace(0, sqrt, out_features_dim))
        ).reshape(1, -1)
        self.freeze(keys=["weight"], recurse=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Encodes ``x`` (any shape, flattened) into an array of shape
        ``[x.size, out_features_dim]``."""
        # elementwise product: a K=1 matmul is less accurate on the GPU
        return mx.cos(x.reshape(-1, 1) * self.weight)


class RWSEEncoder(nn.Module):
    r"""Projection of the random-walk structural encodings (RWSE) of the
    nodes, as in the GPS example of PyG (``examples/graph_gps.py``) and the
    ``RWSE`` node encoder of `GraphGPS <https://github.com/rampasek/GraphGPS>`_.

    The encodings (e.g. computed by
    :class:`~mlx_graphs.transforms.AddRandomWalkPE`) are normalized and
    linearly projected:

    .. math::
        \mathbf{p}_i' = \mathbf{W} \, \mathrm{Norm}(\mathbf{p}_i) + \mathbf{b}

    The result is typically concatenated with an embedding of the node
    features before a :class:`~mlx_graphs.nn.GPSConv`.

    Args:
        pe_dim: Size of the input encodings (the walk length).
        out_features_dim: Size of the projected encodings.
        norm: Normalization of the input encodings: ``"batch"``, ``"layer"``
            or ``None``. Default: ``"batch"``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import RWSEEncoder

        encoder = RWSEEncoder(pe_dim=20, out_features_dim=8)
        random_walk_pe = mx.random.uniform(shape=(10, 20))
        >>> encoder(random_walk_pe).shape
        (10, 8)

    References:
        - `Rampášek et al., "Recipe for a General, Powerful, Scalable Graph
          Transformer", NeurIPS 2022 <https://arxiv.org/abs/2205.12454>`_
        - `Dwivedi et al., "Graph Neural Networks with Learnable Structural
          and Positional Representations", ICLR 2022
          <https://arxiv.org/abs/2110.07875>`_
    """

    def __init__(self, pe_dim: int, out_features_dim: int, norm="batch"):
        super().__init__()
        if norm == "batch":
            self.norm = nn.BatchNorm(pe_dim)
        elif norm == "layer":
            self.norm = nn.LayerNorm(pe_dim)
        elif norm is not None:
            raise ValueError(f"Unknown norm '{norm}'. Available: 'batch', 'layer'")
        self.lin = nn.Linear(pe_dim, out_features_dim)

    def __call__(self, pe: mx.array, batch_indices: Optional[mx.array] = None):
        """Projects the encodings ``pe`` of shape ``[num_nodes, pe_dim]``.

        Args:
            pe: Encodings of shape ``[num_nodes, pe_dim]``.
            batch_indices: Unused, for compatibility with
                :class:`LapPEEncoder`.
        """
        if "norm" in self:
            pe = self.norm(pe)
        return self.lin(pe)


class LapPEEncoder(RWSEEncoder):
    r"""Projection of the Laplacian eigenvector positional encodings (LapPE)
    of the nodes, as the linear ``LapPE`` node encoder of `GraphGPS
    <https://github.com/rampasek/GraphGPS>`_.

    Eigenvectors are only defined up to their sign, so during training
    (``module.train()``) the sign of every eigenvector of every graph is
    flipped at random, which makes the model learn sign-invariant functions
    of the encodings. The encodings (e.g. computed by
    :class:`~mlx_graphs.transforms.AddLaplacianEigenvectorPE`) are then
    normalized (optional) and linearly projected.

    Args:
        pe_dim: Number of eigenvectors :math:`k` in the input encodings.
        out_features_dim: Size of the projected encodings.
        norm: Normalization of the input encodings: ``"batch"``, ``"layer"``
            or ``None``. Default: ``None``
        random_sign_flip: Whether to flip signs at random during training.
            Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import LapPEEncoder

        encoder = LapPEEncoder(pe_dim=4, out_features_dim=8)
        laplacian_pe = mx.random.normal((10, 4))
        batch_indices = mx.array([0] * 6 + [1] * 4)
        >>> encoder(laplacian_pe, batch_indices).shape
        (10, 8)

    References:
        - `Dwivedi et al., "Benchmarking Graph Neural Networks", JMLR 2023
          <https://arxiv.org/abs/2003.00982>`_
        - `Rampášek et al., "Recipe for a General, Powerful, Scalable Graph
          Transformer", NeurIPS 2022 <https://arxiv.org/abs/2205.12454>`_
    """

    def __init__(
        self,
        pe_dim: int,
        out_features_dim: int,
        norm: Optional[str] = None,
        random_sign_flip: bool = True,
    ):
        super().__init__(pe_dim, out_features_dim, norm)
        self.random_sign_flip = random_sign_flip

    def __call__(self, pe: mx.array, batch_indices: Optional[mx.array] = None):
        """Projects the eigenvector encodings ``pe``.

        Args:
            pe: Encodings of shape ``[num_nodes, pe_dim]``.
            batch_indices: Graph of each node, so that each graph gets its own
                random signs during training. Default: ``None`` (one graph).
        """
        if self.training and self.random_sign_flip:
            if batch_indices is None:
                flip = mx.random.bernoulli(0.5, (1, pe.shape[1]))
            else:
                # One row of signs per graph, without reading the number of
                # graphs back from the device (which breaks `mx.compile`):
                # there are at most as many non-empty graphs as nodes. Graph
                # ids beyond that (after empty graphs) wrap around, which
                # keeps one uniformly random sign per graph and eigenvector.
                num_rows = max(pe.shape[0], 1)
                flip = mx.random.bernoulli(0.5, (num_rows, pe.shape[1]))
                flip = flip[batch_indices % num_rows]
            pe = pe * mx.where(flip, -1.0, 1.0).astype(pe.dtype)
        return super().__call__(pe)
