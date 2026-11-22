from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.aggr.base import Aggregation
from mlx_graphs.nn.aggr.utils import DenseLayout


class _RecurrentAggregation(Aggregation):
    def __init__(self, node_features_dim: int, out_features_dim: int):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim

    def _rnn(self, x: mx.array) -> mx.array:
        raise NotImplementedError

    def aggregate(
        self,
        x: mx.array,
        index: mx.array,
        dim_size: int,
        max_num_elements: Optional[int] = None,
        **kwargs,
    ) -> mx.array:
        """Runs the recurrent network over the elements of each set.

        Args:
            x: Array of shape ``[num_elements, node_features_dim]``.
            index: Set of each element.
            dim_size: Number of sets.
            max_num_elements: Only the first ``max_num_elements`` elements of
                each set (in input order) are used. Default: ``None``

        Returns:
            An array of shape ``[dim_size, out_features_dim]``.
        """
        if x.ndim != 2:
            raise ValueError(
                f"{type(self).__name__} requires two-dimensional inputs "
                f"(got {x.ndim})"
            )
        layout = DenseLayout(index, x.shape[0], dim_size, max_num_elements)
        if layout.size == 0:
            return mx.zeros((dim_size, self.out_features_dim), dtype=x.dtype)
        return self._rnn(layout.to_dense(x))[:, -1]


class LSTMAggregation(_RecurrentAggregation):
    r"""Aggregation with an LSTM, from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, as PyG's
    ``LSTMAggregation``.

    The elements of each set are padded with zeros to the size of the
    largest set and fed to an LSTM in input order. The output is the last
    hidden state.

    .. warning::
        The aggregation is not permutation-invariant: it depends on the order
        of the elements in each set.

    Args:
        node_features_dim: Size of each input sample.
        out_features_dim: Size of each output sample (hidden size of the LSTM).
        bias: Whether the LSTM has biases. Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.aggr import LSTMAggregation

        aggr = LSTMAggregation(16, 32)
        x = mx.random.normal((6, 16))
        index = mx.array([0, 0, 1, 1, 1, 2])
        >>> aggr(x, index).shape
        (3, 32)

    References:
        - `Hamilton et al., "Inductive Representation Learning on Large
          Graphs", NeurIPS 2017 <https://arxiv.org/abs/1706.02216>`_
    """

    def __init__(self, node_features_dim: int, out_features_dim: int, bias=True):
        super().__init__(node_features_dim, out_features_dim)
        self.lstm = nn.LSTM(node_features_dim, out_features_dim, bias=bias)

    def _rnn(self, x: mx.array) -> mx.array:
        return self.lstm(x)[0]


class GRUAggregation(_RecurrentAggregation):
    r"""Aggregation with a GRU, from the `"Graph Neural Networks with Adaptive
    Readouts" <https://arxiv.org/abs/2211.04952>`_ paper, as PyG's
    ``GRUAggregation``.

    The elements of each set are padded with zeros to the size of the
    largest set and fed to a GRU in input order. The output is the last
    hidden state.

    .. warning::
        The aggregation is not permutation-invariant: it depends on the order
        of the elements in each set.

    Args:
        node_features_dim: Size of each input sample.
        out_features_dim: Size of each output sample (hidden size of the GRU).
        bias: Whether the GRU has biases. Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.aggr import GRUAggregation

        aggr = GRUAggregation(16, 32)
        x = mx.random.normal((6, 16))
        index = mx.array([0, 0, 1, 1, 1, 2])
        >>> aggr(x, index).shape
        (3, 32)

    References:
        - `Buterez et al., "Graph Neural Networks with Adaptive Readouts",
          NeurIPS 2022 <https://arxiv.org/abs/2211.04952>`_
    """

    def __init__(self, node_features_dim: int, out_features_dim: int, bias=True):
        super().__init__(node_features_dim, out_features_dim)
        self.gru = nn.GRU(node_features_dim, out_features_dim, bias=bias)

    def _rnn(self, x: mx.array) -> mx.array:
        return self.gru(x)
