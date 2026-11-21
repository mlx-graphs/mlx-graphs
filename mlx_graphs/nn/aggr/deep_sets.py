from typing import Callable, Optional

import mlx.core as mx

from mlx_graphs.nn.aggr.base import Aggregation


class DeepSetsAggregation(Aggregation):
    r"""Deep Sets aggregation: the elements are transformed by a network
    :math:`\phi_{\mathbf{\Theta}}`, summed, and the sum is transformed by a
    network :math:`\rho_{\mathbf{\Theta}}`, as in `"Graph Neural Networks with
    Adaptive Readouts" <https://arxiv.org/abs/2211.04952>`_ paper.

    .. math::
        \mathrm{deepsets}(\mathcal{X}) = \rho_{\mathbf{\Theta}} \left(
        \sum_{\mathbf{x}_i \in \mathcal{X}} \phi_{\mathbf{\Theta}}
        (\mathbf{x}_i) \right)

    Args:
        local_nn: Network :math:`\phi_{\mathbf{\Theta}}` applied to every
            element (e.g. an :class:`~mlx_graphs.nn.MLP`). Default: ``None``
        global_nn: Network :math:`\rho_{\mathbf{\Theta}}` applied to every
            sum. Default: ``None``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import MLP
        from mlx_graphs.nn.aggr import DeepSetsAggregation

        aggr = DeepSetsAggregation(MLP([16, 32, 32]), MLP([32, 32, 8]))
        x = mx.random.normal((6, 16))
        index = mx.array([0, 0, 1, 1, 1, 2])
        >>> aggr(x, index).shape
        (3, 8)
    """

    def __init__(
        self,
        local_nn: Optional[Callable[[mx.array], mx.array]] = None,
        global_nn: Optional[Callable[[mx.array], mx.array]] = None,
    ):
        super().__init__()
        self.local_nn = local_nn
        self.global_nn = global_nn

    def aggregate(self, x, index, dim_size, **kwargs):
        if self.local_nn is not None:
            x = self.local_nn(x)
        x = self.reduce(x, index, dim_size, "add")
        if self.global_nn is not None:
            x = self.global_nn(x)
        return x
