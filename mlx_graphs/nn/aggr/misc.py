import math
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.aggr.base import Aggregation
from mlx_graphs.nn.aggr.utils import DenseLayout
from mlx_graphs.utils import degree


class VariancePreservingAggregation(Aggregation):
    r"""The variance-preserving aggregation from the `"GNN-VPA: A
    Variance-Preserving Aggregation Strategy for Graph Neural Networks"
    <https://arxiv.org/abs/2403.04747>`_ paper, as in PyG.

    .. math::
        \mathrm{vpa}(\mathcal{X}) = \frac{1}{\sqrt{|\mathcal{X}|}}
        \sum_{\mathbf{x}_i \in \mathcal{X}} \mathbf{x}_i

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.aggr import VariancePreservingAggregation

        aggr = VariancePreservingAggregation()
        x = mx.random.normal((6, 16))
        index = mx.array([0, 0, 1, 1, 1, 2])
        >>> aggr(x, index).shape
        (3, 16)

    References:
        - `Schneckenreiter et al., "GNN-VPA: A Variance-Preserving
          Aggregation Strategy for Graph Neural Networks", 2024
          <https://arxiv.org/abs/2403.04747>`_
    """

    def aggregate(self, x, index, dim_size, **kwargs):
        out = self.reduce(x, index, dim_size, "add")
        count = degree(index, dim_size).astype(out.dtype)
        count = mx.maximum(mx.sqrt(count), 1.0)
        return out / count.reshape(-1, *([1] * (out.ndim - 1)))


class MLPAggregation(Aggregation):
    r"""Aggregation with a multi-layer perceptron on the concatenated
    elements of each set, from the `"Graph Neural Networks with Adaptive
    Readouts" <https://arxiv.org/abs/2211.04952>`_ paper, as PyG's
    ``MLPAggregation``.

    The elements of each set are padded with zeros (or truncated) to
    ``max_num_elements``, flattened, and fed to an
    :class:`~mlx_graphs.nn.MLP`.

    .. warning::
        The aggregation is not permutation-invariant: it depends on the order
        of the elements in each set.

    Args:
        node_features_dim: Size of each input sample.
        out_features_dim: Size of each output sample.
        max_num_elements: Number of elements per set fed to the MLP.
        hidden_features_dim: Size of the hidden layers of the MLP. Required if
            ``num_layers > 1``. Default: ``None``
        num_layers: Number of layers of the MLP. Default: ``1``
        **mlp_kwargs: Other arguments of :class:`~mlx_graphs.nn.MLP`
            (``dropout``, ``act``, ``norm``, ``plain_last``, ``bias``). As
            in PyG's ``MLP``, ``norm`` defaults to ``"batch"``.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.aggr import MLPAggregation

        aggr = MLPAggregation(16, 32, max_num_elements=3, hidden_features_dim=64,
                              num_layers=2)
        x = mx.random.normal((6, 16))
        index = mx.array([0, 0, 1, 1, 1, 2])
        >>> aggr(x, index).shape
        (3, 32)

    References:
        - `Buterez et al., "Graph Neural Networks with Adaptive Readouts",
          NeurIPS 2022 <https://arxiv.org/abs/2211.04952>`_
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        max_num_elements: int,
        hidden_features_dim: Optional[int] = None,
        num_layers: int = 1,
        **mlp_kwargs: Any,
    ):
        super().__init__()
        from mlx_graphs.nn.models.basic_gnn import MLP

        if num_layers > 1 and hidden_features_dim is None:
            raise ValueError("`hidden_features_dim` is required if `num_layers > 1`")
        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.max_num_elements = max_num_elements
        dims = (
            [node_features_dim * max_num_elements]
            + [hidden_features_dim] * (num_layers - 1)
            + [out_features_dim]
        )
        mlp_kwargs.setdefault("norm", "batch")
        self.mlp = MLP(dims, **mlp_kwargs)

    def aggregate(self, x, index, dim_size, **kwargs):
        if x.ndim != 2:
            raise ValueError(
                f"MLPAggregation requires two-dimensional inputs (got {x.ndim})"
            )
        layout = DenseLayout(index, x.shape[0], dim_size, self.max_num_elements)
        dense = layout.to_dense(x)
        return self.mlp(dense.reshape(dim_size, -1))


class LCMAggregation(Aggregation):
    r"""The Learnable Commutative Monoid aggregation from the `"Learnable
    Commutative Monoids for Graph Neural Networks"
    <https://arxiv.org/abs/2212.08541>`_ paper, as PyG's ``LCMAggregation``.

    The elements of each set are (optionally) projected, then combined in a
    balanced binary tree whose binary operator is a symmetrized GRU cell,
    in :math:`\mathcal{O}(\log |\mathcal{V}|)` depth:

    .. math::
        \mathbf{y} = \frac{1}{2}\left(\mathrm{GRU}(\mathbf{a}, \mathbf{b}) +
        \mathrm{GRU}(\mathbf{b}, \mathbf{a})\right)

    Sets are padded with zeros to the size of the largest set.

    Args:
        node_features_dim: Size of each input sample.
        out_features_dim: Size of each output sample.
        project: Whether to apply a linear projection followed by a ReLU
            before the aggregation. Required if the two sizes differ.
            Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.aggr import LCMAggregation

        aggr = LCMAggregation(16, 32)
        x = mx.random.normal((6, 16))
        index = mx.array([0, 0, 1, 1, 1, 2])
        >>> aggr(x, index).shape
        (3, 32)

    References:
        - `Ong and Veličković, "Learnable Commutative Monoids for Graph
          Neural Networks", LoG 2022 <https://arxiv.org/abs/2212.08541>`_
    """

    def __init__(
        self, node_features_dim: int, out_features_dim: int, project: bool = True
    ):
        super().__init__()
        if node_features_dim != out_features_dim and not project:
            raise ValueError(
                "Inputs of 'LCMAggregation' must be projected if "
                "`node_features_dim != out_features_dim`"
            )
        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.project = project
        if project:
            self.lin = nn.Linear(node_features_dim, out_features_dim)
        self.gru_cell = nn.GRU(out_features_dim, out_features_dim)

    def aggregate(
        self,
        x: mx.array,
        index: mx.array,
        dim_size: int,
        max_num_elements: Optional[int] = None,
        **kwargs,
    ) -> mx.array:
        """Aggregates ``x``.

        Args:
            x: Array of shape ``[num_elements, node_features_dim]``.
            index: Set of each element.
            dim_size: Number of sets.
            max_num_elements: Only the first ``max_num_elements`` elements of
                each set (in input order) are used. Default: ``None``
        """
        if x.ndim != 2:
            raise ValueError(
                f"LCMAggregation requires two-dimensional inputs (got {x.ndim})"
            )
        if self.project:
            x = nn.relu(self.lin(x))
        layout = DenseLayout(index, x.shape[0], dim_size, max_num_elements)
        if layout.size == 0:
            return mx.zeros((dim_size, self.out_features_dim), dtype=x.dtype)
        x = layout.to_dense(x).transpose(1, 0, 2)  # [size, num_sets, features]
        num_features = x.shape[-1]

        for _ in range(math.ceil(math.log2(x.shape[0]))):
            remainder = None
            if x.shape[0] % 2 == 1:
                x, remainder = x[:-1], x[-1:]
            left_right = x.reshape(-1, 2, dim_size, num_features)
            right_left = left_right[:, ::-1]
            out = self.gru_cell(
                left_right.reshape(-1, 1, num_features),
                right_left.reshape(-1, num_features),
            )
            out = out.reshape(-1, 2, dim_size, num_features).mean(axis=1)
            if remainder is not None:
                out = mx.concatenate([out, remainder], axis=0)
            x = out
        return x[0]


# The potential is differentiated twice (the inner gradient steps are
# differentiated during training), which the compiled / fused MLX kernels of
# ``nn.softplus``, ``nn.Tanh`` and ``nn.LayerNorm`` do not support.
def _softplus(x: mx.array) -> mx.array:
    return mx.logaddexp(x, 0.0)


class _LayerNorm(nn.LayerNorm):
    def __call__(self, x: mx.array) -> mx.array:
        mean = x.mean(axis=-1, keepdims=True)
        var = mx.square(x - mean).mean(axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + self.eps) * self.weight + self.bias


class _Tanh(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return mx.tanh(x)


class _ResNetPotential(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_layers: list[int]):
        super().__init__()
        sizes = [in_dim] + list(num_layers) + [out_dim]
        self.layers = [
            nn.Sequential(nn.Linear(i, o), _LayerNorm(o), _Tanh())
            for i, o in zip(sizes[:-2], sizes[1:-1])
        ]
        self.layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.res_trans = [nn.Linear(in_dim, size) for size in sizes[1:]]

    def __call__(self, x, y, index, dim_size):
        inp = mx.concatenate([x, y[index]], axis=1)
        h = inp
        for layer, res in zip(self.layers, self.res_trans):
            h = res(inp) + layer(h)
        return self.reduce_mean(h, index, dim_size).sum()

    @staticmethod
    def reduce_mean(h, index, dim_size):
        from mlx_graphs.utils import scatter

        return scatter(h, index, dim_size, aggr="mean")


class _MomentumOptimizer(nn.Module):
    def __init__(self, learning_rate: float = 0.1, momentum: float = 0.9):
        super().__init__()
        # Raw parameters (``_lr`` and ``_mom`` in PyG).
        self.lr = mx.array([learning_rate])
        self.mom = mx.array([momentum])

    @property
    def learning_rate(self):
        return _softplus(self.lr)

    @property
    def momentum(self):
        return mx.sigmoid(self.mom)

    def __call__(self, x, y, index, dim_size, func, iterations):
        momentum_buffer = mx.zeros_like(y)
        for _ in range(iterations):
            grad = mx.grad(lambda y_: func(x, y_, index, dim_size))(y)
            momentum_buffer = (
                self.momentum * momentum_buffer - self.learning_rate * grad
            )
            y = y + momentum_buffer
        return y


class EquilibriumAggregation(Aggregation):
    r"""The equilibrium aggregation layer from the `"Equilibrium Aggregation:
    Encoding Sets via Optimization" <https://arxiv.org/abs/2202.12795>`_
    paper, as PyG's ``EquilibriumAggregation``.

    The output of each set is the minimizer of a learned energy, found by
    ``grad_iter`` steps of gradient descent with momentum from zero:

    .. math::
        \mathbf{y}_i = \arg\min_{\mathbf{y}} \sum_{j \in \mathcal{X}_i}
        F(\mathbf{x}_j, \mathbf{y}) + \lambda \| \mathbf{y} \|^2

    where :math:`F` is a residual MLP (the potential), and the learning rate,
    momentum and :math:`\lambda` are learnable. The inner gradients are
    computed with :func:`mlx.core.grad`, so the layer is differentiable
    end-to-end.

    Args:
        node_features_dim: Size of each input sample.
        out_features_dim: Size of each output sample.
        num_layers: Sizes of the hidden layers of the potential.
        grad_iter: Number of gradient steps. Default: ``5``
        lamb: Initial regularization strength :math:`\lambda` (before a
            softplus). Default: ``0.1``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn.aggr import EquilibriumAggregation

        aggr = EquilibriumAggregation(16, 8, num_layers=[32, 32])
        x = mx.random.normal((6, 16))
        index = mx.array([0, 0, 1, 1, 1, 2])
        >>> aggr(x, index).shape
        (3, 8)

    References:
        - `Bartunov et al., "Equilibrium Aggregation: Encoding Sets via
          Optimization", UAI 2022 <https://arxiv.org/abs/2202.12795>`_
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        num_layers: list[int],
        grad_iter: int = 5,
        lamb: float = 0.1,
    ):
        super().__init__()
        self.potential = _ResNetPotential(
            node_features_dim + out_features_dim, 1, num_layers
        )
        self.optimizer = _MomentumOptimizer()
        self.lamb = mx.array([lamb])
        self.grad_iter = grad_iter
        self.out_features_dim = out_features_dim

    def reg(self, y: mx.array) -> mx.array:
        return _softplus(self.lamb) * mx.square(y).sum(axis=-1).mean()

    def energy(self, x, y, index, dim_size) -> mx.array:
        return (self.potential(x, y, index, dim_size) + self.reg(y)).sum()

    def aggregate(self, x, index, dim_size, **kwargs):
        if x.ndim != 2:
            raise ValueError(
                f"EquilibriumAggregation requires two-dimensional inputs "
                f"(got {x.ndim})"
            )
        y = mx.zeros((dim_size, self.out_features_dim), dtype=x.dtype)
        return self.optimizer(x, y, index, dim_size, self.energy, self.grad_iter)
