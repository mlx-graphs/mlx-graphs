import math
from functools import lru_cache
from typing import List, Sequence, Tuple, Union

import mlx.core as mx
import numpy as np
from mlx.nn import Linear  # uniform initialization, as PyG's root weight

from mlx_graphs.nn.message_passing import MessagePassing


def _basis_1d(v: mx.array, k_mod: mx.array, degree: int) -> mx.array:
    """B-spline basis functions of the given degree, at the fractional
    positions ``v`` for the local knots ``k_mod`` (as in torch-spline-conv)."""
    if degree == 1:
        return 1.0 - v - k_mod + 2.0 * v * k_mod
    if degree == 2:
        return mx.where(
            k_mod == 0,
            0.5 * v * v - v + 0.5,
            mx.where(k_mod == 1, -v * v + v + 0.5, 0.5 * v * v),
        )
    if degree == 3:
        return mx.where(
            k_mod == 0,
            (1.0 - v) ** 3 / 6.0,
            mx.where(
                k_mod == 1,
                (3.0 * v**3 - 6.0 * v * v + 4.0) / 6.0,
                mx.where(
                    k_mod == 2,
                    (-3.0 * v**3 + 3.0 * v * v + 3.0 * v + 1.0) / 6.0,
                    v**3 / 6.0,
                ),
            ),
        )
    raise ValueError(f"degree must be 1, 2 or 3, got {degree}")


@lru_cache(maxsize=32)
def _knots(dim: int, degree: int) -> np.ndarray:
    """``[S, dim]`` local knot offsets of the ``S = (degree + 1) ** dim``
    basis products (the first dimension varies fastest)."""
    s = np.arange((degree + 1) ** dim)
    return np.stack([(s // (degree + 1) ** d) % (degree + 1) for d in range(dim)], 1)


def spline_basis(
    pseudo: mx.array,
    kernel_size: Sequence[int],
    is_open_spline: Sequence[bool],
    degree: int = 1,
) -> Tuple[mx.array, mx.array]:
    r"""Tensor-product B-spline basis of the `"SplineCNN: Fast Geometric Deep
    Learning with Continuous B-Spline Kernels"
    <https://arxiv.org/abs/1711.08920>`_ paper.

    Each pseudo-coordinate :math:`u \in [0, 1]^D` activates :math:`S = (d +
    1)^D` of the :math:`\prod_i K_i` kernel weights, with the products of the
    1D B-spline bases of degree :math:`d` as coefficients. Differentiable with
    respect to ``pseudo``.

    Args:
        pseudo: Pseudo-coordinates of shape ``[E, D]``, in ``[0, 1]``.
        kernel_size: Number of knots :math:`K_i` in each dimension.
        is_open_spline: Whether each dimension uses an open B-spline (``True``)
            or a closed (periodic) one.
        degree: Degree of the B-splines (1, 2 or 3). Default: ``1``

    Returns:
        A tuple ``(basis, weight_index)`` of shapes ``[E, S]``: the
        coefficients and the index of the kernel weight they multiply.
    """
    pseudo = pseudo.reshape(-1, 1) if pseudo.ndim == 1 else pseudo
    dim = pseudo.shape[1]
    kernel = np.asarray(kernel_size, dtype=np.int64).reshape(-1)
    open_spline = np.asarray(is_open_spline, dtype=np.int64).reshape(-1)
    if kernel.size != dim or open_spline.size != dim:
        raise ValueError("kernel_size and is_open_spline need one entry per dimension")
    knots = mx.array(_knots(dim, degree).astype(np.int32))  # [S, D]

    scale = mx.array((kernel - degree * open_spline).astype(np.float32))
    v = (pseudo * scale)[:, None, :]  # [E, 1, D]
    floor = mx.floor(mx.stop_gradient(v))
    frac = v - floor
    basis = _basis_1d(frac, knots.astype(frac.dtype)[None], degree).prod(axis=-1)

    offsets = np.concatenate([[1], np.cumprod(kernel)[:-1]])
    kernel_arr = mx.array(kernel.astype(np.int32))
    index = (floor.astype(mx.int32) + knots[None]) % kernel_arr
    weight_index = (index * mx.array(offsets.astype(np.int32))).sum(axis=-1)
    return basis, weight_index


def spline_weighting(
    x: mx.array, weight: mx.array, basis: mx.array, weight_index: mx.array
) -> mx.array:
    """Applies the B-spline kernels to the features of each edge:
    ``out[e] = sum_s basis[e, s] * x[e] @ weight[weight_index[e, s]]``.

    Args:
        x: Features of shape ``[E, in_features]``.
        weight: Kernel weights of shape ``[K, in_features, out_features]``.
        basis: Coefficients of shape ``[E, S]`` (see :func:`spline_basis`).
        weight_index: Weight indices of shape ``[E, S]``.

    Returns:
        The weighted features, of shape ``[E, out_features]``.
    """
    num_edges, num_products = basis.shape
    if num_edges == 0:
        return mx.zeros((0, weight.shape[-1]), dtype=x.dtype)
    lhs = mx.repeat(mx.arange(num_edges, dtype=mx.uint32), num_products)
    out = mx.gather_mm(
        x[:, None, :],
        weight,
        lhs_indices=lhs,
        rhs_indices=weight_index.reshape(-1).astype(mx.uint32),
    )  # [E * S, 1, out]
    out = out.reshape(num_edges, num_products, -1)
    return (out * basis[..., None]).sum(axis=1)


def _repeat(value, dim: int) -> List:
    if isinstance(value, (list, tuple)):
        if len(value) != dim:
            raise ValueError(f"Expected {dim} values, got {len(value)}")
        return list(value)
    return [value] * dim


class SplineConv(MessagePassing):
    r"""The spline-based convolutional operator from the `"SplineCNN: Fast
    Geometric Deep Learning with Continuous B-Spline Kernels"
    <https://arxiv.org/abs/1711.08920>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in
        \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}) + \mathbf{\Theta}_r
        \mathbf{x}_i + \mathbf{b},

    where :math:`h_{\mathbf{\Theta}}` is a kernel function defined over the
    weighted B-spline tensor product basis of the pseudo-coordinates
    :math:`\mathbf{e}_{i,j} \in [0, 1]^D` (``edge_features``, e.g. from the
    :class:`~mlx_graphs.transforms.Cartesian` transform).

    Args:
        node_features_dim: Size of the input features, or a ``(src_dim,
            dst_dim)`` tuple for bipartite graphs.
        out_features_dim: Size of the output features.
        dim: Pseudo-coordinate dimensionality :math:`D`.
        kernel_size: Size of the convolving kernel in each dimension.
        is_open_spline: Whether to use open (``True``) or closed B-spline
            bases, in each dimension. Default: ``True``
        degree: B-spline basis degree (1, 2 or 3). Default: ``1``
        aggr: Aggregation of the messages. Default: ``"mean"``
        root_weight: Whether to add the transformed root node features.
            Default: ``True``
        bias: Whether to add a bias. Default: ``True``
        **kwargs: Arguments of :class:`MessagePassing`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import SplineConv

        edge_index = mx.array([[0, 1, 2, 3], [1, 0, 3, 2]])
        x = mx.random.normal((4, 16))
        pseudo = mx.random.uniform(shape=(4, 2))
        conv = SplineConv(16, 32, dim=2, kernel_size=5)
        >>> conv(edge_index, x, pseudo).shape
        (4, 32)
    """

    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        out_features_dim: int,
        dim: int,
        kernel_size: Union[int, List[int]],
        is_open_spline: Union[bool, List[bool]] = True,
        degree: int = 1,
        aggr: str = "mean",
        root_weight: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)
        if degree not in (1, 2, 3):
            raise ValueError(f"degree must be 1, 2 or 3, got {degree}")
        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.dim = dim
        self.degree = degree
        self.root_weight = root_weight
        self.kernel_size = tuple(int(k) for k in _repeat(kernel_size, dim))
        self.is_open_spline = tuple(bool(o) for o in _repeat(is_open_spline, dim))
        self.K = int(np.prod(self.kernel_size))
        src_dim, dst_dim = (
            (node_features_dim, node_features_dim)
            if isinstance(node_features_dim, int)
            else node_features_dim
        )
        bound = 1.0 / math.sqrt(self.K * src_dim)
        self.weight = mx.random.uniform(
            -bound, bound, (self.K, src_dim, out_features_dim)
        )
        if root_weight:
            self.lin = Linear(dst_dim, out_features_dim, bias=False)
        if bias:
            self.bias = mx.zeros((out_features_dim,))

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_features: mx.array,
    ) -> mx.array:
        """Computes the forward pass of SplineConv.

        Args:
            edge_index: Edges of shape ``[2, num_edges]``.
            node_features: Node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs.
            edge_features: Pseudo-coordinates of shape ``[num_edges, dim]``,
                in ``[0, 1]``.

        Returns:
            The features of the destination nodes.
        """
        if isinstance(node_features, tuple):
            dst_features = node_features[1]
        else:
            dst_features = node_features
            node_features = (node_features, node_features)
        out = self.propagate(
            edge_index=edge_index,
            node_features=node_features,
            message_kwargs={"edge_features": edge_features},
        )
        if self.root_weight:
            out = out + self.lin(dst_features)
        if "bias" in self:
            out = out + self.bias
        return out

    def message(
        self,
        src_features: mx.array,
        dst_features: mx.array,
        edge_features: mx.array,
        **kwargs,
    ) -> mx.array:
        basis, weight_index = spline_basis(
            edge_features, self.kernel_size, self.is_open_spline, self.degree
        )
        return spline_weighting(src_features, self.weight, basis, weight_index)
