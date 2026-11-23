import math
from functools import lru_cache
from typing import Callable, List, Literal, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_graphs.nn.models.basic_gnn import _ACTIVATIONS
from mlx_graphs.utils.scatter import scatter
from mlx_graphs.utils.spatial import radius_graph
from mlx_graphs.utils.transformations import _to_numpy

__all__ = [
    "DimeNet",
    "DimeNetPlusPlus",
    "Envelope",
    "BesselBasisLayer",
    "SphericalBasisLayer",
    "EmbeddingBlock",
    "ResidualLayer",
    "InteractionBlock",
    "InteractionPPBlock",
    "OutputBlock",
    "OutputPPBlock",
    "triplets",
    "glorot_orthogonal",
]

OutputInitializer = Literal["zeros", "glorot_orthogonal"]


def _resolve_act(act: Union[str, Callable]) -> Callable:
    if callable(act):
        return act
    activations = {"swish": nn.silu, **_ACTIVATIONS}
    if act not in activations:
        raise ValueError(
            f"Unknown activation '{act}'. Available: {sorted(activations)}, "
            "or pass a callable."
        )
    return activations[act]


def glorot_orthogonal(shape: Tuple[int, int], scale: float) -> mx.array:
    r"""PyG's ``glorot_orthogonal`` initialization: a (semi-)orthogonal
    matrix, rescaled to the variance :math:`\frac{\text{scale}}{n_{in} +
    n_{out}}`.

    Args:
        shape: The shape ``[out_features, in_features]`` of the weight.
        scale: The scale of the variance.
    """
    rows, cols = shape
    flat = np.array(mx.random.normal((rows, cols)), dtype=np.float64)
    if rows < cols:
        flat = flat.T
    q, r = np.linalg.qr(flat)
    q = q * np.sign(np.diag(r))
    if rows < cols:
        q = q.T
    q = q * math.sqrt(scale / ((rows + cols) * q.var(ddof=1)))
    return mx.array(q.astype(np.float32))


class _GlorotOrthogonalLinear(nn.Linear):
    """``nn.Linear`` initialized with :func:`glorot_orthogonal` (scale 2) and
    a zero bias."""

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True):
        super().__init__(input_dims, output_dims, bias=bias)
        self.weight = glorot_orthogonal((output_dims, input_dims), scale=2.0)
        if bias:
            self.bias = mx.zeros((output_dims,))


# Radial and angular bases ---------------------------------------------------


def _spherical_bessel_zeros(num_orders: int, num_zeros: int) -> np.ndarray:
    """The first ``num_zeros`` positive zeros of the spherical Bessel functions
    :math:`j_0, \\dots, j_{n-1}`, found by bisection between the zeros of the
    previous order (they interlace), in float32 as in PyG."""
    from scipy.optimize import brentq
    from scipy.special import spherical_jn

    zeros = np.zeros((num_orders, num_zeros), dtype=np.float32)
    zeros[0] = np.arange(1, num_zeros + 1) * np.pi
    points = np.arange(1, num_zeros + num_orders) * np.pi
    for order in range(1, num_orders):
        roots = np.zeros(num_zeros + num_orders - 1 - order, dtype=np.float32)
        for j in range(roots.size):
            roots[j] = brentq(
                lambda r: spherical_jn(order, r), points[j], points[j + 1]
            )
        points = roots
        zeros[order] = roots[:num_zeros]
    return zeros


@lru_cache(maxsize=None)
def _bessel_basis_expressions(num_orders: int, num_zeros: int) -> list:
    r"""Symbolic normalized spherical Bessel functions
    :math:`\sqrt{2} / |j_{l+1}(z_{ln})| \, j_l(z_{ln} x)`, with the closed
    forms of :math:`j_l` given by Rayleigh's formula (as in DimeNet)."""
    import sympy as sym
    from scipy.special import spherical_jn

    x = sym.symbols("x")
    # Rayleigh: j_l(x) = (-x)^l (1/x d/dx)^l sin(x)/x
    forms = [sym.sin(x) / x]
    current = sym.sin(x) / x
    for order in range(1, num_orders):
        derivative = sym.diff(current, x) / x
        forms.append(sym.simplify(derivative * (-x) ** order))
        current = sym.simplify(derivative)

    zeros = _spherical_bessel_zeros(num_orders, num_zeros)
    basis = []
    for order in range(num_orders):
        norms = 1 / np.sqrt(0.5 * spherical_jn(order + 1, zeros[order]) ** 2)
        basis.append(
            [
                sym.simplify(norms[i] * forms[order].subs(x, zeros[order, i] * x))
                for i in range(num_zeros)
            ]
        )
    return basis


class Envelope(nn.Module):
    r"""The smooth polynomial envelope of DimeNet,
    :math:`u(x) = \frac{1}{x} + a x^{p-1} + b x^p + c x^{p+1}` for
    :math:`x < 1` (and 0 beyond), with :math:`p` = ``exponent + 1``."""

    def __init__(self, exponent: int):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def __call__(self, x: mx.array) -> mx.array:
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x ** (p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        out = 1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2
        return out * (x < 1.0).astype(x.dtype)


class BesselBasisLayer(nn.Module):
    r"""The radial basis of DimeNet,
    :math:`\tilde{e}_n(d) = u(d/c) \sin(f_n d / c)`, with trainable
    frequencies :math:`f_n` initialized to :math:`n \pi`.

    Args:
        num_radial: Number of basis functions.
        cutoff: Cutoff distance :math:`c`. Default: ``5.0``
        envelope_exponent: Exponent of the :class:`Envelope`. Default: ``5``
    """

    def __init__(
        self, num_radial: int, cutoff: float = 5.0, envelope_exponent: int = 5
    ):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.freq = mx.arange(1, num_radial + 1, dtype=mx.float32) * math.pi

    def __call__(self, dist: mx.array) -> mx.array:
        dist = dist[:, None] / self.cutoff
        return self.envelope(dist) * mx.sin(self.freq * dist)


class SphericalBasisLayer(nn.Module):
    r"""The 2D spherical Fourier-Bessel basis of DimeNet, combining the
    normalized spherical Bessel functions of the distance :math:`d_{kj}` with
    the zonal spherical harmonics :math:`Y_l^0` of the angle between the
    edges :math:`kj` and :math:`ji`.

    It requires the ``sympy`` and ``scipy`` packages to build the basis.

    Args:
        num_spherical: Number of spherical harmonics.
        num_radial: Number of radial basis functions.
        cutoff: Cutoff distance. Default: ``5.0``
        envelope_exponent: Exponent of the :class:`Envelope`. Default: ``5``
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
    ):
        super().__init__()
        try:
            import sympy as sym
        except ImportError:
            raise ImportError(
                "DimeNet's spherical basis is built with 'sympy': pip install sympy"
            ) from None
        if num_radial > 64:
            raise ValueError("num_radial must be at most 64")
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        x = sym.symbols("x")
        modules = [{"sin": mx.sin, "cos": mx.cos}]
        self._bessel_funcs: List[Callable] = [
            sym.lambdify([x], expression, modules)
            for expressions in _bessel_basis_expressions(num_spherical, num_radial)
            for expression in expressions
        ]

    def _spherical_harmonics(self, angle: mx.array) -> mx.array:
        """Zonal spherical harmonics ``sqrt((2l+1)/4pi) P_l(cos(angle))``,
        shape ``[num_triplets, num_spherical]``."""
        cos = mx.cos(angle)
        legendre = [mx.ones_like(cos), cos]
        for degree in range(2, self.num_spherical):
            legendre.append(
                ((2 * degree - 1) * cos * legendre[-1] - (degree - 1) * legendre[-2])
                / degree
            )
        return mx.stack(
            [
                math.sqrt((2 * degree + 1) / (4 * math.pi)) * legendre[degree]
                for degree in range(self.num_spherical)
            ],
            axis=1,
        )

    def __call__(self, dist: mx.array, angle: mx.array, idx_kj: mx.array) -> mx.array:
        """Computes the basis of shape ``[num_triplets, num_spherical *
        num_radial]`` from the edge lengths ``dist``, the triplet angles
        ``angle`` and the edge ``kj`` of each triplet ``idx_kj``."""
        dist = dist / self.cutoff
        rbf = mx.stack([f(dist) for f in self._bessel_funcs], axis=1)
        rbf = self.envelope(dist)[:, None] * rbf
        cbf = self._spherical_harmonics(angle)
        n, k = self.num_spherical, self.num_radial
        out = rbf[idx_kj].reshape(-1, n, k) * cbf.reshape(-1, n, 1)
        return out.reshape(-1, n * k)


# Blocks ---------------------------------------------------------------------


class EmbeddingBlock(nn.Module):
    """Embedding of the directed edges ``j -> i`` of DimeNet, from the atom
    types of ``i`` and ``j`` and the radial basis of their distance."""

    def __init__(self, num_radial: int, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act
        self.emb = nn.Embedding(95, hidden_channels)
        self.emb.weight = mx.random.uniform(
            -math.sqrt(3), math.sqrt(3), (95, hidden_channels)
        )
        # initialized like torch.nn.Linear, as in PyG
        self.lin_rbf = nn.Linear(num_radial, hidden_channels)
        self.lin = nn.Linear(3 * hidden_channels, hidden_channels)

    def __call__(self, x: mx.array, rbf: mx.array, i: mx.array, j: mx.array):
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(mx.concatenate([x[i], x[j], rbf], axis=-1)))


class ResidualLayer(nn.Module):
    """Two linear layers with activations and a skip connection."""

    def __init__(self, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act
        self.lin1 = _GlorotOrthogonalLinear(hidden_channels, hidden_channels)
        self.lin2 = _GlorotOrthogonalLinear(hidden_channels, hidden_channels)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class _InteractionBase(nn.Module):
    def _skip(self, h: mx.array, x: mx.array) -> mx.array:
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)
        return h

    def _residual_layers(
        self, hidden_channels: int, num_before_skip: int, num_after_skip: int
    ):
        self.layers_before_skip = [
            ResidualLayer(hidden_channels, self.act) for _ in range(num_before_skip)
        ]
        self.lin = _GlorotOrthogonalLinear(hidden_channels, hidden_channels)
        self.layers_after_skip = [
            ResidualLayer(hidden_channels, self.act) for _ in range(num_after_skip)
        ]


class InteractionBlock(_InteractionBase):
    """The directional message passing block of DimeNet, with a bilinear
    layer combining the messages ``kj`` and the spherical basis of the
    triplets ``k -> j -> i``."""

    def __init__(
        self,
        hidden_channels: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
    ):
        super().__init__()
        self.act = act
        self.lin_rbf = _GlorotOrthogonalLinear(num_radial, hidden_channels, bias=False)
        self.lin_sbf = _GlorotOrthogonalLinear(
            num_spherical * num_radial, num_bilinear, bias=False
        )
        self.lin_kj = _GlorotOrthogonalLinear(hidden_channels, hidden_channels)
        self.lin_ji = _GlorotOrthogonalLinear(hidden_channels, hidden_channels)
        self.W = mx.random.normal(
            (hidden_channels, num_bilinear, hidden_channels),
            scale=2 / hidden_channels,
        )
        self._residual_layers(hidden_channels, num_before_skip, num_after_skip)

    def __call__(
        self,
        x: mx.array,
        rbf: mx.array,
        sbf: mx.array,
        idx_kj: mx.array,
        idx_ji: mx.array,
    ) -> mx.array:
        rbf = self.lin_rbf(rbf)
        sbf = self.lin_sbf(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x)) * rbf
        x_kj = x_kj[idx_kj]
        # einsum("wj,wl,ijl->wi", sbf, x_kj, W) as a single matmul
        outer = (sbf[:, :, None] * x_kj[:, None, :]).reshape(x_kj.shape[0], -1)
        x_kj = outer @ self.W.reshape(self.W.shape[0], -1).T
        x_kj = scatter(x_kj, idx_ji, x.shape[0], "add")
        return self._skip(x_ji + x_kj, x)


class InteractionPPBlock(_InteractionBase):
    """The interaction block of DimeNet++, which replaces the bilinear layer
    of DimeNet by down- and up-projections of the messages."""

    def __init__(
        self,
        hidden_channels: int,
        int_emb_size: int,
        basis_emb_size: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
    ):
        super().__init__()
        self.act = act
        self.lin_rbf1 = _GlorotOrthogonalLinear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = _GlorotOrthogonalLinear(
            basis_emb_size, hidden_channels, bias=False
        )
        self.lin_sbf1 = _GlorotOrthogonalLinear(
            num_spherical * num_radial, basis_emb_size, bias=False
        )
        self.lin_sbf2 = _GlorotOrthogonalLinear(
            basis_emb_size, int_emb_size, bias=False
        )
        self.lin_kj = _GlorotOrthogonalLinear(hidden_channels, hidden_channels)
        self.lin_ji = _GlorotOrthogonalLinear(hidden_channels, hidden_channels)
        self.lin_down = _GlorotOrthogonalLinear(
            hidden_channels, int_emb_size, bias=False
        )
        self.lin_up = _GlorotOrthogonalLinear(int_emb_size, hidden_channels, bias=False)
        self._residual_layers(hidden_channels, num_before_skip, num_after_skip)

    def __call__(
        self,
        x: mx.array,
        rbf: mx.array,
        sbf: mx.array,
        idx_kj: mx.array,
        idx_ji: mx.array,
    ) -> mx.array:
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        x_kj = x_kj * self.lin_rbf2(self.lin_rbf1(rbf))
        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf2(self.lin_sbf1(sbf))
        x_kj = x_kj[idx_kj] * sbf

        x_kj = scatter(x_kj, idx_ji, x.shape[0], "add")
        x_kj = self.act(self.lin_up(x_kj))
        return self._skip(x_ji + x_kj, x)


class _OutputBase(nn.Module):
    def _final(self, in_dim: int, out_channels: int, output_initializer: str):
        if output_initializer not in ("zeros", "glorot_orthogonal"):
            raise ValueError(
                "output_initializer must be 'zeros' or 'glorot_orthogonal' "
                f"(got '{output_initializer}')"
            )
        self.output_initializer = output_initializer
        self.lin = _GlorotOrthogonalLinear(in_dim, out_channels, bias=False)
        if output_initializer == "zeros":
            self.lin.weight = mx.zeros_like(self.lin.weight)


class OutputBlock(_OutputBase):
    """The output block of DimeNet: edge embeddings weighted by the radial
    basis, summed per atom and passed through an MLP."""

    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: OutputInitializer = "zeros",
    ):
        super().__init__()
        self.act = act
        self.lin_rbf = _GlorotOrthogonalLinear(num_radial, hidden_channels, bias=False)
        self.lins = [
            _GlorotOrthogonalLinear(hidden_channels, hidden_channels)
            for _ in range(num_layers)
        ]
        self._final(hidden_channels, out_channels, output_initializer)

    def __call__(
        self, x: mx.array, rbf: mx.array, i: mx.array, num_nodes: Optional[int] = None
    ) -> mx.array:
        x = scatter(self.lin_rbf(rbf) * x, i, num_nodes, "add")
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class OutputPPBlock(_OutputBase):
    """The output block of DimeNet++, with an up-projection before the
    MLP."""

    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_emb_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: OutputInitializer = "zeros",
    ):
        super().__init__()
        self.act = act
        self.lin_rbf = _GlorotOrthogonalLinear(num_radial, hidden_channels, bias=False)
        self.lin_up = _GlorotOrthogonalLinear(
            hidden_channels, out_emb_channels, bias=False
        )
        self.lins = [
            _GlorotOrthogonalLinear(out_emb_channels, out_emb_channels)
            for _ in range(num_layers)
        ]
        self._final(out_emb_channels, out_channels, output_initializer)

    def __call__(
        self, x: mx.array, rbf: mx.array, i: mx.array, num_nodes: Optional[int] = None
    ) -> mx.array:
        x = scatter(self.lin_rbf(rbf) * x, i, num_nodes, "add")
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


def triplets(edge_index: mx.array, num_nodes: int) -> Tuple[mx.array, ...]:
    r"""Computes the triplets ``k -> j -> i`` of a graph (with ``k != i``),
    in the same order as PyG.

    Args:
        edge_index: The edges ``j -> i``, of shape ``[2, num_edges]``.
        num_nodes: The number of nodes.

    Returns:
        ``(i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji)``: the target and
        source of every edge, the three nodes of every triplet, and the
        indices of its edges ``k -> j`` and ``j -> i``.
    """
    row, col = _to_numpy(edge_index).astype(np.int64)  # j -> i
    num_edges = row.size
    # The edges into each node j, sorted by source k (as in PyG's sparse
    # adjacency matrix), and for each edge j -> i the edges into j.
    order = np.lexsort((row, col))
    counts = np.bincount(col, minlength=num_nodes)
    ptr = np.concatenate([[0], np.cumsum(counts)])
    num_triplets = counts[row]
    total = int(num_triplets.sum())
    first = np.repeat(np.cumsum(num_triplets) - num_triplets, num_triplets)
    position = np.repeat(ptr[row], num_triplets) + np.arange(total) - first
    idx_kj = order[position]
    idx_ji = np.repeat(np.arange(num_edges), num_triplets)
    idx_i, idx_j, idx_k = col[idx_ji], row[idx_ji], row[idx_kj]
    mask = idx_i != idx_k
    out = (col, row, idx_i[mask], idx_j[mask], idx_k[mask], idx_kj[mask], idx_ji[mask])
    return tuple(mx.array(a.astype(np.int32)) for a in out)


def _safe_norm(x: mx.array) -> mx.array:
    """Euclidean norm over the last axis, with a zero gradient at 0."""
    sq = (x * x).sum(axis=-1)
    positive = sq > 0
    return mx.where(positive, mx.sqrt(mx.where(positive, sq, 1.0)), 0.0)


def _cross(a: mx.array, b: mx.array) -> mx.array:
    a0, a1, a2 = a[:, 0], a[:, 1], a[:, 2]
    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
    return mx.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=1)


class DimeNet(nn.Module):
    r"""The directional message passing neural network (DimeNet) from the
    `"Directional Message Passing for Molecular Graphs"
    <https://arxiv.org/abs/2003.03123>`_ paper.

    DimeNet embeds the directed edges ``j -> i`` of the graph of atoms closer
    than ``cutoff``, and updates them with messages from the edges ``k -> j``
    transformed with the angle :math:`\angle kji` (with a 2D spherical
    Fourier-Bessel basis), which makes the model invariant to rotations.

    The model follows PyG's :class:`~torch_geometric.nn.models.DimeNet`
    (same arguments, parameters and initialization). It requires the
    ``sympy`` and ``scipy`` packages. The interaction graph and the triplets
    are computed on the host (see :func:`~mlx_graphs.utils.radius_graph` and
    :func:`triplets`), the distances and angles on the device, so that the
    output can be differentiated with respect to ``pos``.

    .. note::

        The mini-batches of molecules all have different numbers of atoms,
        edges and triplets, so MLX rarely reuses the buffers it frees and its buffer
        cache can grow to tens of GB during training. Bound it, e.g. with
        ``mx.set_cache_limit(1024**3)`` before training.

    Args:
        hidden_channels: Hidden embedding size.
        out_channels: Size of each output sample.
        num_blocks: Number of building blocks.
        num_bilinear: Size of the bilinear layer tensor.
        num_spherical: Number of spherical harmonics.
        num_radial: Number of radial basis functions.
        cutoff: Cutoff distance for interatomic interactions. Default: ``5.0``
        max_num_neighbors: The maximum number of neighbors of each atom
            within ``cutoff``. Default: ``32``
        envelope_exponent: Shape of the smooth cutoff. Default: ``5``
        num_before_skip: Number of residual layers in the interaction blocks
            before the skip connection. Default: ``1``
        num_after_skip: Number of residual layers in the interaction blocks
            after the skip connection. Default: ``2``
        num_output_layers: Number of linear layers for the output blocks.
            Default: ``3``
        act: The activation function, a name (e.g. ``"swish"``) or a
            callable. Default: ``"swish"``
        output_initializer: The initialization of the output layer,
            ``"zeros"`` or ``"glorot_orthogonal"``. Default: ``"zeros"``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import DimeNet

        model = DimeNet(
            hidden_channels=32, out_channels=1, num_blocks=2, num_bilinear=4,
            num_spherical=3, num_radial=4,
        )
        z = mx.array([6, 1, 1, 1, 1])
        pos = mx.random.normal((5, 3))
        model(z, pos, mx.zeros(5, dtype=mx.int32)).shape
        >>> (1, 1)
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = "swish",
        output_initializer: OutputInitializer = "zeros",
    ):
        super().__init__()
        if num_spherical < 2:
            raise ValueError("'num_spherical' should be greater than 1")
        act = _resolve_act(act)
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, cutoff, envelope_exponent
        )
        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)
        self.output_blocks = [
            OutputBlock(
                num_radial,
                hidden_channels,
                out_channels,
                num_output_layers,
                act,
                output_initializer,
            )
            for _ in range(num_blocks + 1)
        ]
        self.interaction_blocks = [
            InteractionBlock(
                hidden_channels,
                num_bilinear,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                act,
            )
            for _ in range(num_blocks)
        ]

    def _angles(self, pos, idx_i, idx_j, idx_k) -> mx.array:
        pos_ji = pos[idx_j] - pos[idx_i]
        pos_ki = pos[idx_k] - pos[idx_i]
        a = (pos_ji * pos_ki).sum(axis=-1)
        b = _safe_norm(_cross(pos_ji, pos_ki))
        return mx.arctan2(b, a)

    def __call__(
        self,
        z: mx.array,
        pos: mx.array,
        batch_indices: Optional[mx.array] = None,
    ) -> mx.array:
        """Predicts the target properties of each molecule.

        Args:
            z: Atomic number of each atom, of shape ``[num_atoms]``.
            pos: Coordinates of each atom, of shape ``[num_atoms, 3]``.
            batch_indices: Molecule of each atom, of shape ``[num_atoms]``.
                Default: ``None`` (a single molecule).

        Returns:
            The predictions, of shape ``[num_molecules, out_channels]``, or
            ``[out_channels]`` if ``batch_indices`` is ``None`` (as in PyG).
        """
        num_nodes = z.shape[0]
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch_indices=batch_indices,
            max_num_neighbors=self.max_num_neighbors,
        )
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edge_index, num_nodes)

        dist = _safe_norm(pos[i] - pos[j])
        angle = self._angles(pos, idx_i, idx_j, idx_k)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=num_nodes)
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=num_nodes)

        if batch_indices is None:
            return P.sum(axis=0)
        return scatter(P, batch_indices, int(batch_indices.max().item()) + 1, "add")


class DimeNetPlusPlus(DimeNet):
    r"""DimeNet++ from the `"Fast and Uncertainty-Aware Directional Message
    Passing for Non-Equilibrium Molecules" <https://arxiv.org/abs/2011.14115>`_
    paper.

    DimeNet++ replaces the bilinear layer of :class:`DimeNet` by down- and
    up-projections, which makes it several times faster and more accurate.
    The model follows PyG's :class:`~torch_geometric.nn.models.DimeNetPlusPlus`.

    Args:
        hidden_channels: Hidden embedding size.
        out_channels: Size of each output sample.
        num_blocks: Number of building blocks.
        int_emb_size: Size of embedding in the interaction block.
        basis_emb_size: Size of basis embedding in the interaction block.
        out_emb_channels: Size of embedding in the output block.
        num_spherical: Number of spherical harmonics.
        num_radial: Number of radial basis functions.
        cutoff: Cutoff distance for interatomic interactions. Default: ``5.0``
        max_num_neighbors: The maximum number of neighbors of each atom
            within ``cutoff``. Default: ``32``
        envelope_exponent: Shape of the smooth cutoff. Default: ``5``
        num_before_skip: Number of residual layers in the interaction blocks
            before the skip connection. Default: ``1``
        num_after_skip: Number of residual layers in the interaction blocks
            after the skip connection. Default: ``2``
        num_output_layers: Number of linear layers for the output blocks.
            Default: ``3``
        act: The activation function. Default: ``"swish"``
        output_initializer: The initialization of the output layer,
            ``"zeros"`` or ``"glorot_orthogonal"``. Default: ``"zeros"``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import DimeNetPlusPlus

        model = DimeNetPlusPlus(
            hidden_channels=32, out_channels=1, num_blocks=2, int_emb_size=16,
            basis_emb_size=4, out_emb_channels=32, num_spherical=3,
            num_radial=4,
        )
        z = mx.array([6, 1, 1, 1, 1, 8, 1, 1])
        pos = mx.random.normal((8, 3))
        batch_indices = mx.array([0, 0, 0, 0, 0, 1, 1, 1])
        model(z, pos, batch_indices).shape
        >>> (2, 1)
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        int_emb_size: int,
        basis_emb_size: int,
        out_emb_channels: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = "swish",
        output_initializer: OutputInitializer = "zeros",
    ):
        act = _resolve_act(act)
        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=0,
            num_bilinear=1,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
            output_initializer=output_initializer,
        )
        self.num_blocks = num_blocks
        self.output_blocks = [
            OutputPPBlock(
                num_radial,
                hidden_channels,
                out_emb_channels,
                out_channels,
                num_output_layers,
                act,
                output_initializer,
            )
            for _ in range(num_blocks + 1)
        ]
        self.interaction_blocks = [
            InteractionPPBlock(
                hidden_channels,
                int_emb_size,
                basis_emb_size,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                act,
            )
            for _ in range(num_blocks)
        ]

    def _angles(self, pos, idx_i, idx_j, idx_k) -> mx.array:
        pos_jk = pos[idx_j] - pos[idx_k]
        pos_ij = pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(axis=-1)
        b = _safe_norm(_cross(pos_ij, pos_jk))
        return mx.arctan2(b, a)
