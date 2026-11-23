import math
from typing import Callable, Literal, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_graphs.nn.linear import Linear
from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.nn.pooling import global_add_pool, global_mean_pool
from mlx_graphs.utils.spatial import radius_graph

__all__ = [
    "SchNet",
    "RadiusInteractionGraph",
    "InteractionBlock",
    "CFConv",
    "GaussianSmearing",
    "ShiftedSoftplus",
    "cosine_cutoff",
]


def cosine_cutoff(distances: mx.array, cutoff: float) -> mx.array:
    r"""The cosine cutoff of SchNet,
    :math:`C(d) = \frac{1}{2} \left(\cos\left(\frac{\pi d}{c}\right) + 1\right)`.

    As in PyG's :class:`~torch_geometric.nn.models.SchNet`, it is not set to
    zero beyond the cutoff: the interaction graph only contains pairs of atoms
    closer than ``cutoff``.

    Args:
        distances: Interatomic distances, of any shape.
        cutoff: The cutoff distance :math:`c`.
    """
    return 0.5 * (mx.cos(distances * (math.pi / cutoff)) + 1.0)


class ShiftedSoftplus(nn.Module):
    r"""The shifted softplus activation
    :math:`\mathrm{ssp}(x) = \ln(1 + e^x) - \ln 2`, which is zero at zero."""

    def __init__(self):
        super().__init__()
        self.shift = math.log(2.0)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.softplus(x) - self.shift


class GaussianSmearing(nn.Module):
    r"""Expands distances in a basis of Gaussians
    :math:`\exp(-\gamma (d - \mu_k)^2)`, with ``num_gaussians`` centers
    :math:`\mu_k` evenly spaced in ``[start, stop]`` and
    :math:`\gamma = \frac{1}{2 \Delta\mu^2}`.

    Args:
        start: The first center. Default: ``0.0``
        stop: The last center. Default: ``5.0``
        num_gaussians: The number of Gaussians. Default: ``50``
    """

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = np.linspace(start, stop, num_gaussians, dtype=np.float32)
        self.coeff = -0.5 / float(offset[1] - offset[0]) ** 2
        # a fixed buffer, as in PyG
        self.offset = mx.array(offset)
        self.freeze(keys=["offset"], recurse=False)

    def __call__(self, distances: mx.array) -> mx.array:
        """Expands ``distances`` of shape ``[num_edges]`` into
        ``[num_edges, num_gaussians]``."""
        diff = distances.reshape(-1, 1) - self.offset.reshape(1, -1)
        return mx.exp(self.coeff * diff * diff)


class RadiusInteractionGraph(nn.Module):
    r"""Connects every atom to the atoms closer than ``cutoff`` (with
    :func:`~mlx_graphs.utils.radius_graph`), and returns the interatomic
    distances of these edges.

    The edges are computed on the host and do not depend on the positions
    differentiably, but the distances do (e.g. to compute forces as gradients
    of the energy with respect to ``pos``).

    Args:
        cutoff: The cutoff distance. Default: ``10.0``
        max_num_neighbors: The maximum number of neighbors of each atom (the
            closest ones are kept). Default: ``32``
    """

    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def __call__(
        self, pos: mx.array, batch_indices: Optional[mx.array] = None
    ) -> Tuple[mx.array, mx.array]:
        """Returns ``(edge_index, edge_weight)`` with the edges (from neighbor
        to center atom) and their lengths."""
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch_indices=batch_indices,
            max_num_neighbors=self.max_num_neighbors,
        )
        row, col = edge_index
        edge_weight = mx.linalg.norm(pos[row] - pos[col], axis=-1)
        return edge_index, edge_weight


class CFConv(MessagePassing):
    r"""The continuous-filter convolution of SchNet.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_2 \sum_{j \in \mathcal{N}(i)}
        \mathbf{W}_1 \mathbf{x}_j \odot h(\mathbf{e}_{j,i}) \, C(d_{j,i})

    where :math:`h` is a filter-generating network applied to the expanded
    distances :math:`\mathbf{e}_{j,i}` and :math:`C` is the
    :func:`cosine_cutoff`.

    Args:
        in_channels: Size of the input features.
        out_channels: Size of the output features.
        num_filters: Number of filters.
        nn: The filter-generating network, mapping the expanded distances to
            ``num_filters`` features.
        cutoff: The cutoff distance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: Callable,
        cutoff: float,
    ):
        super().__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weight: mx.array,
        edge_attr: mx.array,
    ) -> mx.array:
        """Computes the convolution.

        Args:
            edge_index: Edges of shape ``[2, num_edges]``.
            node_features: Atom features of shape ``[num_atoms, in_channels]``.
            edge_weight: Interatomic distances of shape ``[num_edges]``.
            edge_attr: Expanded distances of shape ``[num_edges, num_basis]``.
        """
        C = cosine_cutoff(edge_weight, self.cutoff)
        W = self.nn(edge_attr) * C.reshape(-1, 1)
        x = self.lin1(node_features)
        x = self.propagate(edge_index, x, message_kwargs={"W": W})
        return self.lin2(x)

    def message(self, src_features: mx.array, dst_features: mx.array, W: mx.array):
        return src_features * W


class InteractionBlock(nn.Module):
    r"""An interaction block of SchNet: a :class:`CFConv` (whose filters are
    generated by an MLP ``Linear -> ShiftedSoftplus -> Linear`` of the
    expanded distances), a shifted softplus and a linear layer.

    Args:
        hidden_channels: Size of the atom features.
        num_gaussians: Size of the expanded distances.
        num_filters: Number of filters.
        cutoff: The cutoff distance.
    """

    def __init__(
        self, hidden_channels: int, num_gaussians: int, num_filters: int, cutoff: float
    ):
        super().__init__()
        mlp = nn.Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        # The MLP is stored once, in the convolution (PyG's state dict has it
        # twice, as `mlp` and `conv.nn`).
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

    @property
    def mlp(self) -> nn.Sequential:
        """The filter-generating network (``conv.nn``)."""
        return self.conv.nn

    def __call__(
        self,
        node_features: mx.array,
        edge_index: mx.array,
        edge_weight: mx.array,
        edge_attr: mx.array,
    ) -> mx.array:
        x = self.conv(edge_index, node_features, edge_weight, edge_attr)
        return self.lin(self.act(x))


class SchNet(nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper, with
    interaction blocks of the form

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    where :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` the interatomic distances.

    The model follows PyG's :class:`~torch_geometric.nn.models.SchNet` (same
    arguments, parameters and initialization). It takes the atomic numbers
    ``z`` and positions ``pos`` of the atoms and returns one prediction per
    molecule. The interaction graph connects the atoms closer than ``cutoff``
    (with :func:`~mlx_graphs.utils.radius_graph`, which keeps the
    ``max_num_neighbors`` *closest* neighbors when there are more).

    Args:
        hidden_channels: Hidden embedding size. Default: ``128``
        num_filters: The number of filters to use. Default: ``128``
        num_interactions: The number of interaction blocks. Default: ``6``
        num_gaussians: The number of gaussians :math:`\mu`. Default: ``50``
        cutoff: Cutoff distance for interatomic interactions. Default: ``10.0``
        interaction_graph: The function computing the interaction graph and
            the interatomic distances, given ``pos`` and ``batch_indices``. It
            returns ``(edge_index, edge_weight)``. If ``None``, a
            :class:`RadiusInteractionGraph` with ``cutoff`` and
            ``max_num_neighbors``. Default: ``None``
        max_num_neighbors: The maximum number of neighbors of each atom in the
            interaction graph. Default: ``32``
        readout: The global aggregation, ``"add"`` or ``"mean"``.
            Default: ``"add"``
        dipole: If ``True``, predicts the magnitude of the dipole moment
            (e.g. target 0 of :class:`~mlx_graphs.datasets.QM9`), from atomic
            charges and the center of mass. Requires the ``ase`` package.
            Default: ``False``
        mean: The mean of the property to predict. Default: ``None``
        std: The standard deviation of the property to predict.
            Default: ``None``
        atomref: The reference of single-atom properties, of shape
            ``[max_atomic_number, 1]`` (e.g. ``QM9.atomref(target)``).
            Default: ``None``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import SchNet

        model = SchNet(hidden_channels=64, num_interactions=3, cutoff=5.0)
        z = mx.array([6, 1, 1, 1, 1, 8, 1, 1])  # CH4 and H2O
        pos = mx.random.normal((8, 3))
        batch_indices = mx.array([0, 0, 0, 0, 0, 1, 1, 1])
        model(z, pos, batch_indices).shape
        >>> (2, 1)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: Literal["add", "sum", "mean"] = "add",
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: Optional[mx.array] = None,
    ):
        super().__init__()
        if readout not in ("add", "sum", "mean"):
            raise ValueError(f"readout must be 'add' or 'mean' (got '{readout}')")
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.dipole = dipole
        self.readout = "add" if dipole else readout
        self.mean = mean
        self.std = std
        self.scale = None

        if dipole:
            try:
                import ase.data
            except ImportError:
                raise ImportError(
                    "SchNet(dipole=True) needs the atomic masses of the 'ase' "
                    "package: pip install ase"
                ) from None
            self.atomic_mass = mx.array(ase.data.atomic_masses.astype(np.float32))

        # N(0, 1) as torch.nn.Embedding. z == 0 is a padding atom: its
        # embedding is zero and gets no gradient (torch's padding_idx=0).
        self.embedding = nn.Embedding(100, hidden_channels)
        weight = mx.random.normal((100, hidden_channels))
        self.embedding.weight = weight * (mx.arange(100) != 0)[:, None]

        if interaction_graph is not None:
            self.interaction_graph = interaction_graph
        else:
            self.interaction_graph = RadiusInteractionGraph(cutoff, max_num_neighbors)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = [
            InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            for _ in range(num_interactions)
        ]
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        if atomref is not None:
            self.atomref = nn.Embedding(100, 1)
            self.atomref.weight = mx.array(atomref).reshape(100, 1).astype(mx.float32)

        if dipole:
            self.freeze(keys=["atomic_mass"], recurse=False)

    def __call__(
        self,
        z: mx.array,
        pos: mx.array,
        batch_indices: Optional[mx.array] = None,
    ) -> mx.array:
        """Predicts the target property of each molecule.

        Args:
            z: Atomic number of each atom, of shape ``[num_atoms]``.
            pos: Coordinates of each atom, of shape ``[num_atoms, 3]``.
            batch_indices: Molecule of each atom, of shape ``[num_atoms]``.
                Default: ``None`` (a single molecule).

        Returns:
            The predictions, of shape ``[num_molecules, 1]``.
        """
        if batch_indices is None:
            batch_indices = mx.zeros(z.shape, dtype=mx.int32)

        h = self.embedding(z) * (z != 0)[:, None]
        edge_index, edge_weight = self.interaction_graph(pos, batch_indices)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin2(self.act(self.lin1(h)))

        if self.dipole:
            # charges times positions relative to the center of mass
            mass = self.atomic_mass[z].reshape(-1, 1)
            M = global_add_pool(mass, batch_indices)
            c = global_add_pool(mass * pos, batch_indices) / M
            h = h * (pos - c[batch_indices])

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and "atomref" in self:
            h = h + self.atomref(z)

        if self.readout == "mean":
            out = global_mean_pool(h, batch_indices)
        else:
            out = global_add_pool(h, batch_indices)

        if self.dipole:
            # the norm, with a zero gradient at 0 as in torch (e.g. one atom)
            sq = (out * out).sum(axis=-1, keepdims=True)
            out = mx.where(sq > 0, mx.sqrt(mx.where(sq > 0, sq, 1.0)), 0.0)

        if self.scale is not None:
            out = self.scale * out

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_filters={self.num_filters}, "
            f"num_interactions={self.num_interactions}, "
            f"num_gaussians={self.num_gaussians}, "
            f"cutoff={self.cutoff})"
        )
