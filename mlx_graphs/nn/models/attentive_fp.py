from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.conv.gat_conv import GATConv
from mlx_graphs.nn.linear import Linear, glorot
from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.nn.models.tgn import GRUCell
from mlx_graphs.nn.pooling import global_add_pool
from mlx_graphs.utils.scatter import scatter

__all__ = ["AttentiveFP", "GATEConv"]


class GATEConv(MessagePassing):
    r"""The graph attention layer with edge features of the first layer of
    :class:`AttentiveFP`.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}
        \mathbf{W}_2 \mathbf{x}_j + \mathbf{b}, \qquad
        \alpha_{i,j} = \mathrm{softmax}_j \left( \mathrm{LeakyReLU} \left(
        \mathbf{a}_l^{\top} \mathrm{LeakyReLU}(\mathbf{W}_1
        [\mathbf{x}_j \, \Vert \, \mathbf{e}_{j,i}]) +
        \mathbf{a}_r^{\top} \mathbf{x}_i \right) \right)

    Args:
        in_channels: Size of the input node features.
        out_channels: Size of the output node features.
        edge_dim: Size of the edge features.
        dropout: Dropout probability of the attention coefficients.
            Default: ``0.0``
    """

    def __init__(
        self, in_channels: int, out_channels: int, edge_dim: int, dropout: float = 0.0
    ):
        super().__init__(aggr="add")
        self.att_l = glorot((1, out_channels))
        self.att_r = glorot((1, in_channels))
        self.lin1 = Linear(in_channels + edge_dim, out_channels, bias=False)
        self.lin2 = Linear(out_channels, out_channels, bias=False)
        self.bias = mx.zeros((out_channels,))
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self, edge_index: mx.array, node_features: mx.array, edge_features: mx.array
    ) -> mx.array:
        src, dst = edge_index
        x_j = node_features[src]
        h = nn.leaky_relu(self.lin1(mx.concatenate([x_j, edge_features], axis=-1)))
        alpha_j = (h @ self.att_l.T).squeeze(-1)
        alpha_i = (node_features @ self.att_r.T).squeeze(-1)[dst]
        alpha = nn.leaky_relu(alpha_j + alpha_i)
        alpha = scatter(alpha, dst, node_features.shape[0], "softmax")
        alpha = self.dropout(alpha)
        out = self.propagate(edge_index, node_features, message_kwargs={"alpha": alpha})
        return out + self.bias

    def message(self, src_features: mx.array, dst_features: mx.array, alpha: mx.array):
        return self.lin2(src_features) * alpha[:, None]


class AttentiveFP(nn.Module):
    r"""The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Atom embeddings are updated by graph attention layers (a
    :class:`GATEConv` using the bond features, then
    :class:`~mlx_graphs.nn.GATConv` layers) followed by GRU cells; the
    molecule embedding is then refined for ``num_timesteps`` steps by
    attending over the atoms of the molecule (a GAT layer from the atoms to a
    virtual molecule node, followed by a GRU cell).

    The model follows PyG's :class:`~torch_geometric.nn.models.AttentiveFP`
    (same arguments, parameters and initialization).

    Args:
        in_channels: Size of each input sample.
        hidden_channels: Hidden node feature dimensionality.
        out_channels: Size of each output sample.
        edge_dim: Edge feature dimensionality.
        num_layers: Number of GNN layers.
        num_timesteps: Number of iterative refinement steps for global
            readout.
        dropout: Dropout probability. Default: ``0.0``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import AttentiveFP

        model = AttentiveFP(
            in_channels=9, hidden_channels=64, out_channels=1, edge_dim=3,
            num_layers=2, num_timesteps=2,
        )
        x = mx.random.normal((5, 9))
        edge_index = mx.array([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
        edge_features = mx.random.normal((6, 3))
        batch_indices = mx.array([0, 0, 0, 1, 1])
        model(x, edge_index, edge_features, batch_indices).shape
        >>> (2, 1)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps

        # initialized like torch.nn.Linear (PyG's default Linear initializer)
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = [
            GATConv(
                hidden_channels,
                hidden_channels,
                dropout=dropout,
                add_self_loops=False,
                negative_slope=0.01,
            )
            for _ in range(num_layers - 1)
        ]
        self.atom_grus = [
            GRUCell(hidden_channels, hidden_channels) for _ in range(num_layers - 1)
        ]
        self.mol_conv = GATConv(
            hidden_channels,
            hidden_channels,
            dropout=dropout,
            add_self_loops=False,
            negative_slope=0.01,
        )
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def __call__(
        self,
        node_features: mx.array,
        edge_index: mx.array,
        edge_features: mx.array,
        batch_indices: Optional[mx.array] = None,
    ) -> mx.array:
        """Predicts the target properties of each molecule.

        The arguments are in PyG's order ``(x, edge_index, edge_attr, batch)``.

        Args:
            node_features: Atom features, of shape ``[num_atoms, in_channels]``.
            edge_index: Bonds, of shape ``[2, num_bonds]``.
            edge_features: Bond features, of shape ``[num_bonds, edge_dim]``.
            batch_indices: Molecule of each atom, of shape ``[num_atoms]``.
                Default: ``None`` (a single molecule).

        Returns:
            The predictions, of shape ``[num_molecules, out_channels]``.
        """
        if batch_indices is None:
            batch_indices = mx.zeros(node_features.shape[0], dtype=mx.int32)

        # Atom embedding
        x = nn.leaky_relu(self.lin1(node_features))
        h = nn.elu(self.gate_conv(edge_index, x, edge_features))
        h = self.dropout(h)
        x = nn.relu(self.gru(h, x))

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = nn.elu(conv(edge_index, x))
            h = self.dropout(h)
            x = nn.relu(gru(h, x))

        # Molecule embedding: attention from the atoms to their molecule
        num_graphs = int(batch_indices.max().item()) + 1
        atoms = mx.arange(batch_indices.shape[0], dtype=batch_indices.dtype)
        mol_edge_index = mx.stack([atoms, batch_indices], axis=0)

        out = nn.relu(global_add_pool(x, batch_indices, num_graphs))
        for _ in range(self.num_timesteps):
            h = nn.elu(self.mol_conv(mol_edge_index, (x, out)))
            h = self.dropout(h)
            out = nn.relu(self.mol_gru(h, out))

        # Predictor
        out = self.dropout(out)
        return self.lin2(out)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"out_channels={self.out_channels}, "
            f"edge_dim={self.edge_dim}, "
            f"num_layers={self.num_layers}, "
            f"num_timesteps={self.num_timesteps})"
        )
