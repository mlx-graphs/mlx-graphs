# Convert

<!-- api mlx_graphs.utils.convert -->
to_networkx
from_networkx
<!-- /api -->

<!-- api mlx_graphs.utils -->
from_pyg
to_pyg
<!-- /api -->

## Molecules

Conversions between molecules (RDKit molecules or SMILES strings) and
[`GraphData`][mlx_graphs.data.data.GraphData], with the featurization of PyG.
RDKit is an optional dependency, imported only when one of these functions is
called (`pip install rdkit`).

<!-- api mlx_graphs.utils -->
from_smiles
to_smiles
from_rdmol
to_rdmol
<!-- /api -->

The node features of `from_smiles` are, for each atom, the indices of its
properties in the lists of `mlx_graphs.utils.x_map`, and the edge features the
indices of the bond properties in `mlx_graphs.utils.e_map` (the same tables as
PyG's `torch_geometric.utils.smiles`):

| Map     | Features, in column order                                                                                                  |
|---------|----------------------------------------------------------------------------------------------------------------------------|
| `x_map` | `atomic_num`, `chirality`, `degree`, `formal_charge`, `num_hs`, `num_radical_electrons`, `hybridization`, `is_aromatic`, `is_in_ring` |
| `e_map` | `bond_type`, `stereo`, `is_conjugated`                                                                                     |
