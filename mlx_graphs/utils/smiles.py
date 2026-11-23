"""Conversions between molecules (RDKit molecules or SMILES strings) and
:class:`~mlx_graphs.data.GraphData`, with the featurization of PyG.

RDKit is an optional dependency: it is only imported when one of these
functions is called (``pip install rdkit``).
"""

from typing import TYPE_CHECKING, Any, Dict, List

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:  # mlx_graphs.data imports mlx_graphs.utils
    from mlx_graphs.data import GraphData

__all__ = ["x_map", "e_map", "from_rdmol", "to_rdmol", "from_smiles", "to_smiles"]

#: Possible values of each atom feature: the node features of
#: :func:`from_smiles` are the indices of the atom properties in these lists.
x_map: Dict[str, List[Any]] = {
    "atomic_num": list(range(0, 119)),
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "CHI_TETRAHEDRAL",
        "CHI_ALLENE",
        "CHI_SQUAREPLANAR",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
    ],
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

#: Possible values of each bond feature: the edge features of
#: :func:`from_smiles` are the indices of the bond properties in these lists.
e_map: Dict[str, List[Any]] = {
    "bond_type": [
        "UNSPECIFIED",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "AROMATIC",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOANY",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
    ],
    "is_conjugated": [False, True],
}


def _import_rdkit():
    try:
        from rdkit import Chem, RDLogger
    except ImportError:
        raise ImportError(
            "rdkit is required to convert molecules to and from graphs: "
            "run `pip install rdkit`"
        ) from None
    return Chem, RDLogger


def from_rdmol(mol: Any) -> "GraphData":
    r"""Converts an RDKit molecule (``rdkit.Chem.Mol``) to a
    :class:`~mlx_graphs.data.GraphData`, with the features of PyG's
    ``from_rdmol``.

    Each atom has 9 categorical features, the indices in the lists of
    :data:`x_map` of its atomic number, chirality, total degree, formal
    charge, total number of hydrogens, number of radical electrons,
    hybridization, aromaticity and ring membership. Each bond gives two
    directed edges, with 3 categorical features: the indices in the lists of
    :data:`e_map` of its type, stereo configuration and conjugation. The edges
    are sorted by source then destination node, as in PyG.

    Args:
        mol: The RDKit molecule.

    Returns:
        The graph, with ``node_features`` of shape ``[num_atoms, 9]`` and
        ``edge_features`` of shape ``[num_edges, 3]`` (both ``int32``, PyG
        uses ``int64``) and an ``int32`` ``edge_index``.
    """
    from mlx_graphs.data import GraphData

    arrays = _rdmol_arrays(mol)
    return GraphData(**{k: mx.array(v) for k, v in arrays.items()})


def _rdmol_arrays(mol: Any) -> Dict[str, np.ndarray]:
    """The ``edge_index``, ``node_features`` and ``edge_features`` of
    :func:`from_rdmol`, as NumPy arrays."""
    Chem, _ = _import_rdkit()
    if not isinstance(mol, Chem.Mol):
        raise TypeError(f"Expected an rdkit.Chem.Mol (got {type(mol).__name__})")

    xs = []
    for atom in mol.GetAtoms():
        xs.append(
            [
                x_map["atomic_num"].index(atom.GetAtomicNum()),
                x_map["chirality"].index(str(atom.GetChiralTag())),
                x_map["degree"].index(atom.GetTotalDegree()),
                x_map["formal_charge"].index(atom.GetFormalCharge()),
                x_map["num_hs"].index(atom.GetTotalNumHs()),
                x_map["num_radical_electrons"].index(atom.GetNumRadicalElectrons()),
                x_map["hybridization"].index(str(atom.GetHybridization())),
                x_map["is_aromatic"].index(atom.GetIsAromatic()),
                x_map["is_in_ring"].index(atom.IsInRing()),
            ]
        )
    x = np.array(xs, dtype=np.int32).reshape(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        e = [
            e_map["bond_type"].index(str(bond.GetBondType())),
            e_map["stereo"].index(str(bond.GetStereo())),
            e_map["is_conjugated"].index(bond.GetIsConjugated()),
        ]
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = np.array(edge_indices, dtype=np.int64).reshape(-1, 2).T
    edge_attr = np.array(edge_attrs, dtype=np.int32).reshape(-1, 3)
    if edge_index.size > 0:  # sort the edges, as in PyG
        perm = np.argsort(edge_index[0] * x.shape[0] + edge_index[1], kind="stable")
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
    return {
        "edge_index": edge_index.astype(np.int32),
        "node_features": x,
        "edge_features": edge_attr,
    }


def _smiles_to_rdmol(smiles: str, with_hydrogen: bool, kekulize: bool) -> Any:
    Chem, RDLogger = _import_rdkit()
    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles("")
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)
    return mol


def from_smiles(
    smiles: str, with_hydrogen: bool = False, kekulize: bool = False
) -> "GraphData":
    r"""Converts a SMILES string to a :class:`~mlx_graphs.data.GraphData`,
    with the featurization of PyG's ``from_smiles`` (see :func:`from_rdmol`).

    An invalid SMILES string gives an empty molecule (no atoms), as in PyG.
    Requires ``rdkit``.

    Args:
        smiles: The SMILES string.
        with_hydrogen: Whether to add the hydrogens to the graph.
            Default: ``False``
        kekulize: Whether to convert the aromatic bonds to single and double
            bonds. Default: ``False``

    Returns:
        The graph, with ``int32`` ``node_features`` of shape ``[num_atoms,
        9]`` and ``edge_features`` of shape ``[num_edges, 3]``, and the
        SMILES string in its ``smiles`` attribute.

    Example:

    .. code-block:: python

        from mlx_graphs.utils import from_smiles, to_smiles

        graph = from_smiles("CCO")
        >>> GraphData(
                edge_index(shape=(2, 4), int32)
                node_features(shape=(3, 9), int32)
                edge_features(shape=(4, 3), int32)
                smiles=CCO)

        to_smiles(graph)
        >>> 'CCO'
    """
    data = from_rdmol(_smiles_to_rdmol(smiles, with_hydrogen, kekulize))
    data.smiles = smiles
    return data


def _numpy(array) -> np.ndarray:
    return np.array(array) if isinstance(array, mx.array) else np.asarray(array)


def to_rdmol(data: "GraphData", kekulize: bool = False) -> Any:
    r"""Converts a :class:`~mlx_graphs.data.GraphData` with the features of
    :func:`from_rdmol` back to an RDKit molecule, as PyG's ``to_rdmol``.

    Args:
        data: The molecular graph.
        kekulize: Whether to convert the aromatic bonds to single and double
            bonds. Default: ``False``

    Returns:
        The sanitized ``rdkit.Chem.Mol``.
    """
    Chem, _ = _import_rdkit()
    if data.node_features is None or data.edge_features is None:
        raise ValueError("The graph needs node_features and edge_features")

    x = _numpy(data.node_features).astype(np.int64)
    edge_index = _numpy(data.edge_index).astype(np.int64)
    edge_attr = _numpy(data.edge_features).astype(np.int64)

    mol = Chem.RWMol()
    for i in range(x.shape[0]):
        atom = Chem.Atom(int(x[i, 0]))
        atom.SetChiralTag(Chem.rdchem.ChiralType.values[int(x[i, 1])])
        atom.SetFormalCharge(x_map["formal_charge"][int(x[i, 3])])
        atom.SetNumExplicitHs(x_map["num_hs"][int(x[i, 4])])
        atom.SetNumRadicalElectrons(x_map["num_radical_electrons"][int(x[i, 5])])
        atom.SetHybridization(Chem.rdchem.HybridizationType.values[int(x[i, 6])])
        atom.SetIsAromatic(bool(x[i, 7]))
        mol.AddAtom(atom)

    visited = set()
    for k, (src, dst) in enumerate(edge_index.T.tolist()):
        key = (min(src, dst), max(src, dst))
        if key in visited:
            continue
        mol.AddBond(src, dst, Chem.BondType.values[int(edge_attr[k, 0])])

        stereo = Chem.rdchem.BondStereo.values[int(edge_attr[k, 1])]
        if stereo != Chem.rdchem.BondStereo.STEREONONE:
            bond = mol.GetBondBetweenAtoms(src, dst)
            bond.SetStereoAtoms(dst, src)
            bond.SetStereo(stereo)

        mol.GetBondBetweenAtoms(src, dst).SetIsConjugated(bool(edge_attr[k, 2]))
        visited.add(key)

    mol = mol.GetMol()
    if kekulize:
        Chem.Kekulize(mol)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol)
    return mol


def to_smiles(data: "GraphData", kekulize: bool = False) -> str:
    r"""Converts a :class:`~mlx_graphs.data.GraphData` with the features of
    :func:`from_smiles` to a (canonical, isomeric) SMILES string, as PyG's
    ``to_smiles``.

    Args:
        data: The molecular graph.
        kekulize: Whether to convert the aromatic bonds to single and double
            bonds. Default: ``False``

    Returns:
        The SMILES string.
    """
    Chem, _ = _import_rdkit()
    mol = to_rdmol(data, kekulize=kekulize)
    return Chem.MolToSmiles(mol, isomericSmiles=True)
