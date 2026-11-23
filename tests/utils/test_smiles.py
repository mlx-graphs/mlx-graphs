import sys

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.utils import e_map, from_smiles, to_smiles, x_map

pytest.importorskip("rdkit")

SMILES = [
    "C",
    "CCO",
    "O=C=O",
    "C#N",
    "c1ccccc1",
    "c1ccncc1",
    "C1CCCCC1",
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "C[C@H](N)C(=O)O",  # L-alanine
    "C[C@@H](N)C(=O)O",  # D-alanine
    "F/C=C/F",  # trans
    "F/C=C\\F",  # cis
    "[NH4+]",
    "[O-]C(=O)C",
    "C[N+](C)(C)C",
    "[Na+].[Cl-]",
    "[CH3]",  # radical
    "OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O",
    "Brc1ccc(cc1)S(=O)(=O)N",
    "C1=CC=C2C(=C1)C=CC=C2",
    "[2H]C([2H])([2H])O",
]


def assert_same_graph(graph, pyg_data):
    np.testing.assert_array_equal(
        np.array(graph.edge_index), pyg_data.edge_index.numpy()
    )
    np.testing.assert_array_equal(np.array(graph.node_features), pyg_data.x.numpy())
    np.testing.assert_array_equal(
        np.array(graph.edge_features), pyg_data.edge_attr.numpy()
    )
    assert graph.num_nodes == pyg_data.num_nodes
    assert graph.num_edges == pyg_data.num_edges


@pytest.mark.parametrize("with_hydrogen", [False, True])
@pytest.mark.parametrize("kekulize", [False, True])
def test_from_smiles_matches_pyg(with_hydrogen, kekulize):
    from torch_geometric.utils import from_smiles as pyg_from_smiles

    for smiles in SMILES:
        graph = from_smiles(smiles, with_hydrogen=with_hydrogen, kekulize=kekulize)
        expected = pyg_from_smiles(
            smiles, with_hydrogen=with_hydrogen, kekulize=kekulize
        )
        assert_same_graph(graph, expected)
        assert graph.smiles == smiles
        assert graph.node_features.dtype == mx.int32
        assert graph.edge_features.dtype == mx.int32
        assert graph.edge_index.dtype == mx.int32
        assert graph.node_features.shape[1] == len(x_map)
        assert graph.edge_features.shape[1] == len(e_map)


def test_from_smiles_features():
    graph = from_smiles("CCO")
    # atomic numbers, total degrees and numbers of hydrogens
    x = np.array(graph.node_features)
    assert x[:, 0].tolist() == [6, 6, 8]
    assert [x_map["degree"][i] for i in x[:, 2]] == [4, 4, 2]
    assert [x_map["num_hs"][i] for i in x[:, 4]] == [3, 2, 1]
    # 2 bonds, 4 directed edges sorted by source then destination
    assert np.array(graph.edge_index).tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert all(
        e_map["bond_type"][b] == "SINGLE" for b in np.array(graph.edge_features)[:, 0]
    )


def test_from_smiles_invalid_and_no_bonds():
    from torch_geometric.utils import from_smiles as pyg_from_smiles

    empty = from_smiles("not a smiles")
    assert empty.num_nodes == 0
    assert empty.edge_index.shape == (2, 0)
    assert empty.edge_features.shape == (0, 3)
    single = from_smiles("[Na+]")
    assert_same_graph(single, pyg_from_smiles("[Na+]"))
    assert single.edge_index.shape == (2, 0)


def test_to_smiles_round_trip():
    from rdkit import Chem
    from torch_geometric.utils import from_smiles as pyg_from_smiles
    from torch_geometric.utils import to_smiles as pyg_to_smiles

    for smiles in SMILES:
        graph = from_smiles(smiles)
        out = to_smiles(graph)
        assert out == pyg_to_smiles(pyg_from_smiles(smiles))
        if smiles in ("[CH3]", "[2H]C([2H])([2H])O"):
            continue  # radicals and isotopes are not encoded in the features
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        assert Chem.MolToSmiles(Chem.MolFromSmiles(out)) == Chem.MolToSmiles(
            Chem.MolFromSmiles(canonical)
        )
    kekulized = to_smiles(from_smiles("c1ccccc1"), kekulize=True)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(kekulized)) == "c1ccccc1"

    # plain numpy-convertible graphs work too
    graph = from_smiles("CCO")
    graph = GraphData(
        edge_index=graph.edge_index,
        node_features=graph.node_features.astype(mx.float32),
        edge_features=graph.edge_features,
    )
    assert to_smiles(graph) == "CCO"
    with pytest.raises(ValueError, match="edge_features"):
        to_smiles(GraphData(edge_index=graph.edge_index, node_features=graph.x))


def test_rdkit_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "rdkit", None)
    with pytest.raises(ImportError, match="rdkit is required"):
        from_smiles("CCO")
    with pytest.raises(ImportError, match="pip install rdkit"):
        to_smiles(GraphData(edge_index=mx.zeros((2, 0), mx.int32)))
