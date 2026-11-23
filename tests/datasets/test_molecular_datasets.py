import contextlib
import os
import pickle

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.datasets import QM9, ZINC, Dataset, MoleculeNet
from mlx_graphs.datasets.utils.packed import PackedGraphsMixin, graphs_from_records
from mlx_graphs.loaders import Dataloader


def assert_matches_pyg(graph, data, int_features=True):
    """Checks that a graph of mlx-graphs equals a PyG ``Data``."""
    assert graph.num_nodes == data.num_nodes
    assert graph.num_edges == data.num_edges
    assert graph.edge_index.dtype == mx.int32
    np.testing.assert_array_equal(np.array(graph.edge_index), data.edge_index.numpy())
    for ours, theirs in [
        (graph.node_features, data.x),
        (graph.edge_features, data.edge_attr),
    ]:
        assert ours.dtype == (mx.int32 if int_features else mx.float32)
        np.testing.assert_array_equal(np.array(ours), theirs.numpy())
    assert graph.graph_labels.dtype == mx.float32
    np.testing.assert_allclose(
        np.array(graph.graph_labels), data.y.numpy(), rtol=1e-6, equal_nan=True
    )
    for key in ("z", "pos"):
        if key in data:
            np.testing.assert_array_equal(
                np.array(getattr(graph, key)), getattr(data, key).numpy()
            )
    for key in ("smiles", "name"):
        if key in data and key in graph:
            assert getattr(graph, key) == getattr(data, key)
    if "idx" in data:
        assert graph.idx == int(data.idx)


def sample_indices(n, k=50, seed=0):
    rng = np.random.default_rng(seed)
    return sorted({0, n - 1, *rng.integers(0, n, min(k, n)).tolist()})


# --------------------------------------------------------------------------
# Graphs stored as views of a few arrays
# --------------------------------------------------------------------------


class _PackedDataset(PackedGraphsMixin, Dataset):
    def __init__(self, records, base_dir, pre_transform=None):
        self._records = records
        super().__init__(name="packed", base_dir=base_dir, pre_transform=pre_transform)

    def download(self):
        pass

    def process(self):
        self.graphs = graphs_from_records(self._records)


def make_records(num_graphs, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(num_graphs):
        n, e = int(rng.integers(0, 6)), int(rng.integers(0, 9))
        records.append(
            dict(
                edge_index=rng.integers(0, max(n, 1), (2, e)).astype(np.int32),
                node_features=rng.standard_normal((n, 3)).astype(np.float32),
                edge_features=rng.integers(0, 4, e).astype(np.int32),
                graph_labels=rng.standard_normal((1, 2)).astype(np.float32),
                scalar=np.array(i, dtype=np.int64),
                name=f"mol_{i}",
                idx=i,
            )
        )
    return records


def check_records(graphs, records):
    assert len(graphs) == len(records)
    for graph, record in zip(graphs, records):
        for key, value in record.items():
            ours = getattr(graph, key)
            if isinstance(value, np.ndarray):
                assert isinstance(ours, mx.array)
                assert ours.shape == value.shape
                np.testing.assert_array_equal(np.array(ours), value)
            else:
                assert ours == value


def test_graphs_from_records_and_packed_save(tmp_path):
    records = make_records(40)
    graphs = graphs_from_records(records)
    check_records(graphs, records)
    assert graphs_from_records([]) == []

    dataset = _PackedDataset(records, str(tmp_path))
    check_records(dataset.graphs, records)
    files = os.listdir(dataset.processed_path)
    assert files == ["graphs_packed.pkl"]
    reloaded = _PackedDataset(None, str(tmp_path))  # loads, does not process
    check_records(reloaded.graphs, records)

    batch = next(iter(Dataloader(reloaded, 8)))
    assert batch.num_graphs == 8
    assert batch.name == [f"mol_{i}" for i in range(8)]
    assert batch.graph_labels.shape == (8, 2)


def test_packed_save_falls_back_to_pickle(tmp_path):
    records = make_records(5)

    def add_attribute(graph):  # only some graphs have it: cannot be packed
        if graph.idx % 2 == 0:
            graph.extra = mx.ones(2)
        return graph

    dataset = _PackedDataset(records, str(tmp_path), pre_transform=add_attribute)
    assert os.listdir(dataset.processed_path) == ["graphs.pkl"]
    reloaded = _PackedDataset(None, str(tmp_path))
    assert len(reloaded) == 5
    assert "extra" in reloaded[0] and "extra" not in reloaded[1]


def test_many_graphs_do_not_exhaust_device_buffers():
    # 3 arrays x 200,000 graphs exceed the ~500,000 Metal buffers, but the
    # graphs are views of 3 arrays.
    n = 200_000
    records = [
        dict(
            edge_index=np.zeros((2, 2), np.int32),
            node_features=np.full((2, 1), i, np.float32),
            graph_labels=np.array([float(i)], np.float32),
        )
        for i in range(n)
    ]
    graphs = graphs_from_records(records)
    mx.eval([g.node_features for g in graphs], [g.graph_labels for g in graphs])
    assert graphs[-1].node_features.tolist() == [[n - 1], [n - 1]]


# --------------------------------------------------------------------------
# QM9, from small synthetic raw files (offline)
# --------------------------------------------------------------------------

QM9_SMILES = ["C", "N", "O", "CC", "C=O", "C#N", "CC(F)O", "OC(=O)C", "c1ccoc1"]
QM9_SKIPPED = [3]  # 1-based, as in uncharacterized.txt


def write_qm9_raw(raw_dir):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    os.makedirs(raw_dir, exist_ok=True)
    writer = Chem.SDWriter(os.path.join(raw_dir, "gdb9.sdf"))
    writer.SetKekulize(False)  # keep aromatic bonds
    for i, smiles in enumerate(QM9_SMILES):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=i)
        mol.SetProp("_Name", f"gdb_{i + 1}")
        writer.write(mol)
    writer.close()

    rng = np.random.default_rng(0)
    header = (
        "mol_id,A,B,C,mu,alpha,homo,lumo,gap,r2,zpve,u0,u298,h298,g298,cv,"
        "u0_atom,u298_atom,h298_atom,g298_atom"
    )
    lines = [header] + [
        f"gdb_{i + 1}," + ",".join(f"{v:.6f}" for v in rng.normal(0, 10, 19))
        for i in range(len(QM9_SMILES))
    ]
    with open(os.path.join(raw_dir, "gdb9.sdf.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(raw_dir, "uncharacterized.txt"), "w") as f:
        f.write("\n".join(f"header {i}" for i in range(9)) + "\n")
        f.write("\n".join(f"  {i}  C  0.1" for i in QM9_SKIPPED) + "\n")
        f.write("footer\n")


@pytest.fixture
def qm9_pyg(tmp_path):
    pytest.importorskip("rdkit")
    from torch_geometric.datasets import QM9 as QM9_torch

    write_qm9_raw(os.path.join(tmp_path, "pyg", "raw"))
    return QM9_torch(os.path.join(tmp_path, "pyg"))


def test_qm9_rdkit_matches_pyg(tmp_path, qm9_pyg):
    write_qm9_raw(os.path.join(tmp_path, "mxg", "QM9", "raw"))
    dataset = QM9(base_dir=os.path.join(tmp_path, "mxg"))
    assert len(dataset) == len(qm9_pyg) == len(QM9_SMILES) - len(QM9_SKIPPED)
    for graph, data in zip(dataset, qm9_pyg):
        assert_matches_pyg(graph, data, int_features=False)
        assert graph.z.dtype == mx.int32
        assert graph.node_features.shape[1] == 11
        assert graph.edge_features.shape[1] == 4
    # aromatic bonds and hydrogens counts are featurized
    assert np.array(dataset[-1].edge_features)[:, 3].sum() > 0

    for target in (0, 7, 12):
        assert dataset.mean(target) == pytest.approx(qm9_pyg.mean(target), rel=1e-5)
        assert dataset.std(target) == pytest.approx(qm9_pyg.std(target), rel=1e-5)
    for target in range(19):
        expected = qm9_pyg.atomref(target)
        ours = dataset.atomref(target)
        if expected is None:
            assert ours is None
        else:
            np.testing.assert_allclose(np.array(ours), expected.numpy())

    batch = next(iter(Dataloader(dataset, 3)))
    assert batch.z.shape == (batch.num_nodes,)
    assert batch.pos.shape == (batch.num_nodes, 3)
    assert batch.graph_labels.shape == (3, 19)

    reloaded = QM9(base_dir=os.path.join(tmp_path, "mxg"))
    for graph, data in zip(reloaded, qm9_pyg):
        assert_matches_pyg(graph, data, int_features=False)


def test_qm9_without_rdkit_reads_preprocessed_file(tmp_path, qm9_pyg, monkeypatch):
    import torch

    import mlx_graphs.datasets.qm9 as qm9_module

    # PyG's pre-processed file: a list of dicts, without the SMILES
    raw_dir = os.path.join(tmp_path, "mxg", "QM9", "raw")
    os.makedirs(raw_dir)
    data_list = []
    for data in qm9_pyg:
        d = {k: v for k, v in data.to_dict().items() if k != "smiles"}
        d["idx"] = torch.tensor([d["idx"]])
        data_list.append(d)
    torch.save(data_list, os.path.join(raw_dir, "qm9_v3.pt"))

    monkeypatch.setattr(qm9_module, "_rdkit_available", lambda: False)
    dataset = QM9(base_dir=os.path.join(tmp_path, "mxg"))
    assert len(dataset) == len(qm9_pyg)
    for graph, data in zip(dataset, qm9_pyg):
        assert "smiles" not in graph
        assert_matches_pyg(graph, data, int_features=False)


# --------------------------------------------------------------------------
# MoleculeNet, from small synthetic raw files (offline)
# --------------------------------------------------------------------------

ESOL_CSV = """Compound ID,ESOL predicted log solubility in mols per litre,\
Minimum Degree,Molecular Weight,Number of H-Bond Donors,Number of Rings,\
Number of Rotatable Bonds,Polar Surface Area,measured log solubility in mols \
per litre,smiles
Amigdalin,-0.974,1,457.43200000000013,7,3,7,202.32,-0.77,\
OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O
"Benzene, 1,2-dichloro",-3.2,1,147.0,0,1,0,0.0,-3.05,Clc1ccccc1Cl
invalid,-1.0,1,1.0,0,0,0,0.0,-1.5,not_a_smiles
Ethanol,0.02,1,46.07,1,0,0,20.23,1.1,CCO
"""

TOX21_CSV = """NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,\
SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53,mol_id,smiles
0,0,1,,,0,0,1,0,0,0,0,TOX3021,CCOc1ccc2nc(S(N)(=O)=O)sc2c1
0,0,0,0,0,0,0,,0,,0,0,TOX3020,CCN1C(=O)NC(c2ccccc2)C1=O
,,,,,,,,,,,1,TOX3024,C[C@H](N)C(=O)O
"""


def write_molnet_raw(root, name, file_name, content):
    raw_dir = os.path.join(root, name, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    with open(os.path.join(raw_dir, file_name), "w") as f:
        f.write(content)


@pytest.mark.parametrize(
    "name,file_name,content,num_graphs,num_tasks",
    [
        ("ESOL", "delaney-processed.csv", ESOL_CSV, 3, 1),
        ("Tox21", "tox21.csv", TOX21_CSV, 3, 12),
    ],
)
def test_molecule_net_matches_pyg(
    tmp_path, name, file_name, content, num_graphs, num_tasks
):
    pytest.importorskip("rdkit")
    from torch_geometric.datasets import MoleculeNet as MoleculeNet_torch

    for root in ("mxg", "pyg"):
        write_molnet_raw(os.path.join(tmp_path, root), name.lower(), file_name, content)

    def skipped():  # ESOL has an invalid SMILES, which gives no atoms
        if name == "ESOL":
            return pytest.warns(UserWarning, match="zero atoms")
        return contextlib.nullcontext()

    with skipped():
        dataset = MoleculeNet(name, base_dir=os.path.join(tmp_path, "mxg"))
    with skipped():
        dataset_torch = MoleculeNet_torch(os.path.join(tmp_path, "pyg"), name)
    assert (
        repr(dataset)
        == f"{dataset_torch.names[name.lower()][0]}(num_graphs={num_graphs})"
    )
    assert len(dataset) == len(dataset_torch) == num_graphs
    for graph, data in zip(dataset, dataset_torch):
        assert_matches_pyg(graph, data)
        assert graph.graph_labels.shape == (1, num_tasks)
    labels = np.concatenate([np.array(g.graph_labels) for g in dataset])
    if name == "Tox21":
        assert np.isnan(labels).sum() == 2 + 2 + 11
    else:  # the quoted name with commas is removed before splitting
        assert labels[:, 0].tolist() == pytest.approx([-0.77, -3.05, 1.1])

    batch = next(iter(Dataloader(dataset, num_graphs)))
    assert batch.graph_labels.shape == (num_graphs, num_tasks)
    assert batch.smiles == [g.smiles for g in dataset]

    reloaded = MoleculeNet(name.lower(), base_dir=os.path.join(tmp_path, "mxg"))
    for graph, data in zip(reloaded, dataset_torch):
        assert_matches_pyg(graph, data)


def test_molecule_net_custom_from_smiles(tmp_path):
    pytest.importorskip("rdkit")
    from mlx_graphs.utils import from_smiles

    def featurize(smiles):
        graph = from_smiles(smiles, with_hydrogen=True)
        graph.node_features = graph.node_features[:, :1].astype(mx.float32)
        return graph

    write_molnet_raw(str(tmp_path), "esol", "delaney-processed.csv", ESOL_CSV)
    with pytest.warns(UserWarning, match="zero atoms"):
        dataset = MoleculeNet("esol", base_dir=str(tmp_path), from_smiles=featurize)
    assert len(dataset) == 3
    assert dataset[2].num_nodes == 9  # CCO with its hydrogens
    assert dataset[2].node_features.dtype == mx.float32
    assert dataset[2].node_features.shape == (9, 1)


def test_molecule_net_invalid_name():
    with pytest.raises(ValueError, match="Unknown MoleculeNet"):
        MoleculeNet("qm7")


# --------------------------------------------------------------------------
# ZINC, from small synthetic raw files (offline)
# --------------------------------------------------------------------------


def write_zinc_raw(raw_dir, sizes=(12, 5, 4)):
    import torch

    os.makedirs(raw_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, size in zip(("train", "val", "test"), sizes):
        mols = []
        for _ in range(size):
            n = int(rng.integers(1, 8))
            adj = np.triu(rng.integers(0, 4, (n, n)) * (rng.random((n, n)) < 0.4), 1)
            adj = adj + adj.T
            mols.append(
                {
                    "num_atom": n,
                    "atom_type": torch.tensor(rng.integers(0, 28, n), dtype=torch.int8),
                    "bond_type": torch.tensor(adj, dtype=torch.int8),
                    "logP_SA_cycle_normalized": torch.tensor(
                        [rng.normal()], dtype=torch.float32
                    ),
                }
            )
        with open(os.path.join(raw_dir, f"{split}.pickle"), "wb") as f:
            pickle.dump(mols, f)
        subset = sorted(rng.choice(size, size // 2, replace=False).tolist())
        with open(os.path.join(raw_dir, f"{split}.index"), "w") as f:
            f.write(",".join(str(i) for i in subset) + "\n")


@pytest.mark.parametrize("subset", [False, True])
def test_zinc_matches_pyg(tmp_path, subset):
    from torch_geometric.datasets import ZINC as ZINC_torch

    write_zinc_raw(os.path.join(tmp_path, "mxg", "ZINC", "raw"))
    write_zinc_raw(os.path.join(tmp_path, "pyg", "raw"))
    for split in ("train", "val", "test"):
        dataset = ZINC(subset, split, base_dir=os.path.join(tmp_path, "mxg"))
        dataset_torch = ZINC_torch(os.path.join(tmp_path, "pyg"), subset, split)
        assert len(dataset) == len(dataset_torch)
        for graph, data in zip(dataset, dataset_torch):
            assert_matches_pyg(graph, data)
            assert graph.graph_labels.shape == (1,)
        batch = next(iter(Dataloader(dataset, 2)))
        assert batch.graph_labels.shape == (2,)
    # the subset and the full dataset are processed separately
    other = ZINC(not subset, "train", base_dir=os.path.join(tmp_path, "mxg"))
    assert len(other) == (12 if subset else 6)


def test_zinc_invalid_split():
    with pytest.raises(ValueError, match="Split"):
        ZINC(split="validation")


# --------------------------------------------------------------------------
# Full datasets (downloads), compared with PyG: run with --datasets
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "name,size",
    [
        ("ESOL", 1128),
        ("FreeSolv", 642),
        ("Lipo", 4200),
        ("BACE", 1513),
        ("BBBP", None),
        ("Tox21", None),
        ("HIV", None),
    ],
)
def test_molecule_net_full_matches_pyg(tmp_path, name, size):
    from torch_geometric.datasets import MoleculeNet as MoleculeNet_torch

    dataset = MoleculeNet(name, base_dir=os.path.join(tmp_path, "mxg"))
    dataset_torch = MoleculeNet_torch(os.path.join(tmp_path, "pyg"), name)
    assert len(dataset) == len(dataset_torch)
    if size is not None:
        assert len(dataset) == size
    for i in sample_indices(len(dataset)):
        assert_matches_pyg(dataset[i], dataset_torch[i])


@pytest.mark.slow
def test_qm9_full_matches_pyg(tmp_path):
    from torch_geometric.datasets import QM9 as QM9_torch

    dataset = QM9(base_dir=os.path.join(tmp_path, "mxg"))
    dataset_torch = QM9_torch(os.path.join(tmp_path, "pyg"))
    assert len(dataset) == len(dataset_torch) == 130831
    for i in sample_indices(len(dataset), k=200):
        assert_matches_pyg(dataset[i], dataset_torch[i], int_features=False)
    assert dataset.mean(7) == pytest.approx(dataset_torch.mean(7), rel=1e-4)
    assert dataset.std(7) == pytest.approx(dataset_torch.std(7), rel=1e-4)
    num_nodes = sum(g.num_nodes for g in dataset)
    num_edges = sum(g.num_edges for g in dataset)
    assert num_nodes == dataset_torch.x.shape[0]
    assert num_edges == dataset_torch.edge_index.shape[1]


@pytest.mark.slow
def test_zinc_full_matches_pyg(tmp_path):
    from torch_geometric.datasets import ZINC as ZINC_torch

    sizes = {
        True: {"train": 10000, "val": 1000, "test": 1000},
        False: {"train": 220011, "val": 24445, "test": 5000},
    }
    for subset in (True, False):
        for split in ("train", "val", "test"):
            dataset = ZINC(subset, split, base_dir=os.path.join(tmp_path, "mxg"))
            dataset_torch = ZINC_torch(os.path.join(tmp_path, "pyg"), subset, split)
            assert len(dataset) == len(dataset_torch) == sizes[subset][split]
            for i in sample_indices(len(dataset)):
                assert_matches_pyg(dataset[i], dataset_torch[i])
            del dataset_torch
