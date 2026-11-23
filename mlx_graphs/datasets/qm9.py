import os
import sys
from typing import Callable, Optional

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download, extract_archive
from mlx_graphs.datasets.utils.packed import PackedGraphsMixin, graphs_from_records
from mlx_graphs.datasets.utils.torch_serialization import load_torch_file

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

# Unit conversion of the 19 targets (after moving the rotational constants
# A, B and C from the first to the last columns), as in PyG.
_CONVERSION = np.array(
    [1.0, 1.0, HAR2EV, HAR2EV, HAR2EV, 1.0, HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV]
    + [1.0, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1.0, 1.0, 1.0],
    dtype=np.float32,
)

# Reference single-atom properties of H, C, N, O and F, for targets 7 to 10.
_ATOMREFS = {
    6: [0.0, 0.0, 0.0, 0.0, 0.0],
    7: [-13.61312172, -1029.86312267, -1485.30251237, -2042.61123593, -2713.48485589],
    8: [-13.5745904, -1029.82456413, -1485.26398105, -2042.5727046, -2713.44632457],
    9: [-13.54887564, -1029.79887659, -1485.2382935, -2042.54701705, -2713.42063702],
    10: [-13.90303183, -1030.25891228, -1485.71166277, -2043.01812778, -2713.88796536],
    11: [0.0, 0.0, 0.0, 0.0, 0.0],
}

_ATOM_TYPES = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}


def _rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401
    except ImportError:
        return False
    return True


class QM9(PackedGraphsMixin, Dataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting
    of about 130,000 small organic molecules (up to 9 heavy atoms among C, O,
    N and F) with 19 regression targets.

    Each molecule includes the 3D coordinates of the atoms of its low-energy
    conformation, and the atom features of the `"Neural Message Passing for
    Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper. Each graph
    has:

    - ``node_features``: ``float32`` ``[num_atoms, 11]``, the one-hot atom
      type (H, C, N, O, F), atomic number, aromaticity, sp / sp2 / sp3
      hybridization and number of hydrogens;
    - ``z``: ``int32`` ``[num_atoms]``, the atomic numbers;
    - ``pos``: ``float32`` ``[num_atoms, 3]``, the coordinates (in Å);
    - ``edge_index`` and ``edge_features`` (``float32`` ``[num_edges, 4]``,
      the one-hot bond type: single, double, triple, aromatic);
    - ``graph_labels``: ``float32`` ``[1, 19]``, the targets below;
    - ``name`` (e.g. ``"gdb_1"``), ``idx`` (index in the raw data) and,
      when processed with RDKit, ``smiles``.

    .. list-table::
        :widths: 8 25 50 17
        :header-rows: 1

        * - Target
          - Property
          - Description
          - Unit
        * - 0
          - :math:`\mu`
          - Dipole moment
          - D
        * - 1
          - :math:`\alpha`
          - Isotropic polarizability
          - :math:`{a_0}^3`
        * - 2
          - :math:`\epsilon_{\textrm{HOMO}}`
          - Highest occupied molecular orbital energy
          - eV
        * - 3
          - :math:`\epsilon_{\textrm{LUMO}}`
          - Lowest unoccupied molecular orbital energy
          - eV
        * - 4
          - :math:`\Delta \epsilon`
          - Gap between HOMO and LUMO
          - eV
        * - 5
          - :math:`\langle R^2 \rangle`
          - Electronic spatial extent
          - :math:`{a_0}^2`
        * - 6
          - ZPVE
          - Zero point vibrational energy
          - eV
        * - 7
          - :math:`U_0`
          - Internal energy at 0K
          - eV
        * - 8
          - :math:`U`
          - Internal energy at 298.15K
          - eV
        * - 9
          - :math:`H`
          - Enthalpy at 298.15K
          - eV
        * - 10
          - :math:`G`
          - Free energy at 298.15K
          - eV
        * - 11
          - :math:`c_{\textrm{v}}`
          - Heat capacity at 298.15K
          - cal/(mol K)
        * - 12
          - :math:`U_0^{\textrm{ATOM}}`
          - Atomization energy at 0K
          - eV
        * - 13
          - :math:`U^{\textrm{ATOM}}`
          - Atomization energy at 298.15K
          - eV
        * - 14
          - :math:`H^{\textrm{ATOM}}`
          - Atomization enthalpy at 298.15K
          - eV
        * - 15
          - :math:`G^{\textrm{ATOM}}`
          - Atomization free energy at 298.15K
          - eV
        * - 16
          - :math:`A`
          - Rotational constant
          - GHz
        * - 17
          - :math:`B`
          - Rotational constant
          - GHz
        * - 18
          - :math:`C`
          - Rotational constant
          - GHz

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html>`_:
    if ``rdkit`` is installed, the raw SDF files are processed (as PyG does
    with RDKit), otherwise PyG's pre-processed version of the dataset is read
    (without PyTorch). Both give the same graphs, except for the ``smiles``
    attribute that is only available with RDKit.

    .. list-table::
        :widths: 15 15 15 15 15 25
        :header-rows: 1

        * - #graphs
          - #nodes (avg)
          - #edges (avg)
          - #features
          - #tasks
          - Download
        * - 130,831
          - ~18.0
          - ~37.3
          - 11
          - 19
          - 45 MB (raw) / 57 MB (pre-processed)

    Args:
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to each graph once, before
            it is saved.
        transform: A function/transform applied to each graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import QM9

        dataset = QM9()
        >>> QM9(num_graphs=130831)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 8), int32)
                node_features(shape=(5, 11), float32)
                edge_features(shape=(8, 4), float32)
                graph_labels(shape=(1, 19), float32)
                z(shape=(5,), int32)
                pos(shape=(5, 3), float32)
                smiles=[H]C([H])([H])[H]
                name=gdb_1
                idx=0)

        # target 7 (U0), e.g. for SchNet
        z, pos, y = dataset[0].z, dataset[0].pos, dataset[0].graph_labels[:, 7]
    """

    raw_url = (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
        "molnet_publish/qm9.zip"
    )
    raw_url2 = "https://ndownloader.figshare.com/files/3195404"
    processed_url = "https://data.pyg.org/datasets/qm9_v3.zip"

    _raw_files = ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"]
    _preprocessed_file = "qm9_v3.pt"

    def __init__(
        self,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            name="QM9",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    def _labels(self, target: int) -> np.ndarray:
        return np.concatenate(
            [np.array(graph.graph_labels)[:, target] for graph in self.graphs]
        )

    def mean(self, target: int) -> float:
        """The mean of the target ``target`` over the dataset."""
        return float(self._labels(target).mean())

    def std(self, target: int) -> float:
        """The (unbiased) standard deviation of the target ``target`` over the
        dataset."""
        return float(self._labels(target).std(ddof=1))

    def atomref(self, target: int) -> Optional[mx.array]:
        """The reference single-atom properties for the target ``target``
        (indexed by atomic number), of shape ``[100, 1]``, or ``None`` if the
        target has none (only targets 6 to 11 have one). Used as the
        ``atomref`` of :class:`~mlx_graphs.nn.SchNet`."""
        if target not in _ATOMREFS:
            return None
        out = np.zeros(100, dtype=np.float32)
        out[[1, 6, 7, 8, 9]] = _ATOMREFS[target]
        return mx.array(out.reshape(-1, 1))

    def _download_raw(self):
        path = download(self.raw_url, path=os.path.join(self.raw_path, "qm9.zip"))
        extract_archive(path, self.raw_path, overwrite=True)
        os.remove(path)
        download(
            self.raw_url2,
            path=os.path.join(self.raw_path, "uncharacterized.txt"),
        )

    def _download_preprocessed(self):
        path = download(
            self.processed_url, path=os.path.join(self.raw_path, "qm9_v3.zip")
        )
        extract_archive(path, self.raw_path, overwrite=True)
        os.remove(path)

    def _has(self, files: list) -> bool:
        return all(os.path.exists(os.path.join(self.raw_path, f)) for f in files)

    def download(self):
        if _rdkit_available():
            self._download_raw()
        else:
            self._download_preprocessed()

    def process(self):
        # The raw directory may hold the files of the other processing (e.g.
        # downloaded before rdkit was installed): fetch the missing ones.
        if _rdkit_available():
            if not self._has(self._raw_files):
                self._download_raw()
            self._process_rdkit()
        else:
            print(
                "Using a pre-processed version of the dataset. Please install "
                "'rdkit' to alternatively process the raw data.",
                file=sys.stderr,
            )
            if not self._has([self._preprocessed_file]):
                self._download_preprocessed()
            self._process_preprocessed()

    def _process_preprocessed(self):
        data_list = load_torch_file(
            os.path.join(self.raw_path, self._preprocessed_file)
        )
        records = []
        for data in tqdm(data_list, desc="Processing QM9"):
            records.append(
                dict(
                    edge_index=data["edge_index"].astype(np.int32),
                    node_features=data["x"],
                    edge_features=data["edge_attr"],
                    graph_labels=data["y"],
                    z=data["z"].astype(np.int32),
                    pos=data["pos"],
                    name=data["name"],
                    idx=int(np.asarray(data["idx"]).reshape(-1)[0]),
                )
            )
        self.graphs = graphs_from_records(records)

    def _process_rdkit(self):
        from rdkit import Chem, RDLogger
        from rdkit.Chem.rdchem import BondType as BT
        from rdkit.Chem.rdchem import HybridizationType

        RDLogger.DisableLog("rdApp.*")
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        sdf, csv, uncharacterized = (
            os.path.join(self.raw_path, f) for f in self._raw_files
        )
        with open(csv) as f:
            target = [
                [float(x) for x in line.split(",")[1:20]]
                for line in f.read().split("\n")[1:-1]
            ]
        y = np.array(target, dtype=np.float32)
        y = np.concatenate([y[:, 3:], y[:, :3]], axis=-1) * _CONVERSION[None]

        with open(uncharacterized) as f:
            skip = {int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]}

        suppl = Chem.SDMolSupplier(sdf, removeHs=False, sanitize=False)
        records = []
        for i, mol in enumerate(tqdm(suppl, desc="Processing QM9")):
            if i in skip:
                continue
            num_atoms = mol.GetNumAtoms()
            pos = mol.GetConformer().GetPositions().astype(np.float32)

            type_idx, z, aromatic, sp, sp2, sp3 = [], [], [], [], [], []
            for atom in mol.GetAtoms():
                type_idx.append(_ATOM_TYPES[atom.GetSymbol()])
                z.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            z = np.array(z, dtype=np.int64)

            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                edge_types += 2 * [bonds[bond.GetBondType()]]
            edge_index = np.array([rows, cols], dtype=np.int64).reshape(2, -1)
            edge_type = np.array(edge_types, dtype=np.int64)
            perm = np.argsort(edge_index[0] * num_atoms + edge_index[1], kind="stable")
            edge_index, edge_type = edge_index[:, perm], edge_type[perm]
            edge_attr = np.eye(len(bonds), dtype=np.float32)[edge_type]

            row, col = edge_index
            num_hs = np.bincount(
                col, weights=(z[row] == 1).astype(np.float64), minlength=num_atoms
            )
            x1 = np.eye(len(_ATOM_TYPES), dtype=np.float32)[type_idx]
            x2 = np.array(
                [z, aromatic, sp, sp2, sp3, num_hs], dtype=np.float32
            ).T.reshape(num_atoms, 6)
            x = np.concatenate([x1, x2], axis=-1)

            records.append(
                dict(
                    edge_index=edge_index.astype(np.int32),
                    node_features=x,
                    edge_features=edge_attr,
                    graph_labels=y[i : i + 1],
                    z=z.astype(np.int32),
                    pos=pos,
                    smiles=Chem.MolToSmiles(mol, isomericSmiles=True),
                    name=mol.GetProp("_Name"),
                    idx=i,
                )
            )
        self.graphs = graphs_from_records(records)
