import os
import re
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download, extract_archive
from mlx_graphs.datasets.utils.packed import PackedGraphsMixin, graphs_from_records


class MoleculeNet(PackedGraphsMixin, Dataset):
    r"""The `MoleculeNet <https://moleculenet.org/datasets-1>`_ benchmark
    collection from the `"MoleculeNet: A Benchmark for Molecular Machine
    Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
    of physical chemistry, biophysics and physiology.

    The molecules are given as SMILES strings and converted to graphs with
    :func:`~mlx_graphs.utils.from_smiles` (which requires ``rdkit``): each
    graph has ``int32`` ``node_features`` of shape ``[num_atoms, 9]`` and
    ``edge_features`` of shape ``[num_edges, 3]`` (categorical atom and bond
    features, see :data:`~mlx_graphs.utils.x_map` and
    :data:`~mlx_graphs.utils.e_map`), the SMILES string in ``smiles`` and the
    ``float32`` targets in ``graph_labels``, of shape ``[1, num_tasks]``, with
    ``NaN`` for missing labels (classification targets are ``0.0`` / ``1.0``).

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MoleculeNet.html>`_
    and reads the same raw files, without requiring PyTorch. The graphs are
    the same, in the same order (with ``int32`` instead of ``int64``
    indices and features). Molecules without atoms (e.g. SMILES strings that
    RDKit cannot parse) are skipped, with a warning, so the number of graphs
    depends slightly on the RDKit version (e.g. 2,039 graphs for BBBP, 7,823
    for Tox21 and 41,120 for HIV with RDKit 2026.9, as in PyG).

    .. list-table::
        :widths: 20 15 15 15 15 20
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes (avg)
          - #edges (avg)
          - #features
          - #tasks
        * - ESOL
          - 1,128
          - ~13.3
          - ~27.4
          - 9
          - 1
        * - FreeSolv
          - 642
          - ~8.7
          - ~16.8
          - 9
          - 1
        * - Lipophilicity
          - 4,200
          - ~27.0
          - ~59.0
          - 9
          - 1
        * - PCBA
          - 437,929
          - ~26.0
          - ~56.2
          - 9
          - 128
        * - MUV
          - 93,087
          - ~24.2
          - ~52.6
          - 9
          - 17
        * - HIV
          - 41,127
          - ~25.5
          - ~54.9
          - 9
          - 1
        * - BACE
          - 1,513
          - ~34.1
          - ~73.7
          - 9
          - 1
        * - BBBP
          - 2,050
          - ~23.9
          - ~51.6
          - 9
          - 1
        * - Tox21
          - 7,831
          - ~18.6
          - ~38.6
          - 9
          - 12
        * - ToxCast
          - 8,597
          - ~18.7
          - ~38.4
          - 9
          - 617
        * - SIDER
          - 1,427
          - ~33.6
          - ~70.7
          - 9
          - 27
        * - ClinTox
          - 1,484
          - ~26.1
          - ~55.5
          - 9
          - 2

    Args:
        name: The name of the dataset (case insensitive): ``"ESOL"``,
            ``"FreeSolv"``, ``"Lipo"``, ``"PCBA"``, ``"MUV"``, ``"HIV"``,
            ``"BACE"``, ``"BBBP"``, ``"Tox21"``, ``"ToxCast"``, ``"SIDER"`` or
            ``"ClinTox"``.
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to each graph once, before
            it is saved.
        transform: A function/transform applied to each graph at every access.
        from_smiles: A function converting a SMILES string to a
            :class:`~mlx_graphs.data.GraphData`, to use another featurization.
            Default: :func:`~mlx_graphs.utils.from_smiles`.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import MoleculeNet

        dataset = MoleculeNet("ESOL")
        >>> ESOL(num_graphs=1128)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 68), int32)
                node_features(shape=(32, 9), int32)
                edge_features(shape=(68, 3), int32)
                graph_labels(shape=(1, 1), float32)
                smiles=OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O )
    """

    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}"

    # Format: name: (display_name, url_name, csv_name, smiles_idx, y_idx)
    names: Dict[str, Tuple[str, str, str, int, Union[int, slice]]] = {
        "esol": ("ESOL", "delaney-processed.csv", "delaney-processed", -1, -2),
        "freesolv": ("FreeSolv", "SAMPL.csv", "SAMPL", 1, 2),
        "lipo": ("Lipophilicity", "Lipophilicity.csv", "Lipophilicity", 2, 1),
        "pcba": ("PCBA", "pcba.csv.gz", "pcba", -1, slice(0, 128)),
        "muv": ("MUV", "muv.csv.gz", "muv", -1, slice(0, 17)),
        "hiv": ("HIV", "HIV.csv", "HIV", 0, -1),
        "bace": ("BACE", "bace.csv", "bace", 0, 2),
        "bbbp": ("BBBP", "BBBP.csv", "BBBP", -1, -2),
        "tox21": ("Tox21", "tox21.csv.gz", "tox21", -1, slice(0, 12)),
        "toxcast": (
            "ToxCast",
            "toxcast_data.csv.gz",
            "toxcast_data",
            0,
            slice(1, 618),
        ),
        "sider": ("SIDER", "sider.csv.gz", "sider", 0, slice(1, 28)),
        "clintox": ("ClinTox", "clintox.csv.gz", "clintox", 0, slice(1, 3)),
    }

    def __init__(
        self,
        name: str,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        from_smiles: Optional[Callable[[str], GraphData]] = None,
    ):
        key = name.lower()
        if key not in self.names:
            raise ValueError(
                f"Unknown MoleculeNet dataset '{name}' "
                f"(available: {[v[0] for v in self.names.values()]})"
            )
        self.from_smiles = from_smiles
        super().__init__(
            name=key,
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    @property
    def display_name(self) -> str:
        """The name of the dataset as in the MoleculeNet paper, e.g.
        ``"ESOL"``."""
        return self.names[self.name][0]

    def download(self):
        file_name = self.names[self.name][1]
        path = download(
            self.url.format(file_name), path=os.path.join(self.raw_path, file_name)
        )
        if file_name.endswith(".gz"):
            extract_archive(path, self.raw_path, overwrite=True)
            os.remove(path)

    def process(self):
        _, _, csv_name, smiles_idx, y_idx = self.names[self.name]
        with open(os.path.join(self.raw_path, f"{csv_name}.csv")) as f:
            lines = f.read().split("\n")[1:-1]
        lines = [line for line in lines if len(line) > 0]  # empty lines

        records = []
        for line in tqdm(lines, desc=f"Processing {self.display_name}"):
            line = re.sub(r"\".*\"", "", line)  # remove the ".*" strings
            values = line.split(",")
            smiles = values[smiles_idx]
            labels = values[y_idx]
            labels = labels if isinstance(labels, list) else [labels]
            ys = [float(y) if len(y) > 0 else float("NaN") for y in labels]

            record, num_nodes = self._featurize(smiles)
            record["graph_labels"] = np.array(ys, dtype=np.float32).reshape(1, -1)
            if num_nodes == 0:
                warnings.warn(
                    f"Skipping molecule '{smiles}' since it resulted in zero atoms",
                    stacklevel=2,
                )
                continue
            records.append(record)
        # views of a few arrays: one array per molecule and attribute would
        # exceed the number of Metal buffers on the largest datasets (PCBA)
        self.graphs = graphs_from_records(records)

    def _featurize(self, smiles: str) -> Tuple[dict, int]:
        """The attributes of the graph of ``smiles`` (with NumPy arrays) and
        its number of nodes."""
        if self.from_smiles is None:
            from mlx_graphs.utils.smiles import _rdmol_arrays, _smiles_to_rdmol

            record = _rdmol_arrays(_smiles_to_rdmol(smiles, False, False))
            record["smiles"] = smiles
            return record, record["node_features"].shape[0]
        data = self.from_smiles(smiles)
        record = {
            key: np.array(value) if isinstance(value, mx.array) else value
            for key, value in vars(data).items()
        }
        return record, data.num_nodes

    def __repr__(self) -> str:
        return f"{self.display_name}(num_graphs={len(self)})"
