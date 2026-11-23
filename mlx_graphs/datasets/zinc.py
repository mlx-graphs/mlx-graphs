import os
import shutil
from typing import Callable, Literal, Optional

import numpy as np
from tqdm import tqdm

from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download, extract_archive
from mlx_graphs.datasets.utils.packed import PackedGraphsMixin, graphs_from_records
from mlx_graphs.datasets.utils.torch_serialization import load_torch_file

ZINC_SPLITS = Literal["train", "val", "test"]


class ZINC(PackedGraphsMixin, Dataset):
    r"""The ZINC dataset from the `ZINC database
    <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559>`_ and the
    `"Automatic Chemical Design Using a Data-Driven Continuous Representation
    of Molecules" <https://arxiv.org/abs/1610.02415>`_ paper, containing about
    250,000 molecular graphs with up to 38 heavy atoms.

    The task is to regress the penalized ``logP`` (also called constrained
    solubility), given by ``y = logP - SAS - cycles``, where ``logP`` is the
    water-octanol partition coefficient, ``SAS`` the synthetic accessibility
    score and ``cycles`` the number of cycles with more than six atoms.

    Each graph has ``int32`` ``node_features`` of shape ``[num_atoms, 1]``
    (the atom type, 28 types), ``int32`` ``edge_features`` of shape
    ``[num_edges]`` (the bond type: 1 single, 2 double, 3 triple) and a
    ``float32`` ``graph_labels`` of shape ``[1]``.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.ZINC.html>`_
    and reads the same raw files, without requiring PyTorch. The graphs are
    the same, in the same order (with ``int32`` instead of ``int64`` indices
    and features).

    .. list-table::
        :widths: 20 15 15 15 15 20
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes (avg)
          - #edges (avg)
          - #features
          - Download
        * - ZINC Full
          - 249,456
          - ~23.2
          - ~49.8
          - 1
          - 384 MB
        * - ZINC Subset
          - 12,000
          - ~23.2
          - ~49.8
          - 1
          - 384 MB

    The full dataset has 220,011 / 24,445 / 5,000 and the subset 10,000 /
    1,000 / 1,000 train / val / test graphs.

    Args:
        subset: If ``True``, loads the subset of 12,000 graphs of the
            `"Benchmarking Graph Neural Networks"
            <https://arxiv.org/abs/2003.00982>`_ paper. Default: ``False``
        split: The split to load, ``"train"``, ``"val"`` or ``"test"``.
            Default: ``"train"``
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to each graph once, before
            it is saved.
        transform: A function/transform applied to each graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import ZINC

        dataset = ZINC(subset=True, split="train")
        >>> ZINC(num_graphs=10000)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 64), int32)
                node_features(shape=(29, 1), int32)
                edge_features(shape=(64,), int32)
                graph_labels(shape=(1,), float32))
    """

    url = "https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1"
    split_url = (
        "https://raw.githubusercontent.com/graphdeeplearning/"
        "benchmarking-gnns/master/data/molecules/{}.index"
    )

    def __init__(
        self,
        subset: bool = False,
        split: ZINC_SPLITS = "train",
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Split '{split}' found, but expected 'train', 'val' or 'test'"
            )
        self.subset = subset
        self.split = split
        super().__init__(
            name="ZINC",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    @property
    def processed_path(self) -> str:
        return os.path.expanduser(
            os.path.join(
                self._base_dir,
                self.name,
                "subset" if self.subset else "full",
                "processed",
                self.split,
            )
        )

    def download(self):
        path = download(self.url, path=os.path.join(self.raw_path, "molecules.zip"))
        extract_archive(path, self.raw_path, overwrite=True)
        os.remove(path)
        extracted = os.path.join(self.raw_path, "molecules")
        for file_name in os.listdir(extracted):
            shutil.move(
                os.path.join(extracted, file_name),
                os.path.join(self.raw_path, file_name),
            )
        os.rmdir(extracted)
        for split in ("train", "val", "test"):
            download(
                self.split_url.format(split),
                path=os.path.join(self.raw_path, f"{split}.index"),
            )

    def process(self):
        mols = load_torch_file(os.path.join(self.raw_path, f"{self.split}.pickle"))
        indices = range(len(mols))
        if self.subset:
            with open(os.path.join(self.raw_path, f"{self.split}.index")) as f:
                indices = [int(x) for x in f.read()[:-1].split(",")]

        records = []
        for idx in tqdm(indices, desc=f"Processing ZINC {self.split}"):
            mol = mols[idx]
            adj = np.asarray(mol["bond_type"])
            row, col = np.nonzero(adj)
            records.append(
                dict(
                    edge_index=np.stack([row, col]).astype(np.int32),
                    node_features=np.asarray(mol["atom_type"])
                    .astype(np.int32)
                    .reshape(-1, 1),
                    edge_features=adj[row, col].astype(np.int32),
                    graph_labels=np.asarray(
                        mol["logP_SA_cycle_normalized"], np.float32
                    ),
                )
            )
        del mols
        self.graphs = graphs_from_records(records)
