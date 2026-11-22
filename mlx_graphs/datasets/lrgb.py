import os
from typing import Callable, Literal, Optional

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download, extract_archive
from mlx_graphs.datasets.utils.torch_serialization import load_torch_file

LRGB_SPLITS = Literal["train", "val", "test"]


class LRGBDataset(Dataset):
    """The Peptides datasets of the `"Long Range Graph Benchmark"
    <https://arxiv.org/abs/2206.08164>`_ (LRGB).

    Graphs are peptides (molecules made of chains of amino acids) and the
    tasks require long-range interactions. Node and edge features are the
    categorical atom and bond features of OGB (9 and 3 integer features per
    node and edge, to be embedded e.g. with one ``nn.Embedding`` per feature).

    - ``"peptides-func"``: multi-label graph classification of 10 peptide
      functions (``graph_labels`` of shape ``[1, 10]``, float), evaluated
      with the average precision.
    - ``"peptides-struct"``: graph regression of 11 3D structural properties
      (``graph_labels`` of shape ``[1, 11]``, float), evaluated with the
      mean absolute error.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.LRGBDataset.html>`_
    and reads the same raw files, without requiring PyTorch. The graphs are
    the same, in the same order, with ``int32`` indices and features.
    The superpixel datasets ``"pascalvoc-sp"`` and ``"coco-sp"`` (382 MB and
    4.1 GB downloads) and ``"pcqm-contact"`` are not available.

    .. list-table::
        :widths: 20 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes (avg)
          - #edges (avg)
          - #features
          - #tasks
        * - peptides-func
          - 15,535
          - ~150.9
          - ~307.3
          - 9
          - 10
        * - peptides-struct
          - 15,535
          - ~150.9
          - ~307.3
          - 9
          - 11

    The splits have 10,873 (train), 2,331 (val) and 2,331 (test) graphs. The
    download is ~30 MB (~1.1 GB extracted).

    Args:
        name: Name of the dataset, ``"peptides-func"`` or ``"peptides-struct"``
            (case insensitive).
        split: The split to load, ``"train"``, ``"val"`` or ``"test"``.
            Default: ``"train"``
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to each graph once, before
            it is saved.
        transform: A function/transform applied to each graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import LRGBDataset

        dataset = LRGBDataset("peptides-func", split="train")
        >>> peptides-func(num_graphs=10873)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 682), int32)
                node_features(shape=(338, 9), int32)
                edge_features(shape=(682, 3), int32)
                graph_labels(shape=(1, 10), float32))

    References:
        - `Dwivedi et al., "Long Range Graph Benchmark", NeurIPS 2022
          Datasets and Benchmarks Track <https://arxiv.org/abs/2206.08164>`_
    """

    names = ["peptides-func", "peptides-struct"]
    unsupported_names = ["pascalvoc-sp", "coco-sp", "pcqm-contact"]
    urls = {
        "peptides-func": "https://www.dropbox.com/s/ycsq37q8sxs1ou8/"
        "peptidesfunc.zip?dl=1",
        "peptides-struct": "https://www.dropbox.com/s/zgv4z8fcpmknhs8/"
        "peptidesstruct.zip?dl=1",
    }
    _folders = {
        "peptides-func": "peptidesfunc",
        "peptides-struct": "peptidesstruct",
    }

    def __init__(
        self,
        name: str,
        split: LRGB_SPLITS = "train",
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        name = name.lower()
        if name in self.unsupported_names:
            raise ValueError(
                f"LRGB dataset '{name}' is not available in mlx-graphs "
                f"(supported: {self.names})"
            )
        if name not in self.names:
            raise ValueError(f"Unknown LRGB dataset '{name}' (got {self.names})")
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Split '{split}' found, but expected 'train', 'val' or 'test'"
            )
        self.split = split
        super().__init__(
            name=name,
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    @property
    def processed_path(self) -> str:
        return os.path.join(super().processed_path, self.split)

    def download(self):
        path = download(
            self.urls[self.name],
            path=os.path.join(self.raw_path, f"{self._folders[self.name]}.zip"),
        )
        extract_archive(path, self.raw_path, overwrite=True)
        os.remove(path)

    def process(self):
        path = os.path.join(self.raw_path, self._folders[self.name], f"{self.split}.pt")
        graphs = load_torch_file(path)
        for graph in tqdm(graphs, desc=f"Processing {self.name} {self.split}"):
            node_features, edge_features, edge_index, labels = graph
            self.graphs.append(
                GraphData(
                    edge_index=mx.array(edge_index.astype(np.int32)),
                    node_features=mx.array(node_features.astype(np.int32)),
                    edge_features=mx.array(edge_features.astype(np.int32)),
                    graph_labels=mx.array(labels.astype(np.float32)),
                )
            )
