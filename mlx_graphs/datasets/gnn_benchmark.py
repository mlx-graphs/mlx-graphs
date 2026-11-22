import os
import warnings
from typing import Callable, Literal, Optional

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download, extract_archive
from mlx_graphs.datasets.utils.torch_serialization import load_torch_file

GNN_BENCHMARK_SPLITS = Literal["train", "val", "test"]
_SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}


class GNNBenchmarkDataset(Dataset):
    """Datasets of the `"Benchmarking Graph Neural Networks"
    <https://arxiv.org/abs/2003.00982>`_ paper.

    - ``"PATTERN"`` and ``"CLUSTER"``: node classification on graphs sampled
      from stochastic block models. In PATTERN, the task is to find the nodes
      of a planted pattern (2 classes); in CLUSTER, to find the cluster of
      each node, given a single labelled node per cluster (6 classes). Node
      features are given in ``node_features`` and labels in ``node_labels``.
    - ``"CSL"``: graph classification of Circular Skip Link graphs into 10
      isomorphism classes. The graphs have no features (only ``num_nodes``)
      and the labels are in ``graph_labels``. CSL has no standard split:
      only the ``"train"`` split exists and contains all the graphs (5-fold
      stratified cross-validation is recommended), as in PyG.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.GNNBenchmarkDataset.html>`_
    and reads the same raw files, without requiring PyTorch. The graphs are
    the same, in the same order, with ``int32`` indices and labels.
    ``"MNIST"`` and ``"CIFAR10"`` are available as
    :class:`~mlx_graphs.datasets.SuperPixelDataset`, and ``"TSP"`` (a 459 MB
    download) is not available.

    .. list-table::
        :widths: 15 10 10 10 10 10 15
        :header-rows: 1

        * - Name
          - #graphs
          - #nodes (avg)
          - #edges (avg)
          - #features
          - #classes
          - Download
        * - PATTERN
          - 14,000
          - ~118.9
          - ~6,079
          - 3
          - 2
          - 88 MB (1.4 GB extracted)
        * - CLUSTER
          - 12,000
          - ~117.2
          - ~4,302
          - 7
          - 6
          - 64 MB (1.1 GB extracted)
        * - CSL
          - 150
          - 41
          - 164
          - 0
          - 10
          - 27 KB

    PATTERN has 10,000 / 2,000 / 2,000 and CLUSTER 10,000 / 1,000 / 1,000
    train / val / test graphs.

    Args:
        name: Name of the dataset, ``"PATTERN"``, ``"CLUSTER"`` or ``"CSL"``.
        split: The split to load, ``"train"``, ``"val"`` or ``"test"``.
            Default: ``"train"``
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to each graph once, before
            it is saved.
        transform: A function/transform applied to each graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import GNNBenchmarkDataset

        dataset = GNNBenchmarkDataset("PATTERN", split="train")
        >>> PATTERN(num_graphs=10000)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 4884), int32)
                node_features(shape=(108, 3), float32)
                node_labels(shape=(108,), int32))

    References:
        - `Dwivedi et al., "Benchmarking Graph Neural Networks", JMLR 2023
          <https://arxiv.org/abs/2003.00982>`_
    """

    names = ["PATTERN", "CLUSTER", "CSL"]
    unsupported_names = ["MNIST", "CIFAR10", "TSP"]
    _root_url = "https://data.pyg.org/datasets/benchmarking-gnns"
    urls = {
        "PATTERN": f"{_root_url}/PATTERN_v2.zip",
        "CLUSTER": f"{_root_url}/CLUSTER_v2.zip",
        "CSL": "https://www.dropbox.com/s/rnbkp5ubgk82ocu/CSL.zip?dl=1",
    }

    def __init__(
        self,
        name: str,
        split: GNN_BENCHMARK_SPLITS = "train",
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        if name in ("MNIST", "CIFAR10"):
            raise ValueError(f"Use mlx_graphs.datasets.SuperPixelDataset for '{name}'")
        if name in self.unsupported_names:
            raise ValueError(f"GNNBenchmarkDataset '{name}' is not available")
        if name not in self.names:
            raise ValueError(f"Unknown dataset '{name}' (got {self.names})")
        if split not in _SPLIT_INDEX:
            raise ValueError(
                f"Split '{split}' found, but expected 'train', 'val' or 'test'"
            )
        if name == "CSL" and split != "train":
            warnings.warn(
                "Dataset 'CSL' does not provide a standardized splitting. "
                "Instead, it is recommended to perform 5-fold cross "
                "validation with stratifed sampling"
            )
            split = "train"
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
        url = self.urls[self.name]
        path = download(url, path=os.path.join(self.raw_path, f"{self.name}.zip"))
        extract_archive(path, self.raw_path, overwrite=True)
        os.remove(path)

    def process(self):
        if self.name == "CSL":
            self._process_csl()
            return

        file_name = self.urls[self.name].split("/")[-1][:-4]
        splits = load_torch_file(os.path.join(self.raw_path, f"{file_name}.pt"))
        graphs = splits[_SPLIT_INDEX[self.split]]
        del splits
        for graph in tqdm(graphs, desc=f"Processing {self.name} {self.split}"):
            self.graphs.append(
                GraphData(
                    edge_index=mx.array(graph["edge_index"].astype(np.int32)),
                    node_features=mx.array(graph["x"]),
                    node_labels=mx.array(graph["y"].astype(np.int32)),
                )
            )

    def _process_csl(self):
        adjacencies = load_torch_file(
            os.path.join(self.raw_path, "graphs_Kary_Deterministic_Graphs.pkl")
        )
        labels = load_torch_file(
            os.path.join(self.raw_path, "y_Kary_Deterministic_Graphs.pt")
        )
        for adjacency, label in zip(adjacencies, labels.tolist()):
            row = np.asarray(adjacency.row, dtype=np.int32)
            col = np.asarray(adjacency.col, dtype=np.int32)
            keep = row != col  # remove self-loops
            self.graphs.append(
                GraphData(
                    edge_index=mx.array(np.stack([row[keep], col[keep]])),
                    graph_labels=mx.array([label], dtype=mx.int32),
                    num_nodes=int(adjacency.shape[0]),
                )
            )
