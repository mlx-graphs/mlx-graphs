import os
from typing import Callable, Literal, Optional, get_args

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download
from mlx_graphs.utils import to_undirected

HETEROPHILOUS_NAMES = Literal[
    "roman_empire", "amazon_ratings", "minesweeper", "tolokers", "questions"
]


class HeterophilousGraphDataset(Dataset):
    """The heterophilous graphs :obj:`"Roman-empire"`, :obj:`"Amazon-ratings"`,
    :obj:`"Minesweeper"`, :obj:`"Tolokers"` and :obj:`"Questions"` from the
    `"A Critical Look at the Evaluation of GNNs under Heterophily: Are We
    Really Making Progress?" <https://arxiv.org/abs/2302.11640>`_ paper.

    The graphs are made undirected. Each dataset comes with 10 fixed
    train/validation/test splits (50%/25%/25%), stored as masks of shape
    ``[num_nodes, 10]`` in the ``train_mask``, ``val_mask`` and ``test_mask``
    attributes (column ``i`` is the ``i``-th split).

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.HeterophilousGraphDataset.html>`_.

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - roman_empire
          - 22,662
          - 65,854
          - 300
          - 18
        * - amazon_ratings
          - 24,492
          - 186,100
          - 300
          - 5
        * - minesweeper
          - 10,000
          - 78,804
          - 7
          - 2
        * - tolokers
          - 11,758
          - 1,038,000
          - 10
          - 2
        * - questions
          - 48,921
          - 307,080
          - 301
          - 2

    Args:
        name: The name of the dataset (``"roman_empire"``,
            ``"amazon_ratings"``, ``"minesweeper"``, ``"tolokers"``,
            ``"questions"``). Dashes are accepted in place of underscores.
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import HeterophilousGraphDataset

        dataset = HeterophilousGraphDataset("minesweeper")
        >>> minesweeper(num_graphs=1)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 78804), int32)
                node_features(shape=(10000, 7), float32)
                node_labels(shape=(10000,), int32)
                train_mask(shape=(10000, 10), bool)
                val_mask(shape=(10000, 10), bool)
                test_mask(shape=(10000, 10), bool))

    References:
        - `Platonov et al., "A Critical Look at the Evaluation of GNNs under
          Heterophily: Are We Really Making Progress?", ICLR 2023
          <https://arxiv.org/abs/2302.11640>`_
    """

    _url = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data"

    def __init__(
        self,
        name: HETEROPHILOUS_NAMES,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        name = name.lower().replace("-", "_")  # type: ignore
        assert name in get_args(HETEROPHILOUS_NAMES), "Invalid dataset name"
        super().__init__(
            name=name,
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    @property
    def _raw_file(self) -> str:
        return os.path.join(self.raw_path, f"{self.name}.npz")

    def download(self):
        download(f"{self._url}/{self.name}.npz", path=self._raw_file)

    def process(self):
        with np.load(self._raw_file) as raw:
            x = raw["node_features"].astype(np.float32)
            y = raw["node_labels"].astype(np.int32)
            edges = raw["edges"].T.astype(np.int32)
            masks = {
                split: np.ascontiguousarray(raw[f"{split}_masks"].T)
                for split in ["train", "val", "test"]
            }

        edge_index = to_undirected(mx.array(edges), coalesce=True, num_nodes=x.shape[0])
        self.graphs = [
            GraphData(
                edge_index=edge_index,
                node_features=mx.array(x),
                node_labels=mx.array(y),
                train_mask=mx.array(masks["train"]),
                val_mask=mx.array(masks["val"]),
                test_mask=mx.array(masks["test"]),
            )
        ]
