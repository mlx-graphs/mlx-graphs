import json
import os
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download
from mlx_graphs.utils import to_undirected


class WikiCS(Dataset):
    """The semi-supervised Wikipedia-based dataset from the
    `"Wiki-CS: A Wikipedia-Based Benchmark for Graph Neural Networks"
    <https://arxiv.org/abs/2007.02901>`_ paper.

    Nodes represent Computer Science articles and edges hyperlinks between
    them. Node features are the averaged GloVe embeddings of the article
    text and the task is to predict which of the 10 branches of Computer
    Science each article belongs to.

    The dataset comes with 20 train/validation/early-stopping splits stored
    as masks of shape ``[num_nodes, 20]`` in the ``train_mask``, ``val_mask``
    and ``stopping_mask`` attributes, and a single test split stored in
    ``test_mask`` (shape ``[num_nodes]``).

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WikiCS.html>`_.

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - undirected (default)
          - 11,701
          - 431,726
          - 300
          - 10
        * - directed
          - 11,701
          - 297,110
          - 300
          - 10

    Args:
        is_undirected: Whether the graph is made undirected. Default:
            ``True`` (the default of PyG, where the original behavior was
            a directed graph).
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import WikiCS

        dataset = WikiCS()
        >>> wikics(num_graphs=1)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 431726), int32)
                node_features(shape=(11701, 300), float32)
                node_labels(shape=(11701,), int32)
                train_mask(shape=(11701, 20), bool)
                val_mask(shape=(11701, 20), bool)
                test_mask(shape=(11701,), bool)
                stopping_mask(shape=(11701, 20), bool))

    References:
        - `Mernyei and Cangea, "Wiki-CS: A Wikipedia-Based Benchmark for
          Graph Neural Networks", ICML GRL+ Workshop 2020
          <https://arxiv.org/abs/2007.02901>`_
    """

    _url = "https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset"

    def __init__(
        self,
        is_undirected: bool = True,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        self.is_undirected = is_undirected
        super().__init__(
            name="wikics",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    @property
    def processed_path(self) -> str:
        suffix = "undirected" if self.is_undirected else "directed"
        return os.path.join(super().processed_path, suffix)

    @property
    def _raw_file(self) -> str:
        return os.path.join(self.raw_path, "data.json")

    def download(self):
        download(f"{self._url}/data.json", path=self._raw_file)

    def process(self):
        with open(self._raw_file) as f:
            data = json.load(f)

        x = np.array(data["features"], dtype=np.float32)
        y = np.array(data["labels"], dtype=np.int32)
        src = np.repeat(
            np.arange(len(data["links"])), [len(js) for js in data["links"]]
        )
        dst = np.array([j for js in data["links"] for j in js], dtype=np.int64)
        edge_index = mx.array(np.stack([src, dst]).astype(np.int32))
        if self.is_undirected:
            edge_index = to_undirected(edge_index, coalesce=True, num_nodes=x.shape[0])

        def mask(key: str) -> mx.array:
            return mx.array(np.ascontiguousarray(np.array(data[key], dtype=bool).T))

        self.graphs = [
            GraphData(
                edge_index=edge_index,
                node_features=mx.array(x),
                node_labels=mx.array(y),
                train_mask=mask("train_masks"),
                val_mask=mask("val_masks"),
                test_mask=mask("test_mask"),
                stopping_mask=mask("stopping_masks"),
            )
        ]
