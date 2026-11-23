import os
from typing import Callable, Literal, Optional, get_args

import mlx.core as mx
import numpy as np

from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.npz_dataset import read_npz
from mlx_graphs.datasets.utils import download
from mlx_graphs.utils import remove_self_loops

CITATION_FULL_NAMES = Literal["cora", "cora_ml", "citeseer", "dblp", "pubmed"]


class CitationFull(Dataset):
    """The full citation network datasets from the
    `"Deep Gaussian Embedding of Graphs: Unsupervised Inductive Learning via
    Ranking" <https://arxiv.org/abs/1707.03815>`_ paper.

    Nodes represent documents and edges represent citation links. Node
    features are binary bag-of-words vectors and the task is to predict the
    topic of each document. Contrary to :class:`PlanetoidDataset`, these are
    the full graphs (no largest-connected-component filtering) and they come
    without train/validation/test split; create one with e.g.
    :class:`~mlx_graphs.transforms.RandomNodeSplit`.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.CitationFull.html>`_.

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - cora
          - 19,793
          - 126,842
          - 8,710
          - 70
        * - cora_ml
          - 2,995
          - 16,316
          - 2,879
          - 7
        * - citeseer
          - 4,230
          - 10,674
          - 602
          - 6
        * - dblp
          - 17,716
          - 105,734
          - 1,639
          - 4
        * - pubmed
          - 19,717
          - 88,648
          - 500
          - 3

    Args:
        name: The name of the dataset (``"cora"``, ``"cora_ml"``,
            ``"citeseer"``, ``"dblp"``, ``"pubmed"``).
        to_undirected: Whether the original graph is converted to an
            undirected one. Default: ``True``.
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import CitationFull

        dataset = CitationFull("cora_ml")
        >>> cora_ml_full(num_graphs=1)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 16316), int32)
                node_features(shape=(2995, 2879), float32)
                node_labels(shape=(2995,), int32))

    References:
        - `Bojchevski and Günnemann, "Deep Gaussian Embedding of Graphs:
          Unsupervised Inductive Learning via Ranking", ICLR 2018
          <https://arxiv.org/abs/1707.03815>`_
    """

    _url = "https://github.com/abojchevski/graph2gauss/raw/master/data/{}.npz"

    def __init__(
        self,
        name: CITATION_FULL_NAMES,
        to_undirected: bool = True,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        name = name.lower()  # type: ignore
        assert name in get_args(CITATION_FULL_NAMES), "Invalid dataset name"
        self._npz_name = name
        self.to_undirected = to_undirected
        super().__init__(
            name=f"{name}_full",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    @property
    def processed_path(self) -> str:
        suffix = "undirected" if self.to_undirected else "directed"
        return os.path.join(super().processed_path, suffix)

    @property
    def _raw_file(self) -> str:
        return os.path.join(self.raw_path, f"{self._npz_name}.npz")

    def download(self):
        download(self._url.format(self._npz_name), path=self._raw_file)

    def process(self):
        graph = read_npz(self._raw_file, self.to_undirected)
        if not self.to_undirected:
            # keep the edges in their stored (CSR) order, as PyG does
            with np.load(self._raw_file) as f:
                indptr = f["adj_indptr"]
                src = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
                edges = np.stack([src, f["adj_indices"]]).astype(np.int32)
            graph.edge_index = remove_self_loops(mx.array(edges))
        self.graphs = [graph]


class CoraFull(CitationFull):
    """Alias for :class:`CitationFull` with ``name="cora"``: the full Cora
    citation graph with 19,793 nodes, 126,842 edges, 8,710 features and
    70 classes.

    Args:
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import CoraFull

        dataset = CoraFull()
        >>> cora_full(num_graphs=1)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 126842), int32)
                node_features(shape=(19793, 8710), float32)
                node_labels(shape=(19793,), int32))
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            "cora",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )
