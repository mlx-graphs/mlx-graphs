import os
from typing import Callable, Literal, Optional, get_args

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download
from mlx_graphs.utils import coalesce, remove_self_loops, to_undirected

AMAZON_NAMES = Literal["computers", "photo"]
COAUTHOR_NAMES = Literal["cs", "physics"]

_NPZ_URL = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz"


def read_npz(path: str, to_undirected_graph: bool = True) -> GraphData:
    """Reads a graph stored in the ``.npz`` format of the `"Pitfalls of Graph
    Neural Network Evaluation" <https://arxiv.org/abs/1811.05868>`_ benchmark.

    The file holds the adjacency matrix and the (bag-of-words) attribute
    matrix in CSR form, and one label per node. Node features are binarized,
    self-loops are removed and the graph is made undirected, as in PyG.

    Args:
        path: Path of the ``.npz`` file.
        to_undirected_graph: Whether to add the reverse of every edge.

    Returns:
        The graph with ``node_features``, ``edge_index`` and ``node_labels``.
    """
    with np.load(path) as f:
        num_nodes, num_features = (int(dim) for dim in f["attr_shape"])

        # CSR attribute matrix -> dense, binarized
        attr_indptr = f["attr_indptr"]
        rows = np.repeat(np.arange(num_nodes), np.diff(attr_indptr))
        x = np.zeros((num_nodes, num_features), dtype=np.float32)
        x[rows, f["attr_indices"]] = f["attr_data"] > 0

        # CSR adjacency matrix -> COO edge index
        adj_indptr = f["adj_indptr"]
        src = np.repeat(np.arange(len(adj_indptr) - 1), np.diff(adj_indptr))
        edge_index = np.stack([src, f["adj_indices"]]).astype(np.int32)

        labels = f["labels"].astype(np.int32)

    edge_index = remove_self_loops(mx.array(edge_index))
    if to_undirected_graph:
        edge_index = to_undirected(edge_index, coalesce=True, num_nodes=num_nodes)
    else:
        edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return GraphData(
        edge_index=edge_index,
        node_features=mx.array(x),
        node_labels=mx.array(labels),
    )


class _NPZDataset(Dataset):
    """A single graph downloaded as one ``.npz`` file."""

    _file_names: dict[str, str] = {}
    _prefix: str = ""

    def __init__(
        self,
        name: str,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        name = name.lower()
        if name not in self._file_names:
            raise ValueError(
                f"Invalid dataset name '{name}'. Available: {list(self._file_names)}"
            )
        self._npz_name = name
        super().__init__(
            name=f"{self._prefix}_{name}",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    @property
    def _raw_file(self) -> str:
        return os.path.join(self.raw_path, self._file_names[self._npz_name])

    def download(self):
        download(f"{_NPZ_URL}/{self._file_names[self._npz_name]}", path=self._raw_file)

    def process(self):
        self.graphs = [read_npz(self._raw_file)]


class AmazonDataset(_NPZDataset):
    """The Amazon Computers and Amazon Photo co-purchase graphs from the
    `"Pitfalls of Graph Neural Network Evaluation"
    <https://arxiv.org/abs/1811.05868>`_ paper.

    Nodes are products, edges connect products that are frequently bought
    together, node features are bag-of-words of the product reviews and the
    task is to predict the product category.

    The datasets come without train/validation/test split; create one with
    e.g. :class:`~mlx_graphs.transforms.RandomNodeSplit`.

    Args:
        name: ``"computers"`` (13,752 nodes, 491,722 edges, 767 features,
            10 classes) or ``"photo"`` (7,650 nodes, 238,162 edges,
            745 features, 8 classes).
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import AmazonDataset

        dataset = AmazonDataset("computers")
        >>> amazon_computers(num_graphs=1)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 491722), int32)
                node_features(shape=(13752, 767), float32)
                node_labels(shape=(13752,), int32))
    """

    _prefix = "amazon"
    _file_names = {
        "computers": "amazon_electronics_computers.npz",
        "photo": "amazon_electronics_photo.npz",
    }

    def __init__(
        self,
        name: AMAZON_NAMES,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        assert name.lower() in get_args(AMAZON_NAMES), "Invalid dataset name"
        super().__init__(name, base_dir, pre_transform, transform)


class CoauthorDataset(_NPZDataset):
    """The Coauthor CS and Coauthor Physics co-authorship graphs from the
    `"Pitfalls of Graph Neural Network Evaluation"
    <https://arxiv.org/abs/1811.05868>`_ paper, built from the Microsoft
    Academic Graph.

    Nodes are authors, edges connect authors who co-authored a paper, node
    features are the keywords of each author's papers and the task is to
    predict each author's most active field of study.

    The datasets come without train/validation/test split; create one with
    e.g. :class:`~mlx_graphs.transforms.RandomNodeSplit`.

    Args:
        name: ``"cs"`` (18,333 nodes, 163,788 edges, 6,805 features,
            15 classes) or ``"physics"`` (34,493 nodes, 495,924 edges,
            8,415 features, 5 classes).
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import CoauthorDataset

        dataset = CoauthorDataset("cs")
        >>> coauthor_cs(num_graphs=1)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 163788), int32)
                node_features(shape=(18333, 6805), float32)
                node_labels(shape=(18333,), int32))
    """

    _prefix = "coauthor"
    _file_names = {
        "cs": "ms_academic_cs.npz",
        "physics": "ms_academic_phy.npz",
    }

    def __init__(
        self,
        name: COAUTHOR_NAMES,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        assert name.lower() in get_args(COAUTHOR_NAMES), "Invalid dataset name"
        super().__init__(name, base_dir, pre_transform, transform)
