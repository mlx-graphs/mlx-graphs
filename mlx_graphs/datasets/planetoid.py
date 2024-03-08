import os
import warnings
from itertools import repeat
from typing import List, Literal, Optional, get_args

import fsspec
import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import read_txt_array
from mlx_graphs.utils import coalesce, fs, index_to_mask, remove_self_loops

try:
    import cPickle as pickle
except ImportError:
    import pickle


PLANETOID_NAMES = Literal["cora", "citeseer", "pubmed"]
PLANETOID_SPLITS = Literal["public", "full", "geom-gcn"]


class PlanetoidDataset(Dataset):
    """The citation network datasets :obj:`"Cora"`, :obj:`"CiteSeer"` and
    :obj:`"PubMed"` from the `"Revisiting Semi-Supervised Learning with Graph
    Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    This dataset follows a similar implementation as in `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html>`_.

    Args:
        name: The name of the dataset (:obj:`"Cora"`, :obj:`"CiteSeer"`,
            :obj:`"PubMed"`).
        split (str, optional): The type of dataset split (:obj:`"public"`,
            :obj:`"full"`, :obj:`"geom-gcn"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the `"Revisiting Semi-Supervised Learning with Graph
            Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"geom-gcn"`, the 10 public fixed splits from the
            `"Geom-GCN: Geometric Graph Convolutional Networks"
            <https://openreview.net/forum?id=S1e2agrFvS>`_ paper are given.
        without_self_loops: Whether to remove self loops. Default to ``True``.
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import Planetoid

        dataset = Planetoid("cora")
        >>> cora(num_graphs=1)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 10556), int32)
                node_features(shape=(2708, 1433), float32)
                node_labels(shape=(2708,), int32)
                train_mask(shape=(2708,), bool)
                val_mask(shape=(2708,), bool)
                test_mask(shape=(2708,), bool))
    """

    _url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    _geom_gcn_url = (
        "https://raw.githubusercontent.com/graphdml-uiuc-jlu/" "geom-gcn/master"
    )

    def __init__(
        self,
        name: PLANETOID_NAMES,
        split: PLANETOID_SPLITS = "public",
        without_self_loops: bool = True,
        base_dir: Optional[str] = None,
    ):
        name, self.split = name.lower(), split.lower()
        self.without_self_loops = without_self_loops

        assert name in get_args(PLANETOID_NAMES), "Invalid dataset name"
        assert self.split in get_args(PLANETOID_SPLITS), "Invalid split specified"

        super().__init__(name=name, base_dir=base_dir)

        if self.split == "full":
            data = self[0]
            data.train_mask = mx.where(data.val_mask | data.test_mask, False, True)

    @property
    def raw_path(self) -> str:
        # raw path includes split
        return os.path.join(
            f"{super(self.__class__, self).raw_path}",
            self.split,
        )

    @property
    def processed_path(self) -> str:
        # processed path includes split and presence of self loops
        return os.path.join(
            f"{super(self.__class__, self).processed_path}",
            self.split,
            "self_loops_" + str(self.without_self_loops),
        )

    @property
    def raw_file_names(self) -> List[str]:
        names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
        return [f"ind.{self.name.lower()}.{name}" for name in names]

    @property
    def _processed_file_name(self):
        return f"{self.name}_{self.split}_{self.without_self_loops}_graphs.pkl"

    def download(self):
        for name in self.raw_file_names:
            fs.cp(f"{self._url}/{name}", self.raw_path)
        if self.split == "geom-gcn":
            for i in range(10):
                url = f"{self._geom_gcn_url}/splits/{self.name.lower()}"
                fs.cp(f"{url}_split_0.6_0.2_{i}.npz", self.raw_path)

    def process(self):
        graph = read_planetoid_data(
            self.raw_path,
            self.name,
            self.raw_file_names,
            self.without_self_loops,
        )

        if self.split == "geom-gcn":
            train_masks, val_masks, test_masks = [], [], []
            for i in range(10):
                name = f"{self.name.lower()}_split_0.6_0.2_{i}.npz"
                splits = np.load(os.path.join(self.raw_path, name))
                train_masks.append(mx.array(splits["train_mask"]))
                val_masks.append(mx.array(splits["val_mask"]))
                test_masks.append(mx.array(splits["test_mask"]))
            graph.train_mask = mx.stack(train_masks, axis=1)
            graph.val_mask = mx.stack(val_masks, axis=1)
            graph.test_mask = mx.stack(test_masks, axis=1)

        self.graphs = [graph]


def read_planetoid_data(
    folder: str, prefix: str, file_names: list[str], without_self_loops: bool
) -> GraphData:
    items = [read_file(folder, prefix, name) for name in file_names]
    node_features, tx, allx, y, ty, ally, graph, test_index = items
    train_index = mx.arange(y.shape[0], dtype=mx.int32)
    val_index = mx.arange(y.shape[0], y.shape[0] + 500, dtype=mx.int32)
    sorted_test_index = mx.sort(test_index)

    if prefix.lower() == "citeseer":
        # There are some isolated nodes in the Citeseer graph, resulting in
        # none consecutive test indices. We need to identify them and add them
        # as zero vectors to `tx` and `ty`.
        len_test_indices = (test_index.max() - test_index.min()).item() + 1

        tx_ext = mx.zeros((len_test_indices, tx.shape[1]), dtype=tx.dtype)
        tx_ext[sorted_test_index - test_index.min(), :] = tx
        ty_ext = mx.zeros((len_test_indices, ty.shape[1]), dtype=ty.dtype)
        ty_ext[sorted_test_index - test_index.min(), :] = ty

        tx, ty = tx_ext, ty_ext

    node_features = mx.concatenate([allx, tx], axis=0)
    node_features[test_index] = node_features[sorted_test_index]

    y = mx.argmax(mx.concatenate([ally, ty], axis=0), axis=1)
    y[test_index] = y[sorted_test_index]

    train_mask = index_to_mask(train_index, size=y.shape[0])
    val_mask = index_to_mask(val_index, size=y.shape[0])
    test_mask = index_to_mask(test_index, size=y.shape[0])

    edge_index = edge_index_from_dict(
        graph_dict=graph,
        num_nodes=y.shape[0],
        without_self_loops=without_self_loops,
    )

    graph = GraphData(
        edge_index,
        node_features,
        node_labels=y.astype(mx.int32),
    )
    graph.train_mask = train_mask
    graph.val_mask = val_mask
    graph.test_mask = test_mask

    return graph


def read_file(folder: str, prefix: str, name: str) -> mx.array:
    path = os.path.join(folder, name)
    prefix = prefix.lower()

    if name == f"ind.{prefix}.test.index":
        return read_txt_array(path, dtype=mx.int32)

    with fsspec.open(path, "rb") as f:
        warnings.filterwarnings("ignore", ".*`scipy.sparse.csr` name.*")
        out = pickle.load(f, encoding="latin1")

    if name == f"ind.{prefix}.graph":
        return out

    out = out.todense() if hasattr(out, "todense") else out
    out = mx.array(out.tolist(), dtype=mx.float32)
    return out


def edge_index_from_dict(
    graph_dict: dict[int, list[int]],
    num_nodes: Optional[int] = None,
    without_self_loops: bool = True,
) -> mx.array:
    rows: List[int] = []
    cols: List[int] = []
    for key, value in graph_dict.items():
        rows += repeat(key, len(value))
        cols += value

    row = mx.array(rows)
    col = mx.array(cols)
    edge_index = mx.stack([row, col], axis=0)

    # NOTE: There are some duplicated edges and self loops in the datasets.
    #       Other implementations do not remove them!
    if without_self_loops:
        edge_index = remove_self_loops(edge_index)

    edge_index = coalesce(edge_index)

    return edge_index
