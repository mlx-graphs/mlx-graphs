import os.path as osp
import warnings
from itertools import repeat
from typing import List, Literal, Optional

import fsspec
import mlx.core as mx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_torch_csr_tensor,
)

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import read_txt_array
from mlx_graphs.utils import coalesce, fs, index_to_mask, remove_self_loops

try:
    import cPickle as pickle
except ImportError:
    import pickle


class Planetoid(Dataset):
    """ """

    _url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    _geom_gcn_url = (
        "https://raw.githubusercontent.com/graphdml-uiuc-jlu/" "geom-gcn/master"
    )

    def __init__(
        self,
        name: str,
        split: Literal[["public", "full", "geom-gcn", "random"]] = "public",
        num_train_per_class: int = 20,
        num_val: int = 500,
        num_test: int = 1000,
        without_self_loops: bool = True,
        base_dir: Optional[str] = None,
    ):
        self.split = split.lower()
        self.without_self_loops = without_self_loops

        super().__init__(name=name, base_dir=base_dir)

    @property
    def raw_file_names(self) -> List[str]:
        names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
        return [f"ind.{self.name.lower()}.{name}" for name in names]

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
                splits = np.load(osp.join(self.raw_path, name))
                train_masks.append(mx.array(splits["train_mask"]))
                val_masks.append(mx.array(splits["val_mask"]))
                test_masks.append(mx.array(splits["test_mask"]))
            graph.train_mask = mx.stack(train_masks, axis=1)
            graph.val_mask = mx.stack(val_masks, axis=1)
            graph.test_mask = mx.stack(test_masks, axis=1)

        self.graphs = [graph]


def read_planetoid_data(
    folder: str, prefix: str, file_names: list[str], without_self_loops: bool
) -> Data:
    items = [read_file(folder, prefix, name) for name in file_names]
    node_features, tx, allx, y, ty, ally, graph, test_index = items
    train_index = mx.arange(y.shape[0], dtype=mx.int32)
    val_index = mx.arange(y.shape[0], y.shape[0] + 500, dtype=mx.int32)
    sorted_test_index = mx.sort(test_index)

    if prefix.lower() == "citeseer":
        # There are some isolated nodes in the Citeseer graph, resulting in
        # none consecutive test indices. We need to identify them and add them
        # as zero vectors to `tx` and `ty`.
        len_test_indices = int(test_index.max() - test_index.min()) + 1

        tx_ext = mx.zeros(len_test_indices, tx.shape[1], dtype=tx.dtype)
        tx_ext[sorted_test_index - test_index.min(), :] = tx
        ty_ext = mx.zeros(len_test_indices, ty.shape[1], dtype=ty.dtype)
        ty_ext[sorted_test_index - test_index.min(), :] = ty

        tx, ty = tx_ext, ty_ext

    if prefix.lower() == "nell.0.001":
        tx_ext = mx.zeros(len(graph) - allx.shape[0], node_features.shape[1])
        tx_ext[sorted_test_index - allx.shape[0]] = tx

        ty_ext = mx.zeros(len(graph) - ally.shape[0], y.shape[1])
        ty_ext[sorted_test_index - ally.shape[0]] = ty

        tx, ty = tx_ext, ty_ext

        node_features = torch.cat([allx, tx], axis=0)
        node_features[test_index] = node_features[sorted_test_index]

        # Creating feature vectors for relations.
        row, col = node_features.nonzero(as_tuple=True)
        value = node_features[row, col]

        mask = ~index_to_mask(test_index, size=len(graph))
        mask[: allx.shape[0]] = False
        isolated_idx = mask.nonzero().view(-1)

        row = torch.cat([row, isolated_idx])
        col = torch.cat(
            [col, mx.arange(isolated_idx.shape[0]) + node_features.shape[1]]
        )
        value = torch.cat([value, value.new_ones(isolated_idx.shape[0])])

        node_features = to_torch_csr_tensor(
            edge_index=torch.stack([row, col], axis=0),
            edge_attr=value,
            size=(
                node_features.shape[0],
                isolated_idx.shape[0] + node_features.shape[1],
            ),
        )
    else:
        node_features = mx.concatenate([allx, tx], axis=0)
        node_features[test_index] = node_features[sorted_test_index]

    y = mx.argmax(mx.concatenate([ally, ty], axis=0), axis=1)
    y[test_index] = y[sorted_test_index]

    train_mask = index_to_mask(train_index, size=y.shape[0])
    val_mask = index_to_mask(val_index, size=y.shape[0])
    test_mask = index_to_mask(test_index, size=y.shape[0])

    edge_index = edge_index_from_dict(
        graph_dict=graph,  # type: ignore
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
    path = osp.join(folder, name)
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
