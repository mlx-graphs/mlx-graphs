import glob
import os
from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils.download import download, extract_zip
from mlx_graphs.datasets.utils.io import read_txt_array
from mlx_graphs.utils import one_hot, remove_duplicate_directed_edges


class TUDataset(Dataset):
    """
    A collection of over 120 benchmark datasets for graph classification
    and regression, made available by TU Dortmund University.
    Access all these datasets
    `here <https://chrsmrrs.github.io/datasets/docs/datasets/>`_.

    This class also supports `cleaned` dataset versions containing only
    non-isomorphic graphs, and presented in
    `Understanding Isomorphism Bias in Graph Data Sets <https://arxiv.org/abs/1910.12091>`_.

    Args:
        name: Name of the dataset to load (e.g. "MUTAG", "PROTEINS",
            "IMDB-BINARY", etc.).
        cleaned: Whether to use the cleaned or original version of datasets.
            Default is False.
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
    """

    _url = "https://www.chrsmrrs.com/graphkerneldatasets"
    _cleaned_url = (
        "https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets"
    )

    def __init__(
        self,
        name: str,
        cleaned: bool = False,
        base_dir: Optional[str] = None,
    ):
        self.cleaned = cleaned

        super().__init__(name=name, base_dir=base_dir)

    @property
    def raw_path(self) -> str:
        return (
            f"{super(self.__class__, self).raw_path}"
            f"{'_cleaned' if self.cleaned else ''}"
        )

    @property
    def processed_path(self) -> str:
        return (
            f"{super(self.__class__, self).processed_path}"
            f"{'_cleaned' if self.cleaned else ''}"
        )

    def download(self):
        url = self._cleaned_url if self.cleaned else self._url
        file_path = os.path.join(self.raw_path, self.name + ".zip")

        download(f"{url}/{self.name}.zip", file_path)
        extract_zip(file_path, self.raw_path)
        os.unlink(file_path)

    def process(self):
        self.graphs = read_tu_data(self.raw_path, self.name)

        # TODO: graphs shall be saved to disk once mx.array is pickle-able


def read_tu_data(folder: str, prefix: str) -> list[GraphData]:
    folder = os.path.join(folder, prefix)
    files = glob.glob(os.path.join(folder, f"{prefix}_*.txt"))
    names = [f.split(os.sep)[-1][len(prefix) + 1 : -4] for f in files]

    edge_index = read_file(folder, prefix, "A", mx.int64).transpose() - 1
    batch_indices = read_file(folder, prefix, "graph_indicator", mx.int64) - 1

    node_features = node_labels = None
    if "node_features" in names:
        node_features = read_file(folder, prefix, "node_features")
    if "node_labels" in names:
        node_labels = read_file(folder, prefix, "node_labels", mx.int64)
        if node_labels.ndim == 1:
            node_labels = mx.expand_dims(node_labels, axis=1)
        node_labels = node_labels - node_labels.min(axis=0)[0]

        # TODO: use unbind here once implemented in mLX
        # node_labels = node_labels.unbind(dim=-1)
        node_labels = [node_labels]

        node_labels = [one_hot(x) for x in node_labels]
        node_labels = mx.concatenate(node_labels, axis=-1)
    x = cat([node_features, node_labels])

    edge_features, edge_labels = None, None
    if "edge_features" in names:
        edge_features = read_file(folder, prefix, "edge_features")
    if "edge_labels" in names:
        edge_labels = read_file(folder, prefix, "edge_labels", mx.int64)
        if edge_labels.ndim == 1:
            edge_labels = mx.expand_dims(edge_labels, axis=1)
        edge_labels = edge_labels - edge_labels.min(axis=0)[0]

        # TODO: use unbind here once implemented in mLX
        # edge_labels = edge_labels.unbind(dim=-1)
        edge_labels = [edge_labels]

        edge_labels = [one_hot(x) for x in edge_labels]
        edge_labels = mx.concatenate(edge_labels, axis=-1).astype(mx.float32)
    edge_attr = cat([edge_features, edge_labels])

    y = None
    if "graph_features" in names:  # Regression problem.
        y = read_file(folder, prefix, "graph_features")
    elif "graph_labels" in names:  # Classification problem.
        y = read_file(folder, prefix, "graph_labels", mx.int32)
        _, y = np.unique(np.array(y), return_inverse=True)
        y = mx.array(y, dtype=mx.int32)

    edge_index = remove_duplicate_directed_edges(edge_index.astype(mx.int32))
    # TODO: Once we have coalesced(), we can replace remove_duplicate_directed_edges()
    # by the scatter which will remove duplicates and sum duplicates edge features

    data = GraphData(
        edge_index=edge_index, node_features=x, edge_features=edge_attr, graph_labels=y
    )
    data, slices = split(data, batch_indices)

    graphs = []
    for i in range(len(slices["edge_index"]) - 1):
        kwargs = {}
        for k, v in slices.items():
            if k == "edge_index":
                kwargs[k] = data.edge_index[  # TODO: make edge_index required
                    :, slices[k][i].item() : slices[k][i + 1].item()
                ]
            else:
                kwargs[k] = getattr(data, k)[
                    slices[k][i].item() : slices[k][i + 1].item()
                ]
        graphs.append(GraphData(**kwargs))

    return graphs


def split(data: GraphData, batch: mx.array) -> tuple[GraphData, dict]:
    """Borrowed from PyG"""
    node_slice = mx.cumsum(
        mx.array(np.bincount(batch), dtype=mx.int32), 0
    )  # TODO: needs to change to int64 once supported in MLX
    node_slice = mx.concatenate([mx.array([0]), node_slice])

    row, _ = data.edge_index  # TODO: make edge_index required
    edge_slice = mx.cumsum(mx.array(np.bincount(batch[row]), dtype=mx.int32), 0)
    edge_slice = mx.concatenate([mx.array([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= mx.expand_dims(node_slice[batch[row]], 0)

    slices = {"edge_index": edge_slice}
    if data.node_features is not None:
        slices["node_features"] = node_slice
    if data.edge_features is not None:
        slices["edge_features"] = edge_slice
    if data.graph_labels is not None:
        if data.graph_labels.shape[0] == batch.shape[0]:
            slices["graph_labels"] = node_slice
        else:
            slices["graph_labels"] = mx.arange(
                0, (batch[-1] + 2).item(), dtype=mx.int64
            )

    return data, slices


def cat(seq: list[mx.array]) -> mx.array:
    """Borrowed from PyG"""
    seq = [item for item in seq if item is not None]
    seq = [mx.expand_dims(item, -1) if item.ndim == 1 else item for item in seq]
    return mx.concatenate(seq, axis=-1) if len(seq) > 0 else None


def read_file(folder: str, prefix: str, name: str, dtype: mx.Dtype = None) -> mx.array:
    """Borrowed from PyG"""
    path = os.path.join(folder, f"{prefix}_{name}.txt")
    return read_txt_array(path, sep=",", dtype=dtype)
