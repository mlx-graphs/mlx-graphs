import copy
import glob
import os
from collections.abc import Sequence
from typing import Optional, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils.download import download, extract_zip
from mlx_graphs.datasets.utils.io import read_txt_array
from mlx_graphs.utils import one_hot


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [mx.expand_dims(item, -1) if item.ndim == 1 else item for item in seq]
    return mx.concatenate(seq, axis=-1) if len(seq) > 0 else None


def read_file(folder, prefix, name, dtype=None):
    path = os.path.join(folder, f"{prefix}_{name}.txt")
    return read_txt_array(path, sep=",", dtype=dtype)


def split(data, batch):
    node_slice = mx.cumsum(
        mx.array(np.bincount(batch), dtype=mx.int32), 0
    )  # TODO: needs to change to int64 once fixed
    node_slice = mx.concatenate([mx.array([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = mx.cumsum(mx.array(np.bincount(batch[row]), dtype=mx.int32), 0)
    edge_slice = mx.concatenate([mx.array([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= mx.expand_dims(node_slice[batch[row]], 0)

    slices = {"edge_index": edge_slice}
    if data.node_features is not None:
        slices["node_features"] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = np.bincount(batch).tolist()
        data.num_nodes = batch.size
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


def read_tu_data(folder, prefix):
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

        # node_labels = node_labels.unbind(dim=-1)
        node_labels = [node_labels]  # check if unbind needs to be used

        node_labels = [one_hot(x) for x in node_labels]
        node_labels = mx.concatenate(node_labels, axis=-1)
    x = cat([node_features, node_labels])

    edge_features, edge_labels = None, None
    if "edge_features" in names:
        edge_features = read_file(folder, prefix, "edge_features")
    if "edge_labels" in names:  # TODO
        pass
        # edge_labels = read_file(folder, prefix, "edge_labels", mx.int64)
        # if edge_labels.ndim == 1:
        #     edge_labels = edge_labels.unsqueeze(-1)
        # edge_labels = edge_labels - edge_labels.min(dim=0)[0]
        # edge_labels = edge_labels.unbind(dim=-1)
        # edge_labels = [F.one_hot(e, num_classes=-1) for e in edge_labels]
        # edge_labels = torch.cat(edge_labels, dim=-1).to(torch.float)
    edge_attr = cat([edge_features, edge_labels])

    y = None
    if "graph_features" in names:  # Regression problem.
        y = read_file(folder, prefix, "graph_features")
    elif "graph_labels" in names:  # Classification problem.
        y = read_file(folder, prefix, "graph_labels", mx.int64)

        # _, y = y.unique(sorted=True, return_inverse=True)
        y = y - 1  # also check here

    # num_nodes = edge_index.max().item() + 1 if x is None else x.shape[0]
    # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr) # check
    # edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
    #                                  num_nodes) # check

    # NOTE: In PyG, the src/dst nodes are swapped in the `coalesce()` function
    edge_index = edge_index[mx.array([1, 0])]

    data = GraphData(
        edge_index=edge_index, node_features=x, edge_features=edge_attr, graph_labels=y
    )
    data, slices = split(data, batch_indices)

    data_list = []
    for i in range(len(slices["edge_index"]) - 1):
        kwargs = {}
        for k, v in slices.items():
            if k == "edge_index":
                kwargs[k] = data.edge_index[
                    :, slices[k][i].item() : slices[k][i + 1].item()
                ]
            else:
                kwargs[k] = getattr(data, k)[
                    slices[k][i].item() : slices[k][i + 1].item()
                ]
        data_list.append(GraphData(**kwargs))

    return np.array(data_list)


class TUDataset(Dataset):
    _url = "https://www.chrsmrrs.com/graphkerneldatasets"
    _cleaned_url = (
        "https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets"
    )

    def __init__(
        self,
        name: str,
        use_node_attr: bool = False,
        use_edge_attr: bool = False,
        cleaned: bool = False,
    ):
        self.cleaned = cleaned
        self.data_list: np.array[GraphData] = None

        super().__init__(name=name)

    @property
    def raw_path(self) -> Optional[str]:
        return (
            f"{super(self.__class__, self).raw_path}"
            f"{'_cleaned' if self.cleaned else ''}"
        )

    @property
    def processed_path(self) -> Optional[str]:
        return (
            f"{super(self.__class__, self).processed_path}"
            f"{'_cleaned' if self.cleaned else ''}"
        )

    def download(self):
        url = self.cleaned_url if self.cleaned else self._url
        file_path = os.path.join(self.raw_path, self.name + ".zip")

        download(f"{url}/{self.name}.zip", file_path)
        extract_zip(file_path, self.raw_path)
        os.unlink(file_path)

    def process(self):
        self.data_list = read_tu_data(self.raw_path, self.name)

        # TODO: data_list shall be saved to disk once mx.array is pickle-able

    def __getitem__(
        self,
        idx: Union[int, np.integer],
    ) -> Union["Dataset", GraphData]:
        indices = range(len(self))

        if isinstance(idx, (int, np.integer)) or (
            isinstance(idx, mx.array) and idx.ndim == 0
        ):
            index = indices[idx]
            return self.data_list[index]

        if isinstance(idx, slice):
            indices = indices[idx]

        elif isinstance(idx, mx.array) and idx.dtype in [mx.int64, mx.int32, mx.int16]:
            return self[idx.flatten().tolist()]

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self[idx.flatten().tolist()]

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')"
            )

        dataset = copy.copy(self)
        dataset.data_list = self.data_list[indices]
        return dataset

    def __len__(self) -> int:
        return len(self.data_list)
