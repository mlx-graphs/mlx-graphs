from typing import Literal, Optional, Union, get_args, overload

import mlx.core as mx
import numpy as np

from mlx_graphs.data.data import GraphData
from mlx_graphs.datasets import Dataset
from mlx_graphs.utils import index_to_mask

OGB_NODE_DATASET = Literal[
    "ogbn-products",
    "ogbn-proteins",
    "ogbn-arxiv",
    "ogbn-papers100M",
    # "ogbn-mag", # TODO: requires heterogeneous graphs
]
OGB_EDGE_DATASET = Literal[
    "ogbl-ppa",
    "ogbl-collab",
    "ogbl-ddi",
    "ogbl-citation2",
    # "ogbl-wikikg2",  # TODO: requires heterogeneous graphs
    # "ogbl-biokg",  # TODO: requires heterogeneous graphs
    "ogbl-vessel",
]
OGB_GRAPH_DATASET = Literal[
    "ogbg-molhiv",
    "ogbg-molpcba",
    "ogbg-ppa",
    "ogbg-code2",
]


@overload
def to_mx_array(x: None) -> None:
    ...


@overload
def to_mx_array(x: np.ndarray) -> mx.array:
    ...


def to_mx_array(x: Union[np.ndarray, None]) -> Union[mx.array, None]:
    if x is None:
        return None
    else:
        return mx.array(x.tolist())


class OGBDataset(Dataset):
    def __init__(
        self,
        name: Union[OGB_NODE_DATASET, OGB_EDGE_DATASET, OGB_GRAPH_DATASET],
        split: Optional[Literal["train", "val", "test"]] = None,
        base_dir: Optional[str] = None,
    ):
        self.split = split
        super().__init__(name, base_dir)

    def download(self):
        self._load_data()

    def _load_data(self):
        if self.name in get_args(OGB_NODE_DATASET):
            from ogb.nodeproppred import NodePropPredDataset as OGB_dataset
        elif self.name in get_args(OGB_EDGE_DATASET):
            from ogb.linkproppred import LinkPropPredDataset as OGB_dataset
        elif self.name in get_args(OGB_GRAPH_DATASET):
            from ogb.graphproppred import GraphPropPredDataset as OGB_dataset
        self._raw_ogb_dataset = OGB_dataset(name=self.name, root=self.raw_path)

    def process(self):
        try:
            dataset = self._raw_ogb_dataset
        except AttributeError:
            # reload if already downloaded
            self._load_data()
            dataset = self._raw_ogb_dataset

        if self.name in get_args(OGB_NODE_DATASET):
            graph, label = dataset[0]
            split_idx: dict = dataset.get_idx_split()  # type: ignore
            train_idx, valid_idx, test_idx = (
                split_idx["train"],
                split_idx["valid"],
                split_idx["test"],
            )
            num_nodes = graph["num_nodes"]
            self.graphs.append(
                GraphData(
                    edge_index=to_mx_array(graph["edge_index"]),
                    node_features=to_mx_array(graph["node_feat"]),
                    edge_features=to_mx_array(graph["edge_feat"]),
                    node_labels=to_mx_array(label),  # type: ignore
                    train_mask=index_to_mask(to_mx_array(train_idx), size=num_nodes),
                    val_mask=index_to_mask(to_mx_array(valid_idx), size=num_nodes),
                    test_mask=index_to_mask(to_mx_array(test_idx), size=num_nodes),
                )
            )
        if self.name in get_args(OGB_EDGE_DATASET):
            graph: dict = dataset[0]
            split_idx: dict = dataset.get_edge_split()  # type: ignore
            train_idx, valid_idx, test_idx = (
                split_idx["train"]["edge"].reshape(2, -1),
                split_idx["valid"]["edge"].reshape(2, -1),
                split_idx["test"]["edge"].reshape(2, -1),
            )
            self.graphs.append(
                GraphData(
                    edge_index=to_mx_array(graph["edge_index"]),
                    node_features=to_mx_array(graph["node_feat"]),
                    edge_features=to_mx_array(graph["edge_feat"]),
                )
            )
