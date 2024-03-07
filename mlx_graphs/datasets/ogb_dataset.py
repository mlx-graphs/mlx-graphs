import os
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

ALL_DATASETS = Literal[
    "ogbn-products",
    "ogbn-proteins",
    "ogbn-arxiv",
    "ogbn-papers100M",
    # "ogbn-mag", # TODO: requires heterogeneous graphs
    "ogbl-ppa",
    "ogbl-collab",
    "ogbl-ddi",
    "ogbl-citation2",
    # "ogbl-wikikg2",  # TODO: requires heterogeneous graphs
    # "ogbl-biokg",  # TODO: requires heterogeneous graphs
    "ogbl-vessel",
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
    """
    Datasets from the `Open Graph Benchmark (OGB) <https://ogb.stanford.edu>`_
    collection of realistic, large-scale, and diverse benchmark datasets for machine
    learning on graphs.

    Datasets belongs to three fundamental graph machine learning task categories:
    predicting the properties of nodes, links, and graphs.
    Node property prediction datasets consist of a single graph with three additional
    properties: `train_mask`, `val_mask` and `test_mask` specifying the masks for the
    train, validation and test splits.
    Link property prediction datasets also consist of a single graphs with three
    additional properties: `train_edge_index`, `val_edge_index` and `test_edge_index`,
    specifying the edges to be considered for training, validation and testing.
    Graph property prediction datasets consists of multiple graphs. The desired split
    can be specified via the `split` arg.

    See `here <https://ogb.stanford.edu/docs/dataset_overview/>`_ for further details
    and a list of the available datasets with their descriptions

    Args:
        name: Name of the dataset
        split: Split of the dataset to load. Thi parameter has effect only on graph
            property prediction dataset. If `None`, the entire dataset is loaded.
            Defaults to `None`.
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.

    .. note::

       `ogb` needs to be installed to use this dataset

    .. note::

        The ogbn-mag, ogbl-wikikg2 and igbl-biokg and the graphs belonging to the
        largs-scale challenge category are currently not available as
        they require heterogenous graphs which are not yet supported by `mlx-graphs`

    Example:

    .. code-block:: python

        from mlx_graphs.datasets.ogb_dataset import OGBDataset

        ds = OGBDataset("ogbg-molhiv", split="train")
        >>> ogbg-molhiv(num_graphs=32901)

    """

    def __init__(
        self,
        name: ALL_DATASETS,
        split: Optional[Literal["train", "val", "test"]] = None,
        base_dir: Optional[str] = None,
    ):
        self.split = split
        self._raw_ogb_dataset = None
        super().__init__(name, base_dir)

    @property
    def processed_path(self) -> str:
        # processed path includes split
        if self.name in get_args(OGB_GRAPH_DATASET):
            return os.path.join(
                f"{super(self.__class__, self).processed_path}",
                self.split if self.split is not None else "full",
            )
        else:
            return super(self.__class__, self).processed_path

    def download(self):
        self._load_data()

    def _load_data(self):
        try:
            if self.name in get_args(OGB_NODE_DATASET):
                from ogb.nodeproppred import NodePropPredDataset as OGB_dataset
            elif self.name in get_args(OGB_EDGE_DATASET):
                from ogb.linkproppred import LinkPropPredDataset as OGB_dataset
            elif self.name in get_args(OGB_GRAPH_DATASET):
                from ogb.graphproppred import GraphPropPredDataset as OGB_dataset
            self._raw_ogb_dataset = OGB_dataset(name=self.name, root=self.raw_path)
        except ImportError:
            raise ImportError(
                "ogb needs to be installed to use this dataset",
                "you can install it via pip install ogb",
            )

    def process(self):
        if self._raw_ogb_dataset is not None:
            dataset = self._raw_ogb_dataset
        else:
            # reload if already downloaded
            self._load_data()
            dataset = self._raw_ogb_dataset

        if self.name in get_args(OGB_NODE_DATASET):
            graph, label = dataset[0]  # type: ignore - it's NodePropPredDataset
            split_idx: dict = dataset.get_idx_split()  # type: ignore
            train_idx, val_idx, test_idx = (
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
                    val_mask=index_to_mask(to_mx_array(val_idx), size=num_nodes),
                    test_mask=index_to_mask(to_mx_array(test_idx), size=num_nodes),
                )
            )
        elif self.name in get_args(OGB_EDGE_DATASET):
            graph = dataset[0]  # type: ignore - it's LinkPropPredDataset
            split_idx: dict = dataset.get_edge_split()  # type: ignore
            train_edges, val_edges, test_edges = (
                split_idx["train"]["edge"].reshape(2, -1),
                split_idx["valid"]["edge"].reshape(2, -1),
                split_idx["test"]["edge"].reshape(2, -1),
            )
            self.graphs.append(
                GraphData(
                    edge_index=to_mx_array(graph["edge_index"]),  # type: ignore
                    node_features=to_mx_array(graph["node_feat"]),  # type: ignore
                    edge_features=to_mx_array(graph["edge_feat"]),  # type: ignore
                    train_edge_index=to_mx_array(train_edges),
                    val_edge_index=to_mx_array(val_edges),
                    test_edge_index=to_mx_array(test_edges),
                )
            )

        elif self.name in get_args(OGB_GRAPH_DATASET):
            for graph, label in dataset:  # type: ignore - it's GraphPropPredDataset
                self.graphs.append(
                    GraphData(
                        edge_index=to_mx_array(graph["edge_index"]),
                        node_features=to_mx_array(graph["node_feat"]),
                        edge_features=to_mx_array(graph["edge_feat"]),
                        node_labels=to_mx_array(label),  # type: ignore
                    )
                )
            split_idx: dict = dataset.get_idx_split()  # type: ignore
            train_idx, val_idx, test_idx = (
                split_idx["train"],
                split_idx["valid"],
                split_idx["test"],
            )
            if self.split == "train":
                self.graphs = [self.graphs[i] for i in train_idx]
            elif self.split == "val":
                self.graphs = [self.graphs[i] for i in val_idx]
            elif self.split == "test":
                self.graphs = [self.graphs[i] for i in test_idx]
