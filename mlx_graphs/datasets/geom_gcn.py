import os
from typing import Callable, Literal, Optional, get_args

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download
from mlx_graphs.utils import coalesce

WEBKB_NAMES = Literal["cornell", "texas", "wisconsin"]
WIKIPEDIA_NETWORK_NAMES = Literal["chameleon", "squirrel"]

_GEOM_GCN_URL = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master"

_NODE_FILE = "out1_node_feature_label.txt"
_EDGE_FILE = "out1_graph_edges.txt"


def _read_lines(path: str) -> list[str]:
    with open(path) as f:
        # first line is a header, last line is empty
        return f.read().split("\n")[1:-1]


def _read_geom_gcn_graph(
    folder: str, split_prefix: str, sparse_features: bool = False
) -> GraphData:
    """Reads a graph in the format of the `Geom-GCN
    <https://github.com/graphdml-uiuc-jlu/geom-gcn>`_ repository with its
    10 fixed splits."""
    lines = [line.split("\t") for line in _read_lines(os.path.join(folder, _NODE_FILE))]
    if sparse_features:
        # "<node id>\t<comma separated indices of non-zero features>\t<label>"
        node_ids = np.array([int(n_id) for n_id, _, _ in lines])
        indices = [[int(v) for v in feats.split(",")] for _, feats, _ in lines]
        rows = np.repeat(node_ids, [len(i) for i in indices])
        cols = np.concatenate([np.array(i) for i in indices])
        x = np.zeros((rows.max() + 1, cols.max() + 1), dtype=np.float32)
        x[rows, cols] = 1.0
        y = np.empty(len(lines), dtype=np.int32)
        y[node_ids] = [int(label) for _, _, label in lines]
    else:
        # "<node id>\t<comma separated feature values>\t<label>"
        x = np.array(
            [[float(v) for v in feats.split(",")] for _, feats, _ in lines],
            dtype=np.float32,
        )
        y = np.array([int(label) for _, _, label in lines], dtype=np.int32)

    edges = [
        [int(v) for v in line.split("\t")]
        for line in _read_lines(os.path.join(folder, _EDGE_FILE))
    ]
    edge_index = coalesce(
        mx.array(np.array(edges, dtype=np.int32).T), num_nodes=x.shape[0]
    )

    masks: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    for i in range(10):
        with np.load(
            os.path.join(folder, f"{split_prefix}_split_0.6_0.2_{i}.npz")
        ) as f:
            for split in masks:
                masks[split].append(f[f"{split}_mask"].astype(bool))

    return GraphData(
        edge_index=edge_index,
        node_features=mx.array(x),
        node_labels=mx.array(y),
        train_mask=mx.array(np.stack(masks["train"], axis=1)),
        val_mask=mx.array(np.stack(masks["val"], axis=1)),
        test_mask=mx.array(np.stack(masks["test"], axis=1)),
    )


class _GeomGCNDataset(Dataset):
    """A graph from the Geom-GCN repository, with its 10 fixed splits."""

    _url: str = _GEOM_GCN_URL
    _sparse_features: bool = False

    @property
    def _data_name(self) -> str:
        """Name of the graph in the Geom-GCN repository."""
        return self.name

    def download(self):
        for file_name in [_NODE_FILE, _EDGE_FILE]:
            download(
                f"{self._url}/new_data/{self._data_name}/{file_name}",
                path=os.path.join(self.raw_path, file_name),
            )
        for i in range(10):
            file_name = f"{self._data_name}_split_0.6_0.2_{i}.npz"
            download(
                f"{self._url}/splits/{file_name}",
                path=os.path.join(self.raw_path, file_name),
            )

    def process(self):
        self.graphs = [
            _read_geom_gcn_graph(
                self.raw_path, self._data_name, sparse_features=self._sparse_features
            )
        ]


class WebKB(_GeomGCNDataset):
    """The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.

    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories:
    student, project, course, staff, and faculty. The graphs are directed and
    strongly heterophilous.

    The 10 fixed train/validation/test splits (60%/20%/20%) of the Geom-GCN
    paper are stored as masks of shape ``[num_nodes, 10]`` in the
    ``train_mask``, ``val_mask`` and ``test_mask`` attributes.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WebKB.html>`_.

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - cornell
          - 183
          - 298
          - 1,703
          - 5
        * - texas
          - 183
          - 325
          - 1,703
          - 5
        * - wisconsin
          - 251
          - 515
          - 1,703
          - 5

    Args:
        name: The name of the dataset (``"cornell"``, ``"texas"``,
            ``"wisconsin"``).
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import WebKB

        dataset = WebKB("cornell")
        >>> cornell(num_graphs=1)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 298), int32)
                node_features(shape=(183, 1703), float32)
                node_labels(shape=(183,), int32)
                train_mask(shape=(183, 10), bool)
                val_mask(shape=(183, 10), bool)
                test_mask(shape=(183, 10), bool))

    References:
        - `Pei et al., "Geom-GCN: Geometric Graph Convolutional Networks",
          ICLR 2020 <https://openreview.net/forum?id=S1e2agrFvS>`_
    """

    def __init__(
        self,
        name: WEBKB_NAMES,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        name = name.lower()  # type: ignore
        assert name in get_args(WEBKB_NAMES), "Invalid dataset name"
        super().__init__(
            name=name,
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )


class WikipediaNetwork(_GeomGCNDataset):
    """The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper, as pre-processed in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.

    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to classify the nodes into one of five categories according
    to their average monthly traffic. The graphs are directed and
    heterophilous.

    The 10 fixed train/validation/test splits (60%/20%/20%) of the Geom-GCN
    paper are stored as masks of shape ``[num_nodes, 10]`` in the
    ``train_mask``, ``val_mask`` and ``test_mask`` attributes.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WikipediaNetwork.html>`_
    with ``geom_gcn_preprocess=True``. The raw (regression) versions of the
    graphs, including ``"crocodile"``, are not available: PyG downloads them
    from a host that is no longer reachable.

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - chameleon
          - 2,277
          - 36,101
          - 2,325
          - 5
        * - squirrel
          - 5,201
          - 217,073
          - 2,089
          - 5

    Args:
        name: The name of the dataset (``"chameleon"``, ``"squirrel"``).
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import WikipediaNetwork

        dataset = WikipediaNetwork("chameleon")
        >>> chameleon(num_graphs=1)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 36101), int32)
                node_features(shape=(2277, 2325), float32)
                node_labels(shape=(2277,), int32)
                train_mask(shape=(2277, 10), bool)
                val_mask(shape=(2277, 10), bool)
                test_mask(shape=(2277, 10), bool))

    References:
        - `Rozemberczki et al., "Multi-scale Attributed Node Embedding",
          Journal of Complex Networks 2021
          <https://arxiv.org/abs/1909.13021>`_
        - `Pei et al., "Geom-GCN: Geometric Graph Convolutional Networks",
          ICLR 2020 <https://openreview.net/forum?id=S1e2agrFvS>`_
    """

    # PyG pins the commit for these two graphs
    _url = (
        "https://raw.githubusercontent.com/graphdml-uiuc-jlu/"
        "geom-gcn/f1fc0d14b3b019c562737240d06ec83b07d16a8f"
    )

    def __init__(
        self,
        name: WIKIPEDIA_NETWORK_NAMES,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        name = name.lower()  # type: ignore
        assert name in get_args(WIKIPEDIA_NETWORK_NAMES), (
            "Invalid dataset name (only the Geom-GCN pre-processed 'chameleon' "
            "and 'squirrel' graphs are available)"
        )
        super().__init__(
            name=name,
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    @property
    def raw_path(self) -> str:
        return os.path.join(os.path.dirname(super().raw_path), "geom_gcn", "raw")

    @property
    def processed_path(self) -> str:
        return os.path.join(
            os.path.dirname(super().processed_path), "geom_gcn", "processed"
        )


class Actor(_GeomGCNDataset):
    """The actor-only induced subgraph of the film-director-actor-writer
    network used in the `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.

    Each node corresponds to an actor, and the edge between two nodes denotes
    co-occurrence on the same Wikipedia page. Node features correspond to some
    keywords in the Wikipedia pages. The task is to classify the nodes into
    five categories in terms of words of the actor's Wikipedia page.

    The graph has 7,600 nodes, 30,019 edges, 932 features and 5 classes. The
    10 fixed train/validation/test splits (60%/20%/20%) of the Geom-GCN paper
    are stored as masks of shape ``[num_nodes, 10]`` in the ``train_mask``,
    ``val_mask`` and ``test_mask`` attributes.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Actor.html>`_.

    Args:
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import Actor

        dataset = Actor()
        >>> actor(num_graphs=1)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 30019), int32)
                node_features(shape=(7600, 932), float32)
                node_labels(shape=(7600,), int32)
                train_mask(shape=(7600, 10), bool)
                val_mask(shape=(7600, 10), bool)
                test_mask(shape=(7600, 10), bool))

    References:
        - `Tang et al., "Social Influence Analysis in Large-scale Networks",
          KDD 2009 <https://dl.acm.org/doi/10.1145/1557019.1557108>`_
        - `Pei et al., "Geom-GCN: Geometric Graph Convolutional Networks",
          ICLR 2020 <https://openreview.net/forum?id=S1e2agrFvS>`_
    """

    _sparse_features = True

    def __init__(
        self,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            name="actor",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    @property
    def _data_name(self) -> str:
        return "film"
