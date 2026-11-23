"""Large single-graph node classification benchmarks: Reddit, Reddit2,
Flickr, Yelp and AmazonProducts (and the multi-graph PPI), following PyG's
sources, processing and splits."""

import json
import os
import os.path as osp
from typing import Any, Callable, Literal, Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.utils import download, extract_archive

_GOOGLE_DRIVE_URL = "https://drive.usercontent.google.com/download?id={}&confirm=t"


def _coalesce_edges(row: np.ndarray, col: np.ndarray, num_nodes: int) -> np.ndarray:
    """Edges sorted by ``(row, col)`` without duplicates (PyG's
    ``coalesce``), as an int64 ``[2, num_edges]`` array."""
    key = row.astype(np.int64) * num_nodes + col.astype(np.int64)
    if key.size and not np.all(key[1:] > key[:-1]):
        key = np.unique(key)
    out = np.empty((2, key.size), dtype=np.int64)
    np.floor_divide(key, num_nodes, out=out[0])
    np.remainder(key, num_nodes, out=out[1])
    return out


def _index_dtype(num_nodes: int) -> Any:
    return np.int32 if num_nodes <= np.iinfo(np.int32).max else np.int64


class _LargeGraphDataset(Dataset):
    """A dataset of a single large :class:`GraphData`, stored after
    processing as safetensors (no pickle copy, fast loading)."""

    def save(self):
        graph = self.graphs[0]
        arrays = {
            key: value
            for key, value in vars(graph).items()
            if isinstance(value, mx.array)
        }
        mx.save_safetensors(osp.join(self.processed_path, "graph.safetensors"), arrays)

    def load(self):
        path = osp.join(self.processed_path, "graph.safetensors")
        if not osp.exists(path):
            raise FileNotFoundError(path)
        arrays = mx.load(path)
        self.graphs = [GraphData(**arrays)]  # type: ignore


class _GraphSAINTDataset(_LargeGraphDataset):
    """A graph of the `GraphSAINT <https://arxiv.org/abs/1907.04931>`_
    benchmark (``adj_full.npz``, ``feats.npy``, ``class_map.json`` and
    ``role.json``), processed as in PyG: the edges are the entries of the
    full adjacency matrix in CSR order, nodes absent from ``class_map.json``
    get the label ``-1``, and the ``train_mask`` / ``val_mask`` /
    ``test_mask`` come from ``role.json``."""

    _file_ids: dict[str, str] = {}

    def __init__(
        self,
        name: str,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            name=name,
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    def download(self):
        for file_name, file_id in self._file_ids.items():
            download(
                _GOOGLE_DRIVE_URL.format(file_id),
                path=osp.join(self.raw_path, file_name),
            )

    def process(self):
        with np.load(osp.join(self.raw_path, "adj_full.npz")) as f:
            indptr = f["indptr"].astype(np.int64)
            indices = f["indices"]
            num_nodes = int(f["shape"][0])
        dtype = _index_dtype(num_nodes)
        edge_index = np.empty((2, indices.size), dtype=dtype)
        edge_index[0] = np.repeat(np.arange(num_nodes, dtype=dtype), np.diff(indptr))
        edge_index[1] = indices
        del indptr, indices

        x = np.load(osp.join(self.raw_path, "feats.npy"), mmap_mode="r")
        x = np.ascontiguousarray(x, dtype=np.float32)

        with open(osp.join(self.raw_path, "class_map.json")) as f:
            class_map = json.load(f)
        first = next(iter(class_map.values()))
        if isinstance(first, list):  # multi-label: multi-hot float targets
            y = np.full((num_nodes, len(first)), -1, dtype=np.float32)
        else:
            y = np.full(num_nodes, -1, dtype=np.int32)
        ids = np.fromiter((int(k) for k in class_map), dtype=np.int64)
        y[ids] = np.asarray(list(class_map.values()), dtype=y.dtype)
        del class_map

        with open(osp.join(self.raw_path, "role.json")) as f:
            role = json.load(f)
        masks = {}
        for key, name in (("tr", "train"), ("va", "val"), ("te", "test")):
            mask = np.zeros(num_nodes, dtype=bool)
            mask[np.asarray(role[key], dtype=np.int64)] = True
            masks[f"{name}_mask"] = mx.array(mask)

        self.graphs = [
            GraphData(
                edge_index=mx.array(edge_index),
                node_features=mx.array(x),
                node_labels=mx.array(y),
                **masks,
            )
        ]


class Flickr(_GraphSAINTDataset):
    """The Flickr dataset from the `"GraphSAINT: Graph Sampling Based
    Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
    containing descriptions and common properties of images.

    Nodes are images, edges connect images sharing properties (same
    location, gallery, comments by the same user...), node features are
    bag-of-words of the image descriptions, and the task is to predict one
    of 7 image classes. The graph has the GraphSAINT split (50% / 25% / 25%)
    in ``train_mask`` / ``val_mask`` / ``test_mask``.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Flickr.html>`_
    (same files, edges and splits).

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 89,250
          - 899,756
          - 500
          - 7

    Args:
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import Flickr

        dataset = Flickr()
        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 899756), int32)
                node_features(shape=(89250, 500), float32)
                node_labels(shape=(89250,), int32)
                train_mask(shape=(89250,), bool)
                val_mask(shape=(89250,), bool)
                test_mask(shape=(89250,), bool))
    """

    _file_ids = {
        "adj_full.npz": "1crmsTbd1-2sEXsGwa2IKnIB7Zd3TmUsy",
        "feats.npy": "1join-XdvX3anJU_MLVtick7MgeAQiWIZ",
        "class_map.json": "1uxIkbtg5drHTsKt-PAsZZ4_yJmgFmle9",
        "role.json": "1htXCtuktuCW8TR8KiKfrFDAxUgekQoV7",
    }

    def __init__(
        self,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__("flickr", base_dir, pre_transform, transform)


class Reddit2(_GraphSAINTDataset):
    """The Reddit dataset from the `"GraphSAINT: Graph Sampling Based
    Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
    containing Reddit posts belonging to different communities.

    This is a sparser version of :class:`Reddit` (~23M edges instead of
    ~114M), used in papers such as `SGC <https://arxiv.org/abs/1902.07153>`_
    and `GraphSAINT <https://arxiv.org/abs/1907.04931>`_. Nodes missing from
    the label file have the label ``-1``.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Reddit2.html>`_
    (same files, edges and splits).

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 232,965
          - 23,213,838
          - 602
          - 41

    Args:
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.
    """

    _file_ids = {
        "adj_full.npz": "1sncK996BM5lpuDf75lDFqCiDZyErc1c2",
        "feats.npy": "1ZsHaJ0ussP1W722krmEIp_8pwKAoi5b3",
        "class_map.json": "1JF3Pjv9OboMNYs2aXRQGbJbc4t_nDd5u",
        "role.json": "1nJIKd77lcAGU4j-kVNx_AIGEkveIKz3A",
    }

    def __init__(
        self,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__("reddit2", base_dir, pre_transform, transform)


class Yelp(_GraphSAINTDataset):
    """The Yelp dataset from the `"GraphSAINT: Graph Sampling Based
    Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
    containing customer reviewers and their friendship.

    The task is multi-label: ``node_labels`` holds 100 binary (``float32``)
    targets per node.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Yelp.html>`_
    (same files, edges and splits).

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #tasks
        * - 716,847
          - 13,954,819
          - 300
          - 100

    Args:
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.
    """

    _file_ids = {
        "adj_full.npz": "1Juwx8HtDwSzmVIJ31ooVa1WljI4U5JnA",
        "feats.npy": "1Zy6BZH_zLEjKlEFSduKE5tV9qqA_8VtM",
        "class_map.json": "1VUcBGr0T0-klqerjAjxRmAqFuld_SMWU",
        "role.json": "1NI5pa5Chpd-52eSmLW60OnB3WS5ikxq_",
    }

    def __init__(
        self,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__("yelp", base_dir, pre_transform, transform)


class AmazonProducts(_GraphSAINTDataset):
    """The Amazon dataset from the `"GraphSAINT: Graph Sampling Based
    Inductive Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper,
    containing products and their categories.

    The task is multi-label: ``node_labels`` holds 107 binary (``float32``)
    targets per node. The raw files take about 2.5 GB, and the processed
    graph about 3.5 GB (264M edges stored as ``int32``).

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.AmazonProducts.html>`_
    (same files, edges and splits).

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 1,569,960
          - 264,339,468
          - 200
          - 107

    Args:
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.
    """

    _file_ids = {
        "adj_full.npz": "17qhNA8H1IpbkkR-T2BmPQm8QNW5do-aa",
        "feats.npy": "10SW8lCvAj-kb6ckkfTOC5y0l8XXdtMxj",
        "class_map.json": "1LIl4kimLfftj4-7NmValuWyCQE8AaE7P",
        "role.json": "1npK9xlmbnjNkV80hK2Q68wTEVOFjnt4K",
    }

    def __init__(
        self,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__("amazon_products", base_dir, pre_transform, transform)


class Reddit(_LargeGraphDataset):
    """The Reddit dataset from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, containing
    Reddit posts belonging to different communities.

    Nodes are posts, edges connect posts commented by the same user, node
    features are averaged GloVe embeddings of the post contents plus post
    statistics, and the task is to predict the community (subreddit) of
    each post. The split of the original paper is stored in ``train_mask``
    / ``val_mask`` / ``test_mask``.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Reddit.html>`_
    (DGL's ``reddit.zip``, coalesced edges). The download is 1.4 GB, the
    processed graph about 1.5 GB (edges stored as ``int32``); processing
    needs about 6 GB of memory.

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 232,965
          - 114,615,892
          - 602
          - 41

    Args:
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to the graph once, before
            it is saved.
        transform: A function/transform applied to the graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import Reddit

        dataset = Reddit()
        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 114615892), int32)
                node_features(shape=(232965, 602), float32)
                node_labels(shape=(232965,), int32)
                train_mask(shape=(232965,), bool)
                val_mask(shape=(232965,), bool)
                test_mask(shape=(232965,), bool))
    """

    # DGL's dataset bucket (``data.dgl.ai`` serves the same files).
    url = "https://dgl-data.s3-accelerate.amazonaws.com/dataset/reddit.zip"

    def __init__(
        self,
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            name="reddit",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    def download(self):
        path = download(self.url, path=osp.join(self.raw_path, "reddit.zip"))
        extract_archive(path, self.raw_path)
        os.remove(path)

    def process(self):
        with np.load(osp.join(self.raw_path, "reddit_data.npz")) as f:
            x = np.ascontiguousarray(f["feature"], dtype=np.float32)
            y = f["label"].astype(np.int32)
            split = f["node_types"]
        num_nodes = x.shape[0]
        # A scipy COO matrix saved with `scipy.sparse.save_npz`.
        with np.load(osp.join(self.raw_path, "reddit_graph.npz")) as f:
            row, col = f["row"], f["col"]
        edge_index = _coalesce_edges(row, col, num_nodes)
        del row, col
        edge_index = edge_index.astype(_index_dtype(num_nodes))

        self.graphs = [
            GraphData(
                edge_index=mx.array(edge_index),
                node_features=mx.array(x),
                node_labels=mx.array(y),
                train_mask=mx.array(split == 1),
                val_mask=mx.array(split == 2),
                test_mask=mx.array(split == 3),
            )
        ]


def _ppi_graph_ids(num_nodes: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Graph id of every node of the GraphSAGE PPI release, where the 24
    tissue graphs are stored one after the other: graph boundaries are the
    positions that no edge crosses."""
    reach = np.arange(num_nodes)
    np.maximum.at(reach, np.minimum(src, dst), np.maximum(src, dst))
    reach = np.maximum.accumulate(reach)
    starts = np.concatenate(
        [[0], np.nonzero(reach[:-1] < np.arange(1, num_nodes))[0] + 1]
    )
    graph_id = np.zeros(num_nodes, dtype=np.int64)
    graph_id[starts[1:]] = 1
    return np.cumsum(graph_id)


class PPI(Dataset):
    """The protein-protein interaction networks from the `"Predicting
    Multicellular Function through Multi-layer Tissue Networks"
    <https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
    sets, motif gene sets and immunological signatures as features (50 in
    total) and gene ontology sets as labels (121 in total), as used by
    `GraphSAGE <https://arxiv.org/abs/1706.02216>`_.

    The dataset has 24 graphs (one per human tissue): 20 for training, 2
    for validation and 2 for testing. The task is multi-label node
    classification: ``node_labels`` holds 121 binary (``float32``) targets
    per node. Edges are undirected (both directions are stored) and
    self-loops are removed, as in PyG.

    PyG reads DGL's ``ppi.zip``, which is the GraphSAGE release split into
    one file per split with a graph id per node. As DGL's server is no
    longer reachable, this dataset reads the `GraphSAGE release
    <http://snap.stanford.edu/graphsage/>`_ directly: nodes, features,
    labels and edges are the same, and the graphs (stored one after the
    other) are recovered as the node ranges that no edge crosses (this gives
    exactly 20 / 2 / 2 graphs, with PyG's 44,906 / 6,514 / 5,524 nodes and
    1,226,368 training edges). Edges are sorted by ``(source, target)``.

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #tasks
        * - 20
          - ~2,245.3
          - ~61,318.4
          - 50
          - 121

    Args:
        split: ``"train"`` (20 graphs), ``"val"`` (2) or ``"test"`` (2).
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        pre_transform: A function/transform applied to every graph once,
            before it is saved.
        transform: A function/transform applied to a graph at every access.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import PPI
        from mlx_graphs.loaders import Dataloader

        train = PPI("train")
        loader = Dataloader(train, batch_size=2, shuffle=True)
    """

    url = "https://snap.stanford.edu/graphsage/ppi.zip"
    _splits = ("train", "val", "test")

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        base_dir: Optional[str] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        if split not in self._splits:
            raise ValueError(f"`split` must be one of {self._splits} (got '{split}')")
        self.split = split
        super().__init__(
            name="ppi",
            base_dir=base_dir,
            pre_transform=pre_transform,
            transform=transform,
        )

    @property
    def processed_path(self) -> str:
        return osp.join(super().processed_path, self.split)

    def download(self):
        path = download(self.url, path=osp.join(self.raw_path, "ppi.zip"))
        extract_archive(path, self.raw_path)
        os.remove(path)
        walks = osp.join(self.raw_path, "ppi", "ppi-walks.txt")
        if osp.exists(walks):  # random walks of GraphSAGE, not needed
            os.remove(walks)

    def process(self):
        root = osp.join(self.raw_path, "ppi")
        with open(osp.join(root, "ppi-G.json")) as f:
            graph = json.load(f)
        nodes = graph["nodes"]
        num_nodes = len(nodes)
        node_ids = np.array([n["id"] for n in nodes], dtype=np.int64)
        if not np.array_equal(node_ids, np.arange(num_nodes)):
            raise ValueError("Unexpected node ids in ppi-G.json")
        is_val = np.array([bool(n.get("val", False)) for n in nodes])
        is_test = np.array([bool(n.get("test", False)) for n in nodes])
        links = graph["links"]
        src = np.fromiter((e["source"] for e in links), np.int64, len(links))
        dst = np.fromiter((e["target"] for e in links), np.int64, len(links))
        del graph, links

        # Undirected graph -> both directions, no duplicates, no self-loops.
        keep = src != dst
        src, dst = (
            np.concatenate([src[keep], dst[keep]]),
            np.concatenate([dst[keep], src[keep]]),
        )
        edges = _coalesce_edges(src, dst, num_nodes)

        x = np.load(osp.join(root, "ppi-feats.npy")).astype(np.float32)
        with open(osp.join(root, "ppi-class_map.json")) as f:
            class_map = json.load(f)
        y = np.asarray([class_map[str(i)] for i in range(num_nodes)], np.float32)

        graph_id = _ppi_graph_ids(num_nodes, edges[0], edges[1])
        split_of = np.where(is_test, 2, np.where(is_val, 1, 0))
        wanted = self._splits.index(self.split)

        graphs = []
        for g in range(int(graph_id.max()) + 1):
            members = np.nonzero(graph_id == g)[0]
            if split_of[members[0]] != wanted:
                continue
            if np.any(split_of[members] != wanted):
                raise ValueError("A PPI graph spans several splits")
            start, end = int(members[0]), int(members[-1]) + 1
            lo = np.searchsorted(edges[0], start)
            hi = np.searchsorted(edges[0], end)
            edge_index = edges[:, lo:hi] - start
            graphs.append(
                GraphData(
                    edge_index=mx.array(edge_index.astype(np.int32)),
                    node_features=mx.array(x[start:end]),
                    node_labels=mx.array(y[start:end]),
                )
            )
        self.graphs = graphs
