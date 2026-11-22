import json
import os
import os.path as osp
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import HeteroGraphData
from mlx_graphs.datasets.hetero_dataset import HeteroDataset
from mlx_graphs.datasets.utils import download, extract_archive

EDGE_TYPES = [
    ("author", "affiliated_with", "institution"),
    ("author", "writes", "paper"),
    ("paper", "cites", "paper"),
    ("paper", "has_topic", "field_of_study"),
]
NODE_TYPES = ["paper", "author", "institution", "field_of_study"]


def _read_csv(path: str, dtype) -> np.ndarray:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to process the raw ogbn-mag data")
    return pd.read_csv(path, compression="gzip", header=None, dtype=dtype).values


class OGBMAG(HeteroDataset):
    r"""The ``ogbn-mag`` dataset from the `"Open Graph Benchmark: Datasets for
    Machine Learning on Graphs" <https://arxiv.org/abs/2005.00687>`_ paper, a
    heterogeneous subset of the Microsoft Academic Graph, as PyG's
    ``OGB_MAG``.

    It has four node types: papers (736,389), authors (1,134,649),
    institutions (8,740) and fields of study (59,965), and four edge types:
    ``author -affiliated_with-> institution``, ``author -writes-> paper``,
    ``paper -cites-> paper`` and ``paper -has_topic-> field_of_study``
    (21.1M edges in total). The task is to predict the venue (349 classes) of
    each paper. Papers have 128-dimensional word2vec features and a
    publication year; the other node types have no features. The split is by
    year (train: until 2017, validation: 2018, test: 2019).

    The graph is a :class:`~mlx_graphs.data.HeteroGraphData` with:

    - ``edge_index_dict``: the four edge types (``int32``, no reverse edges;
      message passing models such as :func:`~mlx_graphs.nn.to_hetero` need
      reverse edge types so that every node type receives messages, see
      :meth:`add_reverse_edges`),
    - ``node_features_dict["paper"]`` (``[736389, 128]``) and
      ``node_labels_dict["paper"]`` (``[736389]``),
    - ``num_nodes_dict`` with the number of nodes of every node type,
    - ``paper_year`` and the boolean masks ``paper_train_mask``,
      ``paper_val_mask`` and ``paper_test_mask``.

    The raw data (a 430 MB download) is read with ``pandas``. The processed
    graph is stored as safetensors and loaded lazily.

    Args:
        base_dir: Directory where to store dataset files. Default is
            in the local directory ``.mlx_graphs_data/``.
        transform: A function/transform that takes in a ``HeteroGraphData``
            object and returns a transformed version, at every access.
        pre_transform: A function/transform applied before saving the
            processed data.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import OGBMAG
        from mlx_graphs.loaders import HGTLoader

        data = OGBMAG.add_reverse_edges(OGBMAG()[0])
        loader = HGTLoader(
            data,
            num_samples=[512] * 2,
            input_nodes=("paper", data.paper_train_mask),
            batch_size=1024,
        )

    References:
        `OGB ogbn-mag <https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag>`_
    """

    url = "https://snap.stanford.edu/ogb/data/nodeproppred/mag.zip"

    def __init__(
        self,
        base_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(
            name="ogbn-mag",
            base_dir=base_dir,
            transform=transform,
            pre_transform=pre_transform,
        )

    @property
    def _raw_root(self) -> str:
        return osp.join(self.raw_path, "mag")

    def download(self):
        path = download(self.url, path=self.raw_path)
        extract_archive(path, self.raw_path)
        os.remove(path)

    def process(self):
        raw = osp.join(self._raw_root, "raw")
        x = _read_csv(
            osp.join(raw, "node-feat", "paper", "node-feat.csv.gz"), np.float32
        )
        year = _read_csv(
            osp.join(raw, "node-feat", "paper", "node_year.csv.gz"), np.int64
        )
        y = _read_csv(
            osp.join(raw, "node-label", "paper", "node-label.csv.gz"), np.int64
        )
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required to process the raw ogbn-mag data")
        counts = pd.read_csv(osp.join(raw, "num-node-dict.csv.gz"), compression="gzip")
        num_nodes = {t: int(counts[t].tolist()[0]) for t in NODE_TYPES}

        edge_index_dict = {}
        for edge_type in EDGE_TYPES:
            path = osp.join(raw, "relations", "___".join(edge_type), "edge.csv.gz")
            edges = _read_csv(path, np.int64)
            edge_index_dict[edge_type] = mx.array(
                np.ascontiguousarray(edges.T.astype(np.int32))
            )
            del edges

        num_papers = num_nodes["paper"]
        masks = {}
        for split, name in (("train", "train"), ("valid", "val"), ("test", "test")):
            path = osp.join(self._raw_root, "split", "time", "paper", f"{split}.csv.gz")
            mask = np.zeros(num_papers, dtype=bool)
            mask[_read_csv(path, np.int64).reshape(-1)] = True
            masks[f"paper_{name}_mask"] = mx.array(mask)

        data = HeteroGraphData(
            edge_index_dict=edge_index_dict,
            node_features_dict={"paper": mx.array(x)},
            node_labels_dict={"paper": mx.array(y.reshape(-1).astype(np.int32))},
            num_nodes_dict=num_nodes,
            paper_year=mx.array(year.reshape(-1).astype(np.int32)),
            **masks,
        )
        self.graphs = [data]

    # Safetensors instead of pickle: no extra copy of the arrays when saving,
    # and lazy loading.

    def save(self):
        data = self.graphs[0]
        arrays, meta = {}, {"edge_types": [], "num_nodes_dict": data.num_nodes_dict}
        for i, (edge_type, edge_index) in enumerate(data.edge_index_dict.items()):
            arrays[f"edge_index_{i}"] = edge_index
            meta["edge_types"].append(list(edge_type))
        for prefix, values in (
            ("x", data.node_features_dict),
            ("y", data.node_labels_dict),
        ):
            for node_type, value in (values or {}).items():
                arrays[f"{prefix}__{node_type}"] = value
        for key, value in vars(data).items():
            if isinstance(value, mx.array):
                arrays[f"attr__{key}"] = value
        mx.save_safetensors(osp.join(self.processed_path, "graph.safetensors"), arrays)
        with open(osp.join(self.processed_path, "meta.json"), "w") as f:
            json.dump(meta, f)

    def load(self):
        with open(osp.join(self.processed_path, "meta.json")) as f:
            meta = json.load(f)
        arrays = mx.load(osp.join(self.processed_path, "graph.safetensors"))
        edge_index_dict = {
            tuple(et): arrays[f"edge_index_{i}"]
            for i, et in enumerate(meta["edge_types"])
        }
        node_features, node_labels, attrs = {}, {}, {}
        for key, value in arrays.items():
            if key.startswith("x__"):
                node_features[key[3:]] = value
            elif key.startswith("y__"):
                node_labels[key[3:]] = value
            elif key.startswith("attr__"):
                attrs[key[6:]] = value
        self.graphs = [
            HeteroGraphData(
                edge_index_dict=edge_index_dict,
                node_features_dict=node_features,
                node_labels_dict=node_labels,
                num_nodes_dict=meta["num_nodes_dict"],
                **attrs,
            )
        ]

    @staticmethod
    def add_reverse_edges(data: HeteroGraphData) -> HeteroGraphData:
        """Adds a reverse edge type ``(dst, "rev_<rel>", src)`` for every edge
        type between two different node types, and makes ``paper -cites->
        paper`` undirected (both directions, duplicates removed), like PyG's
        ``ToUndirected`` transform, so that every node type receives messages.

        Args:
            data: The ogbn-mag graph.

        Returns:
            The same graph, with the new edge types.
        """
        edge_index_dict = dict(data.edge_index_dict)
        for (src, rel, dst), edge_index in data.edge_index_dict.items():
            if src == dst:
                both = np.concatenate(
                    [np.array(edge_index), np.array(edge_index[::-1])], axis=1
                )
                both = np.unique(both, axis=1)
                edge_index_dict[(src, rel, dst)] = mx.array(both.astype(np.int32))
            else:
                edge_index_dict[(dst, f"rev_{rel}", src)] = edge_index[::-1]
        data.edge_index_dict = edge_index_dict
        return data
