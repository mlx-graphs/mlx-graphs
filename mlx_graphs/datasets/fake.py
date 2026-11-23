from collections import defaultdict
from itertools import product
from typing import Callable, Literal, Optional, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData, HeteroGraphData
from mlx_graphs.datasets.dataset import Dataset
from mlx_graphs.datasets.hetero_dataset import HeteroDataset
from mlx_graphs.utils import coalesce, remove_self_loops, to_undirected

FAKE_TASKS = Literal["auto", "node", "graph"]


def _get_num_nodes(rng: np.random.Generator, avg_num_nodes: int, avg_degree: float):
    min_num_nodes = max(3 * avg_num_nodes // 4, int(avg_degree))
    max_num_nodes = 5 * avg_num_nodes // 4
    return int(rng.integers(min_num_nodes, max_num_nodes, endpoint=True))


def _get_num_channels(rng: np.random.Generator, num_channels: int) -> int:
    min_num_channels = 3 * num_channels // 4
    max_num_channels = 5 * num_channels // 4
    return int(rng.integers(min_num_channels, max_num_channels, endpoint=True))


def _get_edge_index(
    rng: np.random.Generator,
    num_src_nodes: int,
    num_dst_nodes: int,
    avg_degree: float,
    is_undirected: bool = False,
    remove_loops: bool = False,
) -> mx.array:
    num_edges = int(num_src_nodes * avg_degree)
    row = rng.integers(0, num_src_nodes, num_edges)
    col = rng.integers(0, num_dst_nodes, num_edges)
    edge_index = mx.array(np.stack([row, col]).astype(np.int32))
    if remove_loops:
        edge_index = remove_self_loops(edge_index)
    num_nodes = max(num_src_nodes, num_dst_nodes)
    if is_undirected:
        return to_undirected(edge_index, coalesce=True, num_nodes=num_nodes)
    return coalesce(edge_index, num_nodes=num_nodes)


def _randn(rng: np.random.Generator, shape) -> mx.array:
    return mx.array(rng.standard_normal(shape, dtype=np.float32))


def _rand(rng: np.random.Generator, shape) -> mx.array:
    return mx.array(rng.random(shape, dtype=np.float32))


class FakeDataset(Dataset):
    """A fake dataset that returns randomly generated
    :class:`~mlx_graphs.data.GraphData` objects, useful to test models and
    pipelines without downloading anything.

    Node features are standard normal samples shifted by the (node or graph)
    label, so that the classification task is learnable. Graphs are generated
    once at construction and are deterministic for a given ``seed``.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.FakeDataset.html>`_,
    with an additional ``seed`` argument. The random values differ from PyG's
    since they are drawn from NumPy.

    Args:
        num_graphs: The number of graphs. Default: ``1``.
        avg_num_nodes: The average number of nodes in a graph; the number of
            nodes of each graph is drawn uniformly in
            ``[3/4 * avg_num_nodes, 5/4 * avg_num_nodes]``. Default: ``1000``.
        avg_degree: The average degree per node. Default: ``10.0``.
        num_channels: The number of node features. If ``0``, the graphs have
            no node features and ``num_nodes`` is set instead.
            Default: ``64``.
        edge_dim: The number of edge features. If ``1``, a 1-dimensional
            ``edge_weights`` attribute is set instead of ``edge_features``.
            Default: ``0``.
        num_classes: The number of classes in the dataset. Default: ``10``.
        task: Whether to return node-level (``"node"``) or graph-level
            (``"graph"``) labels. If ``"auto"``, graph-level labels are used
            when ``num_graphs > 1``, node-level labels otherwise.
            Default: ``"auto"``.
        is_undirected: Whether the graphs are undirected. Default: ``True``.
        seed: Seed of the random number generator. Default: ``None``
            (non-deterministic).
        pre_transform: A function/transform applied to each graph once, after
            generation.
        transform: A function/transform applied to the graph at every access.
        **kwargs: Additional graph attributes and their shapes, *e.g.*
            ``global_features=5``, filled with standard normal samples.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import FakeDataset

        dataset = FakeDataset(num_graphs=4, avg_num_nodes=20, seed=0)
        >>> fake(num_graphs=4)

        dataset[0]
        >>> GraphData(
                edge_index(shape=(2, 316), int32)
                node_features(shape=(24, 64), float32)
                graph_labels(shape=(1,), int32))
    """

    def __init__(
        self,
        num_graphs: int = 1,
        avg_num_nodes: int = 1000,
        avg_degree: float = 10.0,
        num_channels: int = 64,
        edge_dim: int = 0,
        num_classes: int = 10,
        task: FAKE_TASKS = "auto",
        is_undirected: bool = True,
        seed: Optional[int] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        **kwargs: Union[int, tuple[int, ...]],
    ):
        if task == "auto":
            task = "graph" if num_graphs > 1 else "node"
        assert task in ["node", "graph"], "Invalid task"

        self._num_graphs = max(num_graphs, 1)
        self.avg_num_nodes = max(avg_num_nodes, int(avg_degree))
        self.avg_degree = max(avg_degree, 1)
        self.num_channels = num_channels
        self.edge_dim = edge_dim
        self._num_classes_ = num_classes
        self.task = task
        self.is_undirected = is_undirected
        self.seed = seed
        self.kwargs = kwargs
        super().__init__(name="fake", pre_transform=pre_transform, transform=transform)

    def _load(self):
        # generated in memory, nothing is downloaded or saved
        self._process()

    def download(self):
        pass

    def process(self):
        rng = np.random.default_rng(self.seed)
        self.graphs = [self._generate_graph(rng) for _ in range(self._num_graphs)]

    def _generate_graph(self, rng: np.random.Generator) -> GraphData:
        num_nodes = _get_num_nodes(rng, self.avg_num_nodes, self.avg_degree)
        num_classes = self._num_classes_

        node_labels = graph_labels = None
        if num_classes > 0 and self.task == "node":
            node_labels = rng.integers(0, num_classes, num_nodes).astype(np.int32)
        elif num_classes > 0 and self.task == "graph":
            graph_labels = rng.integers(0, num_classes, 1).astype(np.int32)

        edge_index = _get_edge_index(
            rng,
            num_nodes,
            num_nodes,
            self.avg_degree,
            is_undirected=self.is_undirected,
            remove_loops=True,
        )
        graph = GraphData(edge_index=edge_index)

        if self.num_channels > 0:
            x = rng.standard_normal((num_nodes, self.num_channels), dtype=np.float32)
            if node_labels is not None:
                x = x + node_labels[:, None]
            elif graph_labels is not None:
                x = x + graph_labels
            graph.node_features = mx.array(x)
        else:
            graph.num_nodes = num_nodes

        if node_labels is not None:
            graph.node_labels = mx.array(node_labels)
        if graph_labels is not None:
            graph.graph_labels = mx.array(graph_labels)

        num_edges = edge_index.shape[1]
        if self.edge_dim > 1:
            graph.edge_features = _rand(rng, (num_edges, self.edge_dim))
        elif self.edge_dim == 1:
            graph.edge_weights = _rand(rng, (num_edges,))

        for feature_name, feature_shape in self.kwargs.items():
            setattr(graph, feature_name, _randn(rng, feature_shape))

        return graph


class FakeHeteroDataset(HeteroDataset):
    """A fake dataset that returns randomly generated
    :class:`~mlx_graphs.data.HeteroGraphData` objects, useful to test
    heterogeneous models and pipelines without downloading anything.

    Node types are named ``"v0"``, ``"v1"``, ... and edge types
    ``("v<i>", "e<k>", "v<j>")``. Only the first node type ``"v0"`` has
    node-level labels. Graphs are generated once at construction and are
    deterministic for a given ``seed``.

    This dataset follows the implementation of `PyG
    <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.FakeHeteroDataset.html>`_,
    with an additional ``seed`` argument. The random values differ from PyG's
    since they are drawn from NumPy.

    Args:
        num_graphs: The number of graphs. Default: ``1``.
        num_node_types: The number of node types. Default: ``3``.
        num_edge_types: The number of edge types. Default: ``6``.
        avg_num_nodes: The average number of nodes of each type in a graph.
            Default: ``1000``.
        avg_degree: The average degree per node. Default: ``10.0``.
        avg_num_channels: The average number of node features per node type;
            each node type gets a number of features drawn uniformly in
            ``[3/4 * avg_num_channels, 5/4 * avg_num_channels]``.
            Default: ``64``.
        edge_dim: The number of edge features. If ``1``, 1-dimensional edge
            weights are stored in ``edge_features_dict``. Default: ``0``.
        num_classes: The number of classes in the dataset. Default: ``10``.
        task: Whether to return node-level (``"node"``) or graph-level
            (``"graph"``) labels. If ``"auto"``, graph-level labels are used
            when ``num_graphs > 1``, node-level labels otherwise.
            Default: ``"auto"``.
        seed: Seed of the random number generator. Default: ``None``
            (non-deterministic).
        pre_transform: A function/transform applied to each graph once, after
            generation.
        transform: A function/transform applied to the graph at every access.
        **kwargs: Additional graph attributes and their shapes, *e.g.*
            ``global_features=5``, filled with standard normal samples.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import FakeHeteroDataset

        dataset = FakeHeteroDataset(avg_num_nodes=20, seed=0)
        dataset[0].edge_index_dict.keys()
        >>> dict_keys([('v1', 'e0', 'v1'), ('v1', 'e0', 'v2'), ...])
    """

    def __init__(
        self,
        num_graphs: int = 1,
        num_node_types: int = 3,
        num_edge_types: int = 6,
        avg_num_nodes: int = 1000,
        avg_degree: float = 10.0,
        avg_num_channels: int = 64,
        edge_dim: int = 0,
        num_classes: int = 10,
        task: FAKE_TASKS = "auto",
        seed: Optional[int] = None,
        pre_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        **kwargs: Union[int, tuple[int, ...]],
    ):
        if task == "auto":
            task = "graph" if num_graphs > 1 else "node"
        assert task in ["node", "graph"], "Invalid task"

        rng = np.random.default_rng(seed)
        self._rng = rng
        self._num_graphs = max(num_graphs, 1)

        self.node_types = [f"v{i}" for i in range(max(num_node_types, 1))]
        pairs = list(product(self.node_types, self.node_types))
        edge_pairs: list[tuple[str, str]] = []
        while len(edge_pairs) < max(num_edge_types, 1):
            edge_pairs.extend(pairs)
        edge_pairs = [edge_pairs[i] for i in rng.permutation(len(edge_pairs))]
        self.edge_types: list[tuple[str, str, str]] = []
        count: dict[tuple[str, str], int] = defaultdict(int)
        for src, dst in edge_pairs[: max(num_edge_types, 1)]:
            self.edge_types.append((src, f"e{count[(src, dst)]}", dst))
            count[(src, dst)] += 1

        self.avg_num_nodes = max(avg_num_nodes, int(avg_degree))
        self.avg_degree = max(avg_degree, 1)
        self.num_channels = [
            _get_num_channels(rng, avg_num_channels) for _ in self.node_types
        ]
        self.edge_dim = edge_dim
        self._num_classes_ = num_classes
        self.task = task
        self.kwargs = kwargs
        super().__init__(
            name="fake_hetero", pre_transform=pre_transform, transform=transform
        )

    def _load(self):
        # generated in memory, nothing is downloaded or saved
        self._process()

    def download(self):
        pass

    def process(self):
        self.graphs = [self._generate_graph(self._rng) for _ in range(self._num_graphs)]

    def _generate_graph(self, rng: np.random.Generator) -> HeteroGraphData:
        num_nodes: dict[str, int] = {}
        node_features_dict: dict[str, mx.array] = {}
        node_labels_dict: dict[str, mx.array] = {}
        for i, (node_type, num_channels) in enumerate(
            zip(self.node_types, self.num_channels)
        ):
            num_nodes[node_type] = _get_num_nodes(
                rng, self.avg_num_nodes, self.avg_degree
            )
            if num_channels > 0:
                node_features_dict[node_type] = _randn(
                    rng, (num_nodes[node_type], num_channels)
                )
            if self._num_classes_ > 0 and self.task == "node" and i == 0:
                node_labels_dict[node_type] = mx.array(
                    rng.integers(0, self._num_classes_, num_nodes[node_type]).astype(
                        np.int32
                    )
                )

        edge_index_dict: dict[tuple[str, str, str], mx.array] = {}
        edge_features_dict: dict[tuple[str, str, str], mx.array] = {}
        for src, rel, dst in self.edge_types:
            edge_index = _get_edge_index(
                rng, num_nodes[src], num_nodes[dst], self.avg_degree
            )
            edge_index_dict[(src, rel, dst)] = edge_index
            if self.edge_dim > 1:
                edge_features_dict[(src, rel, dst)] = _rand(
                    rng, (edge_index.shape[1], self.edge_dim)
                )
            elif self.edge_dim == 1:
                edge_features_dict[(src, rel, dst)] = _rand(rng, (edge_index.shape[1],))

        graph = HeteroGraphData(
            edge_index_dict=edge_index_dict,
            node_features_dict=node_features_dict or None,
            edge_features_dict=edge_features_dict or None,
            node_labels_dict=node_labels_dict or None,
            num_nodes_dict=num_nodes,
        )
        if self._num_classes_ > 0 and self.task == "graph":
            graph.graph_labels = mx.array(
                rng.integers(0, self._num_classes_, 1).astype(np.int32)
            )
        for feature_name, feature_shape in self.kwargs.items():
            setattr(graph, feature_name, _randn(rng, feature_shape))
        return graph
