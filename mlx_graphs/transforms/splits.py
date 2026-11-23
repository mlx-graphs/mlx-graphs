import copy
import warnings
from typing import Literal, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.base_transform import BaseTransform
from mlx_graphs.transforms.structure import _attribute_names, _num_nodes
from mlx_graphs.utils.negative_sampling import negative_sampling
from mlx_graphs.utils.subgraph import _take_rows


def _random_permutation(size: int) -> np.ndarray:
    """Random permutation of ``range(size)`` on the host, from ``mx.random``."""
    if size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(mx.random.permutation(size)).astype(np.int64)


def _resolve_size(size: Union[int, float], total: int, rounded: bool) -> int:
    """Number of items for a split given as a count or as a ratio of ``total``."""
    if isinstance(size, float):
        return round(total * size) if rounded else int(total * size)
    return int(size)


class RandomNodeSplit(BaseTransform):
    """Randomly splits the nodes of a graph into training, validation and test
    sets, stored as the boolean masks ``train_mask``, ``val_mask`` and
    ``test_mask`` of shape ``[num_nodes]``.

    Randomness comes from ``mx.random.permutation``, so results are
    reproducible with ``mx.random.seed``.

    Args:
        split: The type of split:

            - ``"train_rest"``: ``num_val`` random nodes for validation,
              ``num_test`` for test, and all the others for training.
            - ``"test_rest"``: ``num_train_per_class`` random nodes of each
              class for training, ``num_val`` for validation, and all the
              others for test.
            - ``"random"``: ``num_train_per_class`` random nodes of each class
              for training, ``num_val`` for validation and ``num_test`` for
              test. The remaining nodes are in no split.

            Default: ``"train_rest"``
        num_val: Number of validation nodes, or ratio of the nodes if given as
            a float. Default: ``500``
        num_test: Number of test nodes, or ratio of the nodes if given as a
            float. Default: ``1000``
        num_train_per_class: Number of training nodes per class for the
            ``"test_rest"`` and ``"random"`` splits. Default: ``20``
        key: Name of the attribute holding the class index of each node, used
            by the ``"test_rest"`` and ``"random"`` splits. Nodes with a
            negative label are never used for training.
            Default: ``"node_labels"``

    Example:

    .. code-block:: python

        graph = GraphData(
            edge_index=mx.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
            node_features=mx.ones((5, 2)),
        )
        graph = RandomNodeSplit(num_val=1, num_test=0.2)(graph)
        graph.train_mask
        # e.g. mx.array([True, False, True, False, True])
    """

    def __init__(
        self,
        split: Literal["train_rest", "test_rest", "random"] = "train_rest",
        num_val: Union[int, float] = 500,
        num_test: Union[int, float] = 1000,
        num_train_per_class: int = 20,
        key: str = "node_labels",
    ):
        if split not in ("train_rest", "test_rest", "random"):
            raise ValueError(
                f"Invalid split '{split}', expected 'train_rest', 'test_rest' "
                "or 'random'"
            )
        self.split = split
        self.num_val = num_val
        self.num_test = num_test
        self.num_train_per_class = num_train_per_class
        self.key = key

    def process(self, data: GraphData) -> GraphData:
        """Adds the ``train_mask``, ``val_mask`` and ``test_mask`` attributes.

        Args:
            data: The graph to split.

        Returns:
            The graph with the three masks.
        """
        num_nodes = _num_nodes(data)
        num_val = _resolve_size(self.num_val, num_nodes, rounded=True)
        num_test = _resolve_size(self.num_test, num_nodes, rounded=True)

        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)

        if self.split == "train_rest":
            perm = _random_permutation(num_nodes)
            val_mask[perm[:num_val]] = True
            test_mask[perm[num_val : num_val + num_test]] = True
            train_mask[perm[num_val + num_test :]] = True
        else:
            labels = getattr(data, self.key, None)
            if labels is None:
                raise ValueError(
                    f"The '{self.split}' split requires the '{self.key}' attribute"
                )
            labels = np.asarray(labels).reshape(-1)
            if labels.size != num_nodes:
                raise ValueError(
                    f"'{self.key}' must hold one class index per node (got "
                    f"{labels.size} entries for {num_nodes} nodes)"
                )
            # Shuffle the nodes, group them by class (keeping the shuffled order
            # within each class) and take the first ones of each class.
            perm = _random_permutation(num_nodes)
            perm = perm[labels[perm] >= 0]
            perm = perm[np.argsort(labels[perm], kind="stable")]
            sorted_labels = labels[perm]
            is_first = np.ones(perm.size, dtype=bool)
            is_first[1:] = sorted_labels[1:] != sorted_labels[:-1]
            class_start = np.maximum.accumulate(
                np.where(is_first, np.arange(perm.size), 0)
            )
            rank_in_class = np.arange(perm.size) - class_start
            train_mask[perm[rank_in_class < self.num_train_per_class]] = True

            remaining = np.nonzero(~train_mask)[0]
            remaining = remaining[_random_permutation(remaining.size)]
            val_mask[remaining[:num_val]] = True
            if self.split == "test_rest":
                test_mask[remaining[num_val:]] = True
            else:
                test_mask[remaining[num_val : num_val + num_test]] = True

        data.train_mask = mx.array(train_mask)
        data.val_mask = mx.array(val_mask)
        data.test_mask = mx.array(test_mask)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(split='{self.split}')"


class RandomLinkSplit(BaseTransform):
    """Randomly splits the edges of a graph into training, validation and test
    sets for link prediction.

    Three graphs are returned. Each one has

    - ``edge_index``: the edges to use for message passing;
    - ``edge_label_index``: the ``[2, num_labels]`` edges to predict, positive
      edges first, then negative edges;
    - ``edge_label``: the ``[num_labels]`` float32 labels of these edges, 1 for
      positive (existing) edges and 0 for negative (sampled) ones.

    There is no leakage between the splits, as in PyG:

    - the training graph uses the training edges both for message passing and
      as positive edges to predict;
    - the validation graph predicts the validation edges, using only the
      training edges for message passing;
    - the test graph predicts the test edges, using the training and
      validation edges for message passing.

    The negative edges are sampled once among the node pairs that are not
    connected in the full graph (see
    :func:`~mlx_graphs.utils.negative_sampling.negative_sampling`) and shared
    out between the splits, so they do not overlap either. Per-edge attributes
    (e.g. ``edge_features``) follow the message passing edges, and the number
    of nodes of each graph is set explicitly to that of the input graph.

    Randomness comes from ``mx.random`` (``mx.random.permutation`` for the
    split, a NumPy generator seeded from ``mx.random`` for the negative
    edges), so results are reproducible with ``mx.random.seed``.

    Args:
        num_val: Number of validation edges, or ratio of the edges if given
            as a float. Default: ``0.1``
        num_test: Number of test edges, or ratio of the edges if given as a
            float. Default: ``0.2``
        is_undirected: If ``True``, the graph is assumed to be undirected, i.e.
            to hold both directions of every edge. Both directions then go to
            the same split: the edges with ``src <= dst`` are split,
            ``num_val`` and ``num_test`` count these, ``edge_label_index``
            holds only that direction, and the message passing ``edge_index``
            holds both directions. Default: ``False``
        add_negative_train_samples: Whether to add negative edges to the
            training graph. Set it to ``False`` to sample new negative edges
            at every training epoch instead. Default: ``True``
        neg_sampling_ratio: Number of negative edges per positive edge.
            Default: ``1.0``

    Example:

    .. code-block:: python

        graph = GraphData(
            edge_index=mx.array(
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
            ),
            node_features=mx.ones((10, 4)),
        )
        train, val, test = RandomLinkSplit(num_val=0.1, num_test=0.2)(graph)

        train.edge_index.shape, train.edge_label_index.shape
        # (2, 7), (2, 14)
        val.edge_index.shape, val.edge_label_index.shape
        # (2, 7), (2, 2)
        test.edge_index.shape, test.edge_label_index.shape
        # (2, 8), (2, 4)
    """

    def __init__(
        self,
        num_val: Union[int, float] = 0.1,
        num_test: Union[int, float] = 0.2,
        is_undirected: bool = False,
        add_negative_train_samples: bool = True,
        neg_sampling_ratio: float = 1.0,
    ):
        self.num_val = num_val
        self.num_test = num_test
        self.is_undirected = is_undirected
        self.add_negative_train_samples = add_negative_train_samples
        self.neg_sampling_ratio = neg_sampling_ratio

    def process(self, data: GraphData) -> tuple[GraphData, GraphData, GraphData]:
        """Splits the edges of ``data``.

        Args:
            data: The graph to split.

        Returns:
            A tuple ``(train, val, test)`` of graphs.
        """
        num_nodes = _num_nodes(data)
        _, edge_keys = _attribute_names(data, num_nodes)
        edge_index = data.edge_index
        edges = np.asarray(edge_index)

        if self.is_undirected:
            candidates = np.nonzero(edges[0] <= edges[1])[0]
        else:
            candidates = np.arange(edges.shape[1])
        perm = candidates[_random_permutation(candidates.size)]

        num_val = _resolve_size(self.num_val, perm.size, rounded=False)
        num_test = _resolve_size(self.num_test, perm.size, rounded=False)
        num_train = perm.size - num_val - num_test
        if num_train <= 0:
            raise ValueError("Insufficient number of edges for training")

        train_edges = perm[:num_train]
        val_edges = perm[num_train : num_train + num_val]
        test_edges = perm[num_train + num_val :]
        train_val_edges = perm[: num_train + num_val]

        num_neg_train = 0
        if self.add_negative_train_samples:
            num_neg_train = int(num_train * self.neg_sampling_ratio)
        num_neg_val = int(num_val * self.neg_sampling_ratio)
        num_neg_test = int(num_test * self.neg_sampling_ratio)
        num_neg = num_neg_train + num_neg_val + num_neg_test

        negatives = np.zeros((2, 0), dtype=edges.dtype)
        if num_neg > 0:
            negatives = np.asarray(
                negative_sampling(edge_index, num_nodes, num_neg_samples=num_neg)
            )
            num_neg_found = negatives.shape[1]
            if num_neg_found < num_neg:
                warnings.warn(
                    "There are not enough negative edges to satisfy the provided "
                    "sampling ratio. The ratio will be adjusted to "
                    f"{num_neg_found / num_neg:.2f}.",
                    stacklevel=2,
                )
                num_neg_train = int((num_neg_train / num_neg) * num_neg_found)
                num_neg_val = int((num_neg_val / num_neg) * num_neg_found)
                num_neg_test = num_neg_found - num_neg_train - num_neg_val

        splits = (
            (train_edges, train_edges, negatives[:, num_neg_val + num_neg_test :]),
            (train_edges, val_edges, negatives[:, :num_neg_val]),
            (
                train_val_edges,
                test_edges,
                negatives[:, num_neg_val : num_neg_val + num_neg_test],
            ),
        )
        train, val, test = (
            self._build(data, edges, edge_keys, num_nodes, *split) for split in splits
        )
        return train, val, test

    def _build(
        self,
        data: GraphData,
        edges: np.ndarray,
        edge_keys: list[str],
        num_nodes: int,
        message_edges: np.ndarray,
        positive_edges: np.ndarray,
        negative_edge_index: np.ndarray,
    ) -> GraphData:
        """Builds the graph of one split.

        ``message_edges`` and ``positive_edges`` are positions in ``edges``.
        """
        out = copy.copy(data)
        dtype = data.edge_index.dtype

        message_index = edges[:, message_edges]
        if self.is_undirected:
            message_index = np.concatenate([message_index, message_index[::-1]], axis=1)
        out.edge_index = mx.array(message_index, dtype=dtype)

        for key in edge_keys:
            values = _take_rows(getattr(data, key), message_edges)
            if self.is_undirected:
                values = mx.concatenate([values, values], axis=0)
            setattr(out, key, values)

        label_index = np.concatenate(
            [edges[:, positive_edges], negative_edge_index], axis=1
        )
        labels = np.zeros(label_index.shape[1], dtype=np.float32)
        labels[: positive_edges.size] = 1.0
        out.edge_label_index = mx.array(label_index, dtype=dtype)
        out.edge_label = mx.array(labels)
        out.num_nodes = num_nodes
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_val={self.num_val}, "
            f"num_test={self.num_test})"
        )
