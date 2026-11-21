from typing import Iterator, Optional, Sequence, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.datasets.dataset import Dataset

from ._utils import to_index, to_numpy


class ImbalancedSampler:
    """Samples elements with probabilities inversely proportional to the
    frequency of their class, to rebalance an imbalanced classification task.

    Indices are drawn with replacement. The sampler can be passed as
    ``sampler`` to:

    - :class:`Dataloader`, for graph classification (``dataset`` is a
      :class:`Dataset` or a list of :class:`GraphData` with one
      ``graph_labels`` entry each);
    - :class:`NeighborLoader`, for node classification (``dataset`` is the
      :class:`GraphData`, and ``input_nodes`` the same seed nodes as given to
      the loader). It then yields positions in ``input_nodes``.

    Args:
        dataset: A :class:`GraphData` (node-level labels ``node_labels``), a
            :class:`Dataset` or list of :class:`GraphData` (graph-level labels
            ``graph_labels``), or directly an array of integer class labels.
        input_nodes: Node indices or boolean mask restricting the labels to a
            subset of nodes, when ``dataset`` is a :class:`GraphData` or an
            array. Defaults to None (all nodes)
        num_samples: Number of indices drawn per epoch. Defaults to the number
            of labels.

    Example:

    .. code-block:: python

        from mlx_graphs.loaders import Dataloader, ImbalancedSampler, NeighborLoader

        # Graph classification
        sampler = ImbalancedSampler(dataset)
        loader = Dataloader(dataset, batch_size=64, sampler=sampler)

        # Node classification
        sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
        loader = NeighborLoader(
            data,
            num_neighbors=[10, 10],
            input_nodes=data.train_mask,
            batch_size=64,
            sampler=sampler,
        )
    """

    def __init__(
        self,
        dataset: Union[GraphData, Dataset, Sequence[GraphData], mx.array, np.ndarray],
        input_nodes: Optional[Union[mx.array, np.ndarray]] = None,
        num_samples: Optional[int] = None,
    ):
        if isinstance(dataset, GraphData):
            if dataset.node_labels is None:
                raise ValueError("The graph has no `node_labels`.")
            y = to_numpy(dataset.node_labels).reshape(-1)
            if y.size != dataset.num_nodes:
                raise ValueError("ImbalancedSampler requires one label per node.")
            y = y[to_index(input_nodes, y.size)]
        elif isinstance(dataset, (mx.array, np.ndarray)):
            y = to_numpy(dataset).reshape(-1)
            y = y[to_index(input_nodes, y.size)]
        else:
            labels = []
            for i in range(len(dataset)):
                graph = dataset[i]
                if graph.graph_labels is None:
                    raise ValueError("Every graph needs `graph_labels`.")
                labels.append(to_numpy(graph.graph_labels).reshape(-1))
            y = np.concatenate(labels) if labels else np.zeros(0, dtype=np.int64)
            if y.size != len(dataset):
                raise ValueError("ImbalancedSampler requires one label per graph.")

        if y.size and not np.all(np.equal(np.mod(y, 1), 0)):
            raise ValueError("ImbalancedSampler requires integer class labels.")
        y = y.astype(np.int64)
        if y.size and y.min() < 0:
            raise ValueError("Class labels must be non-negative.")

        self.labels = y
        self.num_samples = y.size if num_samples is None else int(num_samples)
        counts = np.bincount(y) if y.size else np.zeros(0, dtype=np.int64)
        with np.errstate(divide="ignore"):
            class_weight = 1.0 / counts
        self.weight = class_weight[y]

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        p = self.weight / self.weight.sum()
        indices = np.random.choice(self.labels.size, self.num_samples, p=p)
        return iter(indices.tolist())


class DynamicBatchSampler:
    """Batch sampler grouping graphs into batches with at most ``max_num``
    nodes (or edges) in total, so that batches use a bounded amount of memory.

    Pass it as ``batch_sampler`` to :class:`Dataloader`. A graph larger than
    ``max_num`` on its own is either yielded alone in its batch, or skipped if
    ``skip_too_big`` is set.

    Args:
        dataset: A :class:`Dataset` or list of :class:`GraphData`.
        max_num: Maximum number of nodes (or edges) per batch.
        mode: ``"node"`` or ``"edge"``: what ``max_num`` counts.
            Defaults to ``"node"``
        shuffle: Whether to shuffle the graphs at the start of each epoch.
            Defaults to False
        skip_too_big: Whether to skip graphs larger than ``max_num``.
            Defaults to False
        num_steps: Maximum number of batches per epoch. Defaults to None (all
            graphs are used). The length of the sampler is only defined when
            it is set.

    Example:

    .. code-block:: python

        from mlx_graphs.loaders import Dataloader, DynamicBatchSampler

        sampler = DynamicBatchSampler(dataset, max_num=1000, mode="node")
        loader = Dataloader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[GraphData]],
        max_num: int,
        mode: str = "node",
        shuffle: bool = False,
        skip_too_big: bool = False,
        num_steps: Optional[int] = None,
    ):
        if max_num <= 0:
            raise ValueError(
                f"`max_num` should be a positive integer value (got {max_num})"
            )
        if mode not in ("node", "edge"):
            raise ValueError(
                f"`mode` choice should be either 'node' or 'edge' (got '{mode}')"
            )
        self.dataset = dataset
        self.max_num = max_num
        self.mode = mode
        self.shuffle = shuffle
        self.skip_too_big = skip_too_big
        self.num_steps = num_steps
        self._sizes: Optional[np.ndarray] = None

    def _graph_sizes(self) -> np.ndarray:
        if self._sizes is None:
            self._sizes = np.array(
                [
                    g.num_nodes if self.mode == "node" else g.num_edges
                    for g in (self.dataset[i] for i in range(len(self.dataset)))
                ],
                dtype=np.int64,
            )
        return self._sizes

    def __iter__(self) -> Iterator[list[int]]:
        sizes = self._graph_sizes()
        if self.shuffle:
            indices = np.random.permutation(len(sizes)).tolist()
        else:
            indices = list(range(len(sizes)))
        max_steps = self.num_steps if self.num_steps is not None else len(sizes)

        samples: list[int] = []
        current = 0
        steps = 0
        for i in indices:
            if steps >= max_steps:
                return
            num = int(sizes[i])
            if num > self.max_num and self.skip_too_big:
                continue
            if samples and current + num > self.max_num:
                yield samples
                steps += 1
                samples, current = [], 0
                if steps >= max_steps:
                    return
            samples.append(i)
            current += num
        if samples and steps < max_steps:
            yield samples

    def __len__(self) -> int:
        if self.num_steps is None:
            # A TypeError (not ValueError, as in PyG) so that `list(sampler)`
            # works: Python treats it as "no length".
            raise TypeError(
                f"The length of '{type(self).__name__}' is undefined since the "
                "number of steps per epoch is ambiguous. Either specify "
                "`num_steps` or use a static batch size."
            )
        return self.num_steps
