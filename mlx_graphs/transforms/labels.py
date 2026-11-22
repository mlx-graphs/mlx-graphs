from typing import Sequence

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.base_transform import BaseTransform


class RemoveTrainingClasses(BaseTransform):
    r"""Removes classes from the node-level training set given by
    ``train_mask``, e.g. to build a zero-shot setting, as PyG's
    ``RemoveTrainingClasses``.

    The nodes whose ``node_labels`` is one of ``classes`` are set to ``False``
    in ``train_mask``. Other masks are unchanged.

    Args:
        classes: The classes to remove from the training set.

    Example:

    .. code-block:: python

        graph = GraphData(
            edge_index=mx.array([[0, 1], [1, 2]]),
            node_labels=mx.array([0, 1, 2]),
            train_mask=mx.array([True, True, True]),
        )
        graph = RemoveTrainingClasses([1])(graph)
        graph.train_mask
        # mx.array([True, False, True])
    """

    def __init__(self, classes: Sequence[int]):
        self.classes = list(classes)

    def process(self, data: GraphData) -> GraphData:
        """Removes the classes from the training mask of ``data``.

        Args:
            data: The graph to transform. It must have ``node_labels`` and a
                ``train_mask``.

        Returns:
            The graph with the updated ``train_mask``.
        """
        train_mask = getattr(data, "train_mask", None)
        if train_mask is None or data.node_labels is None:
            raise ValueError(
                "RemoveTrainingClasses requires `node_labels` and a `train_mask`"
            )
        removed = np.isin(np.asarray(data.node_labels), self.classes)
        data.train_mask = train_mask & ~mx.array(removed).reshape(train_mask.shape)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.classes})"
