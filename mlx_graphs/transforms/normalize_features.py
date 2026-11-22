from typing import List, Union

import mlx.core as mx

from mlx_graphs.data import GraphData, GraphDataBatch
from mlx_graphs.transforms import BaseTransform

# PyG attribute names accepted by `NormalizeFeatures`.
_PYG_NAMES = {"x": "node_features", "edge_attr": "edge_features"}


class FeaturesNormalizedTransform(BaseTransform):
    def __init__(self, attributes: List[str] = ["node_features"]):
        self.attributes = attributes

    def process(
        self, data: Union[GraphData, GraphDataBatch]
    ) -> Union[GraphData, GraphDataBatch]:
        """Normalizes the attributes given in the attributes to sum up to one

        Args:
            data: A GraphData object with node/edge features

        Returns:
            A GraphData object with normalized features (either edges/node
            features unless specified in the transform)

        Example:

        .. code-block:: python

            from mlx_graphs.datasets import EllipticBitcoinDataset
            from mlx_graphs.transforms import FeaturesNormalizedTransform

            dataset = EllipticBitcoinDataset(transform=FeaturesNormalizedTransform())
            # All the node features (if present) for this graph are normalized
        """
        if isinstance(data, GraphData):
            for attribute in self.attributes:
                array = getattr(data, attribute, None)
                if array is not None and array.size > 0:
                    array = array - mx.min(array)
                    sum_val = mx.sum(array, axis=-1, keepdims=True)
                    setattr(data, attribute, array / mx.clip(sum_val, 1.0, None))
        return data


class NormalizeFeatures(FeaturesNormalizedTransform):
    r"""Row-normalizes the given attributes to sum up to one, as PyG's
    ``NormalizeFeatures``.

    Each attribute is first shifted by its global minimum, so that all its
    values are non-negative, then every row is divided by its sum (rows that
    sum to less than one are divided by one):

    .. math::
        \mathbf{x}_i \leftarrow \frac{\mathbf{x}_i - \min(\mathbf{X})}
        {\max\left(1, \sum_j (x_{ij} - \min(\mathbf{X}))\right)}

    This is the PyG-named equivalent of :class:`FeaturesNormalizedTransform`.

    Args:
        attrs: Names of the attributes to normalize. PyG names (``"x"``,
            ``"edge_attr"``) are accepted and mapped to ``node_features`` and
            ``edge_features``. Default: ``["node_features"]``

    Example:

    .. code-block:: python

        graph = GraphData(
            edge_index=mx.array([[0, 1], [1, 0]]),
            node_features=mx.array([[1.0, 3.0], [0.0, 0.0]]),
        )
        graph = NormalizeFeatures()(graph)
        graph.node_features
        # mx.array([[0.25, 0.75], [0.0, 0.0]])
    """

    def __init__(self, attrs: List[str] = ["node_features"]):
        super().__init__([_PYG_NAMES.get(attr, attr) for attr in attrs])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
