from typing import List, Union

import mlx.core as mx

from mlx_graphs.data import GraphData, GraphDataBatch
from mlx_graphs.transforms import BaseTransform


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
                array = getattr(data, attribute)
                if array is not None:
                    if array.size > 0:
                        array = array - mx.min(array)
                        sum_val = mx.sum(array, axis=-1, keepdims=True)
                        new_array = array / mx.clip(sum_val, 1.0, None)
                    setattr(data, attribute, new_array)
        return data
