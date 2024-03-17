from typing import List, Union

import mlx.core as mx

from mlx_graphs.data import GraphData, GraphDataBatch
from mlx_graphs.transforms import BaseTransform


class NormalizeFeatures(BaseTransform):
    def __init__(self, attributes: List[str] = ["node_features"]):
        self.attributes = attributes

    def process(
        self, data: Union[GraphData, GraphDataBatch]
    ) -> Union[GraphData, GraphDataBatch]:
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
