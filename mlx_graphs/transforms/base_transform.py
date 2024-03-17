import copy
from abc import ABC
from typing import Any


class BaseTransform(ABC):
    r"""An abstract class for creating transformations
    Transforms are a way to modify and customize
    GraphData by implicitly passing them as arguments
    in mlx_graphs.dataset.Dataset

    Example:

    .. code-block:: python

        from mlx_graphs.datasets import Planetoid
        from mlx_graphs.transforms import FeaturesNormalizedTransform

        dataset = EllipticBitcoinDataset(transform=FeaturesNormalizedTransform())
        # All the node features (if present) for this graph are normalized
    """

    def __call__(self, data: Any) -> Any:
        return self.process(copy.copy(data))

    def process(self, data: Any) -> Any:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
