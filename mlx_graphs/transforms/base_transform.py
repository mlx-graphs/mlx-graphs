import copy
from abc import ABC
from typing import Any


class BaseTransform(ABC):
    r"""An abstract class for creating transformations

    Transforms are a way to modify and customize
    mlx_graphs.data.GraphData into suitable formats

    """

    def __call__(self, data: Any) -> Any:
        return self.process(copy.copy(data))

    def process(self, data: Any) -> Any:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
