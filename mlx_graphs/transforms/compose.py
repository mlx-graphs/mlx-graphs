from typing import Any, Callable, Sequence

from mlx_graphs.transforms.base_transform import BaseTransform


class Compose(BaseTransform):
    """Chains several transforms together.

    The transforms are applied one after the other, each on the output of the
    previous one. If a transform returns several graphs (e.g.
    :class:`~mlx_graphs.transforms.RandomLinkSplit`), the following transforms
    are applied to each of them.

    Args:
        transforms: The transforms (or any callables taking and returning a
            graph) to apply, in order.

    Example:

    .. code-block:: python

        from mlx_graphs.transforms.compose import Compose
        from mlx_graphs.transforms.features import GCNNorm
        from mlx_graphs.transforms.structure import RemoveSelfLoops, ToUndirected

        transform = Compose([RemoveSelfLoops(), ToUndirected(), GCNNorm()])
        graph = transform(graph)
    """

    def __init__(self, transforms: Sequence[Callable[[Any], Any]]):
        self.transforms = list(transforms)

    def process(self, data: Any) -> Any:
        """Applies every transform in order.

        Args:
            data: A graph, or a list or tuple of graphs.

        Returns:
            The output of the last transform.
        """
        for transform in self.transforms:
            if isinstance(data, (list, tuple)):
                data = type(data)(transform(item) for item in data)
            else:
                data = transform(data)
        return data

    def __repr__(self) -> str:
        if not self.transforms:
            return f"{self.__class__.__name__}([])"
        lines = ",\n".join(f"  {transform}" for transform in self.transforms)
        return f"{self.__class__.__name__}([\n{lines}\n])"
