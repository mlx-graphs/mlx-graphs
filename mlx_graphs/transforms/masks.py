from typing import Optional, Sequence, Union

import mlx.core as mx

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.base_transform import BaseTransform
from mlx_graphs.utils.array_ops import index_to_mask
from mlx_graphs.utils.indexing import mask_to_index


def _attrs_with_suffix(
    attrs: Optional[list[str]], data: GraphData, suffix: str
) -> list[str]:
    if attrs is not None:
        return attrs
    return [
        key
        for key, value in data.to_dict().items()
        if key.endswith(suffix) and isinstance(value, mx.array)
    ]


class IndexToMask(BaseTransform):
    """Converts index attributes (named ``*_index``, e.g. ``train_index``)
    to boolean mask attributes (named ``*_mask``, e.g. ``train_mask``).

    Attributes containing ``edge_index`` are never converted.

    Args:
        attrs: Name(s) of the index attributes to convert. Default: ``None``
            (all the attributes ending with ``_index``).
        sizes: Size of the masks, as one int for all the attributes or a list
            with one int per attribute. Default: ``None`` (the number of edges
            for attributes containing ``edge``, the number of nodes
            otherwise).
        replace: If ``True``, the index attributes are removed.
            Default: ``False``

    Example:

    .. code-block:: python

        graph = GraphData(
            edge_index=mx.array([[0, 1, 2], [1, 2, 3]]),
            node_features=mx.ones((4, 1)),
            train_index=mx.array([0, 2]),
        )
        graph = IndexToMask()(graph)
        graph.train_mask
        # mx.array([True, False, True, False])
    """

    def __init__(
        self,
        attrs: Optional[Union[str, list[str]]] = None,
        sizes: Optional[Union[int, Sequence[int]]] = None,
        replace: bool = False,
    ):
        self.attrs = [attrs] if isinstance(attrs, str) else attrs
        self.sizes = sizes
        self.replace = replace

    def process(self, data: GraphData) -> GraphData:
        """Adds the masks to ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with the mask attributes.
        """
        attrs = _attrs_with_suffix(self.attrs, data, "_index")
        if isinstance(self.sizes, int):
            sizes = [self.sizes] * len(attrs)
        elif isinstance(self.sizes, (list, tuple)):
            if len(attrs) != len(self.sizes):
                raise ValueError(
                    f"The number of attributes (got {len(attrs)}) must match "
                    f"the number of sizes provided (got {len(self.sizes)})"
                )
            sizes = list(self.sizes)
        else:
            sizes = [None] * len(attrs)

        for attr, size in zip(attrs, sizes):
            if "edge_index" in attr or getattr(data, attr, None) is None:
                continue
            if size is None:
                size = data.num_edges if "edge" in attr else data.num_nodes
            setattr(data, f"{attr[:-6]}_mask", index_to_mask(getattr(data, attr), size))
            if self.replace:
                delattr(data, attr)
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(attrs={self.attrs}, "
            f"sizes={self.sizes}, replace={self.replace})"
        )


class MaskToIndex(BaseTransform):
    """Converts boolean mask attributes (named ``*_mask``, e.g.
    ``train_mask``) to index attributes (named ``*_index``, e.g.
    ``train_index``).

    Args:
        attrs: Name(s) of the mask attributes to convert. Default: ``None``
            (all the attributes ending with ``_mask``).
        replace: If ``True``, the mask attributes are removed.
            Default: ``False``

    Example:

    .. code-block:: python

        graph = GraphData(
            edge_index=mx.array([[0, 1, 2], [1, 2, 3]]),
            train_mask=mx.array([True, False, True, False]),
        )
        graph = MaskToIndex()(graph)
        graph.train_index
        # mx.array([0, 2])
    """

    def __init__(
        self,
        attrs: Optional[Union[str, list[str]]] = None,
        replace: bool = False,
    ):
        self.attrs = [attrs] if isinstance(attrs, str) else attrs
        self.replace = replace

    def process(self, data: GraphData) -> GraphData:
        """Adds the indices to ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with the index attributes.
        """
        for attr in _attrs_with_suffix(self.attrs, data, "_mask"):
            if getattr(data, attr, None) is None:
                continue
            setattr(data, f"{attr[:-5]}_index", mask_to_index(getattr(data, attr)))
            if self.replace:
                delattr(data, attr)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(attrs={self.attrs}, replace={self.replace})"
