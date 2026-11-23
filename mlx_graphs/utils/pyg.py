from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from torch_geometric.data import Data

    from mlx_graphs.data import GraphData

# PyG attribute name -> mlx-graphs attribute name (`y` is handled separately).
_PYG_TO_MLX_NAMES = {
    "x": "node_features",
    "edge_attr": "edge_features",
    "edge_weight": "edge_weights",
}
_MLX_TO_PYG_NAMES = {value: key for key, value in _PYG_TO_MLX_NAMES.items()}

_SCALAR_TYPES = (bool, int, float, str)


def _import_pyg() -> tuple[Any, Any]:
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as error:
        raise ImportError(
            "torch and torch_geometric are required to convert to/from PyG, "
            "run `pip install torch torch_geometric`"
        ) from error
    return torch, Data


def from_pyg(data: "Data") -> "GraphData":
    """
    Converts a :class:`torch_geometric.data.Data` object to a
    :class:`mlx_graphs.data.GraphData`.

    The attributes are renamed as follows:

    - ``x`` -> ``node_features``
    - ``edge_attr`` -> ``edge_features``
    - ``edge_weight`` -> ``edge_weights``
    - ``y`` -> ``node_labels`` if it has one entry per node (and the graph
      has more than one node), ``graph_labels`` otherwise

    Every other tensor attribute (e.g. ``train_mask``, ``pos``) is converted
    to an ``mx.array`` and kept under the same name, and plain Python scalars
    and strings are carried over as they are. ``num_nodes`` is always set
    explicitly, so isolated nodes are preserved. float64 tensors become
    float32 arrays.

    ``torch`` and ``torch_geometric`` are optional dependencies, imported when
    the function is called.

    Args:
        data: The PyG graph to convert. It must have an ``edge_index``.

    Returns:
        The converted graph.

    Example:

    .. code-block:: python

        from torch_geometric.datasets import KarateClub

        graph = from_pyg(KarateClub()[0])
        # GraphData(
        #     edge_index(shape=(2, 156), int64)
        #     node_features(shape=(34, 34), float32)
        #     node_labels(shape=(34,), int64)
        #     train_mask(shape=(34,), bool))
        graph.num_nodes
        # 34
    """
    torch, Data = _import_pyg()
    from mlx_graphs.data import GraphData

    if not isinstance(data, Data):
        raise TypeError(
            f"from_pyg expects a torch_geometric.data.Data object (got {type(data)})"
        )
    if data.edge_index is None:
        raise ValueError("The PyG graph has no `edge_index`")

    def to_mlx(tensor: Any) -> mx.array:
        return mx.array(tensor.detach().cpu().numpy())

    num_nodes = data.num_nodes
    attributes: dict[str, Any] = {}
    for key, value in data.to_dict().items():
        if key in ("edge_index", "num_nodes"):
            continue
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if isinstance(value, torch.Tensor):
            if key == "y":
                # a one-entry `y` of a single-node graph is a graph label
                per_node = value.dim() > 0 and value.shape[0] == num_nodes > 1
                key = "node_labels" if per_node else "graph_labels"
            attributes[_PYG_TO_MLX_NAMES.get(key, key)] = to_mlx(value)
        elif isinstance(value, _SCALAR_TYPES):
            attributes[key] = value

    graph = GraphData(edge_index=to_mlx(data.edge_index), **attributes)
    graph.num_nodes = num_nodes
    return graph


def to_pyg(graph: "GraphData") -> "Data":
    """
    Converts a :class:`mlx_graphs.data.GraphData` to a
    :class:`torch_geometric.data.Data` object.

    The attributes are renamed as follows:

    - ``node_features`` -> ``x``
    - ``edge_features`` -> ``edge_attr``
    - ``edge_weights`` -> ``edge_weight``
    - ``node_labels`` -> ``y``, or ``graph_labels`` -> ``y`` if the graph has
      no node labels (if it has both, ``graph_labels`` keeps its name)

    Every other array attribute is converted to a tensor and kept under the
    same name, and plain Python scalars and strings are carried over as they
    are. ``num_nodes`` is always set explicitly. Integer arrays (indices and
    labels) become int64 tensors, as PyG expects.

    ``torch`` and ``torch_geometric`` are optional dependencies, imported when
    the function is called.

    Args:
        graph: The graph to convert.

    Returns:
        The converted PyG graph.

    Example:

    .. code-block:: python

        graph = GraphData(
            edge_index=mx.array([[0, 1], [1, 2]]),
            node_features=mx.ones((3, 4)),
            node_labels=mx.array([0, 1, 0]),
        )
        to_pyg(graph)
        # Data(edge_index=[2, 2], x=[3, 4], y=[3], num_nodes=3)
    """
    torch, Data = _import_pyg()

    def to_torch(array: mx.array) -> Any:
        array_np = np.array(array)
        if np.issubdtype(array_np.dtype, np.integer):
            array_np = array_np.astype(np.int64)
        return torch.from_numpy(array_np)

    num_nodes = graph.num_nodes
    if graph.edge_index.size > 0:
        num_nodes = max(num_nodes, int(graph.edge_index.max().item()) + 1)

    attributes = graph.to_dict()
    label_key = "node_labels" if "node_labels" in attributes else "graph_labels"

    data = Data()
    for key, value in attributes.items():
        if key.startswith("_"):
            continue
        if key == label_key:
            key = "y"
        key = _MLX_TO_PYG_NAMES.get(key, key)
        if isinstance(value, mx.array):
            data[key] = to_torch(value)
        elif isinstance(value, _SCALAR_TYPES):
            data[key] = value
    data.num_nodes = num_nodes
    return data
