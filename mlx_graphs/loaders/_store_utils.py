"""Helpers for loaders reading from a ``(FeatureStore, GraphStore)`` pair."""

from typing import Any

import numpy as np

from mlx_graphs.data.feature_store import FeatureStore, TensorAttr
from mlx_graphs.data.graph_store import GraphStore

# Names of feature store arrays that map to `GraphData` attributes.
STORE_ALIASES = {"x": "node_features", "y": "node_labels", "edge_attr": "edge_features"}


def is_store_pair(data: Any) -> bool:
    """Whether ``data`` is a ``(FeatureStore, GraphStore)`` pair."""
    return (
        isinstance(data, (tuple, list))
        and len(data) == 2
        and isinstance(data[0], FeatureStore)
        and isinstance(data[1], GraphStore)
    )


def gather_from_store(
    feature_store: FeatureStore,
    num_nodes: int,
    num_edges: int,
    node_index: np.ndarray,
    edge_index: np.ndarray,
) -> dict[str, Any]:
    """Rows of the sampled nodes (and edges) of every homogeneous array of
    the store, keyed by their :class:`~mlx_graphs.data.GraphData` name.

    Arrays with ``num_nodes`` rows are node-level (indexed by
    ``node_index``), arrays with ``num_edges`` rows edge-level (indexed by
    ``edge_index``); with as many nodes as edges, arrays with ``"edge"`` in
    their name are edge-level. Other arrays are skipped.
    """
    out: dict[str, Any] = {}
    for attr in feature_store.get_all_tensor_attrs():
        if attr.group_name is not None:
            continue
        size = feature_store.get_tensor_size(attr)
        if not size:
            continue
        name = str(attr.attr_name)
        if size[0] == num_nodes and (num_nodes != num_edges or "edge" not in name):
            index = node_index
        elif size[0] == num_edges:
            index = edge_index
        else:
            continue
        out[STORE_ALIASES.get(name, name)] = feature_store.get_tensor(
            TensorAttr(None, name, index)
        )
    return out


def store_time(
    feature_store: FeatureStore, time_attr: str, num_nodes: int, num_edges: int
) -> tuple[np.ndarray, bool]:
    """The ``time_attr`` array of the store and whether it is node-level
    (``False``: edge-level)."""
    size = feature_store.get_tensor_size(TensorAttr(None, time_attr))
    if size is None:
        raise ValueError(f"'{time_attr}' is not in the feature store.")
    time = np.asarray(feature_store.get_tensor(TensorAttr(None, time_attr)))
    if size[0] == num_nodes:
        return time, True
    if size[0] == num_edges:
        return time, False
    raise ValueError(
        f"The time attribute '{time_attr}' has {size[0]} rows, neither the "
        f"number of nodes ({num_nodes}) nor of edges ({num_edges})."
    )
