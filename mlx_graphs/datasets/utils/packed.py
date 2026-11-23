"""Storage of many small graphs as views of a few large arrays.

Every ``mx.array`` created from host data gets its own device buffer, and
Metal limits the number of live buffers (about 500,000). Datasets with more
than ~100,000 small graphs (QM9, ZINC, the large MoleculeNet datasets) would
exceed it with one array per attribute and graph, so their graphs are built
as slices (views sharing the buffer) of one array per attribute, and saved as
these concatenated arrays.
"""

import os
import pickle  # nosec B403 - only loads files written by save_packed
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData

_FILE_NAME = "graphs_packed.pkl"


def _cat_axis(key: str) -> int:
    return 1 if "index" in key else 0


def graphs_from_records(records: list[dict[str, Any]]) -> list[GraphData]:
    """Builds graphs from per-graph dicts of attributes (NumPy arrays or any
    other values), whose arrays are views of one ``mx.array`` per attribute.

    Every record must have the same keys, and the arrays of an attribute the
    same number of dimensions and dtype. The arrays are concatenated along
    axis 1 for the attributes whose name contains ``"index"``, along axis 0
    otherwise (as when batching), and 0-dimensional arrays are stacked.
    """
    if len(records) == 0:
        return []
    columns = {}
    for key in records[0]:
        values = [record[key] for record in records]
        if all(isinstance(v, np.ndarray) for v in values):
            scalar = all(v.ndim == 0 for v in values)
            axis = 0 if scalar else _cat_axis(key)
            arrays = [v.reshape(1) for v in values] if scalar else values
            sizes = np.array([a.shape[axis] for a in arrays], dtype=np.int64)
            columns[key] = (
                "array",
                np.concatenate(arrays, axis=axis),
                sizes,
                axis,
                scalar,
            )
        else:
            columns[key] = ("list", values)
    return _graphs_from_columns(columns, len(records))


def _graphs_from_columns(columns: dict, num_graphs: int) -> list[GraphData]:
    per_key = {}
    for key, column in columns.items():
        if column[0] == "list":
            per_key[key] = column[1]
            continue
        _, data, sizes, axis, scalar = column
        big = mx.array(data)
        offsets = np.concatenate([[0], np.cumsum(sizes)]).tolist()
        slices = []
        for start, stop in zip(offsets[:-1], offsets[1:]):
            view = big[:, start:stop] if axis == 1 else big[start:stop]
            slices.append(view.reshape(()) if scalar else view)
        per_key[key] = slices
    graphs = []
    keys = list(per_key)
    for i in range(num_graphs):
        graph = GraphData(edge_index=None)
        for key in keys:
            setattr(graph, key, per_key[key][i])
        graphs.append(graph)
    return graphs


def _pack(graphs: list[GraphData]) -> Optional[dict]:
    """The columns of :func:`_graphs_from_columns`, or ``None`` if the graphs
    do not have the same attributes."""
    keys = list(vars(graphs[0]))
    columns = {}
    for key in keys:
        values = []
        for graph in graphs:
            attributes = vars(graph)
            if key not in attributes or len(attributes) != len(keys):
                return None
            values.append(attributes[key])
        if all(isinstance(v, mx.array) for v in values):
            arrays = [np.array(v) for v in values]
            scalar = all(a.ndim == 0 for a in arrays)
            axis = 0 if scalar else _cat_axis(key)
            if scalar:
                arrays = [a.reshape(1) for a in arrays]
            ndims = {a.ndim for a in arrays}
            if len(ndims) != 1 or axis >= ndims.pop():
                return None
            shapes = {a.shape[:axis] + a.shape[axis + 1 :] for a in arrays}
            if len(shapes) != 1 or len({a.dtype for a in arrays}) != 1:
                return None
            sizes = np.array([a.shape[axis] for a in arrays], dtype=np.int64)
            columns[key] = (
                "array",
                np.concatenate(arrays, axis=axis),
                sizes,
                axis,
                scalar,
            )
        elif any(isinstance(v, mx.array) for v in values):
            return None
        else:
            columns[key] = ("list", values)
    return columns


class PackedGraphsMixin:
    """Mixin of :class:`~mlx_graphs.datasets.Dataset` saving and loading the
    graphs as a few concatenated arrays (see :func:`graphs_from_records`),
    instead of one array per attribute and graph. Graphs that cannot be
    packed (e.g. with different attributes after a ``pre_transform``) are
    saved as usual."""

    graphs: list
    processed_path: str

    def save(self):
        columns = _pack(self.graphs) if len(self.graphs) > 0 else None
        if columns is None:
            return super().save()  # type: ignore[misc]
        with open(os.path.join(self.processed_path, _FILE_NAME), "wb") as f:
            pickle.dump({"num_graphs": len(self.graphs), "columns": columns}, f)

    def load(self):
        path = os.path.join(self.processed_path, _FILE_NAME)
        if not os.path.exists(path):
            return super().load()  # type: ignore[misc]
        with open(path, "rb") as f:
            obj = pickle.load(f)  # nosec B301 - own pickle from save()
        self.graphs = _graphs_from_columns(obj["columns"], obj["num_graphs"])
