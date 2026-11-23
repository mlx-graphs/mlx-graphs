"""PyG-style graph stores: the edges of a graph, stored outside a
:class:`~mlx_graphs.data.GraphData`.

A :class:`GraphStore` holds the ``edge_index`` of every edge type (``None``
for homogeneous graphs), together with the number of nodes, and converts it
to the CSC layout used by the neighbor samplers. Paired with a
:class:`~mlx_graphs.data.FeatureStore`, it can be passed to
:class:`~mlx_graphs.loaders.NeighborLoader` in place of a graph.
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import mlx.core as mx
import numpy as np

EdgeLayout = Literal["coo", "csc", "csr"]


@dataclass
class EdgeAttr:
    """Identifies the edges of a :class:`GraphStore`.

    Args:
        edge_type: Type of the edges, ``None`` for homogeneous graphs.
        layout: ``"coo"`` (``(row, col)``), ``"csc"`` (``(row, colptr)``)
            or ``"csr"`` (``(rowptr, col)``), as in PyG.
        is_sorted: Whether COO edges are sorted by destination node.
        size: ``(num_src_nodes, num_dst_nodes)``.
    """

    edge_type: Any = None
    layout: EdgeLayout = "coo"
    is_sorted: bool = False
    size: Optional[tuple[int, int]] = None

    def __post_init__(self):
        if self.layout not in ("coo", "csc", "csr"):
            raise ValueError(
                f"`layout` must be 'coo', 'csc' or 'csr' (got '{self.layout}')"
            )
        if self.layout == "csc":
            self.is_sorted = True
        if self.size is not None:
            self.size = (int(self.size[0]), int(self.size[1]))


def _host(array: Union[mx.array, np.ndarray]) -> np.ndarray:
    return np.asarray(array).astype(np.int64, copy=False)


def _ptr(index: np.ndarray, size: int) -> np.ndarray:
    ptr = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(np.bincount(index, minlength=size), out=ptr[1:])
    return ptr


class GraphStore(ABC):
    """Abstract store of the edges of a graph, following PyG's
    ``GraphStore``.

    Subclasses implement :meth:`_put_edge_index`, :meth:`_get_edge_index`,
    :meth:`_remove_edge_index` and :meth:`get_all_edge_attrs`; edges are
    exchanged as pairs of ``np.ndarray`` (``mx.array`` is accepted when
    putting).

    Example:

    .. code-block:: python

        from mlx_graphs.data import InMemoryGraphStore

        graph_store = InMemoryGraphStore()
        graph_store.put_edge_index(
            data.edge_index, layout="coo", size=(data.num_nodes, data.num_nodes)
        )
        colptr, row, perm = graph_store.csc()
    """

    def __init__(self):
        self._csc_cache: dict[
            Any, tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        ] = {}

    @abstractmethod
    def _put_edge_index(
        self, edge_index: tuple[np.ndarray, np.ndarray], attr: EdgeAttr
    ) -> bool:
        """Stores the edges of ``attr``."""

    @abstractmethod
    def _get_edge_index(
        self, attr: EdgeAttr
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """The edges stored for ``attr.edge_type`` in layout ``attr.layout``."""

    @abstractmethod
    def _remove_edge_index(self, attr: EdgeAttr) -> bool:
        """Removes the edges of ``attr``."""

    @abstractmethod
    def get_all_edge_attrs(self) -> list[EdgeAttr]:
        """The attributes of all the stored edges."""

    # Public API ##############################################################

    def put_edge_index(
        self,
        edge_index: Union[mx.array, np.ndarray, tuple],
        edge_type: Any = None,
        layout: EdgeLayout = "coo",
        size: Optional[tuple[int, int]] = None,
        is_sorted: bool = False,
    ) -> bool:
        """Adds edges, as a ``[2, num_edges]`` array or a pair of arrays
        (``(row, col)`` for COO, ``(row, colptr)`` for CSC, ``(rowptr, col)``
        for CSR).

        Args:
            edge_index: The edges.
            edge_type: Their type (``None`` for homogeneous graphs).
            layout: ``"coo"``, ``"csc"`` or ``"csr"``.
            size: ``(num_src_nodes, num_dst_nodes)``. Inferred from the
                largest node ids if not given (isolated nodes with larger
                ids are then missed).
            is_sorted: Whether COO edges are sorted by destination.
        """
        first, second = (_host(edge_index[0]), _host(edge_index[1]))
        if size is None:
            if layout == "coo":
                n = int(max(first.max(initial=-1), second.max(initial=-1))) + 1
                size = (n, n)
            elif layout == "csc":
                size = (int(first.max(initial=-1)) + 1, second.size - 1)
            else:
                size = (first.size - 1, int(second.max(initial=-1)) + 1)
        attr = EdgeAttr(edge_type, layout, is_sorted, size)
        self._csc_cache.pop(edge_type, None)
        return self._put_edge_index((first, second), attr)

    def get_edge_index(
        self, edge_type: Any = None, layout: Optional[EdgeLayout] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The edges of ``edge_type`` as stored (``layout=None``) or
        converted to ``layout``."""
        attr = self._get_attr(edge_type)
        if layout is None or layout == attr.layout:
            out = self._get_edge_index(attr)
            assert out is not None
            return out
        if layout == "coo":
            return self.coo(edge_type)
        if layout == "csc":
            colptr, row, _ = self.csc(edge_type)
            return row, colptr
        rowptr, col, _ = self.csr(edge_type)
        return rowptr, col

    def remove_edge_index(self, edge_type: Any = None) -> bool:
        """Removes the edges of ``edge_type``."""
        self._csc_cache.pop(edge_type, None)
        try:
            attr = self._get_attr(edge_type)
        except KeyError:
            return False
        return self._remove_edge_index(attr)

    def num_nodes(self, edge_type: Any = None) -> tuple[int, int]:
        """``(num_src_nodes, num_dst_nodes)`` of ``edge_type``."""
        size = self._get_attr(edge_type).size
        assert size is not None
        return size

    def num_edges(self, edge_type: Any = None) -> int:
        """Number of edges of ``edge_type``."""
        attr = self._get_attr(edge_type)
        first, second = self._get_edge_index(attr)  # type: ignore
        return int(
            first.size
            if attr.layout == "coo"
            else (first.size if attr.layout == "csc" else second.size)
        )

    def coo(self, edge_type: Any = None) -> tuple[np.ndarray, np.ndarray]:
        """``(row, col)`` of ``edge_type``, in the original edge order for
        edges stored as COO."""
        attr = self._get_attr(edge_type)
        first, second = self._get_edge_index(attr)  # type: ignore
        if attr.layout == "coo":
            return first, second
        if attr.layout == "csc":
            col = np.repeat(np.arange(second.size - 1), np.diff(second))
            return first, col
        row = np.repeat(np.arange(first.size - 1), np.diff(first))
        return row, second

    def csc(
        self, edge_type: Any = None
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """``(colptr, row, perm)`` of ``edge_type``: the in-edges of every
        node, and the position in the stored COO edges of every CSC entry
        (``None`` if the stored edges need no permutation). Cached."""
        if edge_type in self._csc_cache:
            return self._csc_cache[edge_type]
        attr = self._get_attr(edge_type)
        first, second = self._get_edge_index(attr)  # type: ignore
        num_dst = attr.size[1]  # type: ignore
        if attr.layout == "csc":
            out = (second, first, None)
        else:
            row, col = self.coo(edge_type)
            if attr.layout == "coo" and attr.is_sorted:
                perm = None
            else:
                perm = np.argsort(col, kind="stable")
                row, col = row[perm], col[perm]
            out = (_ptr(col, num_dst), row, perm)
        self._csc_cache[edge_type] = out
        return out

    def csr(
        self, edge_type: Any = None
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """``(rowptr, col, perm)`` of ``edge_type`` (out-edges of every
        node)."""
        attr = self._get_attr(edge_type)
        if attr.layout == "csr":
            rowptr, col = self._get_edge_index(attr)  # type: ignore
            return rowptr, col, None
        row, col = self.coo(edge_type)
        perm = np.argsort(row, kind="stable")
        return _ptr(row[perm], attr.size[0]), col[perm], perm  # type: ignore

    def _get_attr(self, edge_type: Any) -> EdgeAttr:
        for attr in self.get_all_edge_attrs():
            if attr.edge_type == edge_type:
                return attr
        raise KeyError(f"No edges of type {edge_type} in the graph store")

    def __repr__(self) -> str:
        items = ", ".join(
            f"{a.edge_type}: {a.layout}, size={a.size}"
            for a in self.get_all_edge_attrs()
        )
        return f"{type(self).__name__}({items})"


class InMemoryGraphStore(GraphStore):
    """A :class:`GraphStore` keeping its edges in memory (as ``np.ndarray``).

    Example:

    .. code-block:: python

        from mlx_graphs.data import InMemoryGraphStore

        graph_store = InMemoryGraphStore.from_data(data)
    """

    def __init__(self):
        super().__init__()
        self._edges: dict[Any, tuple[EdgeAttr, tuple[np.ndarray, np.ndarray]]] = {}

    @classmethod
    def from_data(cls, data: Any) -> "InMemoryGraphStore":
        """A store holding the edges of a :class:`~mlx_graphs.data.GraphData`."""
        store = cls()
        n = data.num_nodes
        store.put_edge_index(data.edge_index, layout="coo", size=(n, n))
        return store

    def _put_edge_index(self, edge_index, attr: EdgeAttr) -> bool:
        self._edges[attr.edge_type] = (attr, edge_index)
        return True

    def _get_edge_index(self, attr: EdgeAttr):
        entry = self._edges.get(attr.edge_type)
        return None if entry is None else entry[1]

    def _remove_edge_index(self, attr: EdgeAttr) -> bool:
        return self._edges.pop(attr.edge_type, None) is not None

    def get_all_edge_attrs(self) -> list[EdgeAttr]:
        return [attr for attr, _ in self._edges.values()]


class MemmapGraphStore(GraphStore):
    """A :class:`GraphStore` keeping its edges in ``.npy`` files, read
    through memory mapping, so that a graph converted once (e.g. to CSC)
    can be reopened instantly, without parsing or sorting its edges again.

    Only homogeneous graphs (``edge_type=None``) and string edge types are
    supported. Files are ``<root>/<edge_type or "_">.{0,1}.npy`` plus a
    ``<root>/edges.json`` index.

    Example:

    .. code-block:: python

        from mlx_graphs.data import MemmapGraphStore

        store = MemmapGraphStore("graph/")
        colptr, row, perm = InMemoryGraphStore.from_data(data).csc()
        store.put_edge_index((row, colptr), layout="csc", size=(n, n))

        # later, in another process
        store = MemmapGraphStore("graph/")
        colptr, row, _ = store.csc()

    Args:
        root: Directory of the store (created if needed).
    """

    def __init__(self, root: str):
        super().__init__()
        self.root = os.path.expanduser(root)
        os.makedirs(self.root, exist_ok=True)
        self._attrs: dict[Any, EdgeAttr] = {}
        index = os.path.join(self.root, "edges.json")
        if os.path.exists(index):
            with open(index) as f:
                for entry in json.load(f):
                    attr = EdgeAttr(**entry)
                    self._attrs[attr.edge_type] = attr

    def _write_index(self):
        entries = [
            {
                "edge_type": a.edge_type,
                "layout": a.layout,
                "is_sorted": a.is_sorted,
                "size": list(a.size) if a.size else None,
            }
            for a in self._attrs.values()
        ]
        with open(os.path.join(self.root, "edges.json"), "w") as f:
            json.dump(entries, f)

    def _files(self, edge_type: Any) -> tuple[str, str]:
        if edge_type is not None and not isinstance(edge_type, str):
            raise ValueError("MemmapGraphStore only supports string edge types")
        name = "_" if edge_type is None else edge_type
        return (
            os.path.join(self.root, f"{name}.0.npy"),
            os.path.join(self.root, f"{name}.1.npy"),
        )

    def _put_edge_index(self, edge_index, attr: EdgeAttr) -> bool:
        for path, array in zip(self._files(attr.edge_type), edge_index):
            np.save(path, np.ascontiguousarray(array))
        self._attrs[attr.edge_type] = attr
        self._write_index()
        return True

    def _get_edge_index(self, attr: EdgeAttr):
        if attr.edge_type not in self._attrs:
            return None
        first, second = self._files(attr.edge_type)
        return (np.load(first, mmap_mode="r"), np.load(second, mmap_mode="r"))

    def _remove_edge_index(self, attr: EdgeAttr) -> bool:
        if self._attrs.pop(attr.edge_type, None) is None:
            return False
        for path in self._files(attr.edge_type):
            os.remove(path)
        self._write_index()
        return True

    def get_all_edge_attrs(self) -> list[EdgeAttr]:
        return list(self._attrs.values())
