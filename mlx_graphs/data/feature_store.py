"""PyG-style feature stores: node/edge attributes stored outside a
:class:`~mlx_graphs.data.GraphData`, gathered row by row for mini-batches.

A :class:`FeatureStore` holds named arrays identified by a
:class:`TensorAttr` ``(group_name, attr_name, index)``. Loaders such as
:class:`~mlx_graphs.loaders.NeighborLoader` take a
``(feature_store, graph_store)`` pair instead of a graph and only gather the
rows of the sampled nodes, so the features never need to be loaded as a
whole. :class:`MemmapFeatureStore` keeps them in ``.npy`` files on disk,
which allows feature matrices larger than the memory.
"""

import os
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, Union

import mlx.core as mx
import numpy as np

IndexType = Union[None, slice, int, mx.array, np.ndarray, list, tuple]

_UNSET = object()


@dataclass
class TensorAttr:
    """Identifies an array of a :class:`FeatureStore`.

    Args:
        group_name: Group of the array, e.g. a node type. ``None`` for
            homogeneous graphs.
        attr_name: Name of the array, e.g. ``"x"`` (or ``"node_features"``),
            ``"y"``, ``"time"``.
        index: Rows to read or write: ``None`` (all), an integer, a slice,
            or an array of row indices.
    """

    group_name: Optional[str] = None
    attr_name: Optional[str] = None
    index: IndexType = None

    @property
    def key(self) -> tuple[Optional[str], Optional[str]]:
        """``(group_name, attr_name)``, the key of the array."""
        return (self.group_name, self.attr_name)

    @classmethod
    def cast(cls, *args: Any, **kwargs: Any) -> "TensorAttr":
        """A :class:`TensorAttr` from another one, a tuple
        ``(group_name, attr_name[, index])`` or keyword arguments."""
        if len(args) == 1 and not kwargs:
            value = args[0]
            if isinstance(value, TensorAttr):
                return value
            if isinstance(value, tuple):
                return cls(*value)
            if isinstance(value, dict):
                return cls(**value)
        return cls(*args, **kwargs)


def _host_index(index: IndexType) -> Union[None, slice, int, np.ndarray]:
    if index is None or isinstance(index, (slice, int, np.integer)):
        return index
    out = np.asarray(index)
    if out.dtype == np.bool_:
        return np.nonzero(out)[0]
    return out.astype(np.int64, copy=False)


class FeatureStore(ABC):
    """Abstract key-value store of arrays, following PyG's ``FeatureStore``.

    Subclasses implement :meth:`_put_tensor`, :meth:`_get_tensor`,
    :meth:`_remove_tensor`, :meth:`_get_tensor_size` and
    :meth:`get_all_tensor_attrs`. Arrays are passed and returned as
    ``mx.array`` (``np.ndarray`` is also accepted when putting).

    Attributes can be given as a :class:`TensorAttr`, as positional
    arguments ``(group_name, attr_name, index)`` or as keywords, and the
    store can be indexed like a dict:

    .. code-block:: python

        store.put_tensor(x, attr_name="x")
        store["x"] = x  # same, homogeneous graphs (group_name=None)
        rows = store.get_tensor(attr_name="x", index=mx.array([0, 5]))
    """

    @abstractmethod
    def _put_tensor(
        self, tensor: Union[mx.array, np.ndarray], attr: TensorAttr
    ) -> bool:
        """Stores ``tensor`` (all rows if ``attr.index`` is ``None``, else
        the rows ``attr.index`` of an existing array)."""

    @abstractmethod
    def _get_tensor(self, attr: TensorAttr) -> Optional[np.ndarray]:
        """The rows ``attr.index`` of the array, as a host array (``None``
        if there is no such array)."""

    @abstractmethod
    def _remove_tensor(self, attr: TensorAttr) -> bool:
        """Removes the array; returns whether it existed."""

    @abstractmethod
    def _get_tensor_size(self, attr: TensorAttr) -> Optional[tuple[int, ...]]:
        """Shape of the full array (``None`` if there is no such array)."""

    @abstractmethod
    def get_all_tensor_attrs(self) -> list[TensorAttr]:
        """The attributes (without index) of all the stored arrays."""

    # Public API ##############################################################

    def put_tensor(self, tensor: Union[mx.array, np.ndarray], *args, **kwargs) -> bool:
        """Adds an array (or overwrites rows of an existing one when an
        ``index`` is given). Returns whether it succeeded."""
        return self._put_tensor(tensor, TensorAttr.cast(*args, **kwargs))

    def get_tensor(self, *args, **kwargs) -> mx.array:
        """Returns the rows ``index`` (all rows if ``None``) of an array.

        Raises:
            KeyError: If the array does not exist.
        """
        attr = TensorAttr.cast(*args, **kwargs)
        out = self._get_tensor(attr)
        if out is None:
            raise KeyError(f"'{attr.key}' is not in the feature store")
        return mx.array(out)

    def multi_get_tensor(self, attrs: list[TensorAttr]) -> list[mx.array]:
        """:meth:`get_tensor` for several attributes."""
        return [self.get_tensor(attr) for attr in attrs]

    def remove_tensor(self, *args, **kwargs) -> bool:
        """Removes an array. Returns whether it existed."""
        return self._remove_tensor(TensorAttr.cast(*args, **kwargs))

    def update_tensor(
        self, tensor: Union[mx.array, np.ndarray], *args, **kwargs
    ) -> bool:
        """Replaces an existing array (or its rows ``index``)."""
        attr = TensorAttr.cast(*args, **kwargs)
        if self._get_tensor_size(attr) is None:
            raise KeyError(f"'{attr.key}' is not in the feature store")
        if attr.index is None:
            self._remove_tensor(attr)
        return self._put_tensor(tensor, attr)

    def get_tensor_size(self, *args, **kwargs) -> Optional[tuple[int, ...]]:
        """Shape of the full array, ``None`` if it does not exist."""
        return self._get_tensor_size(TensorAttr.cast(*args, **kwargs))

    # Python built-ins ########################################################

    @staticmethod
    def _key_attr(key: Any) -> TensorAttr:
        if isinstance(key, str):
            return TensorAttr(None, key)
        return TensorAttr.cast(key)

    def __setitem__(self, key: Any, value: Union[mx.array, np.ndarray]):
        self.put_tensor(value, self._key_attr(key))

    def __getitem__(self, key: Any) -> mx.array:
        return self.get_tensor(self._key_attr(key))

    def __delitem__(self, key: Any):
        if not self.remove_tensor(self._key_attr(key)):
            raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        return self.get_tensor_size(self._key_attr(key)) is not None

    def __len__(self) -> int:
        return len(self.get_all_tensor_attrs())

    def __repr__(self) -> str:
        items = ", ".join(
            f"{a.attr_name if a.group_name is None else a.key}: "
            f"{self._get_tensor_size(a)}"
            for a in self.get_all_tensor_attrs()
        )
        return f"{type(self).__name__}({items})"


class InMemoryFeatureStore(FeatureStore):
    """A :class:`FeatureStore` keeping its arrays in memory (as ``mx.array``).

    Example:

    .. code-block:: python

        from mlx_graphs.data import InMemoryFeatureStore

        feature_store = InMemoryFeatureStore.from_data(data)
        feature_store.get_tensor(attr_name="x", index=mx.array([0, 3]))
    """

    def __init__(self):
        self._store: dict[tuple, mx.array] = {}

    @classmethod
    def from_data(
        cls, data: Any, attrs: Optional[list[str]] = None
    ) -> "InMemoryFeatureStore":
        """A store holding the array attributes of a
        :class:`~mlx_graphs.data.GraphData` (all but ``edge_index`` unless
        ``attrs`` lists the ones to keep)."""
        store = cls()
        for name, value in _graph_arrays(data, attrs).items():
            store.put_tensor(value, attr_name=name)
        return store

    def _put_tensor(self, tensor, attr: TensorAttr) -> bool:
        index = _host_index(attr.index)
        tensor = (
            tensor if isinstance(tensor, mx.array) else mx.array(np.asarray(tensor))
        )
        if index is None:
            self._store[attr.key] = tensor
            return True
        if attr.key not in self._store:
            raise KeyError(
                f"Cannot write rows of '{attr.key}', which is not in the store"
            )
        value = self._store[attr.key]
        if isinstance(index, np.ndarray):
            index = mx.array(index)
        value[index] = tensor
        self._store[attr.key] = value
        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[Any]:
        value = self._store.get(attr.key)
        if value is None:
            return None
        index = _host_index(attr.index)
        if index is None:
            return value
        if isinstance(index, np.ndarray):
            index = mx.array(index)
        return value[index]

    def get_tensor(self, *args, **kwargs) -> mx.array:
        # No host round trip: the arrays already are `mx.array`.
        attr = TensorAttr.cast(*args, **kwargs)
        out = self._get_tensor(attr)
        if out is None:
            raise KeyError(f"'{attr.key}' is not in the feature store")
        return out

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        return self._store.pop(attr.key, None) is not None

    def _get_tensor_size(self, attr: TensorAttr) -> Optional[tuple[int, ...]]:
        value = self._store.get(attr.key)
        return None if value is None else tuple(value.shape)

    def get_all_tensor_attrs(self) -> list[TensorAttr]:
        return [TensorAttr(g, a) for g, a in self._store]


def _safe_name(name: Optional[str]) -> str:
    if name is None:
        return "_"
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name) or name in (".", ".."):
        raise ValueError(
            f"Invalid name '{name}' for a MemmapFeatureStore (letters, digits, "
            "'_', '-' and '.' only)"
        )
    return name


class MemmapFeatureStore(FeatureStore):
    """A :class:`FeatureStore` keeping every array in a ``.npy`` file on
    disk.

    Only the rows that are asked for are read, so the arrays (e.g. node
    features of a large graph) can be larger than the memory. Rows given as
    an index array are read with positional reads (``os.preadv``), in
    increasing order and spread over ``num_threads`` threads: nearby rows
    (less than 64 KB apart) are read with one request, isolated rows one by
    one. The process only holds the requested rows (plus at most 16 MB per
    thread for gaps), without the read-ahead and resident pages of a memory
    mapping (the OS may still keep the file in its cache). Other
    reads (all rows, slices) and writes go through ``np.memmap``.

    Arrays are stored as ``<root>/<group_name or "_">/<attr_name>.npy``.
    Large arrays can be written in chunks without holding them in memory,
    with :meth:`create_tensor`:

    .. code-block:: python

        from mlx_graphs.data import MemmapFeatureStore

        store = MemmapFeatureStore("features/")
        x = store.create_tensor("x", shape=(num_nodes, 512), dtype="float32")
        for start in range(0, num_nodes, 100_000):
            x[start : start + 100_000] = compute_features(start)
        x.flush()
        del x

        store.get_tensor(attr_name="x", index=mx.array([3, 1, 4]))

    Args:
        root: Directory of the store (created if needed). Arrays already in
            it are available.
        num_threads: Number of threads reading rows in parallel (more
            requests in flight help SSDs). Default: ``8``.
    """

    def __init__(self, root: str, num_threads: int = 8):
        self.root = os.path.expanduser(root)
        os.makedirs(self.root, exist_ok=True)
        self.num_threads = max(1, int(num_threads))
        self._meta: dict[tuple, tuple[str, np.dtype, tuple, int]] = {}
        self._pool: Optional[ThreadPoolExecutor] = None
        self._scan()

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_pool"] = None  # thread pools cannot be pickled
        return state

    @classmethod
    def from_data(
        cls, data: Any, root: str, attrs: Optional[list[str]] = None
    ) -> "MemmapFeatureStore":
        """Writes the array attributes of a
        :class:`~mlx_graphs.data.GraphData` (all but ``edge_index`` unless
        ``attrs`` lists the ones to keep) to a new store in ``root``."""
        store = cls(root)
        for name, value in _graph_arrays(data, attrs).items():
            store.put_tensor(value, attr_name=name)
        return store

    def _path(self, attr: TensorAttr) -> str:
        return os.path.join(
            self.root, _safe_name(attr.group_name), f"{_safe_name(attr.attr_name)}.npy"
        )

    def _scan(self):
        for entry in sorted(os.listdir(self.root)):
            folder = os.path.join(self.root, entry)
            if not os.path.isdir(folder):
                continue
            group = None if entry == "_" else entry
            for file in sorted(os.listdir(folder)):
                if file.endswith(".npy"):
                    self._register(TensorAttr(group, file[:-4]))

    def _register(self, attr: TensorAttr):
        path = self._path(attr)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(array, np.memmap):  # empty arrays are not mapped
            self._meta[attr.key] = (path, array.dtype, tuple(array.shape), -1)
            return
        if not array.flags.c_contiguous and array.ndim > 1:
            raise ValueError(f"{path}: Fortran-ordered arrays are not supported")
        self._meta[attr.key] = (path, array.dtype, tuple(array.shape), array.offset)
        del array

    def create_tensor(
        self,
        attr_name: str,
        shape: tuple[int, ...],
        dtype: Any = "float32",
        group_name: Optional[str] = None,
    ) -> np.memmap:
        """Creates an array on disk and returns it as a writable
        ``np.memmap``, to be filled in chunks (call ``flush()`` and drop it
        when done)."""
        attr = TensorAttr(group_name, attr_name)
        path = self._path(attr)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.dtype(dtype), shape=tuple(shape)
        )
        self._register(attr)
        return out

    def _open(self, key: tuple, mode: str = "r") -> np.ndarray:
        path, dtype, shape, offset = self._meta[key]
        if offset < 0:
            return np.empty(shape, dtype=dtype)
        return np.memmap(path, dtype=dtype, mode=mode, offset=offset, shape=shape)

    def _put_tensor(self, tensor, attr: TensorAttr) -> bool:
        index = _host_index(attr.index)
        tensor = np.asarray(tensor)
        if index is None:
            if tensor.size == 0:  # cannot be memory mapped
                path = self._path(attr)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.save(path, tensor)
                self._register(attr)
                return True
            out = self.create_tensor(
                attr.attr_name, tensor.shape, tensor.dtype, attr.group_name
            )
            out[...] = tensor
            out.flush()
            del out
            return True
        if attr.key not in self._meta:
            raise KeyError(
                f"Cannot write rows of '{attr.key}', which is not in the store"
            )
        out = self._open(attr.key, mode="r+")
        out[index] = tensor
        out.flush()
        del out
        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[np.ndarray]:
        if attr.key not in self._meta:
            return None
        index = _host_index(attr.index)
        path, dtype, shape, offset = self._meta[attr.key]
        if isinstance(index, np.ndarray) and offset >= 0 and hasattr(os, "preadv"):
            return self._read_rows(path, dtype, shape, offset, index)
        array = self._open(attr.key)
        out = np.array(array if index is None else array[index])
        del array  # unmap
        return out

    # Rows closer than this (in bytes) are read with one request, the gap
    # included; requests are capped to `_MAX_READ` bytes.
    _MAX_GAP = 1 << 16
    _MAX_READ = 1 << 24

    def _read_rows(
        self,
        path: str,
        dtype: np.dtype,
        shape: tuple,
        offset: int,
        index: np.ndarray,
    ) -> np.ndarray:
        """Rows ``index`` of a C-ordered array stored at ``offset`` in
        ``path``, read with positional reads: rows are sorted, and rows
        less than ``_MAX_GAP`` bytes apart are read with one request (dense
        requests then read large blocks, sparse ones one row each)."""
        num_rows = shape[0]
        flat = index.reshape(-1)
        if flat.size and (flat.min() < -num_rows or flat.max() >= num_rows):
            raise IndexError(f"Row index out of range for {num_rows} rows")
        flat = np.where(flat < 0, flat + num_rows, flat)
        rows, inverse = np.unique(flat, return_inverse=True)
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        out = np.empty((rows.size, *shape[1:]), dtype=dtype)
        if rows.size and row_bytes:
            # Blocks of rows read with one request each.
            max_gap = max(1, self._MAX_GAP // row_bytes)
            max_span = max(1, self._MAX_READ // row_bytes)
            breaks = np.nonzero(np.diff(rows) > max_gap)[0] + 1
            first = np.concatenate([[0], breaks])
            last = np.append(breaks, rows.size)  # exclusive
            out_bytes = out.reshape(-1).view(np.uint8)
            fd = os.open(path, os.O_RDONLY)

            def pread_into(view: memoryview, start: int):
                done = 0
                while done < len(view):
                    n = os.preadv(fd, [view[done:]], start + done)
                    if n <= 0:
                        raise OSError(f"Unexpected end of file in {path}")
                    done += n

            def read(blocks: np.ndarray):
                for i, j in zip(first[blocks].tolist(), last[blocks].tolist()):
                    while i < j:
                        lo = int(rows[i])
                        # rows of this request: within `max_span` rows of `lo`
                        k = i + int(np.searchsorted(rows[i:j], lo + max_span))
                        hi = int(rows[k - 1]) + 1
                        if hi - lo == k - i:  # consecutive rows: read in place
                            pread_into(
                                memoryview(out_bytes[i * row_bytes : k * row_bytes]),
                                offset + lo * row_bytes,
                            )
                        else:
                            block = np.empty((hi - lo) * row_bytes, dtype=np.uint8)
                            pread_into(memoryview(block), offset + lo * row_bytes)
                            block = block.reshape(hi - lo, row_bytes)
                            out_bytes.reshape(-1, row_bytes)[i:k] = block[
                                rows[i:k] - lo
                            ]
                        i = k

            try:
                parts = np.array_split(
                    np.arange(first.size),
                    min(self.num_threads, max(1, first.size // 64)),
                )
                if len(parts) == 1:
                    read(parts[0])
                else:
                    if self._pool is None:
                        self._pool = ThreadPoolExecutor(self.num_threads)
                    list(self._pool.map(read, parts))
            finally:
                os.close(fd)
        return out[inverse.reshape(index.shape)]

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        meta = self._meta.pop(attr.key, None)
        if meta is None:
            return False
        os.remove(meta[0])
        return True

    def _get_tensor_size(self, attr: TensorAttr) -> Optional[tuple[int, ...]]:
        meta = self._meta.get(attr.key)
        return None if meta is None else meta[2]

    def get_all_tensor_attrs(self) -> list[TensorAttr]:
        return [TensorAttr(g, a) for g, a in self._meta]


def _graph_arrays(data: Any, attrs: Optional[list[str]]) -> dict[str, Any]:
    """Array attributes of a graph (by their mlx-graphs names)."""
    out = {}
    for name, value in vars(data).items():
        if name.startswith("_") or name == "edge_index":
            continue
        if attrs is not None and name not in attrs:
            continue
        if isinstance(value, (mx.array, np.ndarray)):
            out[name] = value
    return out
