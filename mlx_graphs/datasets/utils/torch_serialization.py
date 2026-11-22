"""Reads files written by ``torch.save`` without importing PyTorch.

Several benchmark datasets (e.g. LRGB's Peptides or the benchmarking-gnns
datasets) are only distributed as ``torch.save`` files. This module reads such
files into NumPy arrays with a *restricted* unpickler: only the handful of
globals needed to rebuild tensors, NumPy arrays and SciPy COO matrices are
allowed, and any other global raises an :class:`pickle.UnpicklingError`, so a
malicious file cannot execute arbitrary code.
"""

import codecs
import collections
import io
import os
import pickle  # nosec B403 - restricted unpickler, see _RestrictedUnpickler
import struct
import zipfile
from typing import Any, BinaryIO, Union

import numpy as np

try:  # NumPy >= 2
    from numpy._core.multiarray import _reconstruct, scalar  # type: ignore
except ImportError:  # pragma: no cover - NumPy 1.x
    from numpy.core.multiarray import _reconstruct, scalar  # type: ignore

__all__ = ["load_torch_file", "SparseCOOMatrix"]

_LEGACY_MAGIC_NUMBER = 0x1950A86A20F9469CFC6C

# Typed storages of torch, by name, and their element dtype.
_STORAGE_DTYPES = {
    "DoubleStorage": np.float64,
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _StorageType:
    """Stand-in for a ``torch.<Type>Storage`` class."""

    def __init__(self, name: str):
        self.name = name
        self.dtype = np.dtype(_STORAGE_DTYPES[name])


class _Storage:
    """A flat buffer of a torch storage, filled when its bytes are read."""

    def __init__(self, dtype: np.dtype, numel: int):
        self.dtype = dtype
        self.numel = numel
        self.array = None

    def set_bytes(self, buffer: Union[bytes, np.ndarray]):
        dtype = self.dtype.newbyteorder("<")
        if isinstance(buffer, np.ndarray):
            usable = buffer.size - buffer.size % dtype.itemsize
            array = buffer[:usable].view(dtype)
        else:
            array = np.frombuffer(buffer, dtype=dtype)
        if array.size < self.numel:
            raise pickle.UnpicklingError("Truncated tensor storage")
        self.array = array[: self.numel]


class _StorageView:
    """A view of a :class:`_Storage` (legacy format ``view_metadata``)."""

    def __init__(self, storage: _Storage, offset: int, numel: int):
        self.storage = storage
        self.offset = offset
        self.numel = numel


class _PendingTensor:
    """A tensor whose storage may not have been read yet."""

    def __init__(self, storage, storage_offset, size, stride):
        if isinstance(storage, _StorageView):
            storage_offset = storage.offset + storage_offset
            storage = storage.storage
        if not isinstance(storage, _Storage):
            raise pickle.UnpicklingError("Invalid tensor storage")
        self.storage = storage
        self.storage_offset = int(storage_offset)
        self.size = tuple(int(s) for s in size)
        self.stride = tuple(int(s) for s in stride)

    def materialize(self) -> np.ndarray:
        base = self.storage.array
        if base is None:
            raise pickle.UnpicklingError("Missing tensor storage")
        if any(s < 0 for s in self.size + self.stride):
            raise pickle.UnpicklingError("Invalid tensor shape or stride")
        # Bounds check before building a strided view of the buffer.
        last = self.storage_offset + sum(
            (n - 1) * s for n, s in zip(self.size, self.stride) if n > 0
        )
        if 0 in self.size:
            return np.zeros(self.size, dtype=self.storage.dtype)
        if self.storage_offset < 0 or last >= base.size:
            raise pickle.UnpicklingError("Tensor out of the bounds of its storage")
        view = np.lib.stride_tricks.as_strided(
            base[self.storage_offset :],
            shape=self.size,
            strides=[s * base.itemsize for s in self.stride],
            writeable=False,
        )
        return np.array(view, dtype=self.storage.dtype.newbyteorder("="))


def _rebuild_tensor_v2(
    storage, storage_offset, size, stride, requires_grad=False, hooks=None, *args
):
    return _PendingTensor(storage, storage_offset, size, stride)


def _rebuild_tensor(storage, storage_offset, size, stride):
    return _PendingTensor(storage, storage_offset, size, stride)


def _rebuild_parameter(data, requires_grad=False, hooks=None, *args):
    return data


class SparseCOOMatrix:
    """Minimal stand-in for an unpickled ``scipy.sparse.coo_matrix``.

    Exposes the ``row``, ``col`` and ``data`` arrays and the ``shape`` of the
    matrix, so that SciPy is not needed to read the file.
    """

    @property
    def shape(self) -> tuple:
        return tuple(int(s) for s in self.__dict__.get("_shape", ()))

    @property
    def row(self) -> np.ndarray:
        if "row" in self.__dict__:  # SciPy < 1.13
            return self.__dict__["row"]
        return self.__dict__["coords"][0]

    @property
    def col(self) -> np.ndarray:
        if "col" in self.__dict__:
            return self.__dict__["col"]
        return self.__dict__["coords"][1]

    @property
    def data(self) -> np.ndarray:
        return self.__dict__["data"]


_ALLOWED_GLOBALS = {
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    ("torch._utils", "_rebuild_tensor"): _rebuild_tensor,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
    ("collections", "OrderedDict"): collections.OrderedDict,
    # Pickle protocol 2 (used by torch.save) stores `bytes`, e.g. the buffer of
    # a NumPy array, as `_codecs.encode(str, "latin1")`. Also allowed by
    # torch.load(weights_only=True).
    ("_codecs", "encode"): codecs.encode,
    ("torch", "Size"): tuple,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
    ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
    ("numpy.core.multiarray", "scalar"): scalar,
    ("numpy._core.multiarray", "scalar"): scalar,
    ("scipy.sparse.coo", "coo_matrix"): SparseCOOMatrix,
    ("scipy.sparse._coo", "coo_matrix"): SparseCOOMatrix,
}
_ALLOWED_GLOBALS.update(
    {("torch", name): _StorageType(name) for name in _STORAGE_DTYPES}
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only resolves the globals of ``_ALLOWED_GLOBALS``."""

    def __init__(self, file: BinaryIO, storages: dict, load_storage=None):
        super().__init__(file)
        self._storages = storages
        self._load_storage = load_storage

    def find_class(self, module: str, name: str) -> Any:
        try:
            return _ALLOWED_GLOBALS[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(
                f"Global '{module}.{name}' is not allowed in a torch file"
            ) from None

    def persistent_load(self, pid: Any) -> Any:
        if (
            not isinstance(pid, tuple)
            or len(pid) not in (5, 6)
            or pid[0] != "storage"
            or not isinstance(pid[1], _StorageType)
        ):
            raise pickle.UnpicklingError(f"Unsupported persistent id {pid!r}")
        storage_type, key, numel = pid[1], str(pid[2]), int(pid[4])
        if key not in self._storages:
            storage = _Storage(storage_type.dtype, numel)
            if self._load_storage is not None:
                self._load_storage(key, storage)
            self._storages[key] = storage
        storage = self._storages[key]
        view_metadata = pid[5] if len(pid) == 6 else None
        if view_metadata is not None:
            _, offset, view_numel = view_metadata
            return _StorageView(storage, int(offset), int(view_numel))
        return storage


def _materialize(obj: Any) -> Any:
    """Replaces every pending tensor of ``obj`` (recursively) by an array."""
    if isinstance(obj, _PendingTensor):
        return obj.materialize()
    if isinstance(obj, list):
        return [_materialize(o) for o in obj]
    if isinstance(obj, tuple):
        return tuple(_materialize(o) for o in obj)
    if isinstance(obj, dict):
        return obj.__class__((k, _materialize(v)) for k, v in obj.items())
    if isinstance(obj, SparseCOOMatrix):
        obj.__dict__ = {k: _materialize(v) for k, v in obj.__dict__.items()}
    return obj


def _load_zip(path: str) -> Any:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pickles = [n for n in names if n.endswith("data.pkl")]
        if len(pickles) != 1:
            raise pickle.UnpicklingError("Not a torch zip file: no data.pkl")
        prefix = pickles[0][: -len("data.pkl")]
        byteorder = prefix + "byteorder"
        if byteorder in names and archive.read(byteorder).strip() != b"little":
            raise pickle.UnpicklingError("Only little-endian torch files are read")

        # torch stores the entries uncompressed: they are memory-mapped rather
        # than read, so only the tensors themselves are copied in memory.
        file_map = np.memmap(path, dtype=np.uint8, mode="r")

        def load_storage(key: str, storage: _Storage):
            name = f"{prefix}data/{key}"
            if name not in names:
                raise pickle.UnpicklingError(f"Missing storage {key}")
            info = archive.getinfo(name)
            if info.compress_type != zipfile.ZIP_STORED:
                storage.set_bytes(archive.read(name))
                return
            header = bytes(file_map[info.header_offset : info.header_offset + 30])
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise pickle.UnpicklingError("Invalid zip entry")
            name_len, extra_len = struct.unpack("<HH", header[26:30])
            start = info.header_offset + 30 + name_len + extra_len
            if start + info.file_size > file_map.size:
                raise pickle.UnpicklingError("Truncated torch file")
            storage.set_bytes(file_map[start : start + info.file_size])

        storages: dict = {}
        with archive.open(pickles[0]) as f:
            result = _RestrictedUnpickler(
                io.BytesIO(f.read()), storages, load_storage
            ).load()
        result = _materialize(result)
    return result


def _load_legacy(f: BinaryIO) -> Any:
    def load(file):
        return _RestrictedUnpickler(file, {}).load()

    magic_number = load(f)
    if magic_number != _LEGACY_MAGIC_NUMBER:
        raise pickle.UnpicklingError("Invalid magic number: not a torch file")
    load(f)  # protocol version
    sys_info = load(f)
    if isinstance(sys_info, dict) and not sys_info.get("little_endian", True):
        raise pickle.UnpicklingError("Only little-endian torch files are read")

    storages: dict = {}
    result = _RestrictedUnpickler(f, storages).load()
    keys = load(f)
    if not isinstance(keys, list):
        raise pickle.UnpicklingError("Invalid storage keys")
    for key in keys:
        storage = storages.get(str(key))
        if storage is None:
            raise pickle.UnpicklingError(f"Unknown storage {key}")
        header = f.read(8)
        if len(header) != 8:
            raise pickle.UnpicklingError("Truncated torch file")
        (numel,) = struct.unpack("<q", header)
        if numel < 0:
            raise pickle.UnpicklingError("Invalid storage size")
        nbytes = numel * storage.dtype.itemsize
        buffer = f.read(nbytes)
        if len(buffer) != nbytes:
            raise pickle.UnpicklingError("Truncated torch file")
        storage.set_bytes(buffer)
    return _materialize(result)


def load_torch_file(path: Union[str, os.PathLike]) -> Any:
    """Loads a file saved with ``torch.save`` (or a plain pickle file) without
    PyTorch, with every tensor converted to a :class:`numpy.ndarray`.

    Both the zip-based format (default since PyTorch 1.6) and the legacy
    format are supported. Only tensors (on any device, read on the CPU),
    containers (lists, tuples, dicts), NumPy arrays and SciPy COO matrices
    (read as :class:`SparseCOOMatrix`) can be loaded: any other object raises
    a :class:`pickle.UnpicklingError`, which makes the loader safe to use on
    untrusted files. ``bfloat16`` and complex tensors are not supported.

    Args:
        path: Path of the file.

    Returns:
        The saved object, with tensors as NumPy arrays.

    Example:

    .. code-block:: python

        from mlx_graphs.datasets.utils import load_torch_file

        graphs = load_torch_file("peptidesfunc/train.pt")
        node_features, edge_features, edge_index, y = graphs[0]
    """
    path = os.fspath(path)
    if zipfile.is_zipfile(path):
        return _load_zip(path)
    with open(path, "rb") as f:
        first = f.read(64)
        f.seek(0)
        # A legacy torch file starts with the pickled magic number, which is a
        # LONG1 opcode (``\x8a``) right after the protocol header.
        if len(first) > 2 and first[0] == 0x80 and first[2] == 0x8A:
            return _load_legacy(f)
        return _materialize(_RestrictedUnpickler(f, {}).load())
