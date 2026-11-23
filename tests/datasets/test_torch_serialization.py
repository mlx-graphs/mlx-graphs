import collections
import os
import pickle

import numpy as np
import pytest
import torch

from mlx_graphs.datasets.utils import load_torch_file
from mlx_graphs.datasets.utils.torch_serialization import SparseCOOMatrix


def make_object():
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    return {
        "float": base,
        "transposed_view": base.t(),  # shares the storage of `base`
        "slice": base[1:3, ::2],
        "long": torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long),
        "int": torch.tensor([1, -2, 3], dtype=torch.int32),
        "bool": torch.tensor([True, False, True]),
        "uint8": torch.tensor([0, 255], dtype=torch.uint8),
        "half": torch.tensor([0.5, -1.25], dtype=torch.float16),
        "double": torch.tensor([1e-10, 3.5], dtype=torch.float64),
        "scalar": torch.tensor(7),
        "empty": torch.zeros(0, 3),
        "list": [torch.ones(2), (torch.zeros(1, dtype=torch.long), 3)],
        "ordered": collections.OrderedDict(a=torch.tensor([1.0])),
        "python": {"int": 1, "str": "a", "float": 2.0, "none": None},
    }


def assert_same(loaded, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(loaded, np.ndarray)
        np.testing.assert_array_equal(loaded, expected.numpy())
        assert loaded.dtype == expected.numpy().dtype
        assert loaded.shape == tuple(expected.shape)
    elif isinstance(expected, dict):
        assert set(loaded) == set(expected)
        for key in expected:
            assert_same(loaded[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(loaded) is type(expected) and len(loaded) == len(expected)
        for a, b in zip(loaded, expected):
            assert_same(a, b)
    else:
        assert loaded == expected


@pytest.mark.parametrize("zip_format", [True, False])
def test_load_torch_file_matches_torch(tmp_path, zip_format):
    obj = make_object()
    path = os.path.join(tmp_path, "obj.pt")
    torch.save(obj, path, _use_new_zipfile_serialization=zip_format)
    assert_same(load_torch_file(path), obj)


@pytest.mark.parametrize("zip_format", [True, False])
def test_load_torch_file_single_tensor(tmp_path, zip_format):
    tensor = torch.randint(0, 10, (150,))
    path = os.path.join(tmp_path, "y.pt")
    torch.save(tensor, path, _use_new_zipfile_serialization=zip_format)
    assert_same(load_torch_file(path), tensor)


def test_load_plain_pickle_with_numpy_and_coo(tmp_path):
    scipy_sparse = pytest.importorskip("scipy.sparse")
    adjacency = scipy_sparse.coo_matrix(
        (np.ones(3), (np.array([0, 1, 2]), np.array([1, 2, 0]))), shape=(3, 3)
    )
    obj = [adjacency, np.arange(5, dtype=np.int64), np.float32(2.5)]
    path = os.path.join(tmp_path, "graphs.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=3)

    loaded = load_torch_file(path)
    assert isinstance(loaded[0], SparseCOOMatrix)
    assert loaded[0].shape == (3, 3)
    np.testing.assert_array_equal(loaded[0].row, adjacency.row)
    np.testing.assert_array_equal(loaded[0].col, adjacency.col)
    np.testing.assert_array_equal(loaded[1], obj[1])
    assert loaded[2] == obj[2]


class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


@pytest.mark.parametrize("zip_format", [True, False])
def test_load_torch_file_rejects_arbitrary_globals(tmp_path, zip_format):
    path = os.path.join(tmp_path, "evil.pt")
    torch.save({"x": torch.ones(2), "evil": _Evil()}, path,
               _use_new_zipfile_serialization=zip_format)  # fmt: skip
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        load_torch_file(path)

    # codecs other than the byte encoding of protocol 2 are not allowed
    with open(path, "wb") as f:
        pickle.dump(__import__("codecs").decode, f, protocol=2)
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        load_torch_file(path)

    # modules are not allowed either
    torch.save(torch.nn.Linear(2, 2), path, _use_new_zipfile_serialization=zip_format)
    with pytest.raises(pickle.UnpicklingError):
        load_torch_file(path)


def test_load_plain_pickle_rejects_arbitrary_globals(tmp_path):
    path = os.path.join(tmp_path, "evil.pkl")
    with open(path, "wb") as f:
        pickle.dump([_Evil()], f)
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        load_torch_file(path)


@pytest.mark.parametrize("zip_format", [True, False])
def test_load_torch_file_with_numpy_arrays(tmp_path, zip_format):
    # torch.save pickles with protocol 2, where NumPy arrays store their bytes
    # through `_codecs.encode`.
    obj = {"x": torch.ones(2), "np": np.arange(5), "f": np.float32(1.5)}
    path = os.path.join(tmp_path, "mixed.pt")
    torch.save(obj, path, _use_new_zipfile_serialization=zip_format)
    loaded = load_torch_file(path)
    np.testing.assert_array_equal(loaded["np"], obj["np"])
    assert loaded["f"] == obj["f"]
    assert_same(loaded["x"], obj["x"])


@pytest.mark.parametrize("protocol", [2, 3, pickle.HIGHEST_PROTOCOL])
def test_load_plain_pickle_of_tensors(tmp_path, protocol):
    # Tensors pickled with the `pickle` module (e.g. the raw ZINC files) store
    # their storage as the bytes of a legacy torch.save file.
    obj = make_object()
    obj["records"] = [
        {"atom_type": torch.tensor([0, 1, 2], dtype=torch.int8), "num_atom": 3},
        {"adj": torch.eye(3, dtype=torch.int8)[1:], "y": torch.tensor([0.5])},
    ]
    path = os.path.join(tmp_path, "tensors.pickle")
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)
    assert_same(load_torch_file(path), obj)


def test_load_plain_pickle_of_tensors_rejects_arbitrary_storage_bytes(tmp_path):
    from mlx_graphs.datasets.utils.torch_serialization import _load_from_bytes

    with pytest.raises(pickle.UnpicklingError):
        _load_from_bytes(pickle.dumps(_Evil(), protocol=2))
    with pytest.raises(pickle.UnpicklingError, match="storage bytes"):
        _load_from_bytes("not bytes")
    # a legacy torch file holding something else than a storage
    path = os.path.join(tmp_path, "tensor.pt")
    torch.save(torch.ones(2), path, _use_new_zipfile_serialization=False)
    with open(path, "rb") as f:
        with pytest.raises(pickle.UnpicklingError, match="storage"):
            _load_from_bytes(f.read())
