import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils.array_ops import index_to_mask
from mlx_graphs.utils.indexing import cumsum, mask_to_index, narrow, select
from mlx_graphs.utils.softmax import softmax


def test_mask_to_index():
    mask = mx.array([False, True, True, False, True])
    index = mask_to_index(mask)
    assert index.dtype == mx.int32
    assert index.tolist() == [1, 2, 4]
    assert index_to_mask(index, 5).tolist() == mask.tolist()
    assert mask_to_index(mx.zeros((3,), dtype=mx.bool_)).tolist() == []


def test_cumsum_matches_pyg():
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    x = np.random.default_rng(0).integers(0, 5, size=(4, 3)).astype(np.int32)
    for axis in (0, 1, -1):
        out = cumsum(mx.array(x), axis)
        # PyG's cumsum does not support negative dims.
        expected = pyg_utils.cumsum(torch.tensor(x), axis % x.ndim)
        assert out.dtype == mx.int32
        assert np.array_equal(np.asarray(out), expected.numpy())
    assert cumsum(mx.array([2, 4, 1])).tolist() == [0, 2, 6, 7]
    assert cumsum(mx.zeros((0,))).tolist() == [0]


def test_select_and_narrow():
    src = mx.array([[1, 2], [3, 4], [5, 6]])
    assert select(src, mx.array([True, False, True])).tolist() == [[1, 2], [5, 6]]
    assert select(src, mx.array([2, 0])).tolist() == [[5, 6], [1, 2]]
    assert select(src, mx.array([1]), axis=1).tolist() == [[2], [4], [6]]
    assert select(src, mx.zeros((3,), dtype=mx.bool_)).shape == (0, 2)
    assert select(["a", "b", "c"], mx.array([2, 0])) == ["c", "a"]
    assert select(("a", "b", "c"), [False, True, True]) == ["b", "c"]
    with pytest.raises(ValueError):
        select(["a"], mx.array([0]), axis=1)

    assert narrow(src, 0, 1, 2).tolist() == [[3, 4], [5, 6]]
    assert narrow(src, 1, 1, 1).tolist() == [[2], [4], [6]]
    assert narrow(src, -1, 0, 1).tolist() == [[1], [3], [5]]
    assert narrow(["a", "b", "c"], 0, 1, 2) == ["b", "c"]
    with pytest.raises(ValueError):
        narrow("abc", 0, 0, 1)


@pytest.mark.parametrize("shape", [(10,), (10, 3)])
def test_softmax_matches_pyg(shape):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    rng = np.random.default_rng(0)
    src = rng.normal(size=shape).astype(np.float32)
    index = np.array([0, 0, 1, 1, 1, 2, 4, 4, 4, 4])  # group 3 is empty

    out = softmax(mx.array(src), mx.array(index))
    expected = pyg_utils.softmax(torch.tensor(src), torch.tensor(index))
    assert np.allclose(np.asarray(out), expected.numpy(), atol=1e-6)

    out = softmax(mx.array(src), mx.array(index), num_nodes=7)
    assert np.allclose(np.asarray(out), expected.numpy(), atol=1e-6)

    ptr = np.array([0, 2, 5, 6, 6, 10])
    out = softmax(mx.array(src), ptr=mx.array(ptr))
    expected = pyg_utils.softmax(torch.tensor(src), ptr=torch.tensor(ptr))
    assert np.allclose(np.asarray(out), expected.numpy(), atol=1e-6)


def test_softmax_empty():
    """No entries (e.g. a graph without edges): an empty output, as PyG."""
    index = mx.zeros((0,), dtype=mx.int32)
    for num_nodes in (None, 0, 3):
        out = softmax(mx.zeros((0, 2)), index, num_nodes=num_nodes)
        assert out.shape == (0, 2)
    assert softmax(mx.zeros((2, 0)), index, axis=1).shape == (2, 0)
    assert softmax(mx.zeros((0,)), ptr=mx.array([0])).shape == (0,)


def test_softmax_other_axis():
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    src = np.random.default_rng(1).normal(size=(2, 6, 3)).astype(np.float32)
    index = np.array([0, 1, 0, 1, 2, 2])
    out = softmax(mx.array(src), mx.array(index), axis=1)
    expected = pyg_utils.softmax(torch.tensor(src), torch.tensor(index), dim=1)
    assert np.allclose(np.asarray(out), expected.numpy(), atol=1e-6)

    with pytest.raises(ValueError):
        softmax(mx.array(src))
