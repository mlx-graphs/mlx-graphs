import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils.sparse import (
    dense_to_sparse,
    from_scipy_sparse_matrix,
    to_scipy_sparse_matrix,
)


def random_edges(seed: int, num_nodes: int = 10, num_edges: int = 30) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, num_nodes, size=(2, num_edges))


@pytest.mark.parametrize("seed", range(3))
def test_scipy_roundtrip_matches_pyg(seed):
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    edges = random_edges(seed)
    weights = np.random.default_rng(seed).normal(size=edges.shape[1]).astype(np.float32)

    for w_mx, w_pt in ((None, None), (mx.array(weights), torch.tensor(weights))):
        for num_nodes in (None, 13):
            out = to_scipy_sparse_matrix(mx.array(edges), w_mx, num_nodes)
            expected = pyg_utils.to_scipy_sparse_matrix(
                torch.tensor(edges), w_pt, num_nodes
            )
            assert out.shape == expected.shape
            assert np.allclose(out.toarray(), expected.toarray())

    matrix = pyg_utils.to_scipy_sparse_matrix(
        torch.tensor(edges), torch.tensor(weights)
    )
    edge_index, edge_weights = from_scipy_sparse_matrix(matrix.tocsr())
    exp_index, exp_weights = pyg_utils.from_scipy_sparse_matrix(matrix.tocsr())
    assert edge_index.dtype == mx.int32
    assert np.array_equal(np.asarray(edge_index), exp_index.numpy())
    assert np.allclose(np.asarray(edge_weights), exp_weights.numpy())


def test_scipy_conversion_dtypes():
    sp = pytest.importorskip("scipy.sparse")
    matrix = sp.coo_matrix(([1.0, 2.0], ([0, 1], [1, 0])), shape=(3, 3))
    edge_index, edge_weights = from_scipy_sparse_matrix(matrix)
    assert edge_index.tolist() == [[0, 1], [1, 0]]
    assert edge_weights.dtype == mx.float32  # float64 is narrowed

    out = to_scipy_sparse_matrix(edge_index, mx.array([[3.0], [4.0]]), num_nodes=3)
    assert out.toarray().tolist() == [[0, 3, 0], [4, 0, 0], [0, 0, 0]]
    with pytest.raises(ValueError):
        to_scipy_sparse_matrix(edge_index, mx.ones((2, 2)))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("with_mask", [False, True])
def test_dense_to_sparse_matches_pyg(seed, batched, with_mask):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    rng = np.random.default_rng(seed)
    shape = (3, 5, 5) if batched else (5, 5)
    adj = rng.normal(size=shape).astype(np.float32) * (rng.random(shape) < 0.4)
    mask = None
    if with_mask:
        mask = np.zeros((3, 5), dtype=bool)
        for b, n in enumerate([2, 5, 3]):
            mask[b, :n] = True

    out_index, out_weights = dense_to_sparse(
        mx.array(adj), None if mask is None else mx.array(mask)
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp_index, exp_weights = pyg_utils.dense_to_sparse(
            torch.tensor(adj), None if mask is None else torch.tensor(mask)
        )
    assert out_index.dtype == mx.int32
    assert np.array_equal(np.asarray(out_index), exp_index.numpy())
    assert np.allclose(np.asarray(out_weights), exp_weights.numpy())


def test_dense_to_sparse_edge_cases():
    out_index, out_weights = dense_to_sparse(mx.zeros((3, 3)))
    assert out_index.shape == (2, 0) and out_weights.shape == (0,)
    out_index, out_weights = dense_to_sparse(mx.zeros((2, 3, 3)))
    assert out_index.shape == (2, 0) and out_weights.shape == (0,)
    with pytest.raises(ValueError):
        dense_to_sparse(mx.zeros((3,)))
    with pytest.raises(ValueError):
        dense_to_sparse(mx.zeros((2, 3, 3)), mx.ones((6,), dtype=mx.bool_))
