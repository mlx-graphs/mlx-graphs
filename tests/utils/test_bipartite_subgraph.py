import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils.subgraph import bipartite_subgraph


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("relabel_nodes", [False, True])
@pytest.mark.parametrize("as_mask", [False, True])
def test_bipartite_subgraph_matches_pyg(seed, relabel_nodes, as_mask):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    rng = np.random.default_rng(seed)
    num_src, num_dst = 9, 6
    edges = np.stack(
        [rng.integers(0, num_src, size=30), rng.integers(0, num_dst, size=30)]
    )
    features = rng.normal(size=(30, 2)).astype(np.float32)
    src_subset = np.sort(rng.permutation(num_src)[:5])
    dst_subset = np.sort(rng.permutation(num_dst)[:3])
    if as_mask:
        src_mask = np.zeros(num_src, dtype=bool)
        src_mask[src_subset] = True
        dst_mask = np.zeros(num_dst, dtype=bool)
        dst_mask[dst_subset] = True
        src_subset, dst_subset = src_mask, dst_mask

    size = (num_src, num_dst)
    out_index, out_features, edge_mask = bipartite_subgraph(
        (mx.array(src_subset), mx.array(dst_subset)),
        mx.array(edges),
        mx.array(features),
        relabel_nodes=relabel_nodes,
        size=size,
        return_edge_mask=True,
    )
    exp_index, exp_features, exp_mask = pyg_utils.bipartite_subgraph(
        (torch.tensor(src_subset), torch.tensor(dst_subset)),
        torch.tensor(edges),
        torch.tensor(features),
        relabel_nodes=relabel_nodes,
        size=size,
        return_edge_mask=True,
    )
    assert np.array_equal(np.asarray(out_index), exp_index.numpy())
    assert np.allclose(np.asarray(out_features), exp_features.numpy())
    assert np.array_equal(np.asarray(edge_mask), exp_mask.numpy())


def test_bipartite_subgraph_docstring_example():
    edge_index = mx.array(
        [[0, 5, 2, 3, 3, 4, 4, 3, 5, 5, 6], [0, 0, 3, 2, 0, 0, 2, 1, 2, 3, 1]]
    )
    out_index, out_features = bipartite_subgraph(
        ([2, 3, 5], [2, 3]), edge_index, relabel_nodes=True
    )
    assert out_index.tolist() == [[0, 1, 2, 2], [1, 0, 0, 1]]
    assert out_features is None

    # Relabeling follows the order of an index subset.
    out_index, _ = bipartite_subgraph(
        (mx.array([5, 3, 2]), mx.array([3, 2])), edge_index, relabel_nodes=True
    )
    assert out_index.tolist() == [[2, 1, 0, 0], [0, 1, 1, 0]]

    out_index, _ = bipartite_subgraph(([0], [5]), edge_index, relabel_nodes=True)
    assert out_index.shape == (2, 0)
    with pytest.raises(ValueError):
        bipartite_subgraph(([-1], [0]), edge_index)
    with pytest.raises(ValueError):
        bipartite_subgraph((mx.array([True]), mx.array([True])), edge_index)
