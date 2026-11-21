import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.utils.generators import (
    barabasi_albert_graph,
    erdos_renyi_graph,
    grid,
    stochastic_blockmodel_graph,
)
from mlx_graphs.utils.topology import is_undirected


def is_sorted_unique(edges: np.ndarray, num_nodes: int) -> bool:
    keys = edges[0].astype(np.int64) * num_nodes + edges[1]
    return bool(np.all(np.diff(keys) > 0))


@pytest.mark.parametrize("height, width", [(1, 1), (2, 2), (3, 4), (5, 2)])
def test_grid_matches_pyg(height, width):
    torch = pytest.importorskip("torch")
    pyg_utils = pytest.importorskip("torch_geometric.utils")

    edge_index, pos = grid(height, width)
    exp_index, exp_pos = pyg_utils.grid(height, width)
    assert edge_index.dtype == mx.int32 and pos.dtype == mx.float32
    assert np.array_equal(np.asarray(edge_index), exp_index.numpy())
    assert np.array_equal(np.asarray(pos), exp_pos.numpy())
    assert grid(height, width, dtype=mx.int32)[1].dtype == mx.int32
    del torch


@pytest.mark.parametrize("directed", [False, True])
def test_erdos_renyi_graph(directed):
    num_nodes = 40
    mx.random.seed(0)
    edges = np.asarray(erdos_renyi_graph(num_nodes, 0.3, directed))
    assert np.all(edges[0] != edges[1])
    assert is_sorted_unique(edges, num_nodes)
    num_pairs = num_nodes * (num_nodes - 1)
    assert abs(edges.shape[1] / num_pairs - 0.3) < 0.05
    if not directed:
        assert is_undirected(mx.array(edges))

    assert erdos_renyi_graph(10, 0.0, directed).shape == (2, 0)
    assert erdos_renyi_graph(5, 1.0, directed).shape == (2, 20)
    mx.random.seed(1)
    a = erdos_renyi_graph(10, 0.5, directed)
    mx.random.seed(1)
    assert mx.array_equal(a, erdos_renyi_graph(10, 0.5, directed))


def test_barabasi_albert_graph():
    num_nodes, num_edges = 50, 3
    mx.random.seed(0)
    edges = np.asarray(barabasi_albert_graph(num_nodes, num_edges))
    assert edges.dtype == np.int32
    assert np.all(edges[0] != edges[1])
    assert is_sorted_unique(edges, num_nodes)
    assert is_undirected(mx.array(edges))
    degrees = np.bincount(edges[0], minlength=num_nodes)
    # Every node added after the first ones gets at least one edge.
    assert np.all(degrees[num_edges:] >= 1)
    # Preferential attachment: the first nodes are hubs.
    assert degrees[:num_edges].mean() > degrees[-10:].mean()

    mx.random.seed(2)
    a = barabasi_albert_graph(10, 2)
    mx.random.seed(2)
    assert mx.array_equal(a, barabasi_albert_graph(10, 2))
    with pytest.raises(ValueError):
        barabasi_albert_graph(3, 3)


@pytest.mark.parametrize("directed", [False, True])
def test_stochastic_blockmodel_graph(directed):
    block_sizes = [30, 40]
    edge_probs = [[0.6, 0.05], [0.05, 0.3]]
    mx.random.seed(0)
    edges = np.asarray(stochastic_blockmodel_graph(block_sizes, edge_probs, directed))
    num_nodes = 70
    assert np.all(edges[0] != edges[1])
    assert is_sorted_unique(edges, num_nodes)
    if not directed:
        assert is_undirected(mx.array(edges))

    block = np.repeat([0, 1], block_sizes)
    for a, b in [(0, 0), (0, 1), (1, 1)]:
        count = np.sum((block[edges[0]] == a) & (block[edges[1]] == b))
        pairs = block_sizes[a] * (block_sizes[b] - (a == b))
        assert abs(count / pairs - edge_probs[a][b]) < 0.06

    same = stochastic_blockmodel_graph(
        mx.array([3, 2]), mx.array([[1.0, 0.0], [0.0, 1.0]]), directed
    )
    expected = [[0, 0, 1, 1, 2, 2, 3, 4], [1, 2, 0, 2, 0, 1, 4, 3]]
    assert same.tolist() == expected

    with pytest.raises(ValueError):
        stochastic_blockmodel_graph([2, 2], [[0.5, 0.1], [0.2, 0.5]])
    with pytest.raises(ValueError):
        stochastic_blockmodel_graph([2, 2, 2], [[0.5, 0.1], [0.1, 0.5]])
