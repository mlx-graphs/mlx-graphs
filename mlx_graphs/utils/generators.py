from typing import Sequence, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.negative_sampling import _host_rng
from mlx_graphs.utils.transformations import _to_numpy


def _undirected(edges: np.ndarray, num_nodes: int) -> mx.array:
    """Adds the reverse edges, removes duplicates and sorts by (src, dst)."""
    edges = np.concatenate([edges, edges[::-1]], axis=1)
    keys = np.unique(edges[0] * num_nodes + edges[1])
    return mx.array(np.stack([keys // num_nodes, keys % num_nodes]).astype(np.int32))


def _ordered_pairs(num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """All pairs ``(i, j)`` with ``i != j``, sorted."""
    row, col = np.nonzero(~np.eye(num_nodes, dtype=bool))
    return row, col


def grid(
    height: int, width: int, dtype: mx.Dtype = mx.float32
) -> tuple[mx.array, mx.array]:
    """
    Returns the edges and node positions of a 2D grid graph of ``height`` x
    ``width`` nodes, where each node is connected to itself and to its (up to
    8) horizontal, vertical and diagonal neighbors.

    Node ``i`` is at row ``i // width`` and column ``i % width``; rows are
    numbered from the top, so the position of node ``i`` is
    ``(i % width, height - 1 - i // width)``.

    Args:
        height: Number of rows of the grid.
        width: Number of columns of the grid.
        dtype: Data type of the positions. Default: ``mx.float32``

    Returns:
        A tuple ``(edge_index, pos)`` with the sorted int32 edges and the
        positions, of shape ``[height * width, 2]``.

    Example:

    .. code-block:: python

        edge_index, pos = grid(2, 2)
        # edge_index: every pair of the 4 nodes, including self-loops
        # pos: mx.array([[0, 1], [1, 1], [0, 0], [1, 0]])
    """
    num_nodes = height * width
    w = width
    # Same construction as PyG, including its handling of the grid borders.
    kernel = np.array([-w - 1, -1, w - 1, -w, 0, w, -w + 1, 1, w + 1])
    row = np.repeat(np.arange(num_nodes).reshape(-1, 1), kernel.size, axis=1)
    col = row + kernel.reshape(1, -1)
    row, col = row.reshape(height, -1), col.reshape(height, -1)
    index = np.arange(3, row.shape[1] - 3)
    row, col = row[:, index].reshape(-1), col[:, index].reshape(-1)
    mask = (col >= 0) & (col < num_nodes)
    keys = np.unique(row[mask] * num_nodes + col[mask])
    edge_index = np.stack([keys // num_nodes, keys % num_nodes]).astype(np.int32)

    x = np.tile(np.arange(width), height)
    y = np.repeat((height - 1) - np.arange(height), width)
    pos = mx.array(np.stack([x, y], axis=-1)).astype(dtype)
    return mx.array(edge_index), pos


def erdos_renyi_graph(
    num_nodes: int, edge_prob: float, directed: bool = False
) -> mx.array:
    """
    Returns the edges of a random Erdos-Renyi graph: each possible edge
    (without self-loops) is present independently with probability
    ``edge_prob``.

    Randomness comes from ``mx.random``, so results are reproducible with
    ``mx.random.seed``.

    Args:
        num_nodes: Number of nodes.
        edge_prob: Probability of each edge.
        directed: If ``True``, each ordered pair ``(i, j)`` is sampled
            independently. If ``False``, each unordered pair is sampled once
            and returned in both directions. Default: ``False``

    Returns:
        The sorted int32 ``edge_index`` of the graph.

    Example:

    .. code-block:: python

        erdos_renyi_graph(4, 0.5)
        # e.g. mx.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    """
    if directed:
        row, col = _ordered_pairs(num_nodes)
    else:
        row, col = np.triu_indices(num_nodes, k=1)
    keep = _to_numpy(mx.random.uniform(shape=(row.size,))) < edge_prob
    edges = np.stack([row[keep], col[keep]])
    if directed:
        return mx.array(edges.astype(np.int32))
    return _undirected(edges, num_nodes)


def barabasi_albert_graph(num_nodes: int, num_edges: int) -> mx.array:
    """
    Returns the edges of a random Barabasi-Albert graph (preferential
    attachment): each new node is connected to ``num_edges`` existing nodes,
    chosen with a probability proportional to their degree.

    Randomness comes from ``mx.random``, so results are reproducible with
    ``mx.random.seed``.

    Args:
        num_nodes: Number of nodes.
        num_edges: Number of edges added for each new node.

    Returns:
        The sorted, undirected int32 ``edge_index`` of the graph, without
        self-loops and duplicates.

    Example:

    .. code-block:: python

        barabasi_albert_graph(5, 2)
        # e.g. mx.array([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        #                [2, 3, 3, 4, 0, 4, 0, 1, 1, 2]])
    """
    if not 0 < num_edges < num_nodes:
        raise ValueError(
            "`num_edges` must be positive and smaller than `num_nodes` "
            f"(got num_edges={num_edges}, num_nodes={num_nodes})"
        )
    rng = _host_rng()
    row = list(range(num_edges))
    col = rng.permutation(num_edges).tolist()
    for i in range(num_edges, num_nodes):
        choice = rng.choice(np.array(row + col), num_edges)
        row.extend([i] * num_edges)
        col.extend(choice.tolist())

    edges = np.array([row, col], dtype=np.int64)
    edges = edges[:, edges[0] != edges[1]]
    return _undirected(edges, num_nodes)


def stochastic_blockmodel_graph(
    block_sizes: Union[Sequence[int], mx.array],
    edge_probs: Union[Sequence[Sequence[float]], mx.array],
    directed: bool = False,
) -> mx.array:
    """
    Returns the edges of a random graph from the stochastic block model: the
    nodes are split into blocks, and each possible edge (without self-loops)
    between a node of block ``a`` and a node of block ``b`` is present
    independently with probability ``edge_probs[a][b]``.

    Randomness comes from ``mx.random``, so results are reproducible with
    ``mx.random.seed``.

    Args:
        block_sizes: Number of nodes of each block. The nodes of block 0 come
            first, then those of block 1, and so on.
        edge_probs: Square matrix of the edge probabilities between blocks.
            It must be symmetric if ``directed=False``.
        directed: If ``True``, each ordered pair ``(i, j)`` is sampled
            independently. If ``False``, each unordered pair is sampled once
            and returned in both directions. Default: ``False``

    Returns:
        The sorted int32 ``edge_index`` of the graph.

    Example:

    .. code-block:: python

        block_sizes = [2, 3]
        edge_probs = [[0.9, 0.1], [0.1, 0.8]]
        stochastic_blockmodel_graph(block_sizes, edge_probs)
        # e.g. mx.array([[0, 1, 2, 2, 3, 3, 4, 4], [1, 0, 3, 4, 2, 4, 2, 3]])
    """
    size = np.asarray(
        _to_numpy(block_sizes) if isinstance(block_sizes, mx.array) else block_sizes,
        dtype=np.int64,
    )
    prob = np.asarray(
        _to_numpy(edge_probs) if isinstance(edge_probs, mx.array) else edge_probs,
        dtype=np.float64,
    )
    if size.ndim != 1:
        raise ValueError("`block_sizes` must be one-dimensional")
    if prob.ndim != 2 or prob.shape[0] != prob.shape[1]:
        raise ValueError("`edge_probs` must be a square matrix")
    if size.size != prob.shape[0]:
        raise ValueError("`block_sizes` and `edge_probs` must have the same size")
    if not directed and not np.allclose(prob, prob.T):
        raise ValueError("`edge_probs` must be symmetric for an undirected graph")

    node_block = np.repeat(np.arange(size.size), size)
    num_nodes = node_block.size
    if directed:
        row, col = _ordered_pairs(num_nodes)
    else:
        row, col = np.triu_indices(num_nodes, k=1)
    uniform = _to_numpy(mx.random.uniform(shape=(row.size,)))
    keep = uniform < prob[node_block[row], node_block[col]]
    edges = np.stack([row[keep], col[keep]])
    if directed:
        return mx.array(edges.astype(np.int32))
    return _undirected(edges, num_nodes)
