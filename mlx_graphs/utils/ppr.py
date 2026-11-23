from typing import Optional, Sequence, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.transformations import _infer_num_nodes, _to_numpy
from mlx_graphs.utils.validators import validate_edge_index


def _push_ppr(
    rowptr: np.ndarray, col: np.ndarray, source: int, alpha: float, eps: float
) -> tuple[list[int], list[float]]:
    """Approximate PPR vector of ``source`` with the push algorithm of
    Andersen et al., in the order the nodes are first reached."""
    alpha_eps = alpha * eps
    p = {source: 0.0}
    r = {source: alpha}
    queue = [source]
    in_queue = {source}
    while queue:
        u = queue.pop()
        in_queue.discard(u)
        res = r.get(u, 0.0)
        p[u] = p.get(u, 0.0) + res
        r[u] = 0.0
        start, end = rowptr[u], rowptr[u + 1]
        count = end - start
        for v in col[start:end].tolist():
            r[v] = r.get(v, 0.0) + (1 - alpha) * res / count
            if r[v] >= alpha_eps * (rowptr[v + 1] - rowptr[v]) and v not in in_queue:
                queue.append(v)
                in_queue.add(v)
    return list(p.keys()), list(p.values())


@validate_edge_index
def get_ppr(
    edge_index: mx.array,
    alpha: float = 0.2,
    eps: float = 1e-5,
    target: Optional[Union[mx.array, Sequence[int]]] = None,
    num_nodes: Optional[int] = None,
) -> tuple[mx.array, mx.array]:
    """
    Computes approximate personalized PageRank (PPR) vectors with the push
    algorithm of `"Local Graph Partitioning using PageRank Vectors"
    <https://doi.org/10.1109/FOCS.2006.44>`_, as
    used in `"Predict then Propagate" <https://arxiv.org/abs/1810.05997>`_.

    The PPR vector of a node ``i`` is the stationary distribution of a random
    walk along the edges of ``edge_index`` that teleports back to ``i`` with
    probability ``alpha`` at each step. Runs on the host (pure Python), so it
    is meant for small to medium graphs.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        alpha: Teleport (restart) probability. Default: ``0.2``
        eps: Approximation threshold: residuals below ``alpha * eps *
            degree`` are not pushed further. Default: ``1e-5``
        target: Nodes whose PPR vectors are computed. Default: ``None`` (all
            the nodes).
        num_nodes: Number of nodes in the graph. Inferred from ``edge_index``
            if not given.

    Returns:
        A tuple ``(ppr_edge_index, ppr_weights)``: an edge ``(i, j)`` with
        weight ``w`` means that the PPR score of ``j`` for the source node
        ``i`` is ``w``. Only the non-zero scores are returned.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 1, 2], [1, 0, 2, 1]])
        ppr_edge_index, ppr_weights = get_ppr(edge_index, target=mx.array([0]))
        # ppr_edge_index: mx.array([[0, 0, 0], [0, 1, 2]])
        # ppr_weights: e.g. mx.array([0.38, 0.44, 0.18])
    """
    n = _infer_num_nodes(edge_index, num_nodes)
    edges = _to_numpy(edge_index).astype(np.int64)
    order = np.argsort(edges[0], kind="stable")
    row, col = edges[0][order], edges[1][order]
    rowptr = np.concatenate([[0], np.cumsum(np.bincount(row, minlength=n))])

    if target is None:
        sources = np.arange(n)
    else:
        target_np = _to_numpy(target) if isinstance(target, mx.array) else target
        sources = np.asarray(target_np, dtype=np.int64).reshape(-1)

    rows, cols, weights = [], [], []
    for source in sources.tolist():
        nodes, values = _push_ppr(rowptr, col, source, alpha, eps)
        rows.append(np.full(len(nodes), source, dtype=np.int64))
        cols.append(np.asarray(nodes, dtype=np.int64))
        weights.append(np.asarray(values, dtype=np.float32))

    if not rows:
        return mx.zeros((2, 0), dtype=edge_index.dtype), mx.zeros((0,))
    ppr_edge_index = np.stack([np.concatenate(rows), np.concatenate(cols)])
    return (
        mx.array(ppr_edge_index, dtype=edge_index.dtype),
        mx.array(np.concatenate(weights)),
    )
