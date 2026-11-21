import mlx.core as mx
import numpy as np

from mlx_graphs.utils.transformations import _to_numpy
from mlx_graphs.utils.validators import validate_edge_index


@validate_edge_index
def assortativity(edge_index: mx.array) -> float:
    """
    Computes the degree assortativity coefficient of a graph: the Pearson
    correlation between the out-degree of the source and the in-degree of the
    destination of every edge (`"Mixing patterns in networks"
    <https://arxiv.org/abs/cond-mat/0209450>`_).

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge

    Returns:
        The assortativity coefficient, in ``[-1, 1]``. It is ``nan`` when all
        the edges have the same degrees (e.g. a regular graph).

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 2, 3, 2], [1, 2, 0, 1, 3]])
        assortativity(edge_index)
        # -0.6666...
    """
    edges = _to_numpy(edge_index).astype(np.int64)
    row, col = edges[0], edges[1]
    if row.size == 0:
        return float("nan")
    out_deg = np.bincount(row)
    in_deg = np.bincount(col)
    degrees = np.unique(np.concatenate([out_deg, in_deg]))
    mapping = np.zeros(int(degrees.max()) + 1, dtype=np.int64)
    mapping[degrees] = np.arange(degrees.size)

    # Degree mixing matrix (joint distribution of the degrees of the edges).
    num_degrees = degrees.size
    mixing = np.zeros((num_degrees, num_degrees), dtype=np.float64)
    np.add.at(mixing, (mapping[out_deg[row]], mapping[in_deg[col]]), 1.0)
    mixing /= mixing.sum()

    x = degrees.astype(np.float64)
    a, b = mixing.sum(0), mixing.sum(1)
    var_a = (a * x**2).sum() - ((a * x).sum()) ** 2
    var_b = (b * x**2).sum() - ((b * x).sum()) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (np.outer(x, x) * (mixing - np.outer(a, b))).sum() / np.sqrt(
            var_a * var_b
        )
    return float(out)
