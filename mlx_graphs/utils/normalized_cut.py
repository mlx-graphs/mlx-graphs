from typing import Optional

import mlx.core as mx

from mlx_graphs.utils.scatter import degree
from mlx_graphs.utils.transformations import _infer_num_nodes
from mlx_graphs.utils.validators import validate_edge_index_and_features


@validate_edge_index_and_features
def normalized_cut(
    edge_index: mx.array,
    edge_weights: mx.array,
    num_nodes: Optional[int] = None,
) -> mx.array:
    """
    Computes the normalized cut of every edge ``(i, j)``:
    ``w_ij * (1 / deg(i) + 1 / deg(j))``, where ``deg`` is the in-degree
    (number of incoming edges) of a node.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        edge_weights: The weight of each edge, of shape ``[num_edges]`` (or
            ``[num_edges, ...]``).
        num_nodes: Number of nodes in the graph. Inferred from ``edge_index``
            if not given.

    Returns:
        The normalized cut of each edge, with the shape of ``edge_weights``.

    Example:

    .. code-block:: python

        edge_index = mx.array([[1, 1, 2, 3], [3, 3, 1, 2]])
        edge_weights = mx.array([1.0, 1.0, 1.0, 1.0])
        normalized_cut(edge_index, edge_weights)
        # mx.array([1.5, 1.5, 2.0, 1.5])
    """
    num_nodes = _infer_num_nodes(edge_index, num_nodes)
    dtype = (
        edge_weights.dtype
        if mx.issubdtype(edge_weights.dtype, mx.floating)
        else mx.float32
    )
    inv_deg = 1.0 / degree(edge_index[1], num_nodes).astype(dtype)
    cut = inv_deg[edge_index[0]] + inv_deg[edge_index[1]]
    cut = cut.reshape(cut.shape + (1,) * (edge_weights.ndim - 1))
    return edge_weights * cut
