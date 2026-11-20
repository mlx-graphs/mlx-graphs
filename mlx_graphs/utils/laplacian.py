from typing import Literal, Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.utils.transformations import _infer_num_nodes, _to_numpy
from mlx_graphs.utils.validators import validate_edge_index


def _inverse(values: np.ndarray, power: float) -> np.ndarray:
    """``values ** power`` for a negative power, with 0 mapped to 0."""
    out = np.zeros_like(values)
    np.power(values, power, out=out, where=values != 0)
    return out


@validate_edge_index
def get_laplacian(
    edge_index: mx.array,
    edge_weights: Optional[mx.array] = None,
    normalization: Optional[Literal["sym", "rw"]] = None,
    num_nodes: Optional[int] = None,
) -> tuple[mx.array, mx.array]:
    r"""
    Computes the graph Laplacian of the graph given by ``edge_index`` and
    optional ``edge_weights``, in sparse (edge list) form.

    Existing self-loops are removed first. The degree :math:`\mathbf{D}` is the
    weighted out-degree (the sum of the weights of the edges leaving a node).

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        edge_weights: Optional weight of each edge, of shape ``[num_edges]``.
            Default: ``None`` (all ones).
        normalization: The normalization of the Laplacian:

            - ``None``: no normalization,
              :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`
            - ``"sym"``: symmetric normalization,
              :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
              \mathbf{D}^{-1/2}`
            - ``"rw"``: random-walk normalization,
              :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

        num_nodes: Number of nodes in the graph. Inferred from ``edge_index``
            if not given.

    Returns:
        A tuple ``(edge_index, edge_weights)`` with the non-zero entries of the
        Laplacian: the off-diagonal entries (one per edge that is not a
        self-loop, in the input order) followed by the ``num_nodes`` diagonal
        entries.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 1, 2], [1, 0, 2, 1]])

        get_laplacian(edge_index)
        # (mx.array([[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]]),
        #  mx.array([-1, -1, -1, -1, 1, 2, 1]))

        get_laplacian(edge_index, normalization="rw")
        # (mx.array([[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]]),
        #  mx.array([-1, -0.5, -0.5, -1, 1, 1, 1]))
    """
    if normalization not in (None, "sym", "rw"):
        raise ValueError(
            f"Invalid normalization '{normalization}', expected None, 'sym' or 'rw'"
        )
    num_edges = edge_index.shape[1]
    if edge_weights is not None and edge_weights.size != num_edges:
        raise ValueError(
            "edge_weights must be 1 per edge "
            f"(got {num_edges} edges and {edge_weights.size} weights)"
        )
    num_nodes = _infer_num_nodes(edge_index, num_nodes)
    edges = _to_numpy(edge_index)
    if edges.size > 0 and int(edges.max()) >= num_nodes:
        raise ValueError(
            f"num_nodes={num_nodes} but `edge_index` contains node {int(edges.max())}"
        )

    dtype = mx.float32
    if edge_weights is None:
        weights = np.ones(num_edges, dtype=np.float64)
    else:
        if mx.issubdtype(edge_weights.dtype, mx.floating):
            dtype = edge_weights.dtype
        weights = _to_numpy(edge_weights).reshape(-1).astype(np.float64)

    not_loop = edges[0] != edges[1]
    edges, weights = edges[:, not_loop], weights[not_loop]
    # int64 indices: `np.bincount` rejects uint64. It also returns integers for
    # an empty input, even with weights.
    row, col = edges[0].astype(np.int64), edges[1].astype(np.int64)
    deg = np.bincount(row, weights=weights, minlength=num_nodes).astype(np.float64)

    if normalization is None:
        diagonal = deg
        off_diagonal = -weights
    else:
        diagonal = np.ones(num_nodes, dtype=np.float64)
        if normalization == "sym":
            deg_inv_sqrt = _inverse(deg, -0.5)
            off_diagonal = -deg_inv_sqrt[row] * weights * deg_inv_sqrt[col]
        else:
            off_diagonal = -_inverse(deg, -1.0)[row] * weights

    nodes = np.arange(num_nodes, dtype=edges.dtype)
    laplacian_index = np.concatenate([edges, np.stack([nodes, nodes])], axis=1)
    laplacian_weights = np.concatenate([off_diagonal, diagonal])
    return (
        mx.array(laplacian_index, dtype=edge_index.dtype),
        mx.array(laplacian_weights.astype(np.float32)).astype(dtype),
    )


@validate_edge_index
def _homophily(
    edge_index: mx.array, node_labels: mx.array, method: Literal["edge", "node"]
) -> float:
    if method not in ("edge", "node"):
        raise ValueError(f"Invalid method '{method}', expected 'edge' or 'node'")
    if node_labels.ndim > 1 and node_labels.size != node_labels.shape[0]:
        raise ValueError(
            "node_labels must hold one class index per node "
            f"(got shape {node_labels.shape})"
        )

    labels = _to_numpy(node_labels).reshape(-1)
    edges = _to_numpy(edge_index).astype(np.int64)
    src, dst = edges[0], edges[1]
    if src.size > 0 and int(edges.max()) >= labels.size:
        raise ValueError(
            f"node_labels has {labels.size} entries but `edge_index` contains "
            f"node {int(edges.max())}"
        )
    same_label = (labels[src] == labels[dst]).astype(np.float64)

    if method == "edge":
        if same_label.size == 0:
            return float("nan")
        return float(same_label.mean())

    if labels.size == 0:
        return float("nan")
    num_same = np.bincount(dst, weights=same_label, minlength=labels.size)
    in_degree = np.bincount(dst, minlength=labels.size)
    return float((num_same / np.maximum(in_degree, 1)).mean())


def homophily(
    edge_index: mx.array,
    node_labels: mx.array,
    method: Literal["edge", "node"] = "edge",
) -> float:
    """
    Computes the homophily of a graph: how often connected nodes share the
    same label.

    Args:
        edge_index: a [2, num_edges] array representing the source and target
            nodes of each edge
        node_labels: Array of shape ``[num_nodes]`` (or ``[num_nodes, 1]``)
            with the class index of each node.
        method: The homophily measure:

            - ``"edge"``: the fraction of edges that connect two nodes with
              the same label.
            - ``"node"``: for each node, the fraction of its incoming edges
              that come from a node with the same label, averaged over all
              the nodes (nodes without incoming edges count as 0).

            Default: ``"edge"``

    Returns:
        The homophily ratio, between 0 and 1. It is ``nan`` if it is undefined
        (no edges for ``"edge"``, no nodes for ``"node"``).

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 1, 2, 3], [1, 2, 0, 4]])
        node_labels = mx.array([0, 0, 0, 0, 1])

        homophily(edge_index, node_labels, method="edge")
        # 0.75

        homophily(edge_index, node_labels, method="node")
        # 0.6
    """
    return _homophily(edge_index, node_labels, method)
