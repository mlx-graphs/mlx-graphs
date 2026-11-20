from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.base_transform import BaseTransform
from mlx_graphs.transforms.structure import _attribute_names, _num_nodes


def _add_node_features(data: GraphData, features: mx.array, cat: bool) -> GraphData:
    """Sets ``features`` as the node features of ``data``, or appends them to the
    existing ones if ``cat`` is ``True``."""
    existing = data.node_features
    if existing is not None and cat:
        if existing.ndim == 1:
            existing = existing.reshape(-1, 1)
        features = mx.concatenate([existing, features.astype(existing.dtype)], axis=-1)
    data.node_features = features
    return data


def _edge_weights(data: GraphData) -> np.ndarray:
    """Host float64 weight of each edge of ``data`` (ones without weights)."""
    weights = getattr(data, "edge_weights", None)
    if weights is None:
        return np.ones(data.num_edges, dtype=np.float64)
    if weights.size != data.num_edges:
        raise ValueError(
            "edge_weights must hold one scalar per edge "
            f"(got shape {weights.shape} for {data.num_edges} edges)"
        )
    return np.asarray(weights).reshape(-1).astype(np.float64)


def _host_edges(data: GraphData) -> np.ndarray:
    """Host int64 copy of the edges of ``data`` (``np.bincount`` rejects uint64)."""
    return np.asarray(data.edge_index).astype(np.int64)


def _weighted_bincount(index: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    """Sum of ``weights`` per value of ``index``, as float64.

    ``np.bincount`` returns integers for an empty input, even with weights.
    """
    return np.bincount(index, weights=weights, minlength=size).astype(np.float64)


class OneHotDegree(BaseTransform):
    """Adds the one-hot encoded degree of each node to the node features.

    Args:
        max_degree: Maximum degree of the graph(s). The encoding has
            ``max_degree + 1`` columns.
        in_degree: If ``True``, uses the in-degree (number of incoming edges)
            instead of the out-degree. Default: ``False``
        cat: If ``True``, the encoding is appended to the existing node
            features, otherwise it replaces them. Default: ``True``

    Example:

    .. code-block:: python

        graph = GraphData(edge_index=mx.array([[0, 0, 1], [1, 2, 2]]))
        graph = OneHotDegree(max_degree=2)(graph)
        graph.node_features
        # mx.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    """

    def __init__(self, max_degree: int, in_degree: bool = False, cat: bool = True):
        self.max_degree = max_degree
        self.in_degree = in_degree
        self.cat = cat

    def process(self, data: GraphData) -> GraphData:
        """Adds the one-hot degrees to the node features of ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with the updated ``node_features``.
        """
        num_nodes = _num_nodes(data)
        index = _host_edges(data)[1 if self.in_degree else 0]
        degrees = np.bincount(index, minlength=num_nodes)
        if degrees.max(initial=0) > self.max_degree:
            raise ValueError(
                f"The graph has a node of degree {int(degrees.max())}, which is "
                f"greater than max_degree={self.max_degree}"
            )
        encoding = np.zeros((num_nodes, self.max_degree + 1), dtype=np.float32)
        encoding[np.arange(num_nodes), degrees] = 1.0
        return _add_node_features(data, mx.array(encoding), self.cat)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.max_degree})"


class Constant(BaseTransform):
    """Adds a constant value to the node features of each node.

    This is a common way to give features to the nodes of a graph that has
    none.

    Args:
        value: The value to add. Default: ``1.0``
        cat: If ``True``, the value is appended as a new column to the
            existing node features, otherwise it replaces them.
            Default: ``True``

    Example:

    .. code-block:: python

        graph = GraphData(edge_index=mx.array([[0, 1], [1, 2]]))
        graph = Constant()(graph)
        graph.node_features
        # mx.array([[1.0], [1.0], [1.0]])
    """

    def __init__(self, value: float = 1.0, cat: bool = True):
        self.value = value
        self.cat = cat

    def process(self, data: GraphData) -> GraphData:
        """Adds the constant column to the node features of ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with the updated ``node_features``.
        """
        constant = mx.full((_num_nodes(data), 1), self.value, dtype=mx.float32)
        return _add_node_features(data, constant, self.cat)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value={self.value})"


class AddRandomWalkPE(BaseTransform):
    r"""Adds the random walk positional encoding of each node, from
    `"Graph Neural Networks with Learnable Structural and Positional
    Representations" <https://arxiv.org/abs/2110.07875>`_.

    The encoding of a node holds its probabilities of being back at its
    starting point after :math:`1, 2, \ldots,` ``walk_length`` steps of a
    random walk, i.e. the diagonals of the powers of the transition matrix
    :math:`\mathbf{D}^{-1} \mathbf{A}`, where :math:`\mathbf{D}` is the
    out-degree. The ``edge_weights`` attribute of the graph is used as
    :math:`\mathbf{A}` if it exists: the walk then follows an edge with a
    probability proportional to its weight. Duplicate edges are summed.

    The encoding is computed on the host with NumPy: with dense matrices for
    graphs of up to ``DENSE_THRESHOLD`` nodes, and otherwise with ``scipy``
    sparse products on blocks of nodes, which keeps the memory bounded.

    Args:
        walk_length: Number of random walk steps, which is the size of the
            encoding.
        attr_name: Name of the attribute in which the ``[num_nodes,
            walk_length]`` float32 encoding is stored. If ``None``, the
            encoding is appended to the node features instead.
            Default: ``"random_walk_pe"``

    Example:

    .. code-block:: python

        graph = GraphData(edge_index=mx.array([[0, 1, 1, 2], [1, 0, 2, 1]]))
        graph = AddRandomWalkPE(walk_length=2)(graph)
        graph.random_walk_pe
        # mx.array([[0, 0.5], [0, 1], [0, 0.5]])
    """

    # Number of nodes above which the sparse, blocked computation is used.
    DENSE_THRESHOLD: int = 2000
    # Number of entries of the dense blocks of the sparse computation.
    BLOCK_ENTRIES: int = 2**23

    def __init__(self, walk_length: int, attr_name: Optional[str] = "random_walk_pe"):
        self.walk_length = walk_length
        self.attr_name = attr_name

    def process(self, data: GraphData) -> GraphData:
        """Adds the random walk positional encoding to ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with the encoding.
        """
        num_nodes = _num_nodes(data)
        src, dst = _host_edges(data)
        weights = _edge_weights(data)
        out_degree = _weighted_bincount(src, weights, num_nodes)
        transition = np.zeros_like(weights)
        np.divide(weights, out_degree[src], out=transition, where=out_degree[src] != 0)

        if num_nodes <= self.DENSE_THRESHOLD:
            encoding = self._dense(src, dst, transition, num_nodes)
        else:
            encoding = self._sparse(src, dst, transition, num_nodes)
        encoding = mx.array(encoding.astype(np.float32))

        if self.attr_name is None:
            return _add_node_features(data, encoding, cat=True)
        setattr(data, self.attr_name, encoding)
        return data

    def _dense(
        self, src: np.ndarray, dst: np.ndarray, transition: np.ndarray, num_nodes: int
    ) -> np.ndarray:
        # Duplicate edges are summed.
        matrix = _weighted_bincount(src * num_nodes + dst, transition, num_nodes**2)
        matrix = matrix.reshape(num_nodes, num_nodes)
        encoding = np.zeros((num_nodes, self.walk_length), dtype=np.float64)
        power = matrix
        for step in range(self.walk_length):
            if step > 0:
                power = power @ matrix
            encoding[:, step] = np.diagonal(power)
        return encoding

    def _sparse(
        self, src: np.ndarray, dst: np.ndarray, transition: np.ndarray, num_nodes: int
    ) -> np.ndarray:
        try:
            import scipy.sparse as sp
        except ImportError as error:
            raise ImportError(
                "scipy is required by AddRandomWalkPE for graphs of more than "
                f"{self.DENSE_THRESHOLD} nodes, run `pip install scipy`"
            ) from error

        # Duplicate edges are summed.
        matrix = sp.csr_matrix((transition, (src, dst)), shape=(num_nodes, num_nodes))
        encoding = np.zeros((num_nodes, self.walk_length), dtype=np.float64)
        block_size = max(1, self.BLOCK_ENTRIES // num_nodes)
        for start in range(0, num_nodes, block_size):
            # Columns `block` of the successive powers of the transition matrix.
            block = np.arange(start, min(start + block_size, num_nodes))
            local = np.arange(block.size)
            columns = np.zeros((num_nodes, block.size), dtype=np.float64)
            columns[block, local] = 1.0
            for step in range(self.walk_length):
                columns = matrix @ columns
                encoding[block, step] = columns[block, local]
        return encoding

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(walk_length={self.walk_length})"


class GCNNorm(BaseTransform):
    r"""Applies the GCN normalization from `"Semi-supervised Classification
    with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_
    to the edges of a graph and stores the result in ``edge_weights``:

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}

    where :math:`\hat{D}_{ii} = \sum_{j} (\mathbf{A} + \mathbf{I})_{ji}` is the
    weighted in-degree. The ``edge_weights`` attribute of the graph is used as
    :math:`\mathbf{A}` if it exists.

    With ``add_self_loops``, every node gets exactly one self-loop, placed
    after the other edges: a node that already has a self-loop keeps its
    weight, the others get a self-loop of weight 1. The other per-edge
    attributes (e.g. ``edge_features``) are kept aligned with the new
    ``edge_index``: existing self-loops keep their values and the added ones
    get zeros.

    Args:
        add_self_loops: Whether to add self-loops before normalizing.
            Default: ``True``

    Example:

    .. code-block:: python

        graph = GraphData(edge_index=mx.array([[0, 1], [1, 0]]))
        graph = GCNNorm()(graph)
        graph.edge_index
        # mx.array([[0, 1, 0, 1], [1, 0, 0, 1]])
        graph.edge_weights
        # mx.array([0.5, 0.5, 0.5, 0.5])
    """

    def __init__(self, add_self_loops: bool = True):
        self.add_self_loops = add_self_loops

    def process(self, data: GraphData) -> GraphData:
        """Normalizes the edges of ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with the normalized ``edge_weights`` (and the self-loops
            if requested).
        """
        num_nodes = _num_nodes(data)
        edges = _host_edges(data)
        weights = _edge_weights(data)

        if self.add_self_loops:
            _, edge_keys = _attribute_names(data, num_nodes)
            is_loop = edges[0] == edges[1]
            kept = np.nonzero(~is_loop)[0]
            loops = np.nonzero(is_loop)[0]
            loop_nodes = edges[0][loops]

            for key in edge_keys:
                if key == "edge_weights":
                    continue
                values = getattr(data, key)
                loop_values = mx.zeros((num_nodes, *values.shape[1:]), values.dtype)
                if loops.size > 0:
                    loop_values[mx.array(loop_nodes)] = values[mx.array(loops)]
                if kept.size > 0:
                    loop_values = mx.concatenate(
                        [values[mx.array(kept)], loop_values], axis=0
                    )
                setattr(data, key, loop_values)

            loop_weights = np.ones(num_nodes, dtype=np.float64)
            loop_weights[loop_nodes] = weights[loops]
            weights = np.concatenate([weights[kept], loop_weights])
            nodes = np.arange(num_nodes, dtype=np.int64)
            edges = np.concatenate([edges[:, kept], np.stack([nodes, nodes])], axis=1)
            data.edge_index = mx.array(edges, dtype=data.edge_index.dtype)
            data.num_nodes = num_nodes

        in_degree = _weighted_bincount(edges[1], weights, num_nodes)
        deg_inv_sqrt = np.zeros_like(in_degree)
        np.power(in_degree, -0.5, out=deg_inv_sqrt, where=in_degree > 0)
        weights = deg_inv_sqrt[edges[0]] * weights * deg_inv_sqrt[edges[1]]
        data.edge_weights = mx.array(weights.astype(np.float32))
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(add_self_loops={self.add_self_loops})"
