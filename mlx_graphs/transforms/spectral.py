from typing import Any, Optional

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.transforms.base_transform import BaseTransform
from mlx_graphs.transforms.features import _add_node_features, _edge_weights
from mlx_graphs.transforms.structure import _num_nodes
from mlx_graphs.utils.laplacian import get_laplacian


class AddLaplacianEigenvectorPE(BaseTransform):
    r"""Adds the Laplacian eigenvector positional encoding from `"Benchmarking
    Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_ to each node,
    as PyG's ``AddLaplacianEigenvectorPE``.

    The encoding holds the eigenvectors of the ``k`` smallest non-trivial
    eigenvalues of the symmetrically normalized Laplacian
    :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
    \mathbf{D}^{-1/2}` (the eigenvector of the smallest eigenvalue is
    skipped). The ``edge_weights`` attribute is used as :math:`\mathbf{A}` if
    it exists. As eigenvectors are only defined up to their sign, the sign of
    each one is flipped at random (from ``mx.random``, so reproducible with
    ``mx.random.seed``), as in PyG.

    The eigenvectors are computed on the host: with NumPy for graphs of less
    than ``SPARSE_THRESHOLD`` nodes, and with ``scipy.sparse.linalg`` (which
    is then required) otherwise.

    Args:
        k: Number of non-trivial eigenvectors, which is the size of the
            encoding. Must be smaller than the number of nodes.
        attr_name: Name of the attribute in which the ``[num_nodes, k]``
            float32 encoding is stored. If ``None``, the encoding is appended
            to the node features instead.
            Default: ``"laplacian_eigenvector_pe"``
        is_undirected: If ``True``, the graph is assumed to be undirected,
            which allows a faster symmetric eigensolver. Default: ``False``
        **kwargs: Additional arguments of ``scipy.sparse.linalg.eigs`` (or
            ``eigsh`` if ``is_undirected``), used for large graphs.

    Example:

    .. code-block:: python

        graph = GraphData(edge_index=mx.array([[0, 1, 1, 2], [1, 0, 2, 1]]))
        graph = AddLaplacianEigenvectorPE(k=1, is_undirected=True)(graph)
        graph.laplacian_eigenvector_pe
        # mx.array([[0.7071], [0.0], [-0.7071]]) (up to the sign)
    """

    # Number of nodes from which the sparse eigensolver is used.
    SPARSE_THRESHOLD: int = 100

    def __init__(
        self,
        k: int,
        attr_name: Optional[str] = "laplacian_eigenvector_pe",
        is_undirected: bool = False,
        **kwargs: Any,
    ):
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self.kwargs = kwargs

    def process(self, data: GraphData) -> GraphData:
        """Adds the Laplacian eigenvector positional encoding to ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with the encoding.
        """
        num_nodes = _num_nodes(data)
        if self.k >= num_nodes:
            raise ValueError(
                f"k={self.k} must be smaller than the number of nodes ({num_nodes})"
            )
        weights = mx.array(_edge_weights(data).astype(np.float32))
        edge_index, values = get_laplacian(
            data.edge_index, weights, normalization="sym", num_nodes=num_nodes
        )
        rows, cols = np.asarray(edge_index).astype(np.int64)
        values = np.asarray(values).astype(np.float64)

        if num_nodes < self.SPARSE_THRESHOLD:
            laplacian = np.zeros((num_nodes, num_nodes), dtype=np.float64)
            np.add.at(laplacian, (rows, cols), values)
            eig_fn = np.linalg.eigh if self.is_undirected else np.linalg.eig
            eig_vals, eig_vecs = eig_fn(laplacian)
        else:
            try:
                import scipy.sparse as sp
                from scipy.sparse.linalg import eigs, eigsh
            except ImportError as error:
                raise ImportError(
                    "scipy is required by AddLaplacianEigenvectorPE for graphs of "
                    f"{self.SPARSE_THRESHOLD} nodes or more, run `pip install scipy`"
                ) from error
            # float32, as in PyG: the Laplacian values are float32 (rounded),
            # so in float64 ARPACK has to separate clusters of eigenvalues
            # that differ only by that rounding, e.g. the zero eigenvalue of
            # every connected component, and fails to converge (on Cora).
            laplacian = sp.coo_matrix(
                (values.astype(np.float32), (rows, cols)),
                shape=(num_nodes, num_nodes),
            ).tocsr()
            eig_fn = eigsh if self.is_undirected else eigs
            eig_vals, eig_vecs = eig_fn(
                laplacian,
                k=self.k + 1,
                which="SA" if self.is_undirected else "SR",
                return_eigenvectors=True,
                **self.kwargs,
            )

        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        encoding = eig_vecs[:, 1 : self.k + 1].astype(np.float32)
        sign = 2 * mx.random.randint(0, 2, (self.k,)) - 1
        encoding = mx.array(encoding) * sign.astype(mx.float32)

        if self.attr_name is None:
            return _add_node_features(data, encoding, cat=True)
        setattr(data, self.attr_name, encoding)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"


class SVDFeatureReduction(BaseTransform):
    r"""Reduces the dimension of the node features with a Singular Value
    Decomposition, as PyG's ``SVDFeatureReduction``.

    If the node features :math:`\mathbf{X}` have more than ``out_channels``
    columns, they are replaced by :math:`\mathbf{U}_{:, :k}
    \mathrm{diag}(\mathbf{S}_{:k})`, where :math:`\mathbf{X} = \mathbf{U}
    \mathrm{diag}(\mathbf{S}) \mathbf{V}^\top` and :math:`k` is
    ``out_channels``. The SVD is computed on the host with NumPy. Singular
    vectors are only defined up to their sign, so columns may have the
    opposite sign of PyG's.

    Args:
        out_channels: Number of node features after the reduction.

    Example:

    .. code-block:: python

        graph = GraphData(
            edge_index=mx.array([[0, 1], [1, 2]]),
            node_features=mx.random.normal((3, 8)),
        )
        graph = SVDFeatureReduction(2)(graph)
        graph.node_features.shape
        # (3, 2)
    """

    def __init__(self, out_channels: int):
        self.out_channels = out_channels

    def process(self, data: GraphData) -> GraphData:
        """Reduces the node features of ``data``.

        Args:
            data: The graph to transform.

        Returns:
            The graph with the reduced ``node_features``.
        """
        features = data.node_features
        if features is None:
            raise ValueError("SVDFeatureReduction requires node features")
        if features.shape[-1] > self.out_channels:
            matrix = np.asarray(features.astype(mx.float32)).astype(np.float64)
            u, s, _ = np.linalg.svd(matrix, full_matrices=False)
            k = self.out_channels
            reduced = u[:, :k] * s[:k]
            data.node_features = mx.array(reduced.astype(np.float32))
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.out_channels})"
