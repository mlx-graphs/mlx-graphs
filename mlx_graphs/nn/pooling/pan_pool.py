from typing import Callable, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.pooling.select import _SelectTopK, filter_adj
from mlx_graphs.utils.scatter import scatter


class PANPooling(nn.Module):
    r"""The path integral based pooling operator from the `"Path Integral
    Based Convolution and Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/2006.16811>`_ paper.

    PAN pooling performs top-:math:`k` pooling where global node importance
    is measured based on node features and the MET matrix:

    .. math::
        {\rm score} = \beta_1 \mathbf{X} \cdot \mathbf{p} + \beta_2
        {\rm deg}(\mathbf{M})

    where :math:`{\rm deg}(\mathbf{M})_j` sums the column :math:`j` of
    :math:`\mathbf{M}`. The kept nodes are scaled by their (normalized)
    score, and the pooled graph keeps the entries of :math:`\mathbf{M}`
    between them, as edges ``j -> i`` weighted by :math:`\mathbf{M}_{i,j}`.

    Args:
        node_features_dim: Size of the input node features.
        ratio: The graph pooling ratio, used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil` for a float, or
            :math:`k` itself for an integer. Ignored if ``min_score`` is not
            ``None``. Default: ``0.5``
        min_score: Minimal node score of the kept nodes. When given,
            ``ratio`` is ignored. Default: ``None``
        multiplier: Coefficient by which the features are multiplied after
            pooling. Default: ``1.0``
        nonlinearity: The non-linearity of the scores, as a callable or the
            name of a function of ``mlx.core`` or ``mlx.nn``. Default:
            ``"tanh"``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import PANConv, PANPooling

        conv = PANConv(16, 32, filter_size=2)
        pool = PANPooling(32, ratio=0.5)
        edge_index = mx.array([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
        batch_indices = mx.array([0, 0, 0, 1, 1])
        x, (m_index, m_weights) = conv(edge_index, mx.random.normal((5, 16)))
        x, edge_index, edge_weights, batch_indices, perm, score = pool(
            m_index, x, m_weights, batch_indices
        )
        >>> x.shape
        (3, 32)
    """

    def __init__(
        self,
        node_features_dim: int,
        ratio: Union[float, int] = 0.5,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: Union[str, Callable] = "tanh",
    ):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.p = mx.ones((node_features_dim,))
        self.beta = mx.full((2,), 0.5)
        self.select = _SelectTopK(1, ratio, min_score, nonlinearity)

    def __call__(
        self,
        m_index: mx.array,
        node_features: mx.array,
        m_weights: mx.array,
        batch_indices: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Pools the graphs.

        Args:
            m_index: ``[2, nnz]`` (row, column) indices of the MET matrix
                :math:`\\mathbf{M}` returned by :class:`~mlx_graphs.nn.PANConv`.
            node_features: Node features of shape ``[num_nodes,
                node_features_dim]``.
            m_weights: The ``[nnz]`` values of :math:`\\mathbf{M}`.
            batch_indices: Graph of each node, of shape ``[num_nodes]``.
                Default: ``None`` (a single graph).

        Returns:
            A tuple ``(node_features, edge_index, edge_weights,
            batch_indices, perm, score)`` with the pooled graph, the indices
            ``perm`` of the kept nodes and their ``score``.
        """
        x = node_features
        num_nodes = x.shape[0]
        if batch_indices is None:
            batch_indices = mx.zeros((num_nodes,), dtype=mx.int32)
        row, col = m_index[0], m_index[1]

        score1 = (x * self.p).sum(axis=-1)
        score2 = scatter(m_weights, col, num_nodes, "add")
        score = self.beta[0] * score1 + self.beta[1] * score2

        perm, score = self.select(score, batch_indices)
        x = x[perm] * score[:, None]
        if self.multiplier != 1:
            x = self.multiplier * x

        edge_index, edge_weights = filter_adj(
            mx.stack([col, row]), m_weights, perm, num_nodes=num_nodes
        )
        return x, edge_index, edge_weights, batch_indices[perm], perm, score

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f"ratio={self.ratio}"
        else:
            ratio = f"min_score={self.min_score}"
        return (
            f"{self.__class__.__name__}({self.node_features_dim}, {ratio}, "
            f"multiplier={self.multiplier})"
        )
