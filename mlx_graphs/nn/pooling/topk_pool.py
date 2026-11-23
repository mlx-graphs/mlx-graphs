from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.pooling.select import _pool, _SelectTopK


class TopKPooling(nn.Module):
    r""":math:`\mathrm{top}_k` pooling operator from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_, `"Towards Sparse Hierarchical Graph
    Classifiers" <https://arxiv.org/abs/1811.01287>`_ and `"Understanding
    Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>`_ papers.

    If ``min_score`` :math:`\tilde{\alpha}` is ``None``, computes:

    .. math::
        \mathbf{y} &= \sigma \left( \frac{\mathbf{X}\mathbf{p}}{\|
        \mathbf{p} \|} \right)

        \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

        \mathbf{X}^{\prime} &= (\mathbf{X} \odot
        \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

        \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    If ``min_score`` :math:`\tilde{\alpha}` is a value in :math:`[0, 1]`,
    computes:

    .. math::
        \mathbf{y} &= \mathrm{softmax}(\mathbf{X}\mathbf{p})

        \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

        \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

        \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.

    Args:
        node_features_dim: Size of the input node features.
        ratio: The graph pooling ratio, used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil` for a float, or
            :math:`k` itself for an integer. Ignored if ``min_score`` is not
            ``None``. Default: ``0.5``
        min_score: Minimal node score :math:`\tilde{\alpha}` of the kept nodes.
            When given, ``ratio`` is ignored. Default: ``None``
        multiplier: Coefficient by which the features are multiplied after
            pooling. This can be useful for large graphs and when ``min_score``
            is used. Default: ``1.0``
        nonlinearity: The non-linearity :math:`\sigma`, as a callable or the
            name of a function of ``mlx.core`` or ``mlx.nn``. Default: ``"tanh"``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import TopKPooling

        pool = TopKPooling(16, ratio=0.5)
        edge_index = mx.array([[0, 1, 2, 3, 4], [1, 2, 0, 4, 3]])
        node_features = mx.random.normal((5, 16))
        batch_indices = mx.array([0, 0, 0, 1, 1])

        x, edge_index, edge_features, batch_indices, perm, score = pool(
            edge_index, node_features, batch_indices=batch_indices
        )
        >>> x.shape
        (3, 16)
    """

    def __init__(
        self,
        node_features_dim: int,
        ratio: Union[int, float] = 0.5,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: Union[str, Callable] = "tanh",
    ):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.select = _SelectTopK(node_features_dim, ratio, min_score, nonlinearity)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_features: Optional[mx.array] = None,
        batch_indices: Optional[mx.array] = None,
        attn: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array, Optional[mx.array], mx.array, mx.array, mx.array]:
        """Pools the nodes with the best scores.

        Args:
            edge_index: Edges of shape ``[2, num_edges]``.
            node_features: Node features of shape ``[num_nodes, node_features_dim]``.
            edge_features: Optional edge features of shape ``[num_edges, ...]``.
            batch_indices: Graph of each node, of shape ``[num_nodes]``.
                Default: ``None`` (a single graph).
            attn: Optional node-level array used to compute the scores instead
                of ``node_features``. Default: ``None``

        Returns:
            A tuple ``(node_features, edge_index, edge_features, batch_indices,
            perm, score)`` with the pooled graph, the indices ``perm`` of the
            kept nodes and their ``score``.
        """
        if batch_indices is None:
            batch_indices = mx.zeros((node_features.shape[0],), dtype=mx.int32)
        attn = node_features if attn is None else attn
        perm, score = self.select(attn, batch_indices)
        return _pool(
            node_features,
            edge_index,
            edge_features,
            batch_indices,
            perm,
            score,
            self.multiplier,
        )

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f"ratio={self.ratio}"
        else:
            ratio = f"min_score={self.min_score}"
        return (
            f"{self.__class__.__name__}({self.node_features_dim}, {ratio}, "
            f"multiplier={self.multiplier})"
        )
