"""Dense (soft cluster assignment) pooling: DiffPool, MinCutPool and DMoN.

As the dense convolution layers (e.g. :class:`~mlx_graphs.nn.DenseGCNConv`),
they take batched dense adjacency matrices ``[batch_size, num_nodes,
num_nodes]`` and node features ``[batch_size, num_nodes, num_features]``, with
the adjacency first.
"""

from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

EPS = 1e-15


def _selu(x: mx.array) -> mx.array:
    """SELU whose gradient stays finite for large inputs (``mlx.nn.selu``
    computes ``exp(x)`` on both branches, whose gradient overflows)."""
    alpha, scale = 1.6732632423543772848170429916717, 1.0507009873554804934193349852946
    return scale * mx.where(x > 0, x, alpha * (mx.exp(mx.minimum(x, 0.0)) - 1))


def _batched(*arrays: mx.array) -> list[mx.array]:
    return [a[None] if a.ndim == 2 else a for a in arrays]


def _frobenius(x: mx.array, axes=None) -> mx.array:
    return mx.sqrt((x * x).sum(axis=axes))


def _trace(x: mx.array) -> mx.array:
    return mx.diagonal(x, axis1=-2, axis2=-1).sum(axis=-1)


def _ortho_loss(s: mx.array) -> mx.array:
    r"""Mean of :math:`\| \frac{S^T S}{\|S^T S\|_F} - \frac{I}{\|I\|_F} \|_F`."""
    k = s.shape[-1]
    ss = s.swapaxes(1, 2) @ s
    i_s = mx.eye(k, dtype=ss.dtype)
    ss_norm = _frobenius(ss, axes=(-2, -1))[:, None, None]
    return _frobenius(ss / ss_norm - i_s / k**0.5, axes=(-2, -1)).mean()


def _normalize_pooled_adj(out_adj: mx.array) -> mx.array:
    """Zeroes the diagonal and symmetrically normalizes by the degrees."""
    k = out_adj.shape[-1]
    out_adj = out_adj * (1 - mx.eye(k, dtype=out_adj.dtype))
    d = mx.sqrt(out_adj.sum(axis=-1))[:, None] + EPS  # [B, 1, K]
    return (out_adj / d) / d.swapaxes(1, 2)


def dense_diff_pool(
    adj: mx.array,
    node_features: mx.array,
    s: mx.array,
    mask: Optional[mx.array] = None,
    normalize: bool = True,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    r"""The differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper.

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on the dense learned assignments :math:`\mathbf{S} \in
    \mathbb{R}^{B \times N \times C}`. Also returns the auxiliary link
    prediction loss

    .. math::
        \mathcal{L}_{LP} = {\| \mathbf{A} - \mathrm{softmax}(\mathbf{S})
        {\mathrm{softmax}(\mathbf{S})}^{\top} \|}_F,

    divided by the number of entries of :math:`\mathbf{A}` if ``normalize``,
    and the entropy regularization
    :math:`\mathcal{L}_E = \frac{1}{B N} \sum_{b, n} H(\mathbf{S}_{b, n})`.

    Args:
        adj: Adjacency matrices of shape ``[batch_size, num_nodes,
            num_nodes]`` (or ``[num_nodes, num_nodes]``).
        node_features: Node features of shape ``[batch_size, num_nodes,
            num_features]`` (or ``[num_nodes, num_features]``).
        s: Assignment logits of shape ``[batch_size, num_nodes,
            num_clusters]`` (or ``[num_nodes, num_clusters]``). The softmax
            is applied here.
        mask: Optional boolean mask of shape ``[batch_size, num_nodes]`` of
            the real (non-padding) nodes. Default: ``None``
        normalize: Whether to divide the link loss by the number of entries of
            ``adj``. Default: ``True``

    Returns:
        A tuple ``(node_features, adj, link_loss, entropy_loss)`` with the
        pooled features ``[batch_size, num_clusters, num_features]`` and
        adjacency ``[batch_size, num_clusters, num_clusters]``.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import dense_diff_pool

        adj = (mx.random.uniform(shape=(2, 10, 10)) > 0.7).astype(mx.float32)
        x = mx.random.normal((2, 10, 16))
        s = mx.random.normal((2, 10, 3))
        x, adj, link_loss, ent_loss = dense_diff_pool(adj, x, s)
        >>> x.shape, adj.shape
        ((2, 3, 16), (2, 3, 3))
    """
    adj, x, s = _batched(adj, node_features, s)
    batch_size, num_nodes, _ = x.shape

    s = mx.softmax(s, axis=-1)
    if mask is not None:
        mask = mask.reshape(batch_size, num_nodes, 1).astype(x.dtype)
        x, s = x * mask, s * mask

    s_t = s.swapaxes(1, 2)
    out = s_t @ x
    out_adj = (s_t @ adj) @ s

    link_loss = _frobenius(adj - s @ s_t)
    if normalize:
        link_loss = link_loss / adj.size

    ent_loss = (-s * mx.log(s + EPS)).sum(axis=-1).mean()
    return out, out_adj, link_loss, ent_loss


def dense_mincut_pool(
    adj: mx.array,
    node_features: mx.array,
    s: mx.array,
    mask: Optional[mx.array] = None,
    temp: float = 1.0,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    r"""The MinCut pooling operator from the `"Spectral Clustering in Graph
    Neural Networks for Graph Pooling" <https://arxiv.org/abs/1907.00481>`_
    paper.

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on the dense learned assignments :math:`\mathbf{S} \in
    \mathbb{R}^{B \times N \times C}`. Also returns the MinCut loss

    .. math::
        \mathcal{L}_c = - \frac{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{A}
        \mathbf{S})} {\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{D}
        \mathbf{S})}

    where :math:`\mathbf{D}` is the degree matrix, and the orthogonality loss

    .. math::
        \mathcal{L}_o = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
        {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_C}{\sqrt{C}}
        \right\|}_F.

    As in PyG, the returned adjacency has a zero diagonal and is
    symmetrically normalized by its degrees.

    Args:
        adj: Adjacency matrices of shape ``[batch_size, num_nodes,
            num_nodes]`` (or ``[num_nodes, num_nodes]``).
        node_features: Node features of shape ``[batch_size, num_nodes,
            num_features]`` (or ``[num_nodes, num_features]``).
        s: Assignment logits of shape ``[batch_size, num_nodes,
            num_clusters]`` (or ``[num_nodes, num_clusters]``). The softmax
            is applied here.
        mask: Optional boolean mask of shape ``[batch_size, num_nodes]`` of
            the real (non-padding) nodes. Default: ``None``
        temp: Temperature of the softmax. Default: ``1.0``

    Returns:
        A tuple ``(node_features, adj, mincut_loss, ortho_loss)``.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import dense_mincut_pool

        adj = (mx.random.uniform(shape=(2, 10, 10)) > 0.7).astype(mx.float32)
        x = mx.random.normal((2, 10, 16))
        s = mx.random.normal((2, 10, 3))
        x, adj, mincut_loss, ortho_loss = dense_mincut_pool(adj, x, s)
    """
    adj, x, s = _batched(adj, node_features, s)
    batch_size, num_nodes, _ = x.shape

    s = mx.softmax(s / temp if temp != 1.0 else s, axis=-1)
    if mask is not None:
        mask = mask.reshape(batch_size, num_nodes, 1).astype(x.dtype)
        x, s = x * mask, s * mask

    s_t = s.swapaxes(1, 2)
    out = s_t @ x
    out_adj = (s_t @ adj) @ s

    mincut_num = _trace(out_adj)
    # Tr(S^T D S) with D = diag(rowsum(A))
    degrees = adj.sum(axis=-1)
    mincut_den = (degrees[..., None] * s * s).sum(axis=(1, 2))
    mincut_loss = (-(mincut_num / mincut_den)).mean()

    ortho_loss = _ortho_loss(s)
    return out, _normalize_pooled_adj(out_adj), mincut_loss, ortho_loss


class DMoNPooling(nn.Module):
    r"""The spectral modularity pooling operator from the `"Graph Clustering
    with Graph Neural Networks" <https://arxiv.org/abs/2006.16904>`_ paper.

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    where the assignments :math:`\mathbf{S}` are computed by an MLP from the
    node features. Also returns the spectral (modularity) loss

    .. math::
        \mathcal{L}_s = - \frac{1}{2m}
        \cdot{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{B} \mathbf{S})},

    where :math:`\mathbf{B}` is the modularity matrix, the orthogonality loss
    (see :func:`dense_mincut_pool`) and the cluster loss

    .. math::
        \mathcal{L}_c = \frac{\sqrt{C}}{n}
        {\left\|\sum_i\mathbf{C_i}^{\top}\right\|}_F - 1.

    The output features go through a SELU activation, and the output
    adjacency is normalized as in :func:`dense_mincut_pool`. As in PyG (2.8),
    the cluster loss of a batch is the mean of the cluster norms times the
    mean of the inverse graph sizes.

    Args:
        node_features_dim: Size of the input node features, or a list of
            sizes ``[input, hidden_1, ...]`` of the MLP computing the
            assignments (which ends with ``k``).
        k: Number of clusters.
        dropout: Dropout probability of the assignment logits.
            Default: ``0.0``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import DMoNPooling

        pool = DMoNPooling(16, k=4)
        adj = (mx.random.uniform(shape=(2, 10, 10)) > 0.7).astype(mx.float32)
        x = mx.random.normal((2, 10, 16))
        s, x, adj, spectral_loss, ortho_loss, cluster_loss = pool(adj, x)
        >>> x.shape
        (2, 4, 16)
    """

    def __init__(
        self, node_features_dim: Union[int, List[int]], k: int, dropout: float = 0.0
    ):
        super().__init__()
        from mlx_graphs.nn.models.basic_gnn import MLP

        if isinstance(node_features_dim, int):
            node_features_dim = [node_features_dim]
        self.node_features_dim = list(node_features_dim)
        self.k = k
        self.mlp = MLP(self.node_features_dim + [k], act=None, norm=None)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def __call__(
        self,
        adj: mx.array,
        node_features: mx.array,
        mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Pools the graphs.

        Args:
            adj: Adjacency matrices of shape ``[batch_size, num_nodes,
                num_nodes]`` (or ``[num_nodes, num_nodes]``).
            node_features: Node features of shape ``[batch_size, num_nodes,
                num_features]`` (or ``[num_nodes, num_features]``).
            mask: Optional boolean mask of shape ``[batch_size, num_nodes]``
                of the real (non-padding) nodes. Default: ``None``

        Returns:
            A tuple ``(s, node_features, adj, spectral_loss, ortho_loss,
            cluster_loss)`` with the soft assignments ``s`` of shape
            ``[batch_size, num_nodes, k]`` and the pooled graphs.
        """
        adj, x = _batched(adj, node_features)
        s = self.mlp(x)
        if self.dropout is not None:
            s = self.dropout(s)
        s = mx.softmax(s, axis=-1)

        batch_size, num_nodes, _ = x.shape
        C = s.shape[-1]
        if mask is None:
            mask = mx.ones((batch_size, num_nodes), dtype=x.dtype)
        mask = mask.reshape(batch_size, num_nodes, 1).astype(x.dtype)
        x, s = x * mask, s * mask

        s_t = s.swapaxes(1, 2)
        out = _selu(s_t @ x)
        out_adj = (s_t @ adj) @ s

        degrees = adj.sum(axis=-1)[..., None] * mask  # [B, N, 1]
        m = degrees.sum(axis=(1, 2)) / 2  # [B]
        ca = s_t @ degrees  # [B, C, 1]
        cb = degrees.swapaxes(1, 2) @ s  # [B, 1, C]
        normalizer = (ca @ cb) / 2 / m[:, None, None]
        spectral_loss = (-_trace(out_adj - normalizer) / 2 / m).mean()

        ortho_loss = _ortho_loss(s)

        cluster_size = s.sum(axis=1)  # [B, C]
        cluster_norm = _frobenius(cluster_size, axes=1)  # [B]
        # [B] / [B, 1] broadcasts to [B, B], as in PyG
        cluster_loss = cluster_norm[None, :] / mask.sum(axis=1) * C**0.5 - 1
        cluster_loss = cluster_loss.mean()

        return (
            s,
            out,
            _normalize_pooled_adj(out_adj),
            spectral_loss,
            ortho_loss,
            cluster_loss,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.node_features_dim[0]}, "
            f"num_clusters={self.k})"
        )
