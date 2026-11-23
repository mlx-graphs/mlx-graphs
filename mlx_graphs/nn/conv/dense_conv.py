"""Convolution layers on dense, batched adjacency matrices (PyG's
``torch_geometric.nn.dense``).

They take an adjacency matrix ``adj`` of shape ``[batch_size, num_nodes,
num_nodes]`` and node features ``node_features`` of shape ``[batch_size,
num_nodes, node_features_dim]`` (a 2D ``adj`` / ``node_features`` is a batch of
one graph), e.g. from :func:`~mlx_graphs.utils.to_dense_adj` and
:func:`~mlx_graphs.utils.to_dense_batch`, a
:class:`~mlx_graphs.loaders.DenseDataLoader`, or the output of
:func:`~mlx_graphs.nn.dense_diff_pool`. As the other layers of mlx-graphs, the
graph structure (``adj``) comes first.
"""

from typing import Literal, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.linear import Linear


def _batched(adj: mx.array, node_features: mx.array) -> tuple[mx.array, mx.array]:
    if node_features.ndim == 2:
        node_features = node_features[None]
    if adj.ndim == 2:
        adj = adj[None]
    return adj, node_features


def _apply_mask(out: mx.array, mask: Optional[mx.array]) -> mx.array:
    if mask is None:
        return out
    return out * mask.reshape(out.shape[0], out.shape[1], 1).astype(out.dtype)


def _set_diagonal(adj: mx.array, value: float) -> mx.array:
    eye = mx.eye(adj.shape[-1], dtype=adj.dtype)
    return adj * (1 - eye) + value * eye


class DenseGCNConv(nn.Module):
    r"""Dense version of :class:`~mlx_graphs.nn.GCNConv`.

    .. math::
        \mathbf{X}^{\prime} = \hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}}
        \hat{\mathbf{D}}^{-1/2} \mathbf{X} \mathbf{\Theta} + \mathbf{b}

    where :math:`\hat{\mathbf{A}}` is ``adj`` whose diagonal is set to 1 (2 if
    ``improved``) when ``add_loop`` is ``True``, and :math:`\hat{\mathbf{D}}`
    its degree matrix (degrees are clamped to at least 1).

    Args:
        node_features_dim: Size of the input node features.
        out_features_dim: Size of the output node features.
        improved: Whether the self-loops have a weight of 2 instead of 1.
            Default: ``False``
        bias: Whether to add a bias. Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import DenseGCNConv

        conv = DenseGCNConv(16, 32)
        adj = (mx.random.uniform(shape=(2, 10, 10)) > 0.7).astype(mx.float32)
        node_features = mx.random.normal((2, 10, 16))
        >>> conv(adj, node_features).shape
        (2, 10, 32)
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        improved: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.improved = improved
        # glorot, as PyG's `Linear(weight_initializer="glorot")`
        self.lin = Linear(node_features_dim, out_features_dim, bias=False)
        if bias:
            self.bias = mx.zeros((out_features_dim,))

    def __call__(
        self,
        adj: mx.array,
        node_features: mx.array,
        mask: Optional[mx.array] = None,
        add_loop: bool = True,
    ) -> mx.array:
        """Computes the node embeddings.

        Args:
            adj: Adjacency matrices of shape ``[batch_size, num_nodes,
                num_nodes]`` (or ``[num_nodes, num_nodes]``), whose entry
                ``[b, i, j]`` is the weight of the edge from node ``i`` to node
                ``j`` of graph ``b``.
            node_features: Node features of shape ``[batch_size, num_nodes,
                node_features_dim]`` (or ``[num_nodes, node_features_dim]``).
            mask: Optional boolean mask of shape ``[batch_size, num_nodes]``
                of the real (non-padding) nodes. Default: ``None``
            add_loop: Whether to set the diagonal of ``adj`` to the self-loop
                weight. Default: ``True``

        Returns:
            The node embeddings, of shape ``[batch_size, num_nodes,
            out_features_dim]``.
        """
        adj, node_features = _batched(adj, node_features)
        if add_loop:
            adj = _set_diagonal(adj, 2.0 if self.improved else 1.0)

        out = self.lin(node_features)
        deg_inv_sqrt = mx.rsqrt(mx.maximum(adj.sum(axis=-1), 1.0))
        adj = deg_inv_sqrt[..., :, None] * adj * deg_inv_sqrt[..., None, :]
        out = adj @ out

        if "bias" in self:
            out = out + self.bias
        return _apply_mask(out, mask)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.node_features_dim}, "
            f"{self.out_features_dim})"
        )


class DenseSAGEConv(nn.Module):
    r"""Dense version of :class:`~mlx_graphs.nn.SAGEConv` (mean aggregation).

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_1 \mathbf{x}_i +
        \mathbf{\Theta}_2 \frac{\sum_{j} a_{i,j} \mathbf{x}_j}
        {\max(1, \sum_{j} a_{i,j})}

    Args:
        node_features_dim: Size of the input node features.
        out_features_dim: Size of the output node features.
        normalize: Whether to :math:`\ell_2`-normalize the output embeddings.
            Default: ``False``
        bias: Whether to add a bias. Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import DenseSAGEConv

        conv = DenseSAGEConv(16, 32)
        adj = (mx.random.uniform(shape=(2, 10, 10)) > 0.7).astype(mx.float32)
        node_features = mx.random.normal((2, 10, 16))
        >>> conv(adj, node_features).shape
        (2, 10, 32)
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        normalize: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.normalize = normalize
        # PyTorch's default initialization, as in PyG
        self.lin_rel = nn.Linear(node_features_dim, out_features_dim, bias=False)
        self.lin_root = nn.Linear(node_features_dim, out_features_dim, bias=bias)

    def __call__(
        self,
        adj: mx.array,
        node_features: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the node embeddings.

        Args:
            adj: Adjacency matrices of shape ``[batch_size, num_nodes,
                num_nodes]`` (or ``[num_nodes, num_nodes]``).
            node_features: Node features of shape ``[batch_size, num_nodes,
                node_features_dim]`` (or ``[num_nodes, node_features_dim]``).
            mask: Optional boolean mask of shape ``[batch_size, num_nodes]``
                of the real (non-padding) nodes. Default: ``None``

        Returns:
            The node embeddings, of shape ``[batch_size, num_nodes,
            out_features_dim]``.
        """
        adj, x = _batched(adj, node_features)
        out = adj @ x
        out = out / mx.maximum(adj.sum(axis=-1, keepdims=True), 1.0)
        out = self.lin_rel(out) + self.lin_root(x)
        if self.normalize:
            norm = mx.sqrt((out * out).sum(axis=-1, keepdims=True))
            out = out / mx.maximum(norm, 1e-12)
        return _apply_mask(out, mask)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.node_features_dim}, "
            f"{self.out_features_dim})"
        )


class DenseGINConv(nn.Module):
    r"""Dense version of :class:`~mlx_graphs.nn.GINConv`.

    .. math::
        \mathbf{X}^{\prime} = \textrm{MLP} \left( (\mathbf{A} + (1 + \epsilon)
        \mathbf{I}) \mathbf{X} \right)

    Args:
        mlp: The neural network applied to the aggregated features.
        eps: Initial value of :math:`\epsilon`. Default: ``0.0``
        learn_eps: Whether :math:`\epsilon` is a trainable parameter (PyG's
            ``train_eps``). Default: ``False``

    Example:

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_graphs.nn import DenseGINConv

        conv = DenseGINConv(nn.Sequential(nn.Linear(16, 32), nn.ReLU()))
        adj = (mx.random.uniform(shape=(2, 10, 10)) > 0.7).astype(mx.float32)
        node_features = mx.random.normal((2, 10, 16))
        >>> conv(adj, node_features).shape
        (2, 10, 32)
    """

    def __init__(self, mlp: nn.Module, eps: float = 0.0, learn_eps: bool = False):
        super().__init__()
        self.mlp = mlp
        self.initial_eps = eps
        self.eps = mx.array([eps])
        if not learn_eps:
            self.freeze(keys=["eps"], recurse=False)

    def __call__(
        self,
        adj: mx.array,
        node_features: mx.array,
        mask: Optional[mx.array] = None,
        add_loop: bool = True,
    ) -> mx.array:
        """Computes the node embeddings.

        Args:
            adj: Adjacency matrices of shape ``[batch_size, num_nodes,
                num_nodes]`` (or ``[num_nodes, num_nodes]``).
            node_features: Node features of shape ``[batch_size, num_nodes,
                node_features_dim]`` (or ``[num_nodes, node_features_dim]``).
            mask: Optional boolean mask of shape ``[batch_size, num_nodes]``
                of the real (non-padding) nodes. Default: ``None``
            add_loop: Whether to add the :math:`(1 + \\epsilon)`-weighted
                features of each node. Default: ``True``

        Returns:
            The node embeddings, of shape ``[batch_size, num_nodes, ...]``.
        """
        adj, x = _batched(adj, node_features)
        out = adj @ x
        if add_loop:
            out = (1 + self.eps) * x + out
        return _apply_mask(self.mlp(out), mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mlp={self.mlp})"


class DenseGraphConv(nn.Module):
    r"""Dense version of :class:`~mlx_graphs.nn.GraphConv`.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_1 \mathbf{x}_i +
        \mathbf{\Theta}_2 \bigoplus_{j} a_{i,j} \mathbf{x}_j

    With ``aggr="max"``, as in PyG, node :math:`i` takes the feature-wise
    maximum of the nodes :math:`j` with :math:`a_{j,i} \neq 0` (unweighted;
    0 when there are none).

    Args:
        node_features_dim: Size of the input node features.
        out_features_dim: Size of the output node features.
        aggr: The aggregation, ``"add"``, ``"mean"`` or ``"max"``.
            Default: ``"add"``
        bias: Whether to add a bias. Default: ``True``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import DenseGraphConv

        conv = DenseGraphConv(16, 32)
        adj = (mx.random.uniform(shape=(2, 10, 10)) > 0.7).astype(mx.float32)
        node_features = mx.random.normal((2, 10, 16))
        >>> conv(adj, node_features).shape
        (2, 10, 32)
    """

    def __init__(
        self,
        node_features_dim: int,
        out_features_dim: int,
        aggr: Literal["add", "mean", "max"] = "add",
        bias: bool = True,
    ):
        super().__init__()
        if aggr not in ("add", "mean", "max"):
            raise ValueError(f"aggr must be 'add', 'mean' or 'max', got {aggr!r}")
        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.aggr = aggr
        # PyTorch's default initialization, as in PyG
        self.lin_rel = nn.Linear(node_features_dim, out_features_dim, bias=bias)
        self.lin_root = nn.Linear(node_features_dim, out_features_dim, bias=False)

    def __call__(
        self,
        adj: mx.array,
        node_features: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the node embeddings.

        Args:
            adj: Adjacency matrices of shape ``[batch_size, num_nodes,
                num_nodes]`` (or ``[num_nodes, num_nodes]``).
            node_features: Node features of shape ``[batch_size, num_nodes,
                node_features_dim]`` (or ``[num_nodes, node_features_dim]``).
            mask: Optional boolean mask of shape ``[batch_size, num_nodes]``
                of the real (non-padding) nodes. Default: ``None``

        Returns:
            The node embeddings, of shape ``[batch_size, num_nodes,
            out_features_dim]``.
        """
        adj, x = _batched(adj, node_features)
        if self.aggr == "max":
            # [B, j, i, C]: the features of node j for target node i
            has_edge = (adj != 0)[..., None]
            out = mx.where(has_edge, x[:, :, None, :], -mx.inf).max(axis=1)
            out = mx.where(out == -mx.inf, 0.0, out)
        else:
            out = adj @ x
            if self.aggr == "mean":
                out = out / mx.maximum(adj.sum(axis=-1, keepdims=True), 1.0)
        out = self.lin_rel(out) + self.lin_root(x)
        return _apply_mask(out, mask)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.node_features_dim}, "
            f"{self.out_features_dim})"
        )
