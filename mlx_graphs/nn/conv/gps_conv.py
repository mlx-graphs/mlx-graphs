import inspect
from functools import partial
from typing import Any, Callable, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn.aggr.utils import DenseLayout
from mlx_graphs.nn.attention import MultiheadAttention, PerformerAttention
from mlx_graphs.nn.norm import (
    BatchNormalization,
    GraphNormalization,
    InstanceNormalization,
    LayerNormalization,
)

_NORMS = {
    "batch": BatchNormalization,
    "layer": LayerNormalization,
    "graph": GraphNormalization,
    "instance": InstanceNormalization,
}


def _resolve_act(act: Union[str, Callable, None], act_kwargs: Optional[Dict]):
    from mlx_graphs.nn.models.basic_gnn import _resolve_activation

    fn = _resolve_activation(act)
    if fn is not None and act_kwargs:
        fn = partial(fn, **act_kwargs)
    return fn


def _resolve_norm(
    norm: Union[str, Callable[..., nn.Module], None],
    channels: int,
    norm_kwargs: Dict[str, Any],
) -> Optional[nn.Module]:
    if norm is None:
        return None
    if isinstance(norm, str):
        key = norm.lower().replace("_", "").replace("-", "").replace(" ", "")
        if key.endswith("normalization"):
            key = key[: -len("normalization")]
        elif key.endswith("norm"):
            key = key[: -len("norm")]
        if key not in _NORMS:
            raise ValueError(
                f"Unknown normalization '{norm}'. Available: "
                f"{sorted(k + '_norm' for k in _NORMS)}, None or a class"
            )
        norm = _NORMS[key]
    return norm(channels, **norm_kwargs)


def _dense_layout(
    layout: Optional[DenseLayout], batch_indices: Optional[mx.array], num_nodes: int
) -> DenseLayout:
    """The given layout (checked), or the layout of ``batch_indices``."""
    if layout is None:
        return DenseLayout(batch_indices, num_nodes)
    if layout.num_elements != num_nodes:
        raise ValueError(
            f"The layout is for {layout.num_elements} nodes, got {num_nodes}"
        )
    return layout


def _takes_batch(module: Optional[nn.Module]) -> bool:
    if module is None:
        return False
    params = inspect.signature(module.__call__).parameters
    return "batch" in params or "batch_indices" in params


class GPSConv(nn.Module):
    r"""The general, powerful, scalable (GPS) graph transformer layer from the
    `"Recipe for a General, Powerful, Scalable Graph Transformer"
    <https://arxiv.org/abs/2205.12454>`_ paper, as PyG's ``GPSConv``.

    The layer combines a local message passing layer (MPNN) and a global
    attention layer over all the nodes of each graph, each with a residual
    connection and a normalization, followed by a two-layer MLP:

    .. math::
        \mathbf{X}_M &= \mathrm{Norm}(\mathbf{X} +
        \mathrm{Dropout}(\mathrm{MPNN}(\mathbf{X}, \mathbf{A}))) \\
        \mathbf{X}_T &= \mathrm{Norm}(\mathbf{X} +
        \mathrm{Dropout}(\mathrm{GlobalAttn}(\mathbf{X}))) \\
        \mathbf{X}' &= \mathrm{Norm}(\mathbf{X}_M + \mathbf{X}_T +
        \mathrm{MLP}(\mathbf{X}_M + \mathbf{X}_T))

    Positional and structural encodings (the first part of the GPS recipe)
    are added to the input features beforehand, e.g. with the
    :class:`~mlx_graphs.transforms.AddRandomWalkPE` or
    :class:`~mlx_graphs.transforms.AddLaplacianEigenvectorPE` transforms and
    :class:`~mlx_graphs.nn.RWSEEncoder` / :class:`~mlx_graphs.nn.LapPEEncoder`.

    The global attention is dense: it costs
    :math:`\mathcal{O}(N_{\max}^2)` memory per graph and head, where
    :math:`N_{\max}` is the size of the largest graph of the batch. Use
    ``attn_type="performer"`` for linear attention.

    The dense batch of the attention is laid out from ``batch_indices`` on
    the host (a GPU synchronization, which also prevents :func:`mx.compile`)
    unless a precomputed ``layout`` is passed: see :meth:`__call__` and
    :meth:`~mlx_graphs.nn.aggr.DenseLayout.from_sizes`.

    Args:
        channels: Size of the input and output node features.
        conv: The local message passing layer, called as
            ``conv(edge_index, node_features, **kwargs)``, or ``None`` for a
            pure transformer.
        heads: Number of attention heads. Default: ``1``
        dropout: Dropout probability of the intermediate embeddings.
            Default: ``0.0``
        act: Activation of the MLP: a name (``"relu"``, ``"gelu"``, ...) or a
            callable. Default: ``"relu"``
        act_kwargs: Arguments of the activation. Default: ``None``
        norm: Normalization: ``"batch_norm"``, ``"layer_norm"``,
            ``"graph_norm"``, ``"instance_norm"``, a normalization class
            called as ``norm(channels, **norm_kwargs)``, or ``None``. Graph
            normalizations receive ``batch_indices``.
            Default: ``"batch_norm"``
        norm_kwargs: Arguments of the normalization. Default: ``None``
        attn_type: Global attention: ``"multihead"``
            (:class:`~mlx_graphs.nn.attention.MultiheadAttention`) or
            ``"performer"`` (:class:`~mlx_graphs.nn.attention.PerformerAttention`).
            Default: ``"multihead"``
        attn_kwargs: Arguments of the attention, e.g. ``{"dropout": 0.5}`` for
            dropout on the attention weights. Default: ``None``

    Example:

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_graphs.nn import GINEConv, GPSConv

        mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))
        conv = GPSConv(64, GINEConv(mlp), heads=4, attn_kwargs={"dropout": 0.5})

        edge_index = mx.array([[0, 1, 2, 3], [1, 0, 3, 2]])
        node_features = mx.random.normal((4, 64))
        edge_features = mx.random.normal((4, 64))
        batch_indices = mx.array([0, 0, 1, 1])
        >>> conv(edge_index, node_features, batch_indices,
        ...      edge_features=edge_features).shape
        (4, 64)

    References:
        - `Rampášek et al., "Recipe for a General, Powerful, Scalable Graph
          Transformer", NeurIPS 2022 <https://arxiv.org/abs/2205.12454>`_
    """

    def __init__(
        self,
        channels: int,
        conv: Optional[nn.Module],
        heads: int = 1,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable[..., nn.Module], None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        attn_type: str = "multihead",
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.attn_type = attn_type
        attn_kwargs = attn_kwargs or {}
        if attn_type == "multihead":
            self.attn = MultiheadAttention(channels, heads, **attn_kwargs)
        elif attn_type == "performer":
            self.attn = PerformerAttention(channels, heads, **attn_kwargs)
        else:
            raise ValueError(
                f"Unknown attention type '{attn_type}'. Available: 'multihead', "
                "'performer'"
            )

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            _resolve_act(act, act_kwargs) or (lambda x: x),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

        norm_kwargs = norm_kwargs or {}
        if norm is not None:
            self.norm1 = _resolve_norm(norm, channels, norm_kwargs)
            self.norm2 = _resolve_norm(norm, channels, norm_kwargs)
            self.norm3 = _resolve_norm(norm, channels, norm_kwargs)
        self.norm_with_batch = _takes_batch(self.get("norm1"))

    def redraw_projection_matrix(self):
        """Redraws the random projection of the performer attention (no-op
        for the multi-head attention)."""
        if isinstance(self.attn, PerformerAttention):
            self.attn.redraw_projection_matrix()

    def _norm(self, name: str, x: mx.array, batch_indices: Optional[mx.array]):
        if name not in self:
            return x
        if self.norm_with_batch:
            return self[name](x, batch_indices)
        return self[name](x)

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        batch_indices: Optional[mx.array] = None,
        layout: Optional[DenseLayout] = None,
        **kwargs: Any,
    ) -> mx.array:
        """Computes the forward pass of GPSConv.

        Args:
            edge_index: Edge index of shape ``[2, num_edges]``.
            node_features: Node features of shape ``[num_nodes, channels]``.
            batch_indices: Graph of each node, of shape ``[num_nodes]``.
                Attention is computed within each graph. Default: ``None``
                (a single graph).
            layout: Optional precomputed dense layout of the nodes per graph
                (:class:`~mlx_graphs.nn.aggr.DenseLayout`). By default it is
                computed from ``batch_indices`` on the host at every call,
                which synchronizes with the GPU and prevents
                :func:`mx.compile`. Build it once per batch and share it
                between layers, e.g. with
                ``DenseLayout.from_sizes(batch_indices, num_nodes,
                num_graphs, max_num_nodes)``, which needs no
                synchronization. Default: ``None``
            **kwargs: Additional arguments of the local message passing
                layer (e.g. ``edge_features``).

        Returns:
            The updated node features, of shape ``[num_nodes, channels]``.
        """
        x = node_features
        hs = []
        if self.conv is not None:  # local MPNN
            h = self.conv(edge_index, x, **kwargs)
            h = self.drop(h) + x
            hs.append(self._norm("norm1", h, batch_indices))

        # global attention
        layout = _dense_layout(layout, batch_indices, x.shape[0])
        h = layout.to_dense(x)
        if isinstance(self.attn, MultiheadAttention):
            h = self.attn(h, h, h, key_mask=layout.mask)
        else:
            h = self.attn(h, layout.mask)
        h = self.drop(layout.from_dense(h)) + x
        hs.append(self._norm("norm2", h, batch_indices))

        out = sum(hs[1:], hs[0])
        out = out + self.mlp(out)
        return self._norm("norm3", out, batch_indices)
