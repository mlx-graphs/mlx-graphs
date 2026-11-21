from typing import List, Literal, Optional, Tuple, Union, get_args

import mlx.core as mx
import mlx.nn as nn
from mlx.nn import Linear  # PyTorch's default initialization, as in PyG

from mlx_graphs.nn.message_passing import MessagePassing
from mlx_graphs.utils import scatter
from mlx_graphs.utils.array_ops import p_norm
from mlx_graphs.utils.scatter import scatter_softmax

GENAggregations = Literal[
    "softmax", "softmax_sg", "powermean", "power", "add", "sum", "mean", "max", "min"
]
GENNorms = Literal["batch", "layer", "instance"]


class _InstanceNorm(nn.Module):
    """Normalizes every row over its features, without affine parameters.

    This is what ``torch.nn.InstanceNorm1d`` (used by PyG's ``GENConv``) does
    with a 2D ``[num_nodes, channels]`` input, which it treats as a single
    unbatched sample.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + self.eps)


def _mlp(
    channels: list, norm: Optional[str] = None, bias: bool = True
) -> nn.Sequential:
    """MLP of PyG's ``GENConv``: ``Linear -> norm -> ReLU`` between layers."""
    layers = []
    for i in range(1, len(channels)):
        layers.append(Linear(channels[i - 1], channels[i], bias=bias))
        if i < len(channels) - 1:
            if norm == "batch":
                layers.append(nn.BatchNorm(channels[i]))
            elif norm == "layer":
                layers.append(nn.LayerNorm(channels[i]))
            elif norm == "instance":
                layers.append(_InstanceNorm())
            elif norm:
                raise ValueError(f"Normalization layer '{norm}' not supported.")
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class GENConv(MessagePassing):
    r"""The GENeralized Graph Convolution (GENConv) from the `"DeeperGCN: All
    You Need to Train Deeper GCNs" <https://arxiv.org/abs/2006.07739>`_ paper.

    .. math::
        \mathbf{x}_i^{\prime} = \mathrm{MLP} \left( \mathbf{x}_i +
        \mathrm{AGG} \left( \left\{
        \mathrm{ReLU} \left( \mathbf{x}_j + \mathbf{e_{ji}} \right) +\epsilon
        : j \in \mathcal{N}(i) \right\} \right)
        \right)

    The aggregation is, by default, the softmax aggregation of DeeperGCN,
    computed per feature channel with an (optionally learnable) inverse
    temperature :math:`t`:

    .. math::
        \mathrm{AGG}(\{\mathbf{m}_j\}) = \sum_{j} \frac{\exp(t\,\mathbf{m}_j)}
        {\sum_{k} \exp(t\,\mathbf{m}_k)} \odot \mathbf{m}_j

    With ``aggr="powermean"`` it is the power mean
    :math:`\left(\frac{1}{|\mathcal{N}(i)|} \sum_j \mathbf{m}_j^p\right)^{1/p}`
    with an (optionally learnable) power :math:`p`, where the messages and the
    mean are clamped to ``[1e-4, 100]`` (as in PyG) when :math:`p \neq 1`.

    The layer also works on bipartite graphs: pass ``node_features`` as a
    ``(src_features, dst_features)`` tuple and ``node_features_dim`` as a
    ``(src_dim, dst_dim)`` tuple. The output has one row per destination node.

    Args:
        node_features_dim: Size of input node features, or a ``(src_dim, dst_dim)``
            tuple for bipartite graphs. Inputs whose size differs from
            ``out_features_dim`` are first projected with a linear layer.
        out_features_dim: Size of output node embeddings
        aggr: Aggregation: ``"softmax"``, ``"softmax_sg"`` (softmax whose
            weights are not differentiated, "semi-gradient"), ``"powermean"``
            (or ``"power"``), ``"add"`` (or ``"sum"``), ``"mean"``, ``"max"``
            or ``"min"``; or a list of aggregation names (e.g.
            ``["softmax", "mean"]``, with the default arguments of each),
            whose outputs are concatenated and projected back to
            ``out_features_dim`` with a linear layer, as in PyG.
            Default: ``"softmax"``
        t: Initial inverse temperature of the softmax aggregation.
            Default: ``1.0``
        learn_t: Whether to learn ``t``. Default: ``False``
        p: Initial power of the power mean aggregation. Default: ``1.0``
        learn_p: Whether to learn ``p``. Default: ``False``
        msg_norm: Whether to use message normalization (MsgNorm), i.e. rescale
            the aggregated messages to the :math:`\ell_2` norm of the
            destination features. Default: ``False``
        learn_msg_scale: Whether to learn the scale of the message
            normalization. Default: ``False``
        norm: Normalization of the hidden MLP layers: ``"batch"``, ``"layer"``,
            ``"instance"`` or ``None``. Default: ``"batch"``
        num_layers: Number of MLP layers. Default: ``2``
        expansion: Expansion factor of the hidden MLP layers. Default: ``2``
        eps: Value added to every message. Default: ``1e-7``
        bias: Whether the linear layers use a bias. Default: ``False``
        edge_features_dim: Size of edge features. If set and different from
            ``out_features_dim``, edge features are projected to
            ``out_features_dim``; otherwise they must already have
            ``out_features_dim`` features. Default: ``None``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import GENConv

        conv = GENConv(16, 32, aggr="softmax", learn_t=True, edge_features_dim=4)
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))
        edge_features = mx.random.uniform(low=0, high=1, shape=(5, 4))

        h = conv(edge_index, node_features, edge_features)

        >>> h.shape
        (5, 32)

    References:
        `Li et al. DeeperGCN: All You Need to Train Deeper GCNs
        <https://arxiv.org/abs/2006.07739>`_
    """

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        out_features_dim: int,
        aggr: Union[GENAggregations, List[str]] = "softmax",
        t: float = 1.0,
        learn_t: bool = False,
        p: float = 1.0,
        learn_p: bool = False,
        msg_norm: bool = False,
        learn_msg_scale: bool = False,
        norm: Optional[GENNorms] = "batch",
        num_layers: int = 2,
        expansion: int = 2,
        eps: float = 1e-7,
        bias: bool = False,
        edge_features_dim: Optional[int] = None,
    ):
        if isinstance(aggr, (list, tuple)):
            # A MultiAggregation (``self.aggr_module``) of the listed
            # aggregations, with their default arguments, as in PyG.
            super().__init__(aggr=list(aggr))
            self.semi_grad = False
        else:
            if aggr not in get_args(GENAggregations):
                raise ValueError(
                    f"Invalid aggregation '{aggr}'. Available values are "
                    f"{get_args(GENAggregations)}, or a list of aggregations"
                )
            # The base class only knows the scatter aggregations: the softmax
            # and power mean aggregations are computed in `_aggregate`.
            super().__init__(aggr="add")
            # Normalize aliases, as PyG does.
            aggr = {"power": "powermean", "sum": "add"}.get(aggr, aggr)
            self.semi_grad = aggr == "softmax_sg"
            self.aggr = "softmax" if self.semi_grad else aggr

        if isinstance(node_features_dim, int):
            src_dim = dst_dim = node_features_dim
        else:
            src_dim, dst_dim = node_features_dim

        self.node_features_dim = node_features_dim
        self.out_features_dim = out_features_dim
        self.eps = eps

        # Learnable scalars are arrays (parameters), fixed ones plain floats.
        if self.aggr == "softmax":
            self.t = mx.array([t], dtype=mx.float32) if learn_t else t
        elif self.aggr == "powermean":
            self.p = mx.array([p], dtype=mx.float32) if learn_p else p

        if src_dim != out_features_dim:
            self.lin_src = Linear(src_dim, out_features_dim, bias=bias)
        if edge_features_dim is not None and edge_features_dim != out_features_dim:
            self.lin_edge = Linear(edge_features_dim, out_features_dim, bias=bias)
        if "aggr_module" in self:
            aggr_dim = self.aggr_module.get_out_features_dim(out_features_dim)
            if aggr_dim != out_features_dim:
                self.lin_aggr_out = Linear(aggr_dim, out_features_dim, bias=bias)
        if dst_dim != out_features_dim:
            self.lin_dst = Linear(dst_dim, out_features_dim, bias=bias)

        channels = [out_features_dim]
        channels += [out_features_dim * expansion] * (num_layers - 1)
        channels.append(out_features_dim)
        self.mlp = _mlp(channels, norm=norm, bias=bias)

        self.msg_norm = msg_norm
        if msg_norm:
            self.msg_scale = mx.array([1.0]) if learn_msg_scale else 1.0

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of GENConv.

        Args:
            edge_index: Input edge index of shape `[2, num_edges]`
            node_features: Input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs
            edge_features: Optional edge features of shape
                `[num_edges, edge_features_dim]`

        Returns:
            mx.array: The computed node embeddings, one row per destination node
        """
        if isinstance(node_features, tuple):
            x_src, x_dst = node_features
        else:
            x_src = x_dst = node_features

        if "lin_src" in self:
            x_src = self.lin_src(x_src)

        src_idx, dst_idx = edge_index[0], edge_index[1]
        num_nodes = x_dst.shape[0]

        messages = x_src[src_idx]
        if edge_features is not None:
            if edge_features.ndim == 1:
                edge_features = edge_features.reshape(-1, 1)
            if "lin_edge" in self:
                edge_features = self.lin_edge(edge_features)
            if edge_features.shape[-1] != messages.shape[-1]:
                raise ValueError(
                    "Edge features must have `out_features_dim` features, or "
                    "`edge_features_dim` must be set to project them."
                )
            messages = messages + edge_features
        messages = nn.relu(messages) + self.eps

        out = self._aggregate(messages, dst_idx, num_nodes)
        if "lin_aggr_out" in self:
            out = self.lin_aggr_out(out)

        if self.msg_norm:
            # p_norm: zero gradient (not NaN) for rows of zeros
            norm = p_norm(out, axis=-1, keepdims=True)
            out = out / mx.maximum(norm, 1e-12)
            out = out * p_norm(x_dst, axis=-1, keepdims=True) * self.msg_scale

        if "lin_dst" in self:
            x_dst = self.lin_dst(x_dst)

        return self.mlp(out + x_dst)

    def _aggregate(
        self, messages: mx.array, index: mx.array, num_nodes: int
    ) -> mx.array:
        if "aggr_module" in self:
            return self.aggr_module(messages, index, dim_size=num_nodes)
        if self.aggr == "softmax":
            t = self.t
            logits = messages if isinstance(t, float) and t == 1 else messages * t
            alpha = scatter_softmax(logits, index, num_nodes)
            if self.semi_grad:
                alpha = mx.stop_gradient(alpha)
            return scatter(messages * alpha, index, num_nodes, "add")

        if self.aggr == "powermean":
            p = self.p
            if isinstance(p, float) and p == 1:
                return scatter(messages, index, num_nodes, "mean")
            messages = mx.power(mx.clip(messages, 1e-4, 100.0), p)
            out = scatter(messages, index, num_nodes, "mean")
            return mx.power(mx.clip(out, 1e-4, 100.0), 1.0 / p)

        return scatter(messages, index, num_nodes, self.aggr)
