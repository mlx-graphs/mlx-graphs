from contextlib import contextmanager
from typing import Literal, Optional, Tuple, Union, get_args

import mlx.core as mx
import mlx.nn as nn

import mlx_graphs.nn.message_passing as _message_passing
from mlx_graphs.nn.conv._norm import SelfLoopFillValue, attention_self_loops
from mlx_graphs.nn.kernels.gatv2_edge_scores import _reference_scores, gatv2_edge_scores
from mlx_graphs.nn.linear import Linear, glorot
from mlx_graphs.nn.message_passing import (
    _AUTO_CSR_MIN_EDGES,
    HAS_FUSED_KERNELS,
    MessagePassing,
    _csr_cache,
    fused_message_passing,
)

GATv2ScoreKernel = Literal["auto", "metal", "mlx_cluster"]

# mlx_cluster.gatv2_scores computes the attention logits with a backward pass
# that sums each node's edges on the CSR (no atomics); older mlx-cluster
# versions don't have it.
HAS_GATV2_SCORES = False
if HAS_FUSED_KERNELS:
    try:
        from mlx_cluster import gatv2_scores as _mlx_cluster_gatv2_scores

        HAS_GATV2_SCORES = True
    except ImportError:
        pass

# In "auto" mode, graphs with at least this many edges use
# mlx_cluster.gatv2_scores. It is the size at which fused message passing
# (format="auto") converts a graph to CSR on first use, so the CSR that
# gatv2_scores needs costs no extra sort. Below it, the CSR is often never
# built (e.g. with add_self_loops=True, which makes a new edge_index per call).
_MLX_CLUSTER_MIN_EDGES = _AUTO_CSR_MIN_EDGES

_score_kernel: GATv2ScoreKernel = "auto"


def set_gatv2_score_kernel(kernel: GATv2ScoreKernel) -> None:
    """Chooses the kernel that computes the attention scores of
    :class:`GATv2Conv` on the fused message-passing path.

    - ``"auto"`` (default): ``mlx_cluster.gatv2_scores`` on the Metal GPU for
      graphs with at least 1M edges, the in-tree Metal kernel otherwise.
    - ``"metal"``: always the in-tree Metal kernel.
    - ``"mlx_cluster"``: ``mlx_cluster.gatv2_scores`` whenever it can run,
      whatever the graph size. It needs the graph in CSR form, so on graphs
      whose CSR is not already cached this adds a sort of the edges per call.

    In every mode, layers with edge features use the in-tree kernel (which
    supports them), and with fused message passing disabled every layer
    uses gather-scatter.

    Args:
        kernel: ``"auto"``, ``"metal"`` or ``"mlx_cluster"``.
    """
    global _score_kernel
    if kernel not in get_args(GATv2ScoreKernel):
        raise ValueError(
            f"kernel must be 'auto', 'metal' or 'mlx_cluster', got {kernel!r}"
        )
    if kernel == "mlx_cluster" and not HAS_GATV2_SCORES:
        raise ValueError(
            "kernel='mlx_cluster' needs an mlx-cluster with gatv2_scores "
            "(macOS only); install a recent mlx-cluster."
        )
    _score_kernel = kernel


@contextmanager
def gatv2_score_kernel_mode(kernel: GATv2ScoreKernel):
    """Context manager version of :func:`set_gatv2_score_kernel`.

    .. code-block:: python

        with gatv2_score_kernel_mode("metal"):
            out = model(edge_index, x)  # in-tree kernel, whatever the size
    """
    global _score_kernel
    previous = _score_kernel
    set_gatv2_score_kernel(kernel)
    try:
        yield
    finally:
        _score_kernel = previous


def _on_metal() -> bool:
    return mx.default_device().type == mx.gpu and mx.metal.is_available()


@fused_message_passing
class GATv2Conv(MessagePassing):
    """Graph Attention Network convolution layer with dynamic attention.

    Modification of GATConv based off of `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper.

    The layer also works on bipartite graphs (e.g. ``author -> paper`` in a
    heterogeneous graph): pass ``node_features`` as a ``(src_features,
    dst_features)`` tuple and, if the two sides have different feature
    sizes, ``node_features_dim`` as a ``(src_dim, dst_dim)`` tuple. The output
    has one row per destination node.

    Args:
        node_features_dim: Size of input node features, or a
            ``(src_dim, dst_dim)`` tuple for bipartite graphs
        out_features_dim: Size of output node embeddings
        heads: Number of attention heads
        concat: Whether to use concat of heads or mean reduction
        bias: Whether to use bias in the node projection
        negative_slope: Slope for the leaky relu
        dropout: Probability p for dropout
        edge_features_dim: Size of edge features
        add_self_loops: Whether to add a self-loop to every node, as in PyG
            (existing self-loops are replaced by it). Not supported on
            bipartite ``(src_features, dst_features)`` inputs. Default:
            ``True``
        fill_value: Edge features of the added self-loops, if
            ``edge_features`` are given: a constant, or the reduction
            (``"mean"``, ``"add"``, ``"max"``, ``"min"`` or ``"mul"``) of the
            features of the incoming edges of each node. Default: ``"mean"``

    The attention scores are computed by one of two kernels on the fused
    message-passing path. On the Metal GPU, graphs with at least 1M edges
    (and no edge features) use ``mlx_cluster.gatv2_scores``, whose backward
    pass is faster on large graphs; smaller graphs use the in-tree Metal
    kernel. Both give the same results. :func:`set_gatv2_score_kernel`
    overrides the choice.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import GATv2Conv

        conv = GATv2Conv(16, 32, heads=2, concat=True)
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]])
        node_features = mx.random.uniform(low=0, high=1, shape=(5, 16))

        h = conv(edge_index, node_features)

        >>> h.shape
        (5, 64)

    """

    # what follows the aggregation expects the size of one aggregation (as in PyG)
    supports_concat_aggr = False

    def __init__(
        self,
        node_features_dim: Union[int, Tuple[int, int]],
        out_features_dim: int,
        heads: int = 1,
        concat: bool = True,
        bias: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_features_dim: Optional[int] = None,
        add_self_loops: bool = True,
        fill_value: SelfLoopFillValue = "mean",
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(GATv2Conv, self).__init__(**kwargs)

        self.out_features_dim = out_features_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        # read by to_hetero, which rejects self-loops on bipartite edge types
        self._add_self_loops = add_self_loops
        self.fill_value = fill_value

        # Weights
        if isinstance(node_features_dim, int):
            src_dim = dst_dim = node_features_dim
        else:
            src_dim, dst_dim = node_features_dim
        self.W_src = Linear(src_dim, heads * out_features_dim, bias=False)
        self.W_dst = Linear(dst_dim, heads * out_features_dim, bias=False)

        # Attention is applied in message() during message passing stage.
        self.att = glorot((1, heads, out_features_dim))

        if bias:
            bias_shape = (heads * out_features_dim) if concat else (out_features_dim)
            self.bias = mx.zeros(bias_shape)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

        if edge_features_dim is not None:
            self.edge_lin_proj = Linear(
                edge_features_dim, heads * out_features_dim, bias=False
            )
            self.edge_att = glorot((1, heads, out_features_dim))

    def __call__(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_features: Optional[mx.array] = None,
    ) -> mx.array:
        """Computes the forward pass of GATConv.

        Args:
            edge_index: input edge index of shape `[2, num_edges]`
            node_features: input node features, or a ``(src_features,
                dst_features)`` tuple for bipartite graphs
            edge_features: edge features

        Returns:
            mx.array: computed node embeddings
        """
        H, C = self.heads, self.out_features_dim

        if isinstance(node_features, tuple):
            src_x, dst_x = node_features
        else:
            src_x = dst_x = node_features
        if self._add_self_loops:
            if dst_x is not src_x:
                raise ValueError(
                    f"{type(self).__name__} cannot add self-loops on a bipartite "
                    "graph ((src_features, dst_features) input): set "
                    "add_self_loops=False."
                )
            edge_index, edge_features, removed_loops = attention_self_loops(
                edge_index, src_x.shape[0], edge_features, self.fill_value
            )
        src_feats = self.W_src(src_x).reshape(-1, H, C)
        dst_feats = self.W_dst(dst_x).reshape(-1, H, C)

        edge_feats = (
            self._compute_edge_features(edge_features)
            if edge_features is not None
            else None
        )
        # Tie the Metal score paths to the existing fused message-passing mode.
        if self._can_fuse(edge_index, src_feats, None, None):
            csr = self._mlx_cluster_csr(
                edge_index, dst_feats.shape[0], src_feats.shape[0], edge_feats
            )
            if csr is not None:
                alpha = _mlx_cluster_gatv2_scores(
                    edge_index[1],
                    edge_index[0],
                    dst_feats,
                    src_feats,
                    self.att,
                    self.negative_slope,
                    csr=csr,
                )
            else:
                alpha = gatv2_edge_scores(
                    src_feats,
                    dst_feats,
                    self.att,
                    edge_index,
                    self.negative_slope,
                    edge_feats,
                )
        else:
            alpha = _reference_scores(
                src_feats,
                dst_feats,
                self.att,
                edge_index,
                self.negative_slope,
                edge_feats,
            )
        if self._add_self_loops:
            # the input self-loops are replaced by the added ones
            alpha = mx.where(removed_loops[:, None], -float("inf"), alpha)
        node_features = self._softmax_aggregate(
            edge_index=edge_index,
            logits=alpha,
            node_features=(src_feats, dst_feats),
            dropout=self.dropout if "dropout" in self else None,
        )

        if self.concat:
            node_features = node_features.reshape(
                -1, self.heads * self.out_features_dim
            )
        else:
            node_features = mx.mean(node_features, axis=1)

        if "bias" in self:
            node_features = node_features + self.bias

        return node_features

    def _mlx_cluster_csr(
        self,
        edge_index: mx.array,
        num_dst: int,
        num_src: int,
        edge_feats: Optional[mx.array],
    ):
        """The CSR for ``mlx_cluster.gatv2_scores``, or ``None`` for the
        in-tree kernel (see :func:`set_gatv2_score_kernel`)."""
        if _score_kernel == "metal" or not HAS_GATV2_SCORES or edge_feats is not None:
            return None
        if _score_kernel == "auto" and (
            edge_index.shape[1] < _MLX_CLUSTER_MIN_EDGES or not _on_metal()
        ):
            return None
        fmt = _message_passing._fused_format_override or self._fused_format
        if fmt == "coo":
            return None
        # The same edge_index and sizes as this call's aggregation, which
        # then finds this CSR in the cache. None while tracing (mx.compile).
        return _csr_cache.get(edge_index, num_dst, num_src)

    def _compute_edge_features(self, edge_features: mx.array):
        assert all(layer in self for layer in ["edge_lin_proj", "edge_att"]), """Using
        edge features, GATConv layer should be provided argument
        `edge_features_dim`."""

        if edge_features.ndim == 1:
            edge_features = edge_features.reshape(-1, 1)

        edge_features = self.edge_lin_proj(edge_features)
        edge_features = edge_features.reshape(-1, self.heads, self.out_features_dim)

        return edge_features
