from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Literal, Optional, Tuple, Union, get_args

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.utils import ScatterAggregations, get_src_dst_features, scatter

# Fused SpMM kernels live in mlx-cluster. They are optional: without them (or
# with an older mlx-cluster) every layer keeps the gather-scatter path.
try:
    from mlx_cluster import coo_to_csr as _coo_to_csr
    from mlx_cluster import spmm_coo as _spmm_coo
    from mlx_cluster import spmm_csr as _spmm_csr

    HAS_FUSED_KERNELS = True
    FUSED_KERNELS_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as e:
    # Kept so tools can explain *why* (e.g. mlx_cluster built against a
    # different mlx version) instead of just "not found".
    HAS_FUSED_KERNELS = False
    FUSED_KERNELS_IMPORT_ERROR = e

FusedFormat = Literal["coo", "csr"]
_FUSED_AGGRS: Tuple[str, ...] = ("add", "mean")
if HAS_FUSED_KERNELS:
    # max/min need an mlx-cluster with those reductions; older ones fall back
    try:
        from mlx_cluster.sparse import SpMMReduce as _SpMMReduce

        _FUSED_AGGRS += tuple(a for a in ("max", "min") if a in get_args(_SpMMReduce))
    except ImportError:
        pass

# Global switch, mainly for benchmarks and tests (same model, both paths).
_fused_enabled: bool = True
_fused_format_override: Optional[FusedFormat] = None


def set_fused_message_passing(
    enabled: bool, format: Optional[FusedFormat] = None
) -> None:
    """Globally enables/disables fused message passing.

    Args:
        enabled: if ``False``, every layer uses gather-scatter.
        format: if given, overrides the ``format`` chosen in
            :func:`fused_message_passing` for every layer (``"coo"`` or ``"csr"``).
    """
    global _fused_enabled, _fused_format_override
    if format is not None and format not in get_args(FusedFormat):
        raise ValueError(f"format must be 'coo' or 'csr', got {format!r}")
    _fused_enabled = enabled
    _fused_format_override = format


def is_fused_message_passing_enabled() -> bool:
    """Whether fused message passing is enabled and the kernels are available."""
    return _fused_enabled and HAS_FUSED_KERNELS


@contextmanager
def fused_message_passing_mode(enabled: bool, format: Optional[FusedFormat] = None):
    """Context manager version of :func:`set_fused_message_passing`.

    .. code-block:: python

        with fused_message_passing_mode(False):
            out_reference = model(edge_index, x)  # gather-scatter
    """
    global _fused_enabled, _fused_format_override
    previous = (_fused_enabled, _fused_format_override)
    set_fused_message_passing(enabled, format)
    try:
        yield
    finally:
        _fused_enabled, _fused_format_override = previous


class _CSRCache:
    """Small LRU cache of COO -> CSR conversions, keyed on the edge_index array.

    The cache keeps a reference to each edge_index, so its ``id`` cannot be
    reused by another array while the entry is alive.
    """

    def __init__(self, maxsize: int = 8):
        self.maxsize = maxsize
        self._entries: "OrderedDict[Tuple[int, int, int], Tuple[mx.array, Any]]" = (
            OrderedDict()
        )

    def get(self, edge_index: mx.array, num_rows: int, num_cols: int):
        # num_rows = destination nodes (output rows), num_cols = source nodes
        # (rows of x). They differ for bipartite (src, dst) inputs.
        key = (id(edge_index), num_rows, num_cols)
        hit = self._entries.get(key)
        if hit is not None and hit[0] is edge_index:
            self._entries.move_to_end(key)
            return hit[1]
        csr = _coo_to_csr(edge_index[1], edge_index[0], num_rows, num_cols=num_cols)
        self._entries[key] = (edge_index, csr)
        if len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
        return csr

    def clear(self):
        self._entries.clear()


_csr_cache = _CSRCache()


def clear_csr_cache() -> None:
    """Drops all cached CSR conversions used by ``format="csr"``."""
    _csr_cache.clear()


def fused_message_passing(
    cls=None,
    *,
    format: FusedFormat = "coo",
    allow_message_override: bool = False,
):
    """Class decorator: lets a :class:`MessagePassing` layer fuse ``message``
    and ``aggregate`` into a single sparse x dense matmul (SpMM).

    With the decorator, :meth:`MessagePassing.propagate` calls
    :meth:`MessagePassing.message_and_aggregate` instead of gathering one
    message per edge and scattering them. This never materialises the
    ``[num_edges, features]`` message array, which is what limits
    gather-scatter on large graphs.

    The fused path is used only when it gives the same result; otherwise the
    layer silently falls back to gather-scatter. It falls back when:

    * ``mlx_cluster`` (with ``spmm_coo``) is not installed,
    * fused message passing is disabled (:func:`set_fused_message_passing`),
    * ``aggr`` is not ``"add"``, ``"mean"``, ``"max"`` or ``"min"`` (i.e. it
      is ``"softmax"``; max/min also need a recent mlx-cluster),
    * any message/aggregate kwarg other than ``edge_weights`` is set
      (e.g. ``edge_features``), since messages are then no longer a weighted
      sum of neighbour features,
    * the layer is a subclass of a decorated layer that overrides
      ``message()`` or ``aggregate()`` without also overriding
      ``message_and_aggregate()``.

    Args:
        format: ``"coo"`` (default) works directly on ``edge_index``.
            ``"csr"`` sorts the graph once (cached per ``edge_index``) and uses
            a deterministic, atomic-free kernel.
        allow_message_override: a layer that overrides ``message()`` or
            ``aggregate()`` is rejected by default, because the fused kernel
            would ignore those overrides. Pass ``True`` only if the override
            reduces to the default ``edge_weights * x_j`` / scatter whenever
            the extra kwargs are ``None`` (e.g. ``GINConv``).

    .. code-block:: python

        @fused_message_passing
        class MyConv(MessagePassing):
            ...


        @fused_message_passing(format="csr")
        class MyOtherConv(MessagePassing):
            ...
    """
    if format not in get_args(FusedFormat):
        raise ValueError(f"format must be 'coo' or 'csr', got {format!r}")

    def wrap(klass):
        if not (isinstance(klass, type) and issubclass(klass, MessagePassing)):
            raise TypeError(
                "@fused_message_passing can only decorate MessagePassing subclasses."
            )
        overrides = [
            name
            for name in ("message", "aggregate")
            if getattr(klass, name) is not getattr(MessagePassing, name)
        ]
        custom_fused = (
            klass.message_and_aggregate is not MessagePassing.message_and_aggregate
        )
        if overrides and not custom_fused and not allow_message_override:
            raise TypeError(
                f"{klass.__name__} overrides {' and '.join(overrides)}(), which the "
                "fused SpMM kernel would ignore. Implement message_and_aggregate(), "
                "or pass allow_message_override=True if the override reduces to "
                "the default when only edge_weights are given."
            )
        klass._fused_format = format
        return klass

    return wrap(cls) if cls is not None else wrap


class MessagePassing(nn.Module):
    """Base class for creating Message Passing Neural Networks (MPNNs) [1].

    Inherit this class to build arbitrary GNN models based on the message
    passing paradigm. This implementation is inspired from PyTorch Geometric [2].

    Args:
        aggr: Aggregation strategy used to aggregate messages

    References:
        [1] `Gilmer et al. Neural Message Passing for Quantum Chemistry.\
     <https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf>`_

        [2] `Fey et al. PyG <https://pytorch-geometric.readthedocs.io/en/latest/\
    generated/torch_geometric.nn.conv.MessagePassing.html>`_
    """

    # Set by the @fused_message_passing decorator; None = gather-scatter only.
    _fused_format: Optional[FusedFormat] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # A subclass of a fused layer inherits _fused_format, but the fused
        # kernel would ignore a new message()/aggregate(), so fall back to
        # gather-scatter unless it also provides message_and_aggregate().
        # Runs before any decorator on cls, so @fused_message_passing on the
        # subclass itself still re-checks and sets the format.
        overrides = "message" in cls.__dict__ or "aggregate" in cls.__dict__
        if overrides and "message_and_aggregate" not in cls.__dict__:
            cls._fused_format = None

    def __init__(self, aggr: ScatterAggregations = "add"):
        super().__init__()
        if aggr not in get_args(ScatterAggregations):
            raise ValueError(
                "Invalid aggregation function.",
                f"Available values are {get_args(ScatterAggregations)}",
            )
        self.aggr: ScatterAggregations = aggr
        self.num_nodes = None

    def __call__(self, node_features: mx.array, edge_index: mx.array, **kwargs: Any):
        raise NotImplementedError

    def propagate(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        message_kwargs: Optional[Dict] = {},
        aggregate_kwargs: Optional[Dict] = {},
        update_kwargs: Optional[Dict] = {},
    ) -> mx.array:
        """Computes messages from neighbors, aggregates them and updates
        the final node embeddings.

        Args:
            edge_index: Graph representation of shape `[2, num_edges]`
            node_features: Input node features/embeddings.
                Can be either an array or a tuple of arrays, for distinct src and dst
                node features. With a tuple ``(src, dst)``, ``edge_index[0]``
                indexes ``src`` and ``edge_index[1]`` indexes ``dst``, and the
                output has one row per ``dst`` node.
            message_kwargs: Arguments to pass to the `message` method
            aggregate_kwargs: Arguments to pass to the `aggregate` method
            update_kwargs: Arguments to pass to the `update_nodes` method
        """
        if not (isinstance(edge_index, mx.array) and edge_index.shape[0] == 2):
            raise ValueError("Edge index should be an array with shape (2, |E|)")
        if isinstance(node_features, tuple):
            if len(node_features) != 2 or not all(
                isinstance(array, mx.array) for array in node_features
            ):
                raise ValueError(
                    "Invalid shape for `node_features`, should be a tuple of 2 mx.array"
                )
        else:
            if not isinstance(node_features, mx.array):
                raise ValueError(
                    "Invalid shape for `node_features`, should be an `mx.array`,",
                    f"found {type(node_features)}",
                )

        # Messages are aggregated onto destination nodes, so the output has one
        # row per destination node. For a (src, dst) tuple (bipartite graphs,
        # e.g. author -> paper) that is the size of the dst features.
        self.num_nodes = (
            node_features if isinstance(node_features, mx.array) else node_features[1]
        ).shape[0]

        if self._can_fuse(edge_index, message_kwargs, aggregate_kwargs):
            # shapes: (|N| -> |N|), no per-edge message array
            aggregated = self.message_and_aggregate(
                edge_index=edge_index,
                node_features=node_features,
                **message_kwargs,
            )
            return self.update_nodes(aggregated=aggregated, **update_kwargs)

        src_features, dst_features = get_src_dst_features(edge_index, node_features)
        dst_idx = edge_index[1]

        # shapes: (|E| -> |E|)
        messages = self.message(
            src_features=src_features,
            dst_features=dst_features,
            **message_kwargs,
        )

        # shapes: (|E| -> |N|)
        aggregated = self.aggregate(
            messages=messages,
            indices=dst_idx,
            **aggregate_kwargs,
        )

        # shapes: (|N| -> |N|)
        output = self.update_nodes(
            aggregated=aggregated,
            **update_kwargs,
        )

        return output

    def _can_fuse(
        self,
        edge_index: mx.array,
        message_kwargs: Optional[Dict],
        aggregate_kwargs: Optional[Dict],
    ) -> bool:
        if self._fused_format is None or not is_fused_message_passing_enabled():
            return False
        if self.aggr not in _FUSED_AGGRS:
            return False
        message_kwargs = message_kwargs or {}
        if any(v is not None for k, v in message_kwargs.items() if k != "edge_weights"):
            return False
        if any(v is not None for v in (aggregate_kwargs or {}).values()):
            return False
        weights = message_kwargs.get("edge_weights", None)
        if weights is not None and weights.size != edge_index.shape[1]:
            return False  # e.g. multi-dimensional weights: not a scalar per edge
        return True

    def message_and_aggregate(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        edge_weights: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """Fused ``message`` + ``aggregate`` as one sparse x dense matmul.

        Computes, for every node ``i``,
        ``aggr_{j -> i} edge_weights[j -> i] * x_j`` (``aggr`` = add, mean,
        max or min, per feature; 0 for nodes without neighbours), the same
        result as the
        default :meth:`message` followed by :meth:`aggregate`, without
        building the per-edge message array. Used by layers decorated with
        :func:`fused_message_passing`; override it to provide a custom fused
        implementation.

        Args:
            edge_index: Graph representation of shape `[2, num_edges]`
            node_features: Node features, or a ``(src, dst)`` tuple
                (only the source features are used).
            edge_weights: Optional scalar weight per edge.
        """
        x = node_features[0] if isinstance(node_features, tuple) else node_features
        weights = None if edge_weights is None else edge_weights.reshape(-1)
        reduce = "sum" if self.aggr == "add" else self.aggr  # mean / max / min
        fmt = _fused_format_override or self._fused_format

        if fmt == "csr":
            csr = _csr_cache.get(edge_index, self.num_nodes, x.shape[0])
            if weights is not None:
                weights = weights[csr.perm]
            return _spmm_csr(
                csr.rowptr,
                csr.col,
                x,
                weights,
                reduce=reduce,
                # load-balanced path for high-degree nodes (mlx-cluster >= the
                # long-row release; older CSR tuples have no such field)
                long_rows=getattr(csr, "long_rows", None),
                # cached transpose: the backward also runs on the CSR kernel
                transpose=getattr(csr, "transpose", None),
            )

        return _spmm_coo(
            edge_index[1],
            edge_index[0],
            x,
            weights,
            num_rows=self.num_nodes,
            reduce=reduce,
        )

    def message(
        self, src_features: mx.array, dst_features: mx.array, **kwargs
    ) -> mx.array:
        """Computes messages between connected nodes.

        By default, returns the features of source nodes.
        Optional ``edge_weights`` can be directly integrated in ``kwargs``

        Args:
            src_features: Source node embeddings
            dst_features: Destination node embeddings
            edge_weights: Array of scalars with shape (num_edges,) or (num_edges, 1)
                used to weigh neighbor features during aggregation. Default: ``None``
            **kwargs: Optional args to compute messages
        """
        edge_weights = kwargs.get("edge_weights", None)

        return (
            src_features
            if edge_weights is None
            else edge_weights.reshape(-1, 1) * src_features
        )

    def aggregate(self, messages: mx.array, indices: mx.array, **kwargs) -> mx.array:
        """Aggregates the messages using the `self.aggr` strategy.

        Args:
            messages: Computed messages
            indices: Indices representing the nodes that receive messages
            **kwargs: Optional args to aggregate messages
        """
        return scatter(messages, indices, self.num_nodes, self.aggr)

    def update_nodes(self, aggregated: mx.array, **kwargs) -> mx.array:
        """Updates the final embeddings given the aggregated messages.

        Args:
            aggregated: aggregated messages
            **kwargs: optional args to update messages
        """
        return aggregated
