import math
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, get_args

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

FusedFormat = Literal["auto", "coo", "csr"]

# ``format="auto"`` switches from the COO kernel to the faster CSR kernel when
# the one-time COO -> CSR sort pays off: at once for large graphs, and after a
# few uses of the same ``edge_index`` for smaller ones (e.g. full-batch
# training), but not for mini-batches that are seen once per layer.
_AUTO_CSR_MIN_EDGES = 1_000_000
_AUTO_CSR_MIN_USES = 4
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
            :func:`fused_message_passing` for every layer (``"auto"``,
            ``"coo"`` or ``"csr"``).
    """
    global _fused_enabled, _fused_format_override
    if format is not None and format not in get_args(FusedFormat):
        raise ValueError(f"format must be 'auto', 'coo' or 'csr', got {format!r}")
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


class _CSREntry:
    __slots__ = ("edge_index", "uses", "csr", "failed")

    def __init__(self, edge_index: mx.array):
        self.edge_index = edge_index
        self.uses = 0
        self.csr: Any = None
        self.failed = False


class _CSRCache:
    """Small LRU cache of COO -> CSR conversions, keyed on the edge_index array.

    The cache keeps a reference to each edge_index, so its ``id`` cannot be
    reused by another array while the entry is alive. It also counts how often
    each edge_index was used, which ``format="auto"`` uses to decide when the
    conversion is worth it.
    """

    def __init__(self, maxsize: int = 16):
        self.maxsize = maxsize
        self._entries: "OrderedDict[Tuple[int, int, int], _CSREntry]" = OrderedDict()

    def _entry(self, edge_index: mx.array, num_rows: int, num_cols: int) -> _CSREntry:
        # num_rows = destination nodes (output rows), num_cols = source nodes
        # (rows of x). They differ for bipartite (src, dst) inputs.
        key = (id(edge_index), num_rows, num_cols)
        entry = self._entries.get(key)
        if entry is None or entry.edge_index is not edge_index:
            entry = self._entries[key] = _CSREntry(edge_index)
            if len(self._entries) > self.maxsize:
                self._entries.popitem(last=False)
        else:
            self._entries.move_to_end(key)
        entry.uses += 1
        return entry

    def get(self, edge_index: mx.array, num_rows: int, num_cols: int):
        """CSR of ``edge_index``, converted on first use, or ``None`` (the
        COO kernel) when it cannot be converted."""
        entry = self._entry(edge_index, num_rows, num_cols)
        if entry.csr is None and not entry.failed:
            try:
                entry.csr = _coo_to_csr(
                    edge_index[1], edge_index[0], num_rows, num_cols=num_cols
                )
            except ValueError:
                # The conversion reads the graph back on the host, which is
                # not possible while tracing (edge_index is an input of
                # mx.compile): fall back to the COO kernel, as "auto" does.
                entry.failed = True
        return entry.csr

    def get_if_worth_it(self, edge_index: mx.array, num_rows: int, num_cols: int):
        """CSR of ``edge_index`` if it is cached or worth converting now,
        otherwise ``None`` (the caller then uses the COO kernel)."""
        entry = self._entry(edge_index, num_rows, num_cols)
        if entry.csr is not None or entry.failed:
            return entry.csr
        if (
            edge_index.shape[1] < _AUTO_CSR_MIN_EDGES
            and entry.uses < _AUTO_CSR_MIN_USES
        ):
            return None
        try:
            entry.csr = _coo_to_csr(
                edge_index[1], edge_index[0], num_rows, num_cols=num_cols
            )
        except ValueError:
            # The conversion reads the graph back on the host, which is not
            # possible while tracing (edge_index is an input of mx.compile).
            entry.failed = True
        return entry.csr

    def clear(self):
        self._entries.clear()


_csr_cache = _CSRCache()


def clear_csr_cache() -> None:
    """Drops all cached CSR conversions used by the ``"auto"`` and ``"csr"``
    formats of fused message passing."""
    _csr_cache.clear()


def fused_message_passing(
    cls=None,
    *,
    format: FusedFormat = "auto",
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
      is a list or an aggregation module; max/min also need a recent
      mlx-cluster),
    * any message/aggregate kwarg other than ``edge_weights`` is set
      (e.g. ``edge_features``), since messages are then no longer a weighted
      sum of neighbour features,
    * the layer is a subclass of a decorated layer that overrides
      ``message()`` or ``aggregate()`` without also overriding
      ``message_and_aggregate()``.

    Args:
        format: ``"coo"`` works directly on ``edge_index``. ``"csr"`` sorts
            the graph once (cached per ``edge_index``) and uses a faster,
            deterministic, atomic-free kernel. ``"auto"`` (default) starts
            with ``"coo"`` and switches to ``"csr"`` when the sort pays off:
            immediately for graphs with at least a million edges, and once
            the same ``edge_index`` array has been used a few times
            otherwise (e.g. full-batch training).
        allow_message_override: a layer that overrides ``message()`` or
            ``aggregate()`` is rejected by default, because the fused kernel
            would ignore those overrides. Pass ``True`` only if the override
            reduces to the default ``edge_weights * x_j`` / scatter whenever
            the extra kwargs are ``None`` (e.g. ``GINConv``).

    .. note::
        Cached conversions are looked up by the identity of the
        ``edge_index`` array. Build a new array instead of modifying an
        ``edge_index`` in place (``edge_index[0, 3] = 5``) between calls, or
        call :func:`clear_csr_cache` after doing so.

    .. code-block:: python

        @fused_message_passing
        class MyConv(MessagePassing):
            ...


        @fused_message_passing(format="csr")
        class MyOtherConv(MessagePassing):
            ...
    """
    if format not in get_args(FusedFormat):
        raise ValueError(f"format must be 'auto', 'coo' or 'csr', got {format!r}")

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
        aggr: Aggregation strategy used to aggregate messages. Either:

            * a reduction of :func:`~mlx_graphs.utils.scatter`: ``"add"``
              (or ``"sum"``), ``"mean"``, ``"max"``, ``"min"``, ``"mul"``,
              ``"var"``, ``"std"`` or ``"median"``. These use
              :func:`~mlx_graphs.utils.scatter` directly, and the fused SpMM
              kernels for ``"add"``, ``"mean"``, ``"max"`` and ``"min"``;
            * an :class:`~mlx_graphs.nn.aggr.Aggregation` instance, or the
              name of one (e.g. ``"softmax"``, ``"powermean"`` or
              ``"lstm"``), stored in ``self.aggr_module``. As in PyG,
              ``"softmax"`` is :class:`~mlx_graphs.nn.aggr.SoftmaxAggregation`
              (one output per node), not the per-message normalization of
              ``scatter(..., "softmax")``. A name given with ``aggr_kwargs``
              (even a scatter reduction) also builds its aggregation module,
              with these arguments;
            * a list of names and/or aggregations, combined with
              :class:`~mlx_graphs.nn.aggr.MultiAggregation` (concatenated
              along the last axis by default), stored in ``self.aggr_module``.
              As in PyG, only layers built for it accept concatenated
              aggregations (``supports_concat_aggr``, e.g.
              :class:`~mlx_graphs.nn.SAGEConv` and
              :class:`~mlx_graphs.nn.GENConv`, which size their projection
              accordingly, or layers without a projection after aggregating);
              the others raise a ``ValueError``, and accept a list combined
              without changing the size (``aggr_kwargs={"mode": "sum"}``).

            Default: ``"add"``
        aggr_kwargs: Arguments used to build the aggregation module from a
            name or a list (e.g. ``{"mode": "sum"}`` for a list). Default:
            ``None``

    References:
        [1] `Gilmer et al. Neural Message Passing for Quantum Chemistry.\
     <https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf>`_

        [2] `Fey et al. PyG <https://pytorch-geometric.readthedocs.io/en/latest/\
    generated/torch_geometric.nn.conv.MessagePassing.html>`_
    """

    # Set by the @fused_message_passing decorator; None = gather-scatter only.
    _fused_format: Optional[FusedFormat] = None

    #: Whether the layer works with a list of aggregations whose outputs are
    #: concatenated (``MultiAggregation`` in ``"cat"`` mode), i.e. whether
    #: what follows the aggregation accepts ``len(aggr)`` times more features.
    #: Layers whose projections are sized for a single aggregation set it to
    #: ``False``, and then raise a clear error instead of a shape error (or,
    #: for multi-head attention, a silently wrong output).
    supports_concat_aggr: bool = True

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

    def __init__(
        self,
        aggr: Union[ScatterAggregations, List[Union[str, Any]], Any] = "add",
        aggr_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if aggr == "sum":  # PyG's name of "add"
            aggr = "add"
        if (
            isinstance(aggr, str)
            and aggr in get_args(ScatterAggregations)
            # scatter's "softmax" normalizes the messages (one output per
            # edge): as an aggregation it means SoftmaxAggregation, as in PyG.
            and aggr != "softmax"
            # arguments are only taken by aggregation modules
            and not aggr_kwargs
        ):
            # Plain reductions go straight to `scatter` (and to the fused
            # kernels when possible), without an aggregation module.
            self.aggr = aggr
        else:
            from mlx_graphs.nn.aggr import Aggregation, MultiAggregation
            from mlx_graphs.nn.aggr.resolver import aggregation_resolver

            if isinstance(aggr, (list, tuple)):
                self.aggr_module = MultiAggregation(list(aggr), **(aggr_kwargs or {}))
                self.aggr = [str(a) for a in aggr]
                module = self.aggr_module
                if (
                    not self.supports_concat_aggr
                    and module.mode == "cat"
                    and len(module.aggrs) > 1
                ):
                    raise ValueError(
                        f"{type(self).__name__} does not support a list of "
                        "aggregations whose outputs are concatenated: its "
                        "layers expect the size of one aggregation (as in "
                        "PyG). Combine them without changing the size, e.g. "
                        "aggr_kwargs={'mode': 'sum'} ('mean', 'max', ...), or "
                        "use a layer that supports it, such as SAGEConv or "
                        "GENConv."
                    )
            elif isinstance(aggr, (str, Aggregation)):
                try:
                    self.aggr_module = aggregation_resolver(aggr, **(aggr_kwargs or {}))
                except ValueError:
                    raise ValueError(
                        "Invalid aggregation function.",
                        f"Available values are {get_args(ScatterAggregations)}, "
                        "a list of them, or an Aggregation",
                    ) from None
                self.aggr = str(aggr)
            else:
                raise ValueError(
                    "Invalid aggregation function.",
                    f"Available values are {get_args(ScatterAggregations)}, "
                    f"a list of them, or an Aggregation (got {type(aggr).__name__})",
                )
        self.num_nodes = None

    def __call__(self, edge_index: mx.array, node_features: mx.array, **kwargs: Any):
        raise NotImplementedError

    def propagate(
        self,
        edge_index: mx.array,
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        message_kwargs: Optional[Dict] = None,
        aggregate_kwargs: Optional[Dict] = None,
        update_kwargs: Optional[Dict] = None,
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
        message_kwargs = message_kwargs or {}
        aggregate_kwargs = aggregate_kwargs or {}
        update_kwargs = update_kwargs or {}
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

        if self._can_fuse(edge_index, node_features, message_kwargs, aggregate_kwargs):
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
        node_features: Union[mx.array, Tuple[mx.array, mx.array]],
        message_kwargs: Optional[Dict],
        aggregate_kwargs: Optional[Dict],
    ) -> bool:
        if self._fused_format is None or not is_fused_message_passing_enabled():
            return False
        # only plain string reductions have a fused kernel (not lists or
        # aggregation modules, nor "mul"/"var"/"std"/"median")
        if not isinstance(self.aggr, str) or self.aggr not in _FUSED_AGGRS:
            return False
        if "aggr_module" in self:
            return False
        message_kwargs = message_kwargs or {}
        if any(v is not None for k, v in message_kwargs.items() if k != "edge_weights"):
            return False
        if any(v is not None for v in (aggregate_kwargs or {}).values()):
            return False
        x = node_features[0] if isinstance(node_features, tuple) else node_features
        if x.ndim < 2 or not mx.issubdtype(x.dtype, mx.floating):
            return False
        weights = message_kwargs.get("edge_weights", None)
        if weights is not None:
            num_edges = edge_index.shape[1]
            per_edge = weights.size == num_edges  # one scalar per edge
            # one scalar per edge and per head, for [num_nodes, heads, ...] features
            per_head = (
                x.ndim >= 3
                and weights.ndim == 2
                and weights.shape == (num_edges, x.shape[1])
            )
            if not (per_edge or per_head):
                return False
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
            edge_weights: Optional weights: one scalar per edge, or, for
                multi-head features of shape ``[num_nodes, heads, channels]``,
                an array of shape ``[num_edges, heads]`` with one scalar per
                edge and head (e.g. attention coefficients).
        """
        x = node_features[0] if isinstance(node_features, tuple) else node_features
        weights = edge_weights
        if weights is not None and weights.size == edge_index.shape[1]:
            weights = weights.reshape(-1)  # one scalar per edge
        if self.aggr in ("max", "min"):
            return self._fused_extremum(edge_index, x, weights)
        return self._fused_spmm(edge_index, x, weights)

    def _fused_extremum(
        self, edge_index: mx.array, x: mx.array, weights: Optional[mx.array]
    ) -> mx.array:
        """Fused max/min whose gradient is split equally among tied messages,
        as :func:`~mlx_graphs.utils.scatter` max/min. The kernels give the
        full gradient to each tied message, so the backward pass divides the
        gradient of each output by its number of ties first."""
        src, dst = edge_index[0], edge_index[1]

        def spmm(*primals):
            w = primals[1] if len(primals) > 1 else None
            return self._fused_spmm(edge_index, primals[0], w)

        def split_vjp(primals, cotangent, output):
            x = primals[0]
            msg = x[src]
            if len(primals) > 1:
                w = primals[1]
                msg = w.reshape(w.shape + (1,) * (msg.ndim - w.ndim)) * msg
            num_ties = (
                mx.zeros(output.shape, dtype=mx.int32)
                .at[dst]
                .add((msg == output[dst]).astype(mx.int32))
            )
            split = cotangent / mx.maximum(num_ties, 1).astype(cotangent.dtype)
            return mx.vjp(spmm, list(primals), [split])[1]

        if weights is None:

            @mx.custom_function
            def extremum(x):
                return self._fused_spmm(edge_index, x, None)

            @extremum.vjp
            def extremum_vjp(primals, cotangent, output):
                x = primals if isinstance(primals, mx.array) else primals[0]
                return tuple(split_vjp((x,), cotangent, output))

            return extremum(x)

        @mx.custom_function
        def weighted_extremum(x, w):
            return self._fused_spmm(edge_index, x, w)

        @weighted_extremum.vjp
        def weighted_extremum_vjp(primals, cotangent, output):
            return tuple(split_vjp(tuple(primals), cotangent, output))

        return weighted_extremum(x, weights)

    def _fused_spmm(
        self, edge_index: mx.array, x: mx.array, weights: Optional[mx.array]
    ) -> mx.array:
        """The SpMM kernels, on source features ``x`` and optional weights
        (one per edge, flattened, or ``[num_edges, heads]``)."""
        reduce = "sum" if self.aggr == "add" else self.aggr  # mean / max / min
        num_edges = edge_index.shape[1]
        out_shape = (self.num_nodes, *x.shape[1:])
        per_head = weights is not None and weights.size != num_edges

        # The kernels work on [num_nodes, features]: with one weight per
        # (edge, head) each head is its own SpMM, otherwise heads and channels
        # are flattened into the feature axis (explicit sizes: reshape cannot
        # infer them for 0 nodes).
        if per_head:
            heads = x.shape[1]
            xs = [
                x[:, h].reshape(x.shape[0], math.prod(x.shape[2:]))
                for h in range(heads)
            ]
        else:
            heads = 1
            xs = [x.reshape(x.shape[0], math.prod(x.shape[1:]))]

        csr = self._fused_csr(edge_index, x.shape[0])
        if csr is not None:
            if weights is not None:
                weights = weights[csr.perm]
            outs = [
                _spmm_csr(
                    csr.rowptr,
                    csr.col,
                    xs[h],
                    None
                    if weights is None
                    else (weights[:, h] if per_head else weights),
                    reduce=reduce,
                    # load-balanced path for high-degree nodes (mlx-cluster >= the
                    # long-row release; older CSR tuples have no such field)
                    long_rows=getattr(csr, "long_rows", None),
                    # cached transpose: the backward also runs on the CSR kernel
                    transpose=getattr(csr, "transpose", None),
                )
                for h in range(heads)
            ]
        else:
            outs = [
                _spmm_coo(
                    edge_index[1],
                    edge_index[0],
                    xs[h],
                    None
                    if weights is None
                    else (weights[:, h] if per_head else weights),
                    num_rows=self.num_nodes,
                    reduce=reduce,
                )
                for h in range(heads)
            ]

        out = outs[0] if heads == 1 else mx.stack(outs, axis=1)
        return out.reshape(out_shape)

    def _fused_csr(self, edge_index: mx.array, num_src_nodes: int):
        """CSR form of the graph for the fused kernel, or ``None`` for COO."""
        fmt = _fused_format_override or self._fused_format
        if fmt == "csr":
            return _csr_cache.get(edge_index, self.num_nodes, num_src_nodes)
        if fmt == "auto":
            return _csr_cache.get_if_worth_it(edge_index, self.num_nodes, num_src_nodes)
        return None

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
                used to weigh neighbor features during aggregation. For
                multi-head source features of shape (num_edges, heads, channels),
                an array of shape (num_edges, heads) weighs each head separately.
                Default: ``None``
            **kwargs: Optional args to compute messages
        """
        edge_weights = kwargs.get("edge_weights", None)
        if edge_weights is None:
            return src_features
        if edge_weights.size == src_features.shape[0]:
            # one scalar per edge, broadcast over all feature axes
            shape = (-1,) + (1,) * (src_features.ndim - 1)
            return edge_weights.reshape(shape) * src_features
        # one scalar per edge and head
        shape = edge_weights.shape + (1,) * (src_features.ndim - edge_weights.ndim)
        return edge_weights.reshape(shape) * src_features

    def aggregate(self, messages: mx.array, indices: mx.array, **kwargs) -> mx.array:
        """Aggregates the messages using the `self.aggr` strategy, or
        `self.aggr_module` when `aggr` is a list or an aggregation module.

        Args:
            messages: Computed messages
            indices: Indices representing the nodes that receive messages
            **kwargs: Optional args to aggregate messages
        """
        if isinstance(self.aggr, str) and "aggr_module" not in self:
            return scatter(messages, indices, self.num_nodes, self.aggr)
        return self.aggr_module(messages, indices, dim_size=self.num_nodes)

    def update_nodes(self, aggregated: mx.array, **kwargs) -> mx.array:
        """Updates the final embeddings given the aggregated messages.

        Args:
            aggregated: aggregated messages
            **kwargs: optional args to update messages
        """
        return aggregated
