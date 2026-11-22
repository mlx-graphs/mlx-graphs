"""Conversion of homogeneous GNN models into heterogeneous ones (``to_hetero``).

PyG implements ``to_hetero`` by tracing ``forward`` with ``torch.fx`` and
rewriting the traced graph. MLX has no such tracer, so this module uses
*module rewriting* plus *type-packed execution* instead:

1. The model is deep-copied. Every :class:`MessagePassing` layer in it is
   replaced by a dispatcher holding one copy of the layer per edge type, and
   every other sub-module (``Linear``, normalizations, MLPs, Jumping
   Knowledge, ...) by a dispatcher holding one copy per node type. Sub-modules
   that contain message passing layers themselves (e.g. a ``GraphSAGE`` used
   inside a bigger model) are rewritten recursively.
2. At call time, the per-type input dicts are packed into ordinary arrays:
   node-level dicts become one array with the rows of all node types
   concatenated (in metadata order), edge-level dicts one array with the edges
   of all edge types. The model's own, unchanged ``__call__`` then runs on
   these arrays, so every row-wise operation in it (activations, dropout,
   residual sums, concatenation of features, ...) works as written.
3. The dispatchers split the packed arrays back per type: a node-type copy
   runs on its node type's rows, an edge-type copy on its edges with
   ``(src_features, dst_features)`` (or just the features when both ends have
   the same type), and the outputs of edge types sharing a destination are
   combined with ``aggr``. Arrays that reach a layer unchanged from the
   model's inputs are not split but looked up in the input dicts, so packing
   them never costs any memory (MLX evaluates them lazily, and they are never
   needed).
"""

import copy
import inspect
import math
import threading
import warnings
from collections import deque
from typing import Any, Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from mlx_graphs.nn.message_passing import MessagePassing

NodeType = str
EdgeType = tuple[str, str, str]
Metadata = tuple[list[NodeType], list[EdgeType]]

_AGGRS = ("sum", "mean", "max", "min", "mul")
_COMBINE: dict[str, Callable[[mx.array, mx.array], mx.array]] = {
    "sum": mx.add,
    "mean": mx.add,  # summed, then divided by the number of edge types
    "max": mx.maximum,
    "min": mx.minimum,
    "mul": mx.multiply,
}

# Parameter-free modules that act on every row independently. They are kept
# as they are and run once on the packed array instead of once per node type.
_ROW_WISE_MODULES = (
    nn.Dropout,
    nn.ReLU,
    nn.LeakyReLU,
    nn.ELU,
    nn.GELU,
    nn.SiLU,
    nn.Tanh,
    nn.Sigmoid,
    nn.Identity,
)


def _type_key(key: Union[NodeType, EdgeType]) -> str:
    """Name under which the copy of a module for a node/edge type is stored,
    as in PyG (``"author"``, ``"author__writes__paper"``)."""
    return "__".join(key) if isinstance(key, tuple) else key


def _clone(module: nn.Module) -> nn.Module:
    """Deep copy of a module that keeps (compiled) functions such as
    ``nn.relu``, which cannot be deep-copied, shared."""
    memo: dict[int, Any] = {}
    for _, sub in module.named_modules():
        for value in vars(sub).values():
            if callable(value) and not isinstance(value, (nn.Module, mx.array, type)):
                memo[id(value)] = value
    return copy.deepcopy(module, memo)


def _shuffle(value: Any) -> Any:
    if (
        not isinstance(value, mx.array)
        or value.size <= 1
        or not mx.issubdtype(value.dtype, mx.floating)
    ):
        return value
    permutation = mx.random.permutation(value.size)
    return value.reshape(-1)[permutation].reshape(value.shape)


def _reinitialize(module: nn.Module) -> nn.Module:
    """Re-randomizes the parameters of a copy, like PyG's
    ``reset_parameters()`` on the copies made by ``to_hetero``.

    Modules with a ``reset_parameters()`` method are reset with it (a fresh
    initialization, e.g. of the running statistics of the normalizations of
    ``mlx_graphs.nn.norm``). MLX modules usually have none, and their
    initialization cannot be redone without knowing it (layers often replace
    the default initialization of their ``nn.Linear`` sub-layers), so the
    trainable arrays of the other modules are randomly permuted instead.
    Initializations draw every entry independently from the same
    distribution, so a permutation is a draw from that initialization (and
    leaves constant arrays, e.g. biases or norm weights, unchanged).

    Frozen arrays are buffers rather than initialized weights (e.g. the
    running statistics of ``mlx.nn.BatchNorm``, whose per-feature means and
    variances must stay paired, or fixed encoding frequencies), and are kept
    as they are.
    """
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()
    else:
        frozen = getattr(module, "_no_grad", set())
        for key, value in list(module.items()):
            if key.startswith("_") or key in frozen:
                continue
            if isinstance(value, (mx.array, list, tuple, dict)):
                shuffled = tree_map(
                    _shuffle, value, is_leaf=lambda v: isinstance(v, nn.Module)
                )
                if shuffled is not value:
                    module[key] = shuffled
    for _, child in tree_flatten(
        module.children(), is_leaf=lambda v: isinstance(v, nn.Module)
    ):
        _reinitialize(child)
    return module


# --------------------------------------------------------------------------- #
# Changing the input size of copies that read the raw input features
# --------------------------------------------------------------------------- #


def _new_linear(old: nn.Linear, input_dim: int) -> nn.Linear:
    """A fresh linear layer of the same class and output size as ``old``,
    reading ``input_dim`` features."""
    return type(old)(input_dim, old.weight.shape[0], bias="bias" in old)


def _adapt_first_linear(module: nn.Module, input_dim: int) -> nn.Module:
    from mlx_graphs.nn.models.basic_gnn import MLP

    if isinstance(module, nn.Linear):
        return _new_linear(module, input_dim)
    if isinstance(module, MLP):
        module.lins[0] = _new_linear(module.lins[0], input_dim)
        module.dims[0] = input_dim
        return module
    if isinstance(module, nn.Sequential) and len(module.layers) > 0:
        module.layers[0] = _adapt_first_linear(module.layers[0], input_dim)
        return module
    raise _cannot_adapt(module)


def _cannot_adapt(module: nn.Module) -> TypeError:
    return TypeError(
        f"to_hetero(input_dims=...) cannot change the input size of "
        f"{type(module).__name__}, which reads the input node features. "
        "Supported layers are Linear, MLP, nn.Sequential starting with a "
        "Linear, SAGEConv, GraphConv, GATConv, GATv2Conv, GINConv and GCNConv. "
        "Otherwise, project the node features of every type to a common size "
        "first (e.g. with HeteroDictLinear) and call to_hetero without "
        "input_dims."
    )


def _same_dims(module: nn.Module, src_dim: Optional[int], dst_dim: Optional[int]):
    if src_dim is None or dst_dim is None or src_dim != dst_dim:
        raise ValueError(
            f"{type(module).__name__} needs the same number of input features "
            f"for source and destination nodes, got {src_dim} and {dst_dim}. "
            "Project the node features to a common size first (e.g. with "
            "HeteroDictLinear), or use a layer with separate source and "
            "destination weights such as SAGEConv or GraphConv."
        )
    return src_dim


def _adapt_input_dims(
    module: nn.Module, src_dim: Optional[int], dst_dim: Optional[int]
) -> nn.Module:
    """Rebuilds the input weights of ``module`` for ``src_dim`` source and
    ``dst_dim`` destination features (``None``: unchanged). Returns the
    (possibly new) module."""
    from mlx_graphs.nn.conv.gat_conv import GATConv
    from mlx_graphs.nn.conv.gatv2_conv import GATv2Conv
    from mlx_graphs.nn.conv.gcn_conv import GCNConv
    from mlx_graphs.nn.conv.gin_conv import GINConv
    from mlx_graphs.nn.conv.graph_conv import GraphConv
    from mlx_graphs.nn.conv.sage_conv import SAGEConv

    if isinstance(module, (SAGEConv, GraphConv)):
        if src_dim is not None:
            module.neigh_proj = _new_linear(module.neigh_proj, src_dim)
            module.src_features_dim = src_dim
        if dst_dim is not None:
            if "self_proj" in module:  # absent for SAGEConv(root_weight=False)
                module.self_proj = _new_linear(module.self_proj, dst_dim)
            module.dst_features_dim = dst_dim
        src, dst = module.src_features_dim, module.dst_features_dim
        module.node_features_dim = src if src == dst else (src, dst)
        return module
    if isinstance(module, GATConv):
        if "lin_proj" in module and src_dim is not None and src_dim == dst_dim:
            module.lin_proj = _new_linear(module.lin_proj, src_dim)
            return module
        if "lin_proj" in module:
            shared = module.lin_proj
            del module["lin_proj"]
            module.lin_src_proj = _clone(shared)
            module.lin_dst_proj = _clone(shared)
        if src_dim is not None:
            module.lin_src_proj = _new_linear(module.lin_src_proj, src_dim)
        if dst_dim is not None:
            module.lin_dst_proj = _new_linear(module.lin_dst_proj, dst_dim)
        return module
    if isinstance(module, GATv2Conv):
        if src_dim is not None:
            module.W_src = _new_linear(module.W_src, src_dim)
        if dst_dim is not None:
            module.W_dst = _new_linear(module.W_dst, dst_dim)
        return module
    if isinstance(module, GINConv):
        module.mlp = _adapt_first_linear(
            module.mlp, _same_dims(module, src_dim, dst_dim)
        )
        return module
    if isinstance(module, GCNConv):
        module.linear = _new_linear(module.linear, _same_dims(module, src_dim, dst_dim))
        return module
    if isinstance(module, MessagePassing):
        raise _cannot_adapt(module)
    return _adapt_first_linear(module, src_dim if src_dim is not None else dst_dim)


# --------------------------------------------------------------------------- #
# Call context: how packed arrays map to node and edge types
# --------------------------------------------------------------------------- #

_state = threading.local()


def _current_context(module: nn.Module) -> "_Context":
    stack = getattr(_state, "stack", None)
    if not stack:
        raise RuntimeError(
            f"This {type(module).__name__} holds one copy of a layer per node or "
            "edge type and can only run inside the model returned by to_hetero."
        )
    return stack[-1]


def _map_arrays(value: Any, fn: Callable, path: tuple = ()) -> Any:
    """Applies ``fn(array, path)`` to every array in nested tuples, lists and
    dicts, where ``path`` locates the array in the arguments."""
    if isinstance(value, mx.array):
        return fn(value, path)
    if isinstance(value, tuple):
        return tuple(_map_arrays(v, fn, path + (i,)) for i, v in enumerate(value))
    if isinstance(value, list):
        return [_map_arrays(v, fn, path + (i,)) for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {k: _map_arrays(v, fn, path + (k,)) for k, v in value.items()}
    return value


class _Context:
    """Node/edge layout of the packed arrays of one call of a to_hetero model."""

    def __init__(
        self,
        node_types: list[NodeType],
        num_nodes: dict[NodeType, int],
        edge_types: list[EdgeType],
        num_edges: dict[EdgeType, int],
        input_dims: Optional[dict[NodeType, int]],
        probe_width: Optional[int] = None,
    ):
        self.node_types = node_types
        self.num_nodes = num_nodes
        self.node_slices, start = {}, 0
        for node_type in node_types:
            self.node_slices[node_type] = (start, start + num_nodes[node_type])
            start += num_nodes[node_type]
        self.total_nodes = start

        self.edge_types = edge_types
        self.num_edges = num_edges
        self.edge_slices, start = {}, 0
        for edge_type in edge_types:
            self.edge_slices[edge_type] = (start, start + num_edges[edge_type])
            start += num_edges[edge_type]
        self.total_edges = start

        self.input_dims = input_dims
        # Width of the input features during the dry run of to_hetero, used to
        # recognize the arrays derived from them.
        self.probe_width = probe_width
        # id(packed array) -> (level, per-type dict, packed array, is_input)
        self.packed: dict[int, tuple[str, dict, mx.array, bool]] = {}

    # Packed arrays built from the inputs ----------------------------------- #

    def register(self, level: str, values: dict, packed: mx.array, is_input: bool):
        self.packed[id(packed)] = (level, values, packed, is_input)

    def _entry(self, array: mx.array):
        entry = self.packed.get(id(array))
        return entry if entry is not None and entry[2] is array else None

    # Levels and per-type parts --------------------------------------------- #

    def level(self, array: Any) -> Optional[str]:
        """``"node"``, ``"edge"`` (one row per edge), ``"edge_index"``
        (``[2, num_edges]``) or ``None`` (not split per type)."""
        if not isinstance(array, mx.array) or array.ndim == 0:
            return None
        entry = self._entry(array)
        if entry is not None:
            return entry[0]
        if array.shape[0] == self.total_nodes:
            return "node"
        if self.edge_types and array.shape[0] == self.total_edges:
            return "edge"
        if (
            self.edge_types
            and array.ndim == 2
            and array.shape[0] == 2
            and array.shape[1] == self.total_edges
        ):
            return "edge_index"
        return None

    def is_input_features(self, array: mx.array) -> bool:
        """Whether ``array`` holds (or, in the dry run, derives from) the input
        node features."""
        entry = self._entry(array)
        if entry is not None:
            return entry[3]
        return self.probe_width is not None and array.shape[-1] == self.probe_width

    def node_part(self, array: mx.array, node_type: NodeType, raw: bool) -> mx.array:
        entry = self._entry(array)
        if entry is not None:
            return entry[1][node_type]
        start, end = self.node_slices[node_type]
        part = array[start:end]
        if raw and self.input_dims is not None:
            # padded input features: keep the node type's own features
            part = part[..., : self.input_dims[node_type]]
        return part

    def edge_part(self, array: mx.array, edge_type: EdgeType, level: str) -> mx.array:
        entry = self._entry(array)
        if entry is not None:
            return entry[1][edge_type]
        start, end = self.edge_slices[edge_type]
        return array[:, start:end] if level == "edge_index" else array[start:end]

    # Packing per-type outputs ---------------------------------------------- #

    def pack_nodes(self, outputs: dict[NodeType, Any], where: str) -> Any:
        first = outputs[self.node_types[0]]
        if isinstance(first, mx.array):
            for node_type in self.node_types:
                out = outputs[node_type]
                if not isinstance(out, mx.array) or (
                    out.ndim == 0 or out.shape[0] != self.num_nodes[node_type]
                ):
                    raise ValueError(
                        f"{where} returned {getattr(out, 'shape', type(out))} for "
                        f"node type '{node_type}', expected one row per node "
                        f"({self.num_nodes[node_type]}). to_hetero only supports "
                        "layers that return one row per (destination) node."
                    )
            parts = [outputs[t] for t in self.node_types]
            return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
        if isinstance(first, (tuple, list)):
            packed = [
                self.pack_nodes({t: outputs[t][i] for t in self.node_types}, where)
                for i in range(len(first))
            ]
            return type(first)(packed)
        if first is None:
            return None
        raise TypeError(
            f"{where} returned a {type(first).__name__}; to_hetero supports "
            "layers returning arrays (or tuples/lists of arrays)."
        )

    def unpack(self, value: Any) -> Any:
        """Splits the packed arrays in a model output into per-type dicts."""
        if isinstance(value, mx.array):
            entry = self._entry(value)
            if entry is not None:
                return dict(entry[1])
            level = self.level(value)
            if level == "node":
                return {t: self.node_part(value, t, raw=False) for t in self.node_types}
            if level in ("edge", "edge_index"):
                return {et: self.edge_part(value, et, level) for et in self.edge_types}
            return value
        if isinstance(value, (tuple, list)):
            return type(value)(self.unpack(v) for v in value)
        if isinstance(value, dict):
            return {k: self.unpack(v) for k, v in value.items()}
        return value


# --------------------------------------------------------------------------- #
# Dispatchers: one copy of a module per node type / per edge type
# --------------------------------------------------------------------------- #


class _TypeDispatch(nn.Module):
    """Common part of the per-node-type and per-edge-type dispatchers."""

    def __init__(self, module: nn.Module, keys: list, name: str):
        super().__init__()
        self._keys = list(keys)
        self._name = name
        self._module_class = type(module).__name__
        # Paths (in the call arguments) of the inputs that are the input node
        # features, found by the dry run of to_hetero(input_dims=...).
        self._raw_paths: set = set()
        self._adapted = False
        for key in keys:
            copy_ = _clone(module)
            if len(keys) > 1:
                _reinitialize(copy_)
            self[_type_key(key)] = copy_

    def module_for(self, key: Union[NodeType, EdgeType]) -> nn.Module:
        """The copy of the module used for a node type or an edge type."""
        return self[_type_key(key)]

    def _check_raw(self, ctx: _Context, raw_paths: set):
        """In the dry run, rebuilds the copies that read the input features
        for the input size of each node type."""
        if ctx.probe_width is None:
            return
        if self._adapted and raw_paths != self._raw_paths:
            raise ValueError(
                f"'{self._name}' ({self._module_class}) is called both on the "
                "input node features and on other features, which "
                "to_hetero(input_dims=...) does not support."
            )
        if raw_paths and not self._adapted:
            self._raw_paths = set(raw_paths)
            self._adapt(ctx)
            self._adapted = True

    def _adapt(self, ctx: _Context):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self._name}: {self._module_class})"


class _NodeTypeModules(_TypeDispatch):
    """One copy of a (non message passing) module per node type."""

    def _adapt(self, ctx: _Context):
        for node_type in self._keys:
            dim = ctx.input_dims[node_type]
            key = _type_key(node_type)
            self[key] = _adapt_input_dims(self[key], dim, dim)

    def __call__(self, *args, **kwargs):
        ctx = _current_context(self)
        raw_paths: set = set()
        has_nodes = [False]

        def collect(array, path):
            level = ctx.level(array)
            if level == "node":
                has_nodes[0] = True
                if ctx.probe_width is not None and ctx.is_input_features(array):
                    raw_paths.add(path)
            elif level in ("edge", "edge_index"):
                raise NotImplementedError(
                    f"'{self._name}' ({self._module_class}) is called on edge-level "
                    "values. to_hetero copies non message passing modules per "
                    "node type only; apply edge-level modules outside the "
                    "converted model."
                )
            return array

        _map_arrays((args, kwargs), collect)
        if not has_nodes[0]:
            raise NotImplementedError(
                f"'{self._name}' ({self._module_class}) is not called on "
                "node-level features (one row per node). to_hetero copies such "
                "modules per node type, so graph-level uses (e.g. after a "
                "global pooling) are not supported."
            )
        self._check_raw(ctx, raw_paths)

        outputs = {}
        for node_type in self._keys:

            def part(array, path, node_type=node_type):
                if ctx.level(array) != "node":
                    return array
                return ctx.node_part(array, node_type, path in self._raw_paths)

            args_t, kwargs_t = _map_arrays((args, kwargs), part)
            outputs[node_type] = self[_type_key(node_type)](*args_t, **kwargs_t)
        return ctx.pack_nodes(outputs, f"'{self._name}' ({self._module_class})")


class _EdgeTypeModules(_TypeDispatch):
    """One copy of a message passing layer per edge type; the outputs of the
    edge types sharing a destination node type are combined with ``aggr``."""

    def __init__(self, module: nn.Module, edge_types: list, name: str, aggr: str):
        super().__init__(module, edge_types, name)
        self._aggr = aggr

    def _adapt(self, ctx: _Context):
        for edge_type in self._keys:
            src, _, dst = edge_type
            src_raw = dst_raw = False
            for path in self._raw_paths:
                # a (src, dst) tuple argument has one path per side
                side = path[-1] if self._is_pair_path(path) else None
                src_raw |= side in (None, 0)
                dst_raw |= side in (None, 1)
            key = _type_key(edge_type)
            self[key] = _adapt_input_dims(
                self[key],
                ctx.input_dims[src] if src_raw else None,
                ctx.input_dims[dst] if dst_raw else None,
            )

    def _is_pair_path(self, path: tuple) -> bool:
        return path in self._pair_paths

    def _find_pairs(self, ctx: _Context, value: Any, path: tuple, pairs: set):
        """Paths of ``(src_features, dst_features)`` tuples of node-level
        arrays in the arguments."""
        if (
            len(path) >= 2  # inside an argument, not the argument list itself
            and isinstance(value, tuple)
            and len(value) == 2
            and all(ctx.level(v) == "node" for v in value)
        ):
            pairs.add(path + (0,))
            pairs.add(path + (1,))
        elif isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                self._find_pairs(ctx, v, path + (i,), pairs)
        elif isinstance(value, dict):
            for k, v in value.items():
                self._find_pairs(ctx, v, path + (k,), pairs)

    def __call__(self, *args, **kwargs):
        ctx = _current_context(self)
        if not ctx.edge_types:
            raise ValueError(
                f"'{self._name}' ({self._module_class}) needs edges, but the model "
                "was called without an edge index dict (keyed by edge types)."
            )

        pairs: set = set()
        self._find_pairs(ctx, (args, kwargs), (), pairs)
        self._pair_paths = pairs
        raw_paths: set = set()
        where = f"'{self._name}' ({self._module_class})"

        def collect(array, path):
            _check_edge_index(ctx, array, where)
            if (
                ctx.probe_width is not None
                and ctx.level(array) == "node"
                and ctx.is_input_features(array)
            ):
                raw_paths.add(path)
            return array

        _map_arrays((args, kwargs), collect)
        self._check_raw(ctx, raw_paths)

        outputs: dict[NodeType, list] = {}
        for edge_type in self._keys:
            if edge_type not in ctx.edge_slices:
                continue  # not in this call's edge index dict
            src, _, dst = edge_type

            def part(array, path, edge_type=edge_type):
                level = ctx.level(array)
                raw = path in self._raw_paths
                if level == "node":
                    if path in pairs:
                        node_type = src if path[-1] == 0 else dst
                        return ctx.node_part(array, node_type, raw)
                    x_src = ctx.node_part(array, src, raw)
                    if src == dst:
                        return x_src
                    return (x_src, ctx.node_part(array, dst, raw))
                if level in ("edge", "edge_index"):
                    return ctx.edge_part(array, edge_type, level)
                return array

            args_e, kwargs_e = _map_arrays((args, kwargs), part)
            try:
                out = self[_type_key(edge_type)](*args_e, **kwargs_e)
            except Exception as e:
                hint = (
                    " It is called with a (src_features, dst_features) tuple "
                    "since source and destination node types differ; check that "
                    "the layer supports bipartite graphs."
                    if src != dst
                    else ""
                )
                raise RuntimeError(
                    f"'{self._name}' ({self._module_class}) failed on edge type "
                    f"{edge_type}: {e}.{hint}"
                ) from e
            if not isinstance(out, mx.array) or (
                out.ndim == 0 or out.shape[0] != ctx.num_nodes[dst]
            ):
                raise ValueError(
                    f"'{self._name}' ({self._module_class}) returned "
                    f"{getattr(out, 'shape', type(out))} on edge type {edge_type}, "
                    f"expected one row per '{dst}' node ({ctx.num_nodes[dst]})."
                )
            outputs.setdefault(dst, []).append(out)

        missing = [t for t in ctx.node_types if t not in outputs]
        if missing:
            raise ValueError(
                f"Node types {missing} are not the destination of any edge type "
                f"passed to '{self._name}', so they get no new representation. "
                "Add reverse edge types (e.g. ('paper', 'rev_writes', 'author'))."
            )
        combined = {t: self._combine(outs) for t, outs in outputs.items()}
        return ctx.pack_nodes(combined, f"'{self._name}' ({self._module_class})")

    def _combine(self, outs: list[mx.array]) -> mx.array:
        # Pairwise, in the same order as PyG's to_hetero.
        queue = deque(outs)
        op = _COMBINE[self._aggr]
        while len(queue) >= 2:
            first, second = queue.popleft(), queue.popleft()
            queue.append(op(first, second))
        out = queue[0]
        if self._aggr == "mean" and len(outs) > 1:
            out = out / len(outs)
        return out


class _EdgeTypeWeights:
    """Edge type of every message of the current call (not a parameter)."""

    edge_type: Optional[mx.array] = None


def _scale_messages(conv: MessagePassing, state: _EdgeTypeWeights):
    """Makes ``conv`` multiply each message by the weight of its edge type in
    ``conv.edge_type_weight`` (PyG's message forward hook)."""
    message = conv.message  # bound method of the class

    def scaled_message(*args, **kwargs):
        out = message(*args, **kwargs)
        if out.shape[0] != state.edge_type.shape[0]:
            raise ValueError(
                f"Number of messages ({out.shape[0]}) does not match the number "
                f"of edges ({state.edge_type.shape[0]}). Does the layer add "
                "self-loops? Disable them (add_self_loops=False)."
            )
        weight = conv.edge_type_weight.reshape(-1)[state.edge_type]
        return weight.reshape((-1,) + (1,) * (out.ndim - 1)) * out

    # Instance attributes (not parameters): the scaled message function, and
    # no fused kernel, which would skip message().
    object.__setattr__(conv, "message", scaled_message)
    object.__setattr__(conv, "_fused_format", None)


class _BasisModules(nn.Module):
    """``num_bases`` copies of a message passing layer run on the whole graph
    (all node and edge types), each weighting the messages of every edge type
    with a learned coefficient (``edge_type_weight``), summed."""

    def __init__(
        self, module: MessagePassing, edge_types: list, num_bases: int, name: str
    ):
        super().__init__()
        self._edge_types = list(edge_types)
        self._name = name
        self._module_class = type(module).__name__
        self._state = _EdgeTypeWeights()
        bound = math.sqrt(6.0 / (1 + len(edge_types)))  # xavier_uniform
        self.convs = []
        for _ in range(num_bases):
            conv = _clone(module)
            if num_bases > 1:
                _reinitialize(conv)
            conv.edge_type_weight = mx.random.uniform(
                -bound, bound, (1, len(edge_types))
            )
            _scale_messages(conv, self._state)
            self.convs.append(conv)

    def __call__(self, *args, **kwargs):
        ctx = _current_context(self)
        if not ctx.edge_types:
            raise ValueError(
                f"'{self._name}' ({self._module_class}) needs edges, but the model "
                "was called without an edge index dict (keyed by edge types)."
            )
        type_ids = [
            mx.full((ctx.num_edges[et],), self._edge_types.index(et), mx.int32)
            for et in ctx.edge_types
        ]
        edge_type = type_ids[0] if len(type_ids) == 1 else mx.concatenate(type_ids)

        def to_global(array, path):
            _check_edge_index(ctx, array, f"'{self._name}' ({self._module_class})")
            if ctx.level(array) != "edge_index":
                return array
            # one graph over all nodes: shift the local ids of every edge type
            parts = []
            for et in ctx.edge_types:
                src, _, dst = et
                offset = mx.array(
                    [[ctx.node_slices[src][0]], [ctx.node_slices[dst][0]]],
                    dtype=mx.int32,
                )
                parts.append(ctx.edge_part(array, et, "edge_index") + offset)
            return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)

        args, kwargs = _map_arrays((args, kwargs), to_global)
        self._state.edge_type = edge_type
        try:
            out = None
            for conv in self.convs:
                res = conv(*args, **kwargs)
                out = res if out is None else out + res
        finally:
            self._state.edge_type = None
        if not isinstance(out, mx.array) or out.shape[0] != ctx.total_nodes:
            raise ValueError(
                f"'{self._name}' ({self._module_class}) returned "
                f"{getattr(out, 'shape', type(out))}, expected one row per node "
                f"({ctx.total_nodes})."
            )
        return out

    def __repr__(self):
        return (
            f"_BasisModules({self._name}: {self._module_class}, "
            f"num_bases={len(self.convs)})"
        )


# --------------------------------------------------------------------------- #
# Model rewriting
# --------------------------------------------------------------------------- #


def _check_edge_index(ctx: _Context, array: mx.array, where: str):
    """Rejects an edge index that is not the packed one of the call (e.g.
    edges dropped or added in ``__call__``): it cannot be split per edge
    type, and passing it whole to every copy would mix the node ids of
    different types."""
    if (
        ctx.level(array) is None
        and array.ndim == 2
        and array.shape[0] == 2
        and mx.issubdtype(array.dtype, mx.integer)
    ):
        raise ValueError(
            f"{where} is called with an edge index of {array.shape[1]} edges, "
            f"while the edge index dict of the call has {ctx.total_edges}. "
            "to_hetero splits the edge index per edge type and cannot follow "
            "operations that add or remove edges in the model (e.g. "
            "dropout_edge or add_self_loops). Apply them to every edge type "
            "of the edge index dict before calling the model."
        )


def _contains_message_passing(module: nn.Module) -> bool:
    return any(isinstance(m, MessagePassing) for _, m in module.named_modules())


class _Rewriter:
    def __init__(self, metadata: Metadata, aggr: str, num_bases: Optional[int]):
        self.node_types, self.edge_types = metadata
        self.aggr = aggr
        self.num_bases = num_bases
        self.shared_arrays: list[str] = []
        # A module stored under several names (weight tying) is converted
        # once, so that all its uses share the same per-type copies (as PyG,
        # whose tracer resolves a shared module to a single target).
        self._converted: dict[int, Any] = {}

    def convert(self, value: Any, name: str) -> Any:
        if isinstance(value, nn.Module):
            if id(value) not in self._converted:
                self._converted[id(value)] = (value, self._convert(value, name))
            return self._converted[id(value)][1]
        return self._convert(value, name)

    def _convert(self, value: Any, name: str) -> Any:
        if isinstance(value, MessagePassing):
            if self.num_bases is not None:
                return _BasisModules(value, self.edge_types, self.num_bases, name)
            return _EdgeTypeModules(value, self.edge_types, name, self.aggr)
        if isinstance(value, nn.Module):
            if _contains_message_passing(value):
                self.rewrite_children(value, name)
                return value
            if self.num_bases is not None:
                return value  # shared by all node types
            if isinstance(value, _ROW_WISE_MODULES) and not value.parameters():
                return value
            return _NodeTypeModules(value, self.node_types, name)
        if isinstance(value, list):
            return [self.convert(v, f"{name}.{i}") for i, v in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(self.convert(v, f"{name}.{i}") for i, v in enumerate(value))
        if isinstance(value, dict):
            return {k: self.convert(v, f"{name}.{k}") for k, v in value.items()}
        return value

    def rewrite_children(self, module: nn.Module, name: str):
        """Rewrites the sub-modules of a module that contains message passing
        layers, in place."""
        prefix = f"{name}." if name else ""
        for key in list(module.keys()):
            value = module[key]
            if isinstance(value, mx.array):
                if self.num_bases is None:
                    self.shared_arrays.append(prefix + key)
                continue
            module[key] = self.convert(value, prefix + key)


# --------------------------------------------------------------------------- #
# The converted model
# --------------------------------------------------------------------------- #


class ToHeteroModel(nn.Module):
    """Heterogeneous version of a homogeneous model, returned by
    :func:`to_hetero`.

    It is called like the original model, with a dict per node-level input
    (keyed by node type) and per edge-level input (keyed by edge type) instead
    of an array, e.g. ``model(edge_index_dict, node_features_dict)`` for a
    model called as ``model(edge_index, node_features)``. Node-level outputs
    are returned as dicts keyed by node type (edge-level outputs keyed by edge
    type).

    The rewritten copy of the original model is stored as ``self.module``;
    the copy of a layer used for a type is in the layer's dispatcher, under
    the type's name (``"author"`` or ``"author__writes__paper"``), e.g.
    ``model.module.convs[0]["author__writes__paper"]``, or via
    ``model.module.convs[0].module_for(("author", "writes", "paper"))``.
    """

    def __init__(
        self,
        module: nn.Module,
        metadata: Metadata,
        aggr: str = "sum",
        input_map: Optional[dict[str, str]] = None,
        input_dims: Optional[dict[NodeType, int]] = None,
        num_bases: Optional[int] = None,
        in_channels: Optional[dict[str, int]] = None,
    ):
        super().__init__()
        node_types, edge_types = _check_metadata(metadata)
        if num_bases is not None and num_bases < 1:
            raise ValueError(f"num_bases must be at least 1, got {num_bases}")
        if in_channels and input_dims is None:
            raise ValueError(
                "in_channels needs input_dims, the number of input features of "
                "every node type (MLX has no lazy input sizes)."
            )
        if aggr not in _AGGRS:
            raise ValueError(f"Invalid aggregation '{aggr}'. Available: {_AGGRS}")
        if input_map is not None:
            for name, level in input_map.items():
                if level not in ("node", "edge"):
                    raise ValueError(
                        f"input_map values must be 'node' or 'edge', got "
                        f"{level!r} for '{name}'"
                    )
        if input_dims is not None:
            missing = [t for t in node_types if t not in input_dims]
            if missing:
                raise ValueError(f"input_dims is missing node types {missing}")
            input_dims = {t: int(input_dims[t]) for t in node_types}

        self._node_types = node_types
        self._edge_types = edge_types
        self._aggr = aggr
        self._input_map = dict(input_map or {})
        self._input_dims = input_dims if num_bases is None else None
        self._num_bases = num_bases
        self._in_channels = dict(in_channels or {})
        self._signature = _call_signature(module)

        _check_layers(module, edge_types, bases=num_bases is not None)
        if num_bases is not None and self._in_channels:
            # PyG's LinearAlign: one bias-free projection per node type.
            from mlx_graphs.nn.linear import Linear

            self.align = {
                name: {t: Linear(input_dims[t], dim, bias=False) for t in node_types}
                for name, dim in self._in_channels.items()
            }
        if num_bases is None and _contains_message_passing(module):
            dst_types = {dst for _, _, dst in edge_types}
            not_updated = [t for t in node_types if t not in dst_types]
            if not_updated:
                raise ValueError(
                    f"Node types {not_updated} are not the destination of any "
                    "edge type, so message passing layers cannot compute new "
                    "representations for them. Add reverse edge types, e.g. "
                    "('paper', 'rev_writes', 'author') for "
                    "('author', 'writes', 'paper')."
                )

        rewriter = _Rewriter((node_types, edge_types), aggr, num_bases)
        module = _clone(module)
        if isinstance(module, MessagePassing) or not _contains_message_passing(module):
            module = rewriter.convert(module, type(module).__name__)
        else:
            rewriter.rewrite_children(module, "")
        if rewriter.shared_arrays:
            warnings.warn(
                f"Parameters {rewriter.shared_arrays} are used directly by the "
                "model and are shared by all node types (to_hetero only copies "
                "sub-modules per type). Wrap them in a module to get one copy "
                "per type.",
                stacklevel=3,
            )
        self.module = module

        if self._input_dims is not None:
            self._dry_run()

    @property
    def metadata(self) -> Metadata:
        """The ``(node_types, edge_types)`` the model was converted for."""
        return list(self._node_types), list(self._edge_types)

    # Calling ---------------------------------------------------------------- #

    def __call__(self, *args, **kwargs):
        """Runs the model on dict-valued inputs.

        Args:
            *args: The arguments of the original model, with node-level arrays
                replaced by dicts keyed by node type and edge-level arrays by
                dicts keyed by edge type. Other values are passed unchanged.
            **kwargs: Keyword arguments, likewise.

        Returns:
            The output of the original model, with node-level arrays replaced
            by dicts keyed by node type.
        """
        ctx, args, kwargs = self._pack(args, kwargs)
        return self._run(ctx, args, kwargs)

    def _run(self, ctx: _Context, args, kwargs):
        stack = getattr(_state, "stack", None)
        if stack is None:
            stack = _state.stack = []
        stack.append(ctx)
        try:
            out = self.module(*args, **kwargs)
        finally:
            stack.pop()
        return ctx.unpack(out)

    def _arg_names(self, args, kwargs) -> tuple[list, dict]:
        params = [
            p
            for p in self._signature.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        names = [params[i].name if i < len(params) else None for i in range(len(args))]
        return names, {k: k for k in kwargs}

    def _dict_level(self, name: Optional[str], value: dict) -> Optional[str]:
        level = self._input_map.get(name) if name is not None else None
        keys = list(value.keys())
        if level is None:
            if all(isinstance(k, str) for k in keys):
                level = "node"
            elif all(isinstance(k, tuple) and len(k) == 3 for k in keys):
                level = "edge"
            else:
                raise ValueError(
                    f"Argument '{name}' must be keyed either by node types or by "
                    f"(src, rel, dst) edge types, got keys {keys}"
                )
        if not all(isinstance(v, mx.array) for v in value.values()):
            raise TypeError(f"The values of argument '{name}' must be arrays")
        if level == "edge":
            is_index = (name is not None and ("index" in name or "adj" in name)) or all(
                v.ndim == 2 and v.shape[0] == 2 and mx.issubdtype(v.dtype, mx.integer)
                for v in value.values()
            )
            return "edge_index" if is_index else "edge"
        return level

    def _pack(self, args, kwargs, probe_width: Optional[int] = None):
        arg_names, kwarg_names = self._arg_names(args, kwargs)
        entries = [(("args", i), arg_names[i], v) for i, v in enumerate(args)]
        entries += [(("kwargs", k), kwarg_names[k], v) for k, v in kwargs.items()]
        if self._in_channels:
            entries = [
                (where, name, self._align(name, v))
                if name in self._in_channels and isinstance(v, dict)
                else (where, name, v)
                for where, name, v in entries
            ]
            for where, name, v in entries:
                if where[0] == "args":
                    args = args[: where[1]] + (v,) + args[where[1] + 1 :]
                else:
                    kwargs = {**kwargs, where[1]: v}
        levels = {
            where: self._dict_level(name, v)
            for where, name, v in entries
            if isinstance(v, dict) and len(v) > 0
        }

        # Node counts
        num_nodes: dict[NodeType, int] = {}
        for where, name, v in entries:
            if levels.get(where) != "node":
                continue
            missing = [t for t in self._node_types if t not in v]
            if missing:
                raise ValueError(f"Argument '{name}' is missing node types {missing}")
            for t in self._node_types:
                n = v[t].shape[0] if v[t].ndim > 0 else 0
                if num_nodes.setdefault(t, n) != n:
                    raise ValueError(
                        f"Node type '{t}' has {num_nodes[t]} nodes in one argument "
                        f"and {n} in '{name}'"
                    )
        if not num_nodes:
            raise ValueError(
                "The model needs at least one node-level argument: a dict of "
                "arrays keyed by node type, e.g. the node features."
            )

        # Edge types present in this call and their counts
        edge_types: list[EdgeType] = []
        num_edges: dict[EdgeType, int] = {}
        index_entries = [e for e in entries if levels.get(e[0]) == "edge_index"]
        feature_entries = [e for e in entries if levels.get(e[0]) == "edge"]
        if index_entries:
            values = index_entries[0][2]
            edge_types = [et for et in self._edge_types if et in values]
            num_edges = {et: values[et].shape[1] for et in edge_types}
        elif feature_entries:
            values = feature_entries[0][2]
            edge_types = [et for et in self._edge_types if et in values]
            num_edges = {et: values[et].shape[0] for et in edge_types}

        ctx = _Context(
            self._node_types,
            num_nodes,
            edge_types,
            num_edges,
            self._input_dims,
            probe_width,
        )

        packed_args, packed_kwargs = list(args), dict(kwargs)
        for where, name, v in entries:
            level = levels.get(where)
            if level is None:
                continue
            if level == "node":
                packed, is_input = self._pack_nodes(name, v, probe_width)
                values = {t: v[t] for t in self._node_types}
            else:
                values = {et: v[et] for et in edge_types if et in v}
                missing = [et for et in edge_types if et not in v]
                if missing:
                    raise ValueError(
                        f"Argument '{name}' is missing edge types {missing}"
                    )
                packed, is_input = self._pack_edges(name, values, level), False
            ctx.register(level, values, packed, is_input)
            if where[0] == "args":
                packed_args[where[1]] = packed
            else:
                packed_kwargs[where[1]] = packed
        return ctx, tuple(packed_args), packed_kwargs

    def _align(self, name: str, values: dict) -> dict:
        return {t: self.align[name][t](v) for t, v in values.items()}

    def _pack_nodes(self, name, values: dict, probe_width: Optional[int]):
        arrays = [values[t] for t in self._node_types]
        widths = {t: (a.shape[-1] if a.ndim > 1 else None) for t, a in values.items()}
        is_input = (
            self._input_dims is not None
            and all(a.ndim == 2 for a in arrays)
            and all(widths[t] == self._input_dims[t] for t in self._node_types)
        )
        if len({a.shape[1:] for a in arrays}) > 1 or probe_width is not None:
            if not all(a.ndim == 2 for a in arrays):
                raise ValueError(
                    f"The arrays of argument '{name}' have different shapes per "
                    f"node type: { {t: a.shape for t, a in values.items()} }"
                )
            if not is_input and self._input_dims is not None:
                raise ValueError(
                    f"Argument '{name}' has {widths} features per node type, but "
                    f"the model was converted for input_dims={self._input_dims}."
                )
            if not is_input and self._num_bases is not None:
                raise ValueError(
                    f"Argument '{name}' has a different number of features per "
                    f"node type ({widths}). Pass in_channels={{'{name}': size}} "
                    f"and input_dims={widths} to to_hetero_with_bases to project "
                    "them to a common size."
                )
            if not is_input:
                raise ValueError(
                    f"Argument '{name}' has a different number of features per "
                    f"node type ({widths}). Pass input_dims={widths} to "
                    "to_hetero, or project the features of every node type to "
                    "a common size first (e.g. with HeteroDictLinear)."
                )
            # Padded to a common width. As long as the model passes the input
            # features unchanged to its first layers (which look them up per
            # type), MLX never evaluates this array.
            width = probe_width or max(widths.values())
            arrays = [
                mx.pad(a, [(0, 0), (0, width - a.shape[1])])
                if a.shape[1] < width
                else a
                for a in arrays
            ]
        packed = arrays[0] if len(arrays) == 1 else mx.concatenate(arrays, axis=0)
        return packed, is_input

    def _pack_edges(self, name, values: dict, level: str) -> mx.array:
        arrays = list(values.values())
        if not arrays:
            return mx.zeros((2, 0) if level == "edge_index" else (0,))
        if level == "edge_index":
            return arrays[0] if len(arrays) == 1 else mx.concatenate(arrays, axis=1)
        if len({a.shape[1:] for a in arrays}) > 1:
            if not all(a.ndim == 2 for a in arrays):
                raise ValueError(
                    f"The arrays of argument '{name}' have different shapes per "
                    "edge type"
                )
            width = max(a.shape[1] for a in arrays)
            arrays = [mx.pad(a, [(0, 0), (0, width - a.shape[1])]) for a in arrays]
        return arrays[0] if len(arrays) == 1 else mx.concatenate(arrays, axis=0)

    # Dry run for input_dims ------------------------------------------------- #

    def _dry_run(self):
        """Runs the model once on a tiny graph whose node features have an
        unusual width, to find the layers reading the input features and
        rebuild their copies for the input size of each node type."""
        dims = [d for d in self._input_dims.values()]
        for _, value in tree_flatten(self.module.parameters()):
            dims.extend(value.shape)
        probe_width = 64 * max(dims) + 1

        features = {t: mx.zeros((2, d)) for t, d in self._input_dims.items()}
        edge_index = {
            et: mx.array([[0, 1], [1, 0]], dtype=mx.int32) for et in self._edge_types
        }
        args, kwargs = [], {}
        for p in list(self._signature.parameters.values()):
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.default is not p.empty:
                continue
            level = self._input_map.get(p.name)
            is_edge = level == "edge" or (
                level is None and ("edge" in p.name or "adj" in p.name)
            )
            value = edge_index if is_edge else features
            if p.kind == p.KEYWORD_ONLY:
                kwargs[p.name] = value
            else:
                args.append(value)

        flags = [(m, m.training) for _, m in self.module.named_modules()]
        self.module.eval()
        try:
            ctx, packed_args, packed_kwargs = self._pack(
                tuple(args), kwargs, probe_width
            )
            out = self._run(ctx, packed_args, packed_kwargs)
            mx.eval(out)
        except Exception as e:
            raise RuntimeError(
                "to_hetero(input_dims=...) runs the model once on a tiny graph "
                "to find the layers that read the input node features, and "
                f"this failed: {e}. The dry run passes an edge index dict to "
                "every required argument whose name contains 'edge' and the "
                "node features to the others (use input_map to change this). "
                "Alternatively, project the node features to a common size "
                "(e.g. with HeteroDictLinear) and call to_hetero without "
                "input_dims."
            ) from e
        finally:
            for m, training in flags:
                m._training = training
        mx.eval(self.parameters())

    def __repr__(self):
        extra = (
            f"aggr='{self._aggr}'"
            if self._num_bases is None
            else f"num_bases={self._num_bases}"
        )
        return (
            f"ToHeteroModel(node_types={self._node_types}, "
            f"edge_types={self._edge_types}, {extra})"
        )


def _call_signature(module: nn.Module) -> inspect.Signature:
    signature = inspect.signature(type(module).__call__)
    params = list(signature.parameters.values())[1:]  # drop self
    return signature.replace(parameters=params)


def _check_metadata(metadata: Metadata) -> tuple[list[NodeType], list[EdgeType]]:
    if not (isinstance(metadata, (tuple, list)) and len(metadata) == 2):
        raise ValueError(
            "metadata must be a (node_types, edge_types) tuple, e.g. " "data.metadata()"
        )
    node_types, edge_types = list(metadata[0]), [tuple(et) for et in metadata[1]]
    if not node_types or not edge_types:
        raise ValueError("metadata needs at least one node type and one edge type")
    for node_type in node_types:
        if not isinstance(node_type, str) or "." in node_type:
            raise ValueError(
                f"Node types must be strings without '.', got {node_type!r}"
            )
    keys = set()
    for edge_type in edge_types:
        if len(edge_type) != 3 or not all(isinstance(p, str) for p in edge_type):
            raise ValueError(f"Edge types must be (src, rel, dst), got {edge_type!r}")
        if "." in edge_type[1]:
            raise ValueError(f"Relation names may not contain '.': {edge_type!r}")
        src, _, dst = edge_type
        for node_type in (src, dst):
            if node_type not in node_types:
                raise ValueError(
                    f"Edge type {edge_type} uses node type '{node_type}', which "
                    "is not in the node types of the metadata"
                )
        keys.add(_type_key(edge_type))
    if len(keys) != len(edge_types) or len(set(node_types)) != len(node_types):
        raise ValueError("metadata contains duplicate (or ambiguous) types")
    return node_types, edge_types


def _check_layers(module: nn.Module, edge_types: list[EdgeType], bases: bool):
    """Rejects layers that cannot work on bipartite edge types (or, with
    bases, layers adding self-loops), like PyG."""
    if bases:
        for name, sub in module.named_modules():
            if isinstance(sub, MessagePassing) and getattr(sub, "_add_self_loops", 0):
                raise ValueError(
                    f"'{name or type(module).__name__}' ({type(sub).__name__}) "
                    "adds self-loops, which to_hetero_with_bases does not "
                    "support (the messages of self-loops have no edge type). "
                    "Disable them (add_self_loops=False)."
                )
        return
    bipartite = [et for et in edge_types if et[0] != et[-1]]
    if not bipartite:
        return
    from mlx_graphs.nn.conv.gcn_conv import GCNConv

    for name, sub in module.named_modules():
        if not isinstance(sub, MessagePassing):
            continue
        if isinstance(sub, GCNConv) or getattr(sub, "_add_self_loops", False):
            raise ValueError(
                f"'{name or type(module).__name__}' ({type(sub).__name__}) "
                f"cannot be used on bipartite edge types such as {bipartite[0]}: "
                + (
                    "GCNConv does not support bipartite message passing. Use e.g. "
                    "SAGEConv or GraphConv instead."
                    if isinstance(sub, GCNConv)
                    else "it adds self-loops. Disable them (add_self_loops=False)."
                )
            )


def to_hetero(
    module: nn.Module,
    metadata: Metadata,
    aggr: str = "sum",
    input_map: Optional[dict[str, str]] = None,
    input_dims: Optional[dict[NodeType, int]] = None,
) -> ToHeteroModel:
    r"""Converts a homogeneous GNN model into its heterogeneous equivalent, in
    which node representations are learned for each node type in
    ``metadata[0]`` and messages are exchanged along each edge type in
    ``metadata[1]``, as in the `"Modeling Relational Data with Graph
    Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ paper (PyG's
    ``to_hetero``).

    Every message passing layer :math:`f_{\theta}^{(\ell)}` of the model is
    copied once per relation :math:`r` (edge type), and every other
    sub-module (``Linear``, normalizations, MLPs, Jumping Knowledge, ...) once
    per node type. Message passing then computes

    .. math::
        \mathbf{h}^{(\ell)}_v = \bigoplus_{r \in \mathcal{R}}
        f_{\theta}^{(\ell, r)} ( \mathbf{h}^{(\ell - 1)}_v, \{
        \mathbf{h}^{(\ell - 1)}_w : w \in \mathcal{N}^{(r)}(v) \}),

    where :math:`\bigoplus` is ``aggr``. Edge types between two different node
    types call their layer with a ``(src_features, dst_features)`` tuple, so
    it must support bipartite graphs (e.g. ``SAGEConv``, ``GraphConv``,
    ``GATConv``, ``GATv2Conv``, ``GINConv``; not ``GCNConv``).

    The converted model is called like the original one, with dicts instead
    of arrays: ``model(edge_index_dict, node_features_dict)`` for a model
    called as ``model(edge_index, node_features)``. Dicts keyed by node types
    are node-level inputs, dicts keyed by ``(src, rel, dst)`` edge types are
    edge-level inputs (e.g. ``edge_weights``), and other values are passed
    unchanged. Node-level outputs are returned as dicts keyed by node type.

    **Input sizes.** MLX layers have no lazy input sizes (PyG's ``-1``), so
    there are two ways to handle node types with different numbers of
    features:

    * pass ``input_dims`` (e.g. ``data.num_node_features``): the copies of the
      layers reading the input features are rebuilt for each node type's
      size. The homogeneous model can then be built with any input size. The
      layers reading the input features are found by running the model once
      on a tiny graph when converting it. Supported first layers:
      ``Linear``, ``MLP``, ``nn.Sequential`` starting with a ``Linear``,
      ``SAGEConv``, ``GraphConv``, ``GATConv``, ``GATv2Conv``, and
      ``GINConv``/``GCNConv`` when both ends of every edge type have the same
      size;
    * or project every node type to a common size first (e.g. with
      :class:`~mlx_graphs.nn.HeteroDictLinear`), and convert a model built for
      that size without ``input_dims``.

    **How it works.** PyG traces ``forward`` with ``torch.fx``, which has no
    MLX counterpart. Instead, the model is copied and its layers are replaced
    by dispatchers holding one copy per type. At call time the dicts are
    packed into arrays (the rows of all node types, resp. the edges of all
    edge types, concatenated in metadata order) and the model's own
    ``__call__`` runs on them unchanged: row-wise operations (activations,
    dropout, residual connections, concatenating features) just work, and the
    dispatchers split the arrays per type, run the copies and concatenate the
    results. Inputs passed unchanged to the first layers are looked up in the
    input dicts, so packing them costs nothing.

    **Initialization of the copies.** As PyG, which calls
    ``reset_parameters()`` on every copy, the copies are re-randomized (when
    there are several types). Modules with a ``reset_parameters()`` method
    are reset with it. Most MLX modules have none, so the trainable arrays of
    the others are randomly permuted: for initializations drawing every
    entry independently (the usual uniform/normal/Glorot ones) this is a
    draw from the same distribution, but the copies hold the same values in
    different orders (not independent draws: arrays of a single value, e.g.
    the bias of a one-output layer, are identical in all copies), and an
    initialization with structure (e.g. orthogonal) is not preserved: give
    such modules a ``reset_parameters()`` method. Frozen arrays (buffers,
    e.g. batch norm running statistics) are kept.

    **Supported:** models written in the PyG style (message passing layers,
    ``Linear``/MLP/normalization modules, functional activations, dropout,
    residual and skip connections such as ``conv(edge_index, x) + lin(x)``),
    the built-in :class:`~mlx_graphs.nn.BasicGNN` models (``GCN``,
    ``GraphSAGE``, ``GIN``, ``GAT``, including ``norm`` and ``jk``), models
    nesting such models, edge-level arguments of the message passing layers
    (``edge_weights``, ``edge_features``), and edge types missing from a
    call's edge index dict (skipped). Parameters live in ``model.parameters()``
    like any MLX module (``nn.value_and_grad``, optimizers, ``save_weights``).

    **Not supported** (as in PyG, or because of the packing):

    * operations in ``__call__`` that mix or select rows, e.g. ``x[:n]``,
      ``x.mean(axis=0)``, global pooling or sorting: they would act on the
      rows of all node types together. Do them on the output dicts instead;
    * modules (other than message passing layers) called on edge-level or
      graph-level values;
    * arrays stored directly on the model and used in ``__call__`` are shared
      by all node types (a warning is raised); modules are copied per type;
    * control flow depending on the data, and modifying ``edge_index`` before
      passing it to the layers: adding or removing edges (e.g.
      ``dropout_edge``) raises an error, reordering them is not detected;
    * every node type must be the destination of at least one edge type
      (add reverse edges); ``GCNConv`` and layers adding self-loops cannot be
      used with bipartite edge types;
    * if the total number of nodes equals the total number of edges, arrays
      with that many rows are treated as node-level.

    Args:
        module: The homogeneous model to convert. It is copied, the original
            is left unchanged.
        metadata: The node types and edge types of the graph, e.g.
            ``data.metadata()`` of a :class:`~mlx_graphs.data.HeteroGraphData`.
        aggr: How to combine the outputs of the edge types sharing a
            destination node type: ``"sum"``, ``"mean"``, ``"max"``,
            ``"min"`` or ``"mul"``. Default: ``"sum"``
        input_map: Optional ``{argument name: "node" | "edge"}`` overriding
            the level of dict arguments (otherwise found from their keys) and
            telling the dry run of ``input_dims`` which required arguments are
            edge-level (otherwise those whose name contains ``"edge"``).
        input_dims: Optional number of input features per node type, e.g.
            ``data.num_node_features``. See above. Default: ``None`` (every node
            type has the input size the model was built for).

    Returns:
        A :class:`ToHeteroModel`.

    Example:

    .. code-block:: python

        import mlx.core as mx
        import mlx.nn as nn
        from mlx_graphs.nn import Linear, SAGEConv, to_hetero

        class GNN(nn.Module):
            def __init__(self, hidden, out):
                super().__init__()
                self.conv1 = SAGEConv(hidden, hidden)
                self.lin1 = Linear(hidden, hidden)
                self.conv2 = SAGEConv(hidden, out)
                self.lin2 = Linear(hidden, out)

            def __call__(self, edge_index, x):
                x = nn.relu(self.conv1(edge_index, x) + self.lin1(x))
                return self.conv2(edge_index, x) + self.lin2(x)

        x_dict = {"author": mx.ones((4, 16)), "paper": mx.ones((5, 8))}
        edge_index_dict = {
            ("author", "writes", "paper"): mx.array([[0, 1, 3], [0, 0, 4]]),
            ("paper", "rev_writes", "author"): mx.array([[0, 0, 4], [0, 1, 3]]),
            ("paper", "cites", "paper"): mx.array([[1, 2], [0, 0]]),
        }
        metadata = (list(x_dict), list(edge_index_dict))

        # GNN(-1, ...) is not possible in MLX: give the input sizes instead
        model = to_hetero(GNN(32, 4), metadata, input_dims={"author": 16, "paper": 8})
        out = model(edge_index_dict, x_dict)
        >>> {k: v.shape for k, v in out.items()}
        {'author': (4, 4), 'paper': (5, 4)}

        # Built-in models work too
        from mlx_graphs.nn import GraphSAGE
        model = to_hetero(
            GraphSAGE(16, 32, num_layers=2, out_features_dim=4, jk="cat"),
            metadata,
            input_dims={"author": 16, "paper": 8},
        )

    References:
        `PyG's to_hetero <https://pytorch-geometric.readthedocs.io/en/latest/
        generated/torch_geometric.nn.to_hetero_transformer.to_hetero.html>`_
    """
    return ToHeteroModel(module, metadata, aggr, input_map, input_dims)


def to_hetero_with_bases(
    module: nn.Module,
    metadata: Metadata,
    num_bases: int,
    in_channels: Optional[dict[str, int]] = None,
    input_map: Optional[dict[str, str]] = None,
    input_dims: Optional[dict[NodeType, int]] = None,
) -> ToHeteroModel:
    r"""Converts a homogeneous GNN model into its heterogeneous equivalent via
    the basis-decomposition technique of the `"Modeling Relational Data with
    Graph Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ paper
    (PyG's ``to_hetero_with_bases``).

    Every message passing layer :math:`f_{\theta}^{(\ell)}` is copied
    ``num_bases`` times, and every edge type :math:`r` gets learned
    coefficients :math:`a^{(\ell)}_{r,b}` (``edge_type_weight``). Message
    passing runs on one graph made of all node and edge types:

    .. math::
        \mathbf{h}^{(\ell)}_v = \sum_{b=1}^{B} f_{\theta_b}^{(\ell)} (
        \mathbf{h}^{(\ell - 1)}_v, \{ a^{(\ell)}_{r,b} \cdot
        \mathbf{h}^{(\ell - 1)}_w : w \in \mathcal{N}^{(r)}(v), r \in
        \mathcal{R} \}),

    where every message of an edge of type :math:`r` is scaled by
    :math:`a^{(\ell)}_{r,b}` (``message()`` of the copies is wrapped, and the
    fused SpMM path is disabled for them). The number of parameters grows with
    ``num_bases`` instead of with the number of edge types, which helps with
    many relations. Unlike :func:`to_hetero`, the other modules (``Linear``,
    normalizations, ...) are shared by all node types, and all node types
    need the same number of features: give ``in_channels`` (with
    ``input_dims``) to project node-level inputs of every type to a common size
    first, with one bias-free linear layer per node type (PyG's
    ``LinearAlign``).

    The converted model is called and returns its outputs like
    :func:`to_hetero`'s. Layers must not add self-loops (the messages would
    not belong to any edge type) and must compute their messages with
    :meth:`~mlx_graphs.nn.MessagePassing.message`.

    Args:
        module: The homogeneous model to convert (copied).
        metadata: The node types and edge types of the graph, e.g.
            ``data.metadata()``.
        num_bases: The number of bases (copies of every message passing layer).
        in_channels: Optional ``{argument name: size}``: node-level arguments
            (e.g. ``{"node_features": 64}``) whose features are projected to
            ``size`` per node type before running the model.
        input_map: Optional ``{argument name: "node" | "edge"}``, see
            :func:`to_hetero`.
        input_dims: Number of input features per node type, needed with
            ``in_channels`` (MLX has no lazy input sizes).

    Returns:
        A :class:`ToHeteroModel`.

    Example:

    .. code-block:: python

        from mlx_graphs.nn import GraphSAGE, to_hetero_with_bases

        model = to_hetero_with_bases(
            GraphSAGE(32, 32, num_layers=2, out_features_dim=4),
            data.metadata(),
            num_bases=3,
            in_channels={"node_features": 32},
            input_dims=data.num_node_features,
        )
        out = model(data.edge_index_dict, data.node_features_dict)

    References:
        `PyG's to_hetero_with_bases <https://pytorch-geometric.readthedocs.io/
        en/latest/generated/torch_geometric.nn.to_hetero_with_bases_transformer.
        to_hetero_with_bases.html>`_
    """
    return ToHeteroModel(
        module,
        metadata,
        input_map=input_map,
        input_dims=input_dims,
        num_bases=num_bases,
        in_channels=in_channels,
    )
