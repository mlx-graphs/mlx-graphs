"""Temporal Graph Networks (TGN), mirroring ``torch_geometric.nn.models.tgn``.

Design of the memory state
--------------------------

PyTorch keeps the TGN memory in mutable module buffers that autograd tracks
through in-place writes, and relies on ``memory.detach()`` to cut the graph
between mini-batches. MLX has neither buffers nor in-place autograd, so here:

* The learnable parts (message modules, time encoder, GRU cell) are regular
  MLX parameters, and are the only thing :func:`mlx.nn.value_and_grad` sees.
* The state (``memory``, ``last_update`` and the message stores) lives in
  private attributes (names starting with ``_``), which MLX excludes from
  :meth:`~mlx.nn.Module.parameters`, so optimizers and ``value_and_grad``
  never touch it. Index bookkeeping (``last_update``, message stores, the
  neighbor loader) is kept on the host in NumPy; only the memory matrix is an
  MLX array.
* Reading the memory (``memory(n_id)``) is a pure function of the parameters
  and of the current state, so it can be called inside the loss function: in
  training mode the gradient flows, as in PyG, through the memory update
  computed from the stored messages (GRU, message modules, time encoder).
* :meth:`TGNMemory.update_state` replaces the state arrays with new ones
  (never mutates an array in place). In PyG the stored memory is detached
  before it is used again, so it never carries gradient to later batches;
  here the stored state is likewise a constant for any later
  ``value_and_grad`` call. ``update_state`` can therefore be called either
  inside the loss function (PyG's order) or after it, before the optimizer
  step: both give the same values. :meth:`TGNMemory.detach` applies
  ``mx.stop_gradient`` and evaluates the state, which keeps the lazy graph
  from growing across mini-batches.
"""

import copy
import math
from typing import Callable, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_graphs.utils import scatter

IndexLike = Union[mx.array, np.ndarray, list]


def _host(index: IndexLike, dtype=np.int64) -> np.ndarray:
    """Host (NumPy) copy of an index or value array."""
    out = np.array(index) if isinstance(index, mx.array) else np.asarray(index)
    return out.reshape(-1).astype(dtype, copy=False)


def _host_time(t: IndexLike) -> np.ndarray:
    """Host copy of integer timestamps (int64, as PyG's ``TemporalData.t``).

    Floating timestamps are accepted only if they hold integer values, so
    that they are never silently truncated."""
    out = np.array(t) if isinstance(t, mx.array) else np.asarray(t)
    out = out.reshape(-1)
    if np.issubdtype(out.dtype, np.floating):
        integral = np.isfinite(out) & (out == np.floor(out))
        if not integral.all():
            bad = out[~integral][0]
            raise ValueError(
                "TGNMemory needs integer timestamps (int64, as PyG's "
                f"TemporalData.t), got the non-integer value {bad}. Convert "
                "the timestamps to an integer unit first (e.g. seconds)."
            )
    return out.astype(np.int64, copy=False)


def _out_dim(module) -> int:
    dim = getattr(module, "out_features_dim", None)
    if dim is None:
        dim = module.out_channels
    return dim


class GRUCell(nn.Module):
    r"""A gated recurrent unit cell with the parameters and initialization of
    ``torch.nn.GRUCell``.

    .. math::
        \begin{aligned}
        r &= \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z &= \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n &= \tanh(W_{in} x + b_{in} + r \odot (W_{hn} h + b_{hn})) \\
        h' &= (1 - z) \odot n + z \odot h
        \end{aligned}

    Args:
        input_dim: Size of the input
        hidden_dim: Size of the hidden state
        bias: Whether to use biases. Default: ``True``
    """

    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.reset_parameters()

    def reset_parameters(self):
        """Draws all parameters from :math:`U(-1/\\sqrt{H}, 1/\\sqrt{H})`."""
        k = 1.0 / math.sqrt(self.hidden_dim)
        H, D = self.hidden_dim, self.input_dim
        self.weight_ih = mx.random.uniform(-k, k, (3 * H, D))
        self.weight_hh = mx.random.uniform(-k, k, (3 * H, H))
        if self.bias:
            self.bias_ih = mx.random.uniform(-k, k, (3 * H,))
            self.bias_hh = mx.random.uniform(-k, k, (3 * H,))

    def __call__(self, x: mx.array, h: mx.array) -> mx.array:
        """Computes one step.

        Args:
            x: Inputs, shape `[batch, input_dim]`
            h: Hidden states, shape `[batch, hidden_dim]`

        Returns:
            The new hidden states, shape `[batch, hidden_dim]`
        """
        gi = x @ self.weight_ih.T
        gh = h @ self.weight_hh.T
        if self.bias:
            gi = gi + self.bias_ih
            gh = gh + self.bias_hh
        i_r, i_z, i_n = mx.split(gi, 3, axis=-1)
        h_r, h_z, h_n = mx.split(gh, 3, axis=-1)
        r = mx.sigmoid(i_r + h_r)
        z = mx.sigmoid(i_z + h_z)
        n = mx.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h


class TimeEncoder(nn.Module):
    r"""The time encoder of TGN, :math:`\cos(\mathbf{w} t + \mathbf{b})`.

    Args:
        out_features_dim: Size of the encoding

    Example:

    .. code-block:: python

        from mlx_graphs.nn.models.tgn import TimeEncoder

        enc = TimeEncoder(8)
        enc(mx.array([0.0, 10.0, 100.0])).shape
        >>> (3, 8)
    """

    def __init__(self, out_features_dim: int):
        super().__init__()
        self.out_features_dim = out_features_dim
        self.lin = nn.Linear(1, out_features_dim)  # torch.nn.Linear init

    @property
    def out_channels(self) -> int:
        """PyG name of :attr:`out_features_dim`."""
        return self.out_features_dim

    def reset_parameters(self):
        """Re-initializes the linear layer."""
        self.lin = nn.Linear(1, self.out_features_dim)

    def __call__(self, t: mx.array) -> mx.array:
        return mx.cos(self.lin(t.reshape(-1, 1)))


class IdentityMessage(nn.Module):
    """The identity message function of TGN: the concatenation of the source
    and destination memories, the raw message and the time encoding.

    Args:
        raw_msg_dim: Size of the raw messages
        memory_dim: Size of the memory
        time_dim: Size of the time encoding
    """

    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_features_dim = raw_msg_dim + 2 * memory_dim + time_dim

    @property
    def out_channels(self) -> int:
        """PyG name of :attr:`out_features_dim`."""
        return self.out_features_dim

    def __call__(
        self, z_src: mx.array, z_dst: mx.array, raw_msg: mx.array, t_enc: mx.array
    ) -> mx.array:
        return mx.concatenate([z_src, z_dst, raw_msg, t_enc], axis=-1)


class LastAggregator(nn.Module):
    """Keeps, for every node, the message with the latest timestamp (the
    message stored last among equal timestamps). Nodes without messages get
    zeros.

    Called as ``aggr(msg, index, t, dim_size)``.
    """

    def __call__(
        self, msg: mx.array, index: mx.array, t: mx.array, dim_size: int
    ) -> mx.array:
        idx = _host(index)
        tt = np.array(t).reshape(-1) if isinstance(t, mx.array) else np.asarray(t)
        n = idx.size
        arg = np.full(dim_size, n, dtype=np.int64)
        if n > 0:
            # Sort by node, then time, then position: the last entry of every
            # group is its latest message (largest position among ties).
            order = np.lexsort((np.arange(n), tt, idx))
            sorted_idx = idx[order]
            is_last = np.r_[sorted_idx[1:] != sorted_idx[:-1], True]
            arg[sorted_idx[is_last]] = order[is_last]
        padded = mx.concatenate([msg, mx.zeros((1, msg.shape[-1]), dtype=msg.dtype)])
        return padded[mx.array(arg.astype(np.int32))]


class MeanAggregator(nn.Module):
    """Averages the messages of every node (zeros for nodes without messages).

    Called as ``aggr(msg, index, t, dim_size)``.
    """

    def __call__(
        self, msg: mx.array, index: mx.array, t: mx.array, dim_size: int
    ) -> mx.array:
        return scatter(msg, index, dim_size, aggr="mean")


class _MessageStore:
    """Per-node store of the raw messages of the last update in which a node
    took part (as source), equivalent to PyG's ``Dict[int, (src, dst, t,
    msg)]``.

    The messages live in flat host buffers, every node owning one contiguous
    range; updates append and the buffers are compacted when more than half
    of them is stale.
    """

    def __init__(self, num_nodes: int, raw_msg_dim: int):
        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.reset()

    def reset(self):
        self.start = np.zeros(self.num_nodes, dtype=np.int64)
        self.count = np.zeros(self.num_nodes, dtype=np.int64)
        self.size = 0  # used entries of the buffers (live or stale)
        self.live = 0
        self._alloc(1024)

    def _alloc(self, capacity: int):
        self.src = np.zeros(capacity, dtype=np.int64)
        self.dst = np.zeros(capacity, dtype=np.int64)
        self.t = np.zeros(capacity, dtype=np.int64)
        self.msg = np.zeros((capacity, self.raw_msg_dim), dtype=np.float32)

    def _positions(self, n_id: np.ndarray) -> np.ndarray:
        counts = self.count[n_id]
        total = int(counts.sum())
        offsets = np.repeat(self.start[n_id] - (np.cumsum(counts) - counts), counts)
        return offsets + np.arange(total, dtype=np.int64)

    def _reserve(self, extra: int):
        capacity = self.src.shape[0]
        if self.size + extra <= capacity:
            return
        nodes = np.nonzero(self.count)[0]
        pos = self._positions(nodes)
        old = (self.src[pos], self.dst[pos], self.t[pos], self.msg[pos])
        while self.live + extra > capacity // 2:
            capacity *= 2
        self._alloc(capacity)
        n = pos.size
        self.src[:n], self.dst[:n], self.t[:n], self.msg[:n] = old
        counts = self.count[nodes]
        self.start[nodes] = np.cumsum(counts) - counts
        self.size = n

    def update(self, src: np.ndarray, dst: np.ndarray, t: np.ndarray, msg):
        perm = np.argsort(src, kind="stable")
        nodes, counts = np.unique(src[perm], return_counts=True)
        self.live -= int(self.count[nodes].sum())
        self.count[nodes] = 0
        n = src.size
        self._reserve(n)
        sl = slice(self.size, self.size + n)
        self.src[sl], self.dst[sl], self.t[sl] = src[perm], dst[perm], t[perm]
        self.msg[sl] = msg[perm]
        self.start[nodes] = self.size + np.cumsum(counts) - counts
        self.count[nodes] = counts
        self.size += n
        self.live += n

    def gather(self, n_id: np.ndarray):
        pos = self._positions(n_id)
        return self.src[pos], self.dst[pos], self.t[pos], self.msg[pos]


class TGNMemory(nn.Module):
    r"""The Temporal Graph Network (TGN) memory model from the `"Temporal Graph
    Networks for Deep Learning on Dynamic Graphs"
    <https://arxiv.org/abs/2006.10637>`_ paper, as PyG's
    ``torch_geometric.nn.TGNMemory``.

    Every node has a memory vector, updated with a GRU cell from the messages
    of the events it takes part in. The events of a mini-batch are stored as
    raw messages by :meth:`update_state` and only turned into memory updates
    when the nodes are next accessed, so that the memory read for a
    mini-batch never contains information from that mini-batch itself.

    In training mode, ``memory(n_id)`` computes the memory of ``n_id`` updated
    with the pending messages (differentiable w.r.t. the message modules, the
    time encoder and the GRU), and :meth:`update_state` first writes those
    updates to the state and then stores the new messages. In evaluation
    mode, ``memory(n_id)`` returns the stored memory and :meth:`update_state`
    applies the new messages right away. Switching from training to
    evaluation mode (``.eval()`` / ``.train(False)``, on this module or on a
    model containing it) flushes all pending messages into the memory, as in
    PyG.

    The state is not part of :meth:`~mlx.nn.Module.parameters` (see the
    design notes of :mod:`mlx_graphs.nn.models.tgn`): read it with
    :attr:`memory` and :attr:`last_update`. In particular,
    :meth:`~mlx.nn.Module.save_weights` / ``load_weights`` only save and
    restore the learnable parameters, not the memory (PyG's ``state_dict``
    also holds the ``memory`` and ``last_update`` buffers). To checkpoint
    them, use :meth:`memory_state` and :meth:`load_memory_state`, e.g.
    ``mx.savez("memory.npz", **memory.memory_state())``. As with PyG's
    buffers, the pending messages are not part of it.

    Timestamps are integers (int64, as PyG's ``TemporalData.t``).

    Args:
        num_nodes: Number of nodes to keep a memory for
        raw_msg_dim: Size of the raw messages (event features)
        memory_dim: Size of the memory
        time_dim: Size of the time encoding
        message_module: Message function, called as ``message_module(z_src,
            z_dst, raw_msg, t_enc)``, with an ``out_features_dim`` (or PyG's
            ``out_channels``) attribute, e.g. :class:`IdentityMessage`
        aggregator_module: Aggregation of the messages of each node, called as
            ``aggregator_module(msg, index, t, dim_size)``, e.g.
            :class:`LastAggregator` or :class:`MeanAggregator`

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import TGNMemory
        from mlx_graphs.nn.models.tgn import IdentityMessage, LastAggregator

        memory = TGNMemory(
            num_nodes=100,
            raw_msg_dim=16,
            memory_dim=32,
            time_dim=32,
            message_module=IdentityMessage(16, 32, 32),
            aggregator_module=LastAggregator(),
        )
        n_id = mx.array([0, 1, 2])
        z, last_update = memory(n_id)  # [3, 32], [3]
        memory.update_state(src, dst, t, msg)  # after computing the loss
        memory.detach()

    References:
        - `PyG TGNMemory <https://pytorch-geometric.readthedocs.io/en/latest/
          generated/torch_geometric.nn.models.TGNMemory.html>`_
        - Rossi et al., `Temporal Graph Networks for Deep Learning on Dynamic
          Graphs <https://arxiv.org/abs/2006.10637>`_
    """

    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        message_module: Callable,
        aggregator_module: Callable,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim

        self.msg_s_module = message_module
        self.msg_d_module = copy.deepcopy(message_module)
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        self.gru = GRUCell(_out_dim(message_module), memory_dim)

        self._assoc = np.zeros(num_nodes, dtype=np.int64)
        self._msg_s_store = _MessageStore(num_nodes, raw_msg_dim)
        self._msg_d_store = _MessageStore(num_nodes, raw_msg_dim)
        self.reset_parameters()

    @property
    def memory(self) -> mx.array:
        """The memory of all nodes, shape `[num_nodes, memory_dim]`."""
        return self._memory

    @property
    def last_update(self) -> mx.array:
        """The time of the last memory update of every node, shape
        `[num_nodes]`."""
        return mx.array(self._last_update)

    def reset_parameters(self):
        """Re-initializes all learnable parameters and resets the state."""
        for module in (self.msg_s_module, self.msg_d_module, self.aggr_module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self.time_enc.reset_parameters()
        self.gru.reset_parameters()
        self.reset_state()

    def reset_state(self):
        """Resets the memory, the last update times and the message stores."""
        self._memory = mx.zeros((self.num_nodes, self.memory_dim))
        self._last_update = np.zeros(self.num_nodes, dtype=np.int64)
        self._reset_message_store()

    def memory_state(self) -> dict:
        """The memory state as arrays, the equivalent of PyG's ``memory``
        and ``last_update`` buffers in ``state_dict``: a dict with
        ``"memory"`` (``[num_nodes, memory_dim]``) and ``"last_update"``
        (``[num_nodes]``, int64)."""
        return {"memory": self._memory, "last_update": self.last_update}

    def load_memory_state(self, state: dict):
        """Restores a state returned by :meth:`memory_state` (e.g. read back
        with ``mx.load``). The pending messages are cleared.

        Args:
            state: A dict with ``"memory"`` and ``"last_update"``.
        """
        memory = mx.array(state["memory"]).astype(mx.float32)
        last_update = _host_time(state["last_update"])
        if memory.shape != (self.num_nodes, self.memory_dim):
            raise ValueError(
                f"memory has shape {memory.shape}, expected "
                f"{(self.num_nodes, self.memory_dim)}"
            )
        if last_update.shape != (self.num_nodes,):
            raise ValueError(
                f"last_update has shape {last_update.shape}, expected "
                f"{(self.num_nodes,)}"
            )
        self._memory = memory
        self._last_update = last_update.copy()
        self._reset_message_store()

    def detach(self):
        """Detaches the memory from gradient computation (and evaluates it)."""
        self._memory = mx.stop_gradient(self._memory)
        mx.eval(self._memory)

    def __call__(self, n_id: IndexLike) -> Tuple[mx.array, mx.array]:
        """Returns the current memory and last update time of the nodes ``n_id``
        (with the pending messages applied in training mode).

        Args:
            n_id: Node indices, shape `[n]`

        Returns:
            The memory, shape `[n, memory_dim]`, and the last update times,
            shape `[n]`.
        """
        n_id = _host(n_id)
        if self.training:
            return self._get_updated_memory(n_id)
        memory = self._memory[mx.array(n_id.astype(np.int32))]
        return memory, mx.array(self._last_update[n_id])

    def update_state(
        self, src: IndexLike, dst: IndexLike, t: IndexLike, raw_msg: mx.array
    ):
        """Updates the state with the new events ``(src, dst, t, raw_msg)``.

        Args:
            src: Source nodes, shape `[num_events]`
            dst: Destination nodes, shape `[num_events]`
            t: Integer timestamps, shape `[num_events]` (as PyG's int64
                ``TemporalData.t``; floating arrays must hold integer values,
                otherwise a ``ValueError`` is raised instead of truncating
                them)
            raw_msg: Raw messages, shape `[num_events, raw_msg_dim]`
        """
        src, dst, t = _host(src), _host(dst), _host_time(t)
        raw_msg = np.asarray(np.array(raw_msg), dtype=np.float32)
        raw_msg = raw_msg.reshape(src.size, self.raw_msg_dim)
        n_id = np.unique(np.concatenate([src, dst]))

        if self.training:
            self._update_memory(n_id)
            self._msg_s_store.update(src, dst, t, raw_msg)
            self._msg_d_store.update(dst, src, t, raw_msg)
        else:
            self._msg_s_store.update(src, dst, t, raw_msg)
            self._msg_d_store.update(dst, src, t, raw_msg)
            self._update_memory(n_id)

    def _reset_message_store(self):
        self._msg_s_store.reset()
        self._msg_d_store.reset()

    def _update_memory(self, n_id: np.ndarray):
        memory, last_update = self._get_updated_memory(n_id)
        new_memory = mx.array(self._memory)  # never mutate a shared array
        new_memory[mx.array(n_id.astype(np.int32))] = memory
        self._memory = new_memory
        new_last_update = self._last_update.copy()
        new_last_update[n_id] = np.array(last_update)
        self._last_update = new_last_update

    def _get_updated_memory(self, n_id: np.ndarray) -> Tuple[mx.array, mx.array]:
        self._assoc[n_id] = np.arange(n_id.size)

        # Messages (src -> dst) and (dst -> src).
        msg_s, t_s, src_s = self._compute_msg(
            n_id, self._msg_s_store, self.msg_s_module
        )
        msg_d, t_d, src_d = self._compute_msg(
            n_id, self._msg_d_store, self.msg_d_module
        )

        # Aggregate the messages of every node.
        idx = np.concatenate([src_s, src_d])
        msg = mx.concatenate([msg_s, msg_d], axis=0)
        t = np.concatenate([t_s, t_d])
        aggr = self.aggr_module(
            msg,
            mx.array(self._assoc[idx].astype(np.int32)),
            mx.array(t),
            n_id.size,
        )

        # Local copy of the updated memory.
        memory = self.gru(aggr, self._memory[mx.array(n_id.astype(np.int32))])

        # Local copy of the updated `last_update`: the latest message time
        # (0 for nodes without messages, as PyG's scatter-max).
        sentinel = np.iinfo(np.int64).min
        latest = np.full(self.num_nodes, sentinel, dtype=np.int64)
        np.maximum.at(latest, idx, t)
        latest[latest == sentinel] = 0
        return memory, mx.array(latest[n_id])

    def _compute_msg(
        self, n_id: np.ndarray, store: _MessageStore, msg_module: Callable
    ) -> Tuple[mx.array, np.ndarray, np.ndarray]:
        src, dst, t, raw_msg = store.gather(n_id)
        t_rel = t - self._last_update[src]
        t_enc = self.time_enc(mx.array(t_rel.astype(np.float32)))
        msg = msg_module(
            self._memory[mx.array(src.astype(np.int32))],
            self._memory[mx.array(dst.astype(np.int32))],
            mx.array(raw_msg),
            t_enc,
        )
        return msg, t, src

    def _set_training_mode(self, mode: bool):
        # Called by `train()` / `eval()` on this module or on any model that
        # contains it.
        if self.training and not mode:
            # Flush the message store to the memory when entering eval mode.
            self._update_memory(np.arange(self.num_nodes, dtype=np.int64))
            self._reset_message_store()
            mx.eval(self._memory)
        super()._set_training_mode(mode)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_nodes={self.num_nodes}, "
            f"raw_msg_dim={self.raw_msg_dim}, memory_dim={self.memory_dim}, "
            f"time_dim={self.time_dim})"
        )


class LastNeighborLoader:
    """Keeps, for every node, its ``size`` most recent neighbors in an
    ever-growing (undirected) temporal graph, as PyG's
    ``torch_geometric.nn.models.tgn.LastNeighborLoader``.

    The state is kept on the host (NumPy); the returned arrays are MLX arrays.

    Args:
        num_nodes: Number of nodes
        size: Number of most recent neighbors to keep per node

    Example:

    .. code-block:: python

        from mlx_graphs.nn.models.tgn import LastNeighborLoader

        loader = LastNeighborLoader(num_nodes=10, size=5)
        loader.insert(mx.array([0, 1]), mx.array([2, 3]))
        n_id, edge_index, e_id = loader(mx.array([0, 3]))
        # edge_index (in the local indexing of `n_id`) points from the
        # neighbors to the nodes, e_id are the events of the edges
    """

    def __init__(self, num_nodes: int, size: int):
        self.num_nodes = num_nodes
        self.size = size
        self.neighbors = np.zeros((num_nodes, size), dtype=np.int64)
        self.e_id = np.full((num_nodes, size), -1, dtype=np.int64)
        self._assoc = np.zeros(num_nodes, dtype=np.int64)
        self.reset_state()

    def __call__(self, n_id: IndexLike) -> Tuple[mx.array, mx.array, mx.array]:
        """Returns the most recent neighborhood of the nodes ``n_id``.

        Args:
            n_id: Node indices

        Returns:
            The sorted unique nodes of the neighborhood (``n_id`` and their
            neighbors), the ``[2, num_edges]`` edges from neighbors to nodes
            in the local indexing of these nodes, and the event index of every
            edge.
        """
        n_id = _host(n_id)
        neighbors = self.neighbors[n_id]
        nodes = np.repeat(n_id.reshape(-1, 1), self.size, axis=1)
        e_id = self.e_id[n_id]

        # Filter invalid neighbors (identified by `e_id < 0`).
        mask = e_id >= 0
        neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]

        # Relabel node indices.
        n_id = np.unique(np.concatenate([n_id, neighbors]))
        self._assoc[n_id] = np.arange(n_id.size)
        neighbors, nodes = self._assoc[neighbors], self._assoc[nodes]

        edge_index = np.stack([neighbors, nodes]).astype(np.int32)
        return (
            mx.array(n_id.astype(np.int32)),
            mx.array(edge_index),
            mx.array(e_id.astype(np.int32)),
        )

    def insert(self, src: IndexLike, dst: IndexLike):
        """Inserts new events (in time order) into the temporal graph.

        Args:
            src: Source nodes, shape `[num_events]`
            dst: Destination nodes, shape `[num_events]`
        """
        src, dst = _host(src), _host(dst)

        # Central nodes, their neighbors and the current event ids.
        neighbors = np.concatenate([src, dst])
        nodes = np.concatenate([dst, src])
        e_id = np.tile(np.arange(self.cur_e_id, self.cur_e_id + src.size), 2)
        self.cur_e_id += src.size

        # Locations of the new interactions in a dense [num_nodes, size] format.
        perm = np.argsort(nodes, kind="stable")
        nodes, neighbors, e_id = nodes[perm], neighbors[perm], e_id[perm]

        n_id = np.unique(nodes)
        self._assoc[n_id] = np.arange(n_id.size)

        dense_id = np.arange(nodes.size) % self.size
        dense_id += self._assoc[nodes] * self.size

        dense_e_id = np.full(n_id.size * self.size, -1, dtype=np.int64)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.reshape(-1, self.size)

        dense_neighbors = np.zeros(n_id.size * self.size, dtype=np.int64)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.reshape(-1, self.size)

        # Merge the old and new interactions and keep the most recent ones.
        e_id = np.concatenate([self.e_id[n_id], dense_e_id], axis=-1)
        neighbors = np.concatenate([self.neighbors[n_id], dense_neighbors], axis=-1)
        perm = np.argsort(-e_id, axis=-1, kind="stable")[:, : self.size]
        self.e_id[n_id] = np.take_along_axis(e_id, perm, axis=-1)
        self.neighbors[n_id] = np.take_along_axis(neighbors, perm, axis=-1)

    def reset_state(self):
        """Removes all the inserted events."""
        self.cur_e_id = 0
        self.e_id.fill(-1)
