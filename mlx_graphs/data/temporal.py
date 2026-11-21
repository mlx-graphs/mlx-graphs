import copy
from typing import Any, Iterator, Optional, Sequence, Union

import mlx.core as mx
import numpy as np

_CORE_KEYS = ("src", "dst", "t", "msg")


def _to_numpy(value: Any) -> np.ndarray:
    return np.array(value) if isinstance(value, mx.array) else np.asarray(value)


def _prepare_index(idx: Any, num_events: int) -> Union[slice, mx.array]:
    """Converts an event index to a slice or an int32 array of positions, with
    the same accepted types as PyG's ``TemporalData``."""
    if isinstance(idx, (int, np.integer)):
        idx = int(idx)
        if idx < 0:
            idx += num_events
        return slice(idx, idx + 1)
    if isinstance(idx, slice):
        return idx
    if isinstance(idx, (list, tuple, np.ndarray, mx.array)):
        idx_np = _to_numpy(idx).reshape(-1)
        if idx_np.dtype == np.bool_:
            if idx_np.size != num_events:
                raise IndexError(
                    f"The boolean mask has {idx_np.size} entries but there are "
                    f"{num_events} events"
                )
            idx_np = np.nonzero(idx_np)[0]
        elif idx_np.size > 0 and not np.issubdtype(idx_np.dtype, np.integer):
            raise IndexError(
                f"Event indices must be integers or booleans (got '{idx_np.dtype}')"
            )
        idx_np = idx_np.astype(np.int64)
        # MLX does not check bounds: out-of-range positions would silently
        # read other events.
        if idx_np.size > 0 and (
            idx_np.min() < -num_events or idx_np.max() >= num_events
        ):
            raise IndexError(
                f"Event indices out of range for {num_events} events "
                f"(got {idx_np.min()} to {idx_np.max()})"
            )
        idx_np = np.where(idx_np < 0, idx_np + num_events, idx_np)
        return mx.array(idx_np.astype(np.int32))
    raise IndexError(
        "Only strings, integers, slices (`:`), lists, tuples, and integer or "
        f"boolean arrays are valid indices (got '{type(idx).__name__}')"
    )


class TemporalData:
    r"""A stream of events describing a temporal graph (a *Continuous-Time
    Dynamic Graph*), as PyG's ``torch_geometric.data.TemporalData``.

    Each event ``i`` is a (temporal) edge from the source node ``src[i]`` to the
    destination node ``dst[i]`` at time ``t[i]``, with a message (event
    features) ``msg[i]``. Any other per-event attribute (e.g. labels ``y``) can
    be passed to the constructor or set afterwards.

    Indexing with a string returns an attribute (``events["src"]``); any other
    index (an integer, a slice, a list, an integer array or a boolean mask)
    selects events and returns a new :class:`TemporalData` in which every
    array whose first dimension is ``num_events`` is indexed, as in PyG.

    Args:
        src: Source node of each event, shape `[num_events]`
        dst: Destination node of each event, shape `[num_events]`
        t: Timestamp of each event, shape `[num_events]`
        msg: Message (features) of each event, shape
            `[num_events, num_msg_features]`
        **kwargs: Additional attributes, e.g. event labels ``y``

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.data import TemporalData

        events = TemporalData(
            src=mx.array([1, 2, 3, 4]),
            dst=mx.array([2, 3, 4, 5]),
            t=mx.array([1000, 1010, 1100, 2000]),
            msg=mx.array([[1.0], [1.0], [0.0], [0.0]]),
            y=mx.array([1, 1, 0, 0]),
        )
        events.num_events
        >>> 4
        events.num_nodes
        >>> 6
        train, val, test = events.train_val_test_split(0.25, 0.25)

    References:
        `PyG TemporalData <https://pytorch-geometric.readthedocs.io/en/latest/
        generated/torch_geometric.data.TemporalData.html>`_
    """

    def __init__(
        self,
        src: Optional[mx.array] = None,
        dst: Optional[mx.array] = None,
        t: Optional[mx.array] = None,
        msg: Optional[mx.array] = None,
        **kwargs,
    ):
        object.__setattr__(self, "_store", {})
        self.src = src
        self.dst = dst
        self.t = t
        self.msg = msg
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, mapping: dict) -> "TemporalData":
        """Creates a :class:`TemporalData` from a dictionary of attributes."""
        return cls(**mapping)

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, key: str) -> Any:
        store = self.__dict__.get("_store")
        if store is None:
            raise AttributeError(key)
        if key in store:
            return store[key]
        if key in _CORE_KEYS:
            return None
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        prop = getattr(type(self), key, None)
        if isinstance(prop, property) and key != "edge_index":
            raise AttributeError(f"'{key}' is a read-only property")
        if value is None:
            self._store.pop(key, None)
        else:
            self._store[key] = value

    def __delattr__(self, key: str):
        self._store.pop(key, None)

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, str):
            return self._store[idx]
        return self.index_select(idx)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def __delitem__(self, key: str):
        self._store.pop(key, None)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def keys(self) -> list[str]:
        """Names of the attributes that are set."""
        return list(self._store.keys())

    def items(self):
        """``(name, value)`` pairs of the attributes that are set."""
        return self._store.items()

    def __call__(self, *keys: str) -> Iterator:
        """Iterates over ``(name, value)`` pairs, restricted to ``keys`` if
        given."""
        for key, value in self._store.items():
            if not keys or key in keys:
                yield key, value

    def to_dict(self) -> dict:
        """Returns the attributes as a dictionary."""
        return dict(self._store)

    def __copy__(self) -> "TemporalData":
        out = self.__class__.__new__(self.__class__)
        object.__setattr__(out, "_store", dict(self._store))
        return out

    def __deepcopy__(self, memo) -> "TemporalData":
        out = self.__class__.__new__(self.__class__)
        store = {}
        for key, value in self._store.items():
            if isinstance(value, mx.array):
                store[key] = mx.array(value)
            else:
                store[key] = copy.deepcopy(value, memo)
        object.__setattr__(out, "_store", store)
        return out

    def clone(self) -> "TemporalData":
        """Returns a deep copy, with copies of all the arrays."""
        return copy.deepcopy(self)

    def __getstate__(self) -> dict:
        return {"_store": self._store}

    def __setstate__(self, state: dict):
        object.__setattr__(self, "_store", state["_store"])

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """The number of nodes, ``max(src.max(), dst.max()) + 1``."""
        return max(int(self.src.max().item()), int(self.dst.max().item())) + 1

    @property
    def num_events(self) -> int:
        """The number of events (temporal edges)."""
        return self.src.shape[0]

    @property
    def num_edges(self) -> int:
        """Alias of :attr:`num_events`."""
        return self.num_events

    @property
    def edge_index(self) -> mx.array:
        """The ``[2, num_events]`` stack of ``src`` and ``dst`` (or the
        ``edge_index`` attribute if one was set)."""
        if "edge_index" in self._store:
            return self._store["edge_index"]
        if self.src is not None and self.dst is not None:
            return mx.stack([self.src, self.dst], axis=0)
        raise ValueError(
            f"{type(self).__name__} does not contain 'edge_index' information"
        )

    def size(self, dim: Optional[int] = None):
        """``(src.max(), dst.max())`` as in PyG, or one of the two."""
        size = (int(self.src.max().item()), int(self.dst.max().item()))
        return size if dim is None else size[dim]

    def __len__(self) -> int:
        return self.num_events

    def __iter__(self) -> Iterator["TemporalData"]:
        for i in range(self.num_events):
            yield self[i]

    def index_select(
        self, idx: Union[int, slice, Sequence[int], np.ndarray, mx.array]
    ) -> "TemporalData":
        """Selects events.

        Every array attribute whose first dimension is ``num_events`` is
        indexed; the others are kept as they are.

        Args:
            idx: An integer (kept as a one-event selection), a slice, a list or
                array of event positions, or a boolean mask over the events

        Returns:
            A new :class:`TemporalData` with the selected events.
        """
        num_events = self.num_events
        idx = _prepare_index(idx, num_events)
        out = copy.copy(self)
        for key, value in self._store.items():
            if (
                isinstance(value, mx.array)
                and value.ndim > 0
                and value.shape[0] == num_events
            ):
                out._store[key] = value[idx]
        return out

    def train_val_test_split(
        self, val_ratio: float = 0.15, test_ratio: float = 0.15
    ) -> tuple["TemporalData", "TemporalData", "TemporalData"]:
        """Splits the events into training, validation and test sets according to
        time, as in PyG.

        The split times are the ``1 - val_ratio - test_ratio`` and
        ``1 - test_ratio`` quantiles of ``t``, and events are assumed to be
        sorted by time.

        Args:
            val_ratio: Proportion of events in the validation set. Default:
                ``0.15``
            test_ratio: Proportion of events in the test set. Default: ``0.15``

        Returns:
            The training, validation and test events.
        """
        t = _to_numpy(self.t)
        val_time, test_time = np.quantile(
            t, [1.0 - val_ratio - test_ratio, 1.0 - test_ratio]
        )
        val_idx = int((t <= val_time).sum())
        test_idx = int((t <= test_time).sum())
        return self[:val_idx], self[val_idx:test_idx], self[test_idx:]

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def __cat_dim__(self, key: str, *args, **kwargs) -> int:
        """All attributes are concatenated along the first (event) axis."""
        return 0

    def __inc__(self, key: str, *args, **kwargs) -> Optional[int]:
        """Increment applied to ``key`` when batching: ``num_nodes`` for
        ``src`` and ``dst``, ``max + 1`` for attributes containing ``"batch"``,
        and no increment otherwise (as PyG)."""
        value = self._store.get(key)
        if "batch" in key and isinstance(value, mx.array):
            return int(value.max().item()) + 1 if value.size > 0 else 0
        if key in ("src", "dst"):
            return self.num_nodes
        return None

    @classmethod
    def from_data_list(cls, data_list: Sequence["TemporalData"]) -> "TemporalData":
        """Concatenates several event streams into one, as PyG's
        ``Batch.from_data_list`` does for ``TemporalData``.

        ``src`` and ``dst`` of each stream are offset by the number of nodes of
        the previous streams (see :meth:`__inc__`), attributes containing
        ``"batch"`` by their number of groups, and all the other array
        attributes are concatenated as they are. Non-array attributes become
        lists.

        Args:
            data_list: The event streams, with the same attributes

        Returns:
            The concatenated events.
        """
        if len(data_list) == 0:
            raise ValueError("Cannot batch an empty list of TemporalData")
        first = data_list[0]
        out = cls()
        for key in first.keys():
            values = [data[key] for data in data_list]
            if not all(isinstance(v, mx.array) for v in values):
                out[key] = values
                continue
            cat_dim = first.__cat_dim__(key)
            if first.__inc__(key) is not None:
                incs = np.cumsum([0] + [data.__inc__(key) for data in data_list])
                values = [v + int(inc) for v, inc in zip(values, incs[:-1])]
            out[key] = mx.concatenate(values, axis=cat_dim)
        return out

    def __repr__(self) -> str:
        parts = []
        for key, value in self._store.items():
            if isinstance(value, mx.array):
                parts.append(f"{key}={list(value.shape)}")
            else:
                parts.append(f"{key}={value}")
        return f"{type(self).__name__}({', '.join(parts)})"
