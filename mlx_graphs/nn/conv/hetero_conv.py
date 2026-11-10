from typing import Any, Literal, get_args

import mlx.core as mx
import mlx.nn as nn

EdgeType = tuple[str, str, str]
HeteroAggregations = Literal["sum", "mean", "max"]

_DELIM = "__"


def _edge_type_to_key(edge_type: EdgeType) -> str:
    if not (isinstance(edge_type, tuple) and len(edge_type) == 3):
        raise ValueError(
            f"Edge types must be (src, rel, dst) tuples, got {edge_type!r}"
        )
    if any(_DELIM in part for part in edge_type):
        raise ValueError(
            f"Node/relation names may not contain '{_DELIM}': {edge_type!r}"
        )
    return _DELIM.join(edge_type)


class HeteroConv(nn.Module):
    """Wrapper that runs one graph convolution per edge type of a heterogeneous
    graph and combines the results per destination node type.

    For every edge type ``(src, rel, dst)`` the corresponding layer computes new
    embeddings for the ``dst`` nodes from the ``src`` nodes. When several edge
    types point to the same node type (e.g. ``author -> paper`` and
    ``paper -> paper``), their outputs are combined with ``aggr``.

    Edge types between two different node types are called with a
    ``(src_features, dst_features)`` tuple, so the wrapped layer must support
    bipartite input (e.g. :class:`SAGEConv`, with
    ``node_features_dim=(src_dim, dst_dim)`` if the sizes differ). Edge types
    within one node type are called with a single array.

    Node types that no edge type points to are not in the output. Edge types
    missing from ``edge_index_dict``, or whose source or destination node type
    is missing from ``x_dict``, are skipped.

    Args:
        convs: A dict mapping each edge type ``(src, rel, dst)`` to the layer
            to apply to it.
        aggr: How to combine the outputs of edge types that share a
            destination node type: ``"sum"``, ``"mean"`` or ``"max"``.
            Default: ``"sum"``.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import HeteroConv, SAGEConv

        x_dict = {"author": mx.ones((4, 16)), "paper": mx.ones((5, 8))}
        edge_index_dict = {
            ("author", "writes", "paper"): mx.array([[0, 1, 3], [0, 0, 4]]),
            ("paper", "cites", "paper"): mx.array([[1, 2], [0, 0]]),
            ("paper", "rev_writes", "author"): mx.array([[0, 0, 4], [0, 1, 3]]),
        }
        conv = HeteroConv(
            {
                ("author", "writes", "paper"): SAGEConv((16, 8), 32),
                ("paper", "cites", "paper"): SAGEConv(8, 32),
                ("paper", "rev_writes", "author"): SAGEConv((8, 16), 32),
            },
            aggr="sum",
        )
        out = conv(x_dict, edge_index_dict)
        >>> {k: v.shape for k, v in out.items()}
        {'paper': (5, 32), 'author': (4, 32)}
    """

    def __init__(
        self,
        convs: dict[EdgeType, nn.Module],
        aggr: HeteroAggregations = "sum",
    ):
        super().__init__()
        if aggr not in get_args(HeteroAggregations):
            raise ValueError(
                f"Invalid aggregation '{aggr}'. "
                f"Available values are {get_args(HeteroAggregations)}"
            )
        if not convs:
            raise ValueError("convs must contain at least one edge type")

        self.aggr = aggr
        self.edge_types = list(convs.keys())
        # MLX tracks parameters of sub-modules stored in dicts with string
        # keys, so (src, rel, dst) tuples are stored as "src__rel__dst".
        self.convs = {_edge_type_to_key(et): conv for et, conv in convs.items()}

    def __call__(
        self,
        x_dict: dict[str, mx.array],
        edge_index_dict: dict[EdgeType, mx.array],
        **kwargs_dict: dict[EdgeType, Any],
    ) -> dict[str, mx.array]:
        """Computes new node embeddings for every destination node type.

        Args:
            x_dict: Node features per node type.
            edge_index_dict: Edge index of shape ``[2, num_edges]`` per edge type.
            **kwargs_dict: Optional extra per-edge-type arguments for the wrapped
                layers, passed as dicts keyed by edge type and named
                ``<argument>_dict``. For example
                ``edge_weights_dict={et: weights}`` calls the layer of ``et``
                with ``edge_weights=weights``.

        Returns:
            Node embeddings per destination node type.
        """
        outputs: dict[str, list[mx.array]] = {}

        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict:
                continue
            src, _, dst = edge_type
            if src not in x_dict or dst not in x_dict:
                continue

            conv = self.convs[_edge_type_to_key(edge_type)]
            edge_index = edge_index_dict[edge_type]

            kwargs = {}
            for name, values in kwargs_dict.items():
                if not name.endswith("_dict"):
                    raise ValueError(
                        f"Extra arguments must be dicts named '<argument>_dict', "
                        f"got '{name}'"
                    )
                if edge_type in values:
                    kwargs[name[: -len("_dict")]] = values[edge_type]

            if src == dst:
                out = conv(edge_index, x_dict[src], **kwargs)
            else:
                out = conv(edge_index, (x_dict[src], x_dict[dst]), **kwargs)

            outputs.setdefault(dst, []).append(out)

        return {node_type: self._combine(outs) for node_type, outs in outputs.items()}

    def _combine(self, outs: list[mx.array]) -> mx.array:
        if len(outs) == 1:
            return outs[0]
        stacked = mx.stack(outs, axis=0)
        if self.aggr == "sum":
            return stacked.sum(axis=0)
        if self.aggr == "mean":
            return stacked.mean(axis=0)
        return stacked.max(axis=0)
