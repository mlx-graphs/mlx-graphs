import inspect
from typing import Any, Dict, List, Type, Union

from mlx_graphs.nn.aggr.base import Aggregation

# PyG names of the size arguments, accepted as aliases of the mlx-graphs ones.
_PYG_ARGUMENTS = {
    "in_channels": "node_features_dim",
    "out_channels": "out_features_dim",
}


def _normalize(name: str) -> str:
    key = name.lower().replace("_", "").replace("-", "")
    if key.endswith("aggregation"):
        key = key[: -len("aggregation")]
    return key


def _aggregations() -> Dict[str, Type[Aggregation]]:
    """All the aggregation classes of :mod:`mlx_graphs.nn.aggr`, by
    normalized name (lower case, without underscores and without the
    ``Aggregation`` suffix), as PyG's resolver."""
    import mlx_graphs.nn.aggr as aggr_module

    classes = {
        _normalize(name): obj
        for name, obj in vars(aggr_module).items()
        if isinstance(obj, type)
        and issubclass(obj, Aggregation)
        and obj is not Aggregation
    }
    classes["add"] = classes["sum"]
    return classes


def aggregation_resolver(
    aggr: Union[str, Aggregation, List[Union[str, Aggregation]]], **kwargs: Any
) -> Aggregation:
    """Returns the :class:`Aggregation` described by ``aggr``, as PyG's
    ``aggregation_resolver``.

    Args:
        aggr: An :class:`Aggregation` (returned as is), a list of them or of
            names (combined with :class:`MultiAggregation`), or the name of
            any aggregation class of :mod:`mlx_graphs.nn.aggr`, either its
            short name (``"sum"`` or ``"add"``, ``"mean"``, ``"max"``,
            ``"min"``, ``"mul"``, ``"var"``, ``"std"``, ``"softmax"``,
            ``"powermean"``, ``"median"``, ``"quantile"``, ``"lstm"``,
            ``"gru"``, ``"sort"``, ``"set_transformer"``, ``"mlp"``,
            ``"lcm"``, ``"deep_sets"``, ...) or its class name (e.g.
            ``"SoftmaxAggregation"``), case-insensitive.
        **kwargs: Arguments of the aggregation class when ``aggr`` is a name
            or a list. PyG's ``in_channels`` and ``out_channels`` are accepted
            for ``node_features_dim`` and ``out_features_dim``.

    Returns:
        An :class:`Aggregation` instance.
    """
    if isinstance(aggr, Aggregation):
        return aggr
    if isinstance(aggr, (list, tuple)):
        from mlx_graphs.nn.aggr.multi import MultiAggregation

        return MultiAggregation(list(aggr), **kwargs)
    if not isinstance(aggr, str):
        raise ValueError(
            f"Expected an Aggregation or a string, got {type(aggr).__name__}"
        )
    classes = _aggregations()
    key = _normalize(aggr)
    if key not in classes:
        raise ValueError(f"Unknown aggregation '{aggr}'. Available: {sorted(classes)}")
    cls = classes[key]
    parameters = inspect.signature(cls.__init__).parameters
    for pyg_name, name in _PYG_ARGUMENTS.items():
        if pyg_name in kwargs and pyg_name not in parameters and name in parameters:
            kwargs[name] = kwargs.pop(pyg_name)
    return cls(**kwargs)
