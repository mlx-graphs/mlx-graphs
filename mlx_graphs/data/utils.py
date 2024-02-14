import functools

from .data import GraphData


def validate_list_of_graph_data(func):
    """Decorator function to check the validity of a list of `GraphData`."""

    @functools.wraps(func)
    def wrapper(graph_list: list[GraphData], *args, **kwargs):
        if not isinstance(graph_list, list):
            raise ValueError(f"Expected list of GraphData, got {type(graph_list)}.")
        try:
            expected_attr = set(graph_list[0].to_dict())
        except AttributeError:
            raise ValueError(
                "Expected list of GraphData. "
                "Graph at index 0 in the batch is not of type `GraphData`."
            )
        for i, graph in enumerate(graph_list):
            if not isinstance(graph, GraphData):
                raise ValueError(
                    "Expected list of GraphData. "
                    f"Graph at index {i} in the batch is not of type `GraphData`."
                )
            graph_attr = set(graph.to_dict())
            if graph_attr != expected_attr:
                raise ValueError(
                    "A graph in the batch has mismatching attributes. "
                    f"Found attributes at graph index {i}: {graph_attr}."
                )

        return func(graph_list, *args, **kwargs)

    return wrapper
