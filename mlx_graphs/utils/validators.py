import functools

import mlx.core as mx

from mlx_graphs.data.data import GraphData


def validate_adjacency_matrix(func):
    """Decorator function to check the validity of an adjacency matrix."""

    @functools.wraps(func)
    def wrapper(adjacency_matrix, *args, **kwargs):
        if adjacency_matrix.ndim != 2:
            raise ValueError(
                f"Adjacency matrix must be two-dimensional (got {adjacency_matrix.ndim} dimensions)"
            )
        if not mx.equal(*adjacency_matrix.shape):
            raise ValueError(
                f"Adjacency matrix must be a square matrix (got {adjacency_matrix.shape} shape)"
            )
        return func(adjacency_matrix, *args, **kwargs)

    return wrapper


def validate_edge_index(func):
    """Decorator function to check the validity of an edge_index."""

    @functools.wraps(func)
    def wrapper(edge_index, *args, **kwargs):
        if edge_index.ndim != 2:
            raise ValueError(
                "edge_index must be 2-dimensional with shape [2, num_edges]",
                f"(got {edge_index.ndim} dimensions)",
            )
        if edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must be 2-dimensional with shape [2, num_edges]",
                f"(got {edge_index.shape} shape)",
            )
        return func(edge_index, *args, **kwargs)

    return wrapper


def validate_edge_index_and_features(func):
    """Decorator function to check the validity of an edge_index and edge_features."""

    @functools.wraps(func)
    @validate_edge_index
    def wrapper(edge_index, edge_features=None, *args, **kwargs):
        if edge_features is not None:
            if edge_index.shape[1] != edge_features.shape[0]:
                raise ValueError(
                    "edge_features must be 1 per edge ",
                    f"(got {edge_index.shape[1]} edges and {edge_features.shape[0]} features)",
                )
        return func(edge_index, edge_features, *args, **kwargs)

    return wrapper


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
