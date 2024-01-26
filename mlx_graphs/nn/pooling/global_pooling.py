from typing import Optional

import mlx.core as mx

from mlx_graphs.utils import scatter


def global_add_pool(
    node_features: mx.array, batch: Optional[mx.array] = None
) -> mx.array:
    """Sums all node features to obtain a global graph-level representation.

    If `batch` is provided, applies pooling for each graph in the batch.
    The returned shape is (1, node_features_dim) if `batch` is None, otherise
    the shape is (num_batches, node_features_dim).

    Args:
        node_features: Node features array.
        batch: Batch array of shape (node_features.shape[0]),
            indicating for each node its batch index.

    Returns:
        An array with summed node features for all provided graphs.
    """
    if batch is None:
        return node_features.sum(axis=0, keepdims=True)

    out_size = batch.max().item() + 1
    return scatter(node_features, batch, out_size=out_size, axis=0, aggr="add")


def global_max_pool(
    node_features: mx.array, batch: Optional[mx.array] = None
) -> mx.array:
    """Takes the feature-wise maximum value along all node features to obtain
    a global graph-level representation.

    If `batch` is provided, applies pooling for each graph in the batch.
    The returned shape is (1, node_features_dim) if `batch` is None, otherise
    the shape is (num_batches, node_features_dim).

    Args:
        node_features: Node features array.
        batch: Batch array of shape (node_features.shape[0]),
            indicating for each node its batch index.

    Returns:
        An array with maximum node features for all provided graphs.
    """
    if batch is None:
        return node_features.max(axis=0, keepdims=True)

    out_size = batch.max().item() + 1
    return scatter(node_features, batch, out_size=out_size, axis=0, aggr="max")


def global_mean_pool(
    node_features: mx.array, batch: Optional[mx.array] = None
) -> mx.array:
    """Takes the feature-wise mean value along all node features to obtain
    a global graph-level representation.

    If `batch` is provided, applies pooling for each graph in the batch.
    The returned shape is (1, node_features_dim) if `batch` is None, otherise
    the shape is (num_batches, node_features_dim).

    Args:
        node_features: Node features array.
        batch: Batch array of shape (node_features.shape[0]),
            indicating for each node its batch index.

    Returns:
        An array with averaged node features for all provided graphs.
    """
    if batch is None:
        return node_features.mean(axis=0, keepdims=True)

    out_size = batch.max().item() + 1
    return scatter(node_features, batch, out_size=out_size, axis=0, aggr="mean")
