from typing import Optional

import mlx.core as mx

from mlx_graphs.utils import scatter
from mlx_graphs.utils.dense import _batch_layout


def global_add_pool(
    node_features: mx.array,
    batch_indices: Optional[mx.array] = None,
    num_graphs: Optional[int] = None,
) -> mx.array:
    """Sums all node features to obtain a global graph-level representation.

    If `batch_indices` is provided, applies pooling for each graph in the batch.
    The returned shape is (1, node_features_dim) if `batch_indices` is None, otherise
    the shape is (num_batches, node_features_dim).

    Args:
        node_features: Node features array.
        batch_indices: Batch array of shape (node_features.shape[0]),
            indicating for each node its batch index.
        num_graphs: Number of graphs in the batch, i.e. number of rows of the
            output. By default, inferred as ``batch_indices.max() + 1``.

    Returns:
        An array with summed node features for all provided graphs.
    """
    if batch_indices is None:
        return node_features.sum(axis=0, keepdims=True)

    out_size = num_graphs if num_graphs is not None else batch_indices.max().item() + 1
    return scatter(node_features, batch_indices, out_size=out_size, axis=0, aggr="add")


def global_max_pool(
    node_features: mx.array,
    batch_indices: Optional[mx.array] = None,
    num_graphs: Optional[int] = None,
) -> mx.array:
    """Takes the feature-wise maximum value along all node features to obtain
    a global graph-level representation.

    If `batch_indices` is provided, applies pooling for each graph in the batch.
    The returned shape is (1, node_features_dim) if `batch_indices` is None, otherise
    the shape is (num_batches, node_features_dim).

    Args:
        node_features: Node features array.
        batch_indices: Batch array of shape (node_features.shape[0]),
            indicating for each node its batch index.
        num_graphs: Number of graphs in the batch, i.e. number of rows of the
            output. By default, inferred as ``batch_indices.max() + 1``.

    Returns:
        An array with maximum node features for all provided graphs.
    """
    if batch_indices is None:
        return node_features.max(axis=0, keepdims=True)

    out_size = num_graphs if num_graphs is not None else batch_indices.max().item() + 1
    return scatter(node_features, batch_indices, out_size=out_size, axis=0, aggr="max")


def global_mean_pool(
    node_features: mx.array,
    batch_indices: Optional[mx.array] = None,
    num_graphs: Optional[int] = None,
) -> mx.array:
    """Takes the feature-wise mean value along all node features to obtain
    a global graph-level representation.

    If `batch_indices` is provided, applies pooling for each graph in the batch.
    The returned shape is (1, node_features_dim) if `batch_indices` is None, otherise
    the shape is (num_batches, node_features_dim).

    Args:
        node_features: Node features array.
        batch_indices: Batch array of shape (node_features.shape[0]),
            indicating for each node its batch index.
        num_graphs: Number of graphs in the batch, i.e. number of rows of the
            output. By default, inferred as ``batch_indices.max() + 1``.

    Returns:
        An array with averaged node features for all provided graphs.
    """
    if batch_indices is None:
        return node_features.mean(axis=0, keepdims=True)

    out_size = num_graphs if num_graphs is not None else batch_indices.max().item() + 1
    return scatter(node_features, batch_indices, out_size=out_size, axis=0, aggr="mean")


def global_sort_pool(
    node_features: mx.array,
    batch_indices: Optional[mx.array],
    k: int,
    num_graphs: Optional[int] = None,
) -> mx.array:
    r"""The global pooling operator from the `"An End-to-End Deep Learning
    Architecture for Graph Classification"
    <https://doi.org/10.1609/aaai.v32i1.11782>`_ paper.

    The nodes of each graph are sorted in descending order of their last
    feature, and the features of the first ``k`` nodes are concatenated. Graphs
    with fewer than ``k`` nodes are padded with zeros. This is PyG's
    ``SortAggregation`` (formerly ``global_sort_pool``).

    Args:
        node_features: Node features array of shape ``[num_nodes, num_features]``.
        batch_indices: Batch array of shape (node_features.shape[0]),
            indicating for each node its batch index. ``None`` for a single
            graph.
        k: The number of nodes kept for each graph.
        num_graphs: Number of graphs in the batch, i.e. number of rows of the
            output. By default, inferred as ``batch_indices.max() + 1``.

    Returns:
        An array of shape ``[num_graphs, k * num_features]``.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import global_sort_pool

        node_features = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        batch_indices = mx.array([0, 0, 1])

        >>> global_sort_pool(node_features, batch_indices, k=2)
        array([[3, 4, 1, 2],
               [5, 6, 0, 0]], dtype=float32)
    """
    num_nodes, D = node_features.shape
    batch, local, batch_size, N = _batch_layout(batch_indices, num_nodes)
    B = batch_size if num_graphs is None else num_graphs
    if batch.size > 0 and batch.max() >= B:
        raise ValueError(
            f"num_graphs ({B}) is smaller than the number of graphs in "
            f"batch_indices ({batch_size})"
        )

    # Padding entries hold `fill_value`, smaller than every feature, so that
    # they are sorted last and can be zeroed at the end.
    fill_value = mx.stop_gradient(node_features.min()) - 1
    flat = mx.array(batch * N + local)
    dense = mx.zeros((B * N, D), dtype=node_features.dtype).at[flat].add(node_features)
    mask = mx.zeros((B * N,), dtype=mx.int32).at[flat].add(1) > 0
    dense = mx.where(mask[:, None], dense, fill_value).reshape(B, N, D)

    perm = mx.stop_gradient(mx.argsort(-dense[:, :, -1], axis=-1))
    dense = mx.take_along_axis(dense, perm[..., None], axis=1)

    if N >= k:
        dense = dense[:, :k]
    else:
        fill = mx.full((B, k - N, D), fill_value, dtype=dense.dtype)
        dense = mx.concatenate([dense, fill], axis=1)

    dense = mx.where(dense == fill_value, 0, dense)
    return dense.reshape(B, k * D)
