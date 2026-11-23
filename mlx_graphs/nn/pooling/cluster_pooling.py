"""Pooling of the nodes of each cluster, e.g. of a voxel grid or of graclus."""

from typing import Callable, Literal, Optional, Tuple

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.utils.clustering import consecutive_cluster
from mlx_graphs.utils.scatter import scatter
from mlx_graphs.utils.transformations import _to_numpy, coalesce


def _pool_x(
    cluster: mx.array,
    x: mx.array,
    batch: mx.array,
    aggr: Literal["mean", "max"],
    batch_size: Optional[int],
    size: Optional[int],
) -> Tuple[mx.array, Optional[mx.array]]:
    if size is not None:
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1
        return scatter(x, cluster, batch_size * size, aggr), None
    cluster, perm = consecutive_cluster(cluster)
    x = scatter(x, cluster, perm.shape[0], aggr)
    return x, batch[perm]


def avg_pool_x(
    cluster: mx.array,
    x: mx.array,
    batch: mx.array,
    batch_size: Optional[int] = None,
    size: Optional[int] = None,
) -> Tuple[mx.array, Optional[mx.array]]:
    r"""Averages the features of the nodes of each cluster.

    Args:
        cluster: Cluster of each node, of shape ``[N]`` (e.g. from
            :func:`~mlx_graphs.utils.voxel_grid` or
            :func:`~mlx_graphs.utils.graclus`).
        x: Node features, of shape ``[N, F]``.
        batch: Graph of each node, of shape ``[N]``.
        batch_size: The number of graphs (only used with ``size``). Default:
            ``None`` (inferred from ``batch``).
        size: If given, the maximum number of clusters per graph: the output
            then has ``batch_size * size`` rows indexed by ``cluster`` directly
            (empty clusters are zeros), and no batch vector is returned.
            Default: ``None``

    Returns:
        A tuple ``(x, batch)``: the features of each cluster, in increasing
        order of cluster id, and the graph of each cluster (``None`` when
        ``size`` is given).
    """
    return _pool_x(cluster, x, batch, "mean", batch_size, size)


def max_pool_x(
    cluster: mx.array,
    x: mx.array,
    batch: mx.array,
    batch_size: Optional[int] = None,
    size: Optional[int] = None,
) -> Tuple[mx.array, Optional[mx.array]]:
    r"""Takes the feature-wise maximum of the nodes of each cluster.

    Args:
        cluster: Cluster of each node, of shape ``[N]``.
        x: Node features, of shape ``[N, F]``.
        batch: Graph of each node, of shape ``[N]``.
        batch_size: The number of graphs (only used with ``size``). Default:
            ``None`` (inferred from ``batch``).
        size: If given, the maximum number of clusters per graph: the output
            then has ``batch_size * size`` rows indexed by ``cluster`` directly,
            and no batch vector is returned. Default: ``None``

    Returns:
        A tuple ``(x, batch)``: the features of each cluster, in increasing
        order of cluster id, and the graph of each cluster (``None`` when
        ``size`` is given).
    """
    return _pool_x(cluster, x, batch, "max", batch_size, size)


def pool_edge(
    cluster: mx.array,
    edge_index: mx.array,
    edge_features: Optional[mx.array] = None,
    reduce: str = "add",
) -> Tuple[mx.array, Optional[mx.array]]:
    """Maps the edges onto the clusters of their nodes, removing the self-loops
    and merging duplicate edges (``reduce`` merges their features)."""
    num_nodes = cluster.shape[0]
    edge_index = cluster[edge_index.reshape(-1)].reshape(2, -1).astype(mx.int64)
    ei = _to_numpy(edge_index)
    keep = np.nonzero(ei[0] != ei[1])[0]
    edge_index = mx.array(ei[:, keep])
    if edge_features is not None:
        edge_features = edge_features[mx.array(keep)] if keep.size else None
    if edge_index.shape[1] > 0:
        if edge_features is not None:
            edge_index, edge_features = coalesce(
                edge_index, edge_features, num_nodes, reduce
            )
        else:
            edge_index = coalesce(edge_index, num_nodes=num_nodes)
    return edge_index, edge_features


def pool_pos(cluster: mx.array, pos: mx.array) -> mx.array:
    """Average position of the nodes of each cluster (consecutive ids)."""
    return scatter(pos, cluster, int(cluster.max().item()) + 1, "mean")


def _pool(
    cluster: mx.array,
    data: GraphData,
    aggr: Literal["mean", "max"],
    transform: Optional[Callable],
) -> GraphData:
    cluster, perm = consecutive_cluster(cluster)
    num_clusters = perm.shape[0]
    x = data.node_features
    if x is not None:
        x = scatter(x, cluster, num_clusters, aggr)
    edge_index, edge_features = pool_edge(
        cluster, data.edge_index, getattr(data, "edge_features", None)
    )
    try:
        batch = getattr(data, "batch_indices", None)
    except AttributeError:
        batch = None
    pos = getattr(data, "pos", None)
    kwargs = {"num_nodes": num_clusters}
    if batch is not None:
        kwargs["batch_indices"] = batch[perm]
    if pos is not None:
        kwargs["pos"] = scatter(pos, cluster, num_clusters, "mean")
    out = GraphData(
        edge_index=edge_index.astype(mx.int32),
        node_features=x,
        edge_features=edge_features,
        **kwargs,
    )
    return transform(out) if transform is not None else out


def avg_pool(
    cluster: mx.array, data: GraphData, transform: Optional[Callable] = None
) -> GraphData:
    r"""Pools and coarsens a graph given by a clustering, averaging the node
    features and the positions of each cluster (as PyG's ``avg_pool``).

    Edges are mapped onto the clusters of their nodes; self-loops are removed
    and duplicate edges merged (their ``edge_features`` are summed).

    Args:
        cluster: Cluster of each node, of shape ``[N]``.
        data: The graph (or ``GraphDataBatch``) to coarsen.
        transform: A function/transform applied to the coarsened graph.
            Default: ``None``

    Returns:
        The coarsened graph, with one node per cluster (in increasing order of
        cluster id), the average ``pos`` and, for a batch, the
        ``batch_indices`` of each cluster.
    """
    return _pool(cluster, data, "mean", transform)


def max_pool(
    cluster: mx.array, data: GraphData, transform: Optional[Callable] = None
) -> GraphData:
    r"""Pools and coarsens a graph given by a clustering, taking the
    feature-wise maximum of the nodes of each cluster (as PyG's ``max_pool``).
    The positions are averaged.

    Edges are mapped onto the clusters of their nodes; self-loops are removed
    and duplicate edges merged (their ``edge_features`` are summed).

    Args:
        cluster: Cluster of each node, of shape ``[N]``.
        data: The graph (or ``GraphDataBatch``) to coarsen.
        transform: A function/transform applied to the coarsened graph.
            Default: ``None``

    Returns:
        The coarsened graph, with one node per cluster (in increasing order of
        cluster id), the average ``pos`` and, for a batch, the
        ``batch_indices`` of each cluster.

    Example:

    .. code-block:: python

        from mlx_graphs.nn import max_pool
        from mlx_graphs.utils import voxel_grid

        cluster = voxel_grid(batch.pos, size=0.2, batch=batch.batch_indices)
        coarse = max_pool(cluster, batch)
    """
    return _pool(cluster, data, "max", transform)
