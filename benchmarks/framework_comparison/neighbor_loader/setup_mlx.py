import mlx.core as mx

from mlx_graphs.data import GraphData
from mlx_graphs.loaders import NeighborLoader


def create_mlx_graph(num_nodes: int, num_edges: int) -> GraphData:
    """Create a synthetic graph with random edges and node features."""
    # Create random edges (ensuring valid node indices)
    sources = mx.random.randint(0, num_nodes, shape=(num_edges,))
    targets = mx.random.randint(0, num_nodes, shape=(num_edges,))
    edge_index = mx.stack([sources, targets], axis=0)

    # Create random node features
    node_features = mx.random.normal(shape=(num_nodes, 64))

    return GraphData(edge_index=edge_index, node_features=node_features)


def benchmark_mlx_loader(
    data: GraphData,
    batch_size: int,
    num_neighbors: list[int],
    num_batches: int,
) -> None:
    """Run MLX NeighborLoader for a specified number of batches."""
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
    )

    for i, batch in enumerate(loader):
        # Force evaluation to ensure computation is complete
        mx.eval(batch.edge_index)
        if batch.node_features is not None:
            mx.eval(batch.node_features)
        if i >= num_batches - 1:
            break
