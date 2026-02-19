import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


def create_pyg_graph(num_nodes: int, num_edges: int) -> Data:
    """Create a synthetic graph with random edges and node features."""
    # Create random edges (ensuring valid node indices)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

    # Create random node features
    x = torch.randn(num_nodes, 64)

    return Data(x=x, edge_index=edge_index)


def benchmark_pyg_loader(
    data: Data,
    batch_size: int,
    num_neighbors: list[int],
    num_batches: int,
) -> None:
    """Run PyG NeighborLoader for a specified number of batches."""
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=None,  # Use all nodes
        shuffle=False,
    )

    for i, batch in enumerate(loader):
        # Access data to ensure loading is complete
        _ = batch.x
        _ = batch.edge_index
        if i >= num_batches - 1:
            break
