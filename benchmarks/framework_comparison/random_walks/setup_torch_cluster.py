import torch
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import index2ptr


def torch_cluster_random_walk(edge_index, start_indices, walk_length, compile=True):
    num_nodes = maybe_num_nodes(edge_index=edge_index)
    row, col = sort_edge_index(edge_index=edge_index, num_nodes=num_nodes)
    row_ptr, col = index2ptr(row, num_nodes), col
    random_walks = torch.ops.torch_cluster.random_walk(
        row_ptr, col, start_indices, walk_length, 1.0, 1.0
    )
    return random_walks
