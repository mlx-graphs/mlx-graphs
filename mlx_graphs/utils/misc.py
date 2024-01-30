import mlx.nn as nn


def get_num_hops(model: nn.Module) -> int:
    """
    Returns the number of hops the model is aggregating information
    from.

    Args:
        model: The GNN Model.

    Returns:
        number of hops the model is aggregating information

    Example:
        >>> class GNN(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = GCNConv(4, 16)
        ...         self.conv2 = GCNConv(16, 16)
        ...         self.lin = nn.linear(16, 2)
        ...
        ...     def __call__(self, edge_index: mx.array, node_features: mx.array) -> mx.array:
        ...         x = nn.relu(self.conv1(node_features, edge_index))
        ...         x = self.conv2(node_features, edge_index)
        ...         return self.lin(x)
        >>> get_num_hops(GNN())
        2
    """
    from mlx_graphs.nn import MessagePassing

    num_hops = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            num_hops += 1
    return num_hops
