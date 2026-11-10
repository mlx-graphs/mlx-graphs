import mlx.core as mx
import mlx.nn as nn


class Linear(nn.Linear):
    """Linear layer with Xavier Glorot weight inititalization.

    This Linear class inherits from `mx.nn.Linear`, but uses glorot
    initialization instead of the default initialization in mlx's Linear.

    Args:
        input_dims: Dimensionality of the input features
        output_dims: Dimensionality of the output features
        bias: If set to ``False`` then the layer will
          not use a bias. Default is ``True``.
    """

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        super().__init__(input_dims, output_dims, bias)

        glorot_init = nn.init.glorot_uniform()
        self.weight = glorot_init(mx.zeros((output_dims, input_dims)))
        if bias:
            self.bias = glorot_init(mx.zeros((output_dims, 1))).flatten()


class HeteroDictLinear(nn.Module):
    """Applies a separate :class:`Linear` layer to each node type.

    Node types in a heterogeneous graph usually have different numbers of
    features. This layer maps every node type to the same size, so later layers
    (e.g. :class:`HeteroConv` with ``SAGEConv(out_dim, out_dim)``) can use one
    size for all node types.

    Node types in ``x_dict`` without a layer are left out of the output, and
    layers whose node type is missing from ``x_dict`` are skipped.

    Args:
        input_dims: Number of input features per node type, e.g.
            ``{"author": 334, "paper": 4231}`` or ``data.num_node_features``.
        output_dims: Number of output features, the same for every node type.
        bias: If set to ``False`` the layers will not use a bias.
            Default is ``True``.

    Example:

    .. code-block:: python

        import mlx.core as mx
        from mlx_graphs.nn import HeteroDictLinear

        x_dict = {"author": mx.ones((4, 334)), "paper": mx.ones((5, 4231))}
        proj = HeteroDictLinear({"author": 334, "paper": 4231}, 64)
        h = proj(x_dict)
        >>> {k: v.shape for k, v in h.items()}
        {'author': (4, 64), 'paper': (5, 64)}
    """

    def __init__(
        self, input_dims: dict[str, int], output_dims: int, bias: bool = True
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims must contain at least one node type")
        self.input_dims = dict(input_dims)
        self.output_dims = output_dims
        self.lins = {
            node_type: Linear(dims, output_dims, bias=bias)
            for node_type, dims in input_dims.items()
        }

    def __call__(self, x_dict: dict[str, mx.array]) -> dict[str, mx.array]:
        """Projects the features of every node type that has a layer.

        Args:
            x_dict: Node features per node type.

        Returns:
            Projected node features per node type, each with ``output_dims``
            features.
        """
        return {
            node_type: lin(x_dict[node_type])
            for node_type, lin in self.lins.items()
            if node_type in x_dict
        }
