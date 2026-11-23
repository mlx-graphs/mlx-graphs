import math

import mlx.core as mx
import mlx.nn as nn


def glorot(shape: tuple) -> mx.array:
    r"""Glorot (Xavier) uniform initialization over the last two dimensions.

    Matches PyG's ``glorot``: values are drawn from
    :math:`\mathcal{U}(-a, a)` with :math:`a = \sqrt{6 / (n_{-2} + n_{-1})}`,
    where :math:`n_{-2}, n_{-1}` are the sizes of the last two dimensions.
    Unlike ``mlx.nn.init.glorot_uniform``, it does not treat arrays with more
    than two dimensions as convolution kernels, which is what attention
    vectors of shape ``[1, heads, channels]`` need.

    Args:
        shape: Shape of the array to initialize (at least 2 dimensions).
    """
    if len(shape) < 2:
        raise ValueError(f"glorot needs at least 2 dimensions, got shape {shape}")
    limit = math.sqrt(6.0 / (shape[-2] + shape[-1]))
    return mx.random.uniform(-limit, limit, shape)


class Linear(nn.Linear):
    """Linear layer with Xavier Glorot weight initialization and a zero bias.

    This Linear class inherits from `mx.nn.Linear`, but uses glorot
    initialization instead of the default initialization in mlx's Linear,
    like PyG's ``Linear(weight_initializer="glorot")`` used by e.g. GCN and
    GAT. Layers that PyG initializes like ``torch.nn.Linear`` (e.g.
    GraphSAGE) use ``mlx.nn.Linear``, which has the same default
    initialization as PyTorch.

    Args:
        input_dims: Dimensionality of the input features
        output_dims: Dimensionality of the output features
        bias: If set to ``False`` then the layer will
            not use a bias. Default is ``True``.
    """

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        super().__init__(input_dims, output_dims, bias)

        self.weight = glorot((output_dims, input_dims))
        if bias:
            self.bias = mx.zeros((output_dims,))


class HeteroDictLinear(nn.Module):
    """Applies a separate linear layer to each node type.

    The layers are initialized like PyG's ``HeteroDictLinear`` (and
    ``torch.nn.Linear``): weights and biases uniform in
    ``[-1/sqrt(input_dims), 1/sqrt(input_dims)]``, i.e. ``mlx.nn.Linear``'s
    default.

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
            node_type: nn.Linear(dims, output_dims, bias=bias)
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
