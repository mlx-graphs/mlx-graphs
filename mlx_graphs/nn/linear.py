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
            self.bias = glorot_init(mx.zeros((output_dims,)))
