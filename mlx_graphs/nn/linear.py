import mlx.nn as nn

from mlx_graphs.utils import glorot_init


class Linear(nn.Linear):
    r"""Linear layer with Xavier Glorot weight inititalization.

    This Linear class inherits from `mx.nn.Linear`, but uses glorot
    initialization instead of the default initialization in mlx's Linear.

    Args:
        input_dims (int): Dimensionality of the input features
        output_dims (int): Dimensionality of the output features
        bias (bool, optional): If set to ``False`` then the layer will
          not use a bias. Default is ``True``.
    """

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        super().__init__(input_dims, output_dims, bias)

        self.weight = glorot_init((output_dims, input_dims))
        if bias:
            self.bias = glorot_init((output_dims,))