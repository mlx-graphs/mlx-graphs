from math import sqrt
from typing import Union, Tuple, List

import mlx.core as mx


def glorot_init(shape: Union[Tuple, List]) -> mx.array:
    """
    Glorot/Xavier initialization for a weight matrix.
    
    Args:
    shape (Union[Tuple, List]): the shape of the created tensor.
    
    Returns:
    mx.array: Initialized weight matrix.
    """
    assert len(shape) > 2, "Glorot init requires a weight matrix with for than 2 dims."

    scale = sqrt(6.0 / (shape[-2] + shape[-1]))
    return mx.random.uniform(-scale, scale, shape)
