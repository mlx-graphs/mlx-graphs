from typing import Union, List

import mlx.core as mx

from mlx_graphs.utils.utils import max_nodes

scattering_operations = ["add", "sum", "max"]


def scatter(values: mx.array, indices: mx.array, out_shape: Union[List[int], int], aggr: str, axis: int = 0):
    if aggr not in scattering_operations:
        raise NotImplementedError(f"Aggregation {aggr} not implemented yet.")

    empty_tensor = mx.zeros(out_shape, dtype=values.dtype)

    update_dim = (values.shape[0], 1, *values.shape[1:])
    values = values.reshape(update_dim)

    if aggr in ["add", "sum"]:
        return scatter_add(empty_tensor, indices, values)
    if aggr == "max":
        return scatter_max(empty_tensor, indices, values)


def scatter_add(src: mx.array, indices: mx.array, values: mx.array, axis: int = 0):
    return mx.scatter_add(src, indices, values, axis)


def scatter_max(src: mx.array, indices: mx.array, values: mx.array, axis: int = 0):
    return mx.scatter_max(src, indices, values, axis)


def scatter_softmax(src: mx.array, index: mx.array, out_size: int) -> mx.array:
    # TODO: proper broadcast should be added here.
    # this `out_shape` is a workaround for the moment.
    out_shape = src.shape
    
    scatt_max = scatter(src, index, out_shape, aggr='max')
    scatt_max = scatt_max[index]
    out = (src - scatt_max).exp()

    scatt_sum = scatter(out, index, out_shape, aggr='sum')
    scatt_sum = scatt_sum[index]

    eps = 1e-16
    return out / (scatt_sum + eps)
