from typing import Optional

import mlx.core as mx

scattering_operations = ["add", "sum", "max", "softmax"]


def scatter(
    values: mx.array,
    index: mx.array,
    out_size: Optional[int] = None,
    aggr: str = None,
    axis: Optional[int] = 0,
) -> mx.array:
    """
    Default function for performing all scattering operations.
    Scatters `values` at `index` in an empty array of `out_size` elements.

    Args:
        values (mx.array): array with all the values to scatter in the output tensor
        index (mx.array): array with index to which scatter the values
        out_size (optional, int): number of elements in the output array (size of the first dimension).
            If not provided, use the number of elements in `values`
        aggr (str) ["add" | "sum" | "max" | "softmax"]: scattering method employed for reduction at index
        axis (optional, int): axis on which applying the scattering

    Returns:
        mx.array: array with `out_size` elements containing the scattered values at given index
            following the given `aggr` reduction method

    Example:
        src = mx.array([1., 1., 1., 1.])
        index = mx.array([0, 0, 1, 2])
        num_nodes = src.shape[0]

        scatter(src, index, num_nodes, "softmax")
        >>>  mx.array([0.5, 0.5, 1, 1])

        num_nodes = index.max().item() + 1
        scatter(src, index, num_nodes, "add")
        >>>  mx.array([2, 1, 1])
    """
    if aggr not in scattering_operations:
        raise NotImplementedError(f"Aggregation {aggr} not implemented yet.")

    out_size = out_size if out_size is not None else values.shape[0]

    if aggr == "softmax":
        return scatter_softmax(values, index, out_size)

    out_shape = values.shape
    out_shape[axis] = out_size

    empty_tensor = mx.zeros(out_shape, dtype=values.dtype)
    values = mx.expand_dims(values, 1)

    if aggr in ["add", "sum"]:
        return scatter_add(empty_tensor, index, values)
    if aggr == "max":
        return scatter_max(empty_tensor, index, values)


def scatter_add(src: mx.array, index: mx.array, values: mx.array, axis: int = 0):
    """
    Scatters `values` at `index` within `src`. If duplicate indices are present, the sum
    of the values will be assigned to these index.

    Parameters:
        src (mx.array): Source array where the values will be scattered (often an empty array)
        index (mx.array): Array containing indices that determine the scatter of the 'values'.
        values (mx.array): Input array containing values to be scattered.
        axis (int, optional): Axis along which to scatter

    Returns:
        mx.array: The resulting array after applying scatter and sum operations on the values
            at duplicate indices
    """
    return mx.scatter_add(src, index, values, axis)


def scatter_max(src: mx.array, index: mx.array, values: mx.array, axis: int = 0):
    """
    Scatters `values` at `index` within `src`. If duplicate indices are present, the maximum
    value is kept at these indices.

    Parameters:
        src (mx.array): Source array where the values will be scattered (often an empty array)
        index (mx.array): Array containing indices that determine the scatter of the 'values'.
        values (mx.array): Input array containing values to be scattered.
        axis (int, optional): Axis along which to scatter

    Returns:
        mx.array: The resulting array after applying scatter and max operations on the values
            at duplicate indices
    """
    return mx.scatter_max(src, index, values, axis)


def scatter_softmax(
    values: mx.array, index: mx.array, out_size: int, axis: int = 0
) -> mx.array:
    """
    Compute the softmax of values that are scattered along a specified axis, grouped by index.

    Parameters:
        values (mx.array): Input array containing values to be scattered. These values will
                        undergo a scatter and softmax operation
        index (mx.array): Array containing indices that determine the scatter of the 'values'.
        out_size (int, optional): Size of the output array
        axis (int, optional): Axis along which to scatter and compute the softmax

    Returns:
        mx.array: The resulting array after applying scatter and softmax operations on the input 'values'.

    Example:
        src = mx.array([1., 1., 1., 1.])
        index = mx.array([0, 0, 1, 2])
        num_nodes = src.shape[0]

        scatter_softmax(src, index, num_nodes)
        >>>  mx.array([0.5, 0.5, 1, 1])
    """
    # index = broadcast(index, values, axis) # NOTE: may be used in future.
    scatt_max = scatter(values, index, out_size, aggr="max", axis=axis)
    scatt_max = scatt_max[index]
    out = (values - scatt_max).exp()

    scatt_sum = scatter(out, index, out_size, aggr="sum", axis=axis)
    scatt_sum = scatt_sum[index]

    eps = 1e-16
    return out / (scatt_sum + eps)
