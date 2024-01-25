from typing import Literal, Optional, get_args

import mlx.core as mx

from mlx_graphs.utils.array_ops import broadcast

ScatterAggregations = Literal["add", "max", "mean", "softmax"]


def scatter(
    values: mx.array,
    index: mx.array,
    out_size: Optional[int] = None,
    aggr: ScatterAggregations = "add",
    axis: Optional[int] = 0,
) -> mx.array:
    r"""Default function for performing all scattering operations.
    Scatters `values` at `index` in an empty array of `out_size` elements.

    Args:
        values (mx.array): array with all the values to scatter in the output tensor
        index (mx.array): array with index to which scatter the values
        out_size (int, optional): number of elements in the output array (size of the first dimension).
            If not provided, uses the number of elements in `values`
        aggr (Literal) ["add" | "max" | "softmax"]: scattering method employed for reduction at index
        axis (int, optional): axis on which applying the scattering

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
    if aggr not in get_args(ScatterAggregations):
        raise ValueError(
            "Invalid aggregation function.",
            f"Available values are {get_args(ScatterAggregations)}",
        )

    out_size = out_size if out_size is not None else values.shape[0]

    if aggr == "softmax":
        return scatter_softmax(values, index, out_size)
    if aggr == "mean":
        return scatter_mean(values, index, out_size)

    out_shape = values.shape
    out_shape[axis] = out_size

    empty_tensor = mx.zeros(out_shape, dtype=values.dtype)

    if aggr == "add":
        return scatter_add(empty_tensor, index, values)
    if aggr == "max":
        return scatter_max(empty_tensor, index, values)

    raise NotImplementedError(f"Aggregation {aggr} not implemented yet.")


def scatter_add(src: mx.array, index: mx.array, values: mx.array):
    r"""Scatters `values` at `index` within `src`. If duplicate indices are present, the sum
    of the values will be assigned to these index.

    Args:
        src (mx.array): Source array where the values will be scattered (often an empty array)
        index (mx.array): Array containing indices that determine the scatter of the 'values'.
        values (mx.array): Input array containing values to be scattered.

    Returns:
        mx.array: The resulting array after applying scatter and sum operations on the values
            at duplicate indices
    """
    return src.at[index].add(values)


def scatter_max(src: mx.array, index: mx.array, values: mx.array):
    r"""Scatters `values` at `index` within `src`. If duplicate indices are present, the maximum
    value is kept at these indices.

    Args:
        src (mx.array): Source array where the values will be scattered (often an empty array)
        index (mx.array): Array containing indices that determine the scatter of the 'values'.
        values (mx.array): Input array containing values to be scattered.

    Returns:
        mx.array: The resulting array after applying scatter and max operations on the values
            at duplicate indices
    """
    return src.at[index].maximum(values)


def scatter_mean(
    values: mx.array, index: mx.array, out_size: int, axis: int = 0
) -> mx.array:
    r"""Computes the mean of values that are scattered along a specified axis, grouped by index.

    Args:
        values (mx.array): Input array containing values to be scattered. These values will
            undergo a scatter and mean operation.
        index (mx.array): Array containing indices that determine the scatter of the `values`.
        out_size (int): Size of the output array.
        axis (int, optional): Axis along which to scatter.

    Returns:
        mx.array: An array containing mean of `values` grouped by `index`.
    """
    scatt_add = scatter(values, index, out_size, aggr="add", axis=axis)
    out_size = scatt_add.shape[axis]

    degrees = degree(index, out_size)
    degrees = mx.where(degrees < 1, 1, degrees)  # Avoid 0 division
    degrees = broadcast(degrees, scatt_add, axis)  # Match the shapes for division

    return mx.divide(scatt_add, degrees)


def scatter_softmax(
    values: mx.array, index: mx.array, out_size: int, axis: int = 0
) -> mx.array:
    r"""Computes the softmax of values that are scattered along a specified axis, grouped by index.

    Args:
        values (mx.array): Input array containing values to be scattered. These values will
            undergo a scatter and softmax operation
        index (mx.array): Array containing indices that determine the scatter of the 'values'.
        out_size (int): Size of the output array
        axis (int, optional): Axis along which to scatter

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

    scatt_sum = scatter(out, index, out_size, aggr="add", axis=axis)
    scatt_sum = scatt_sum[index]

    eps = 1e-16
    return out / (scatt_sum + eps)


def degree(index: mx.array, num_nodes: int = None) -> mx.array:
    r"""Counts the number of ocurrences of each node in the given `index`.

    Args:
        index (mx.array): Array with node indices, usually src or dst of an `edge_index`.
        num_nodes (int, optional): Size of the output degree array. If not provided, the number
            of nodes will be inferred from the `index`.

    Returns:
        mx.array: Array of length `num_nodes` with the degree of each node.
    """
    if index.ndim != 1:
        raise ValueError(
            f"The `degree` function requires a 1D index array, found {index.ndim}."
        )

    num_nodes = num_nodes if num_nodes is not None else index.max().item() + 1

    ones = mx.ones((index.shape[0],))
    return scatter(ones, index, num_nodes, "add")
