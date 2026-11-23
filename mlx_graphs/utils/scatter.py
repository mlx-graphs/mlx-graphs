import math
from typing import List, Literal, Optional, Sequence, Union, get_args

import mlx.core as mx

from mlx_graphs.utils.array_ops import broadcast

ScatterAggregations = Literal[
    "add", "max", "mean", "softmax", "min", "mul", "var", "std", "median"
]
QuantileInterpolations = Literal["linear", "lower", "higher", "nearest", "midpoint"]


def scatter(
    values: mx.array,
    index: mx.array,
    out_size: Optional[int] = None,
    aggr: ScatterAggregations = "add",
    axis: int = 0,
) -> mx.array:
    """Default function for performing all scattering operations.
    Scatters `values` at `index` in an empty array of `out_size` elements.

    Args:
        values: array with all the values to scatter in the output tensor
        index: array with index to which scatter the values
        out_size: number of elements in the output array (size of the first dimension).
            If not provided, uses the number of elements in `values`
        aggr: scattering method employed for reduction at index:

            * ``"add"``, ``"mean"``, ``"max"``, ``"min"``: sum, average,
              feature-wise maximum and minimum (0 for empty groups)
            * ``"mul"``: product (1 for empty groups, as in PyG)
            * ``"var"``, ``"std"``: population variance and standard deviation
              (see :func:`scatter_var` and :func:`scatter_std`)
            * ``"median"``: lower median (see :func:`scatter_quantile`)
            * ``"softmax"``: softmax normalization of the values within each
              group. Unlike the other reductions, the output has one row per
              *value*, not per group.
        axis: axis on which applying the scattering

    Returns:
        Array with `out_size` elements containing the scattered values at given index
        following the given `aggr` reduction method

    Example:

    .. code-block:: python

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

    _out_size: int = out_size if out_size is not None else values.shape[0]

    if aggr == "softmax":
        return scatter_softmax(values, index, _out_size)
    if aggr == "mean":
        return scatter_mean(values, index, _out_size)
    if aggr == "std":
        return scatter_std(values, index, _out_size)
    if aggr == "var":
        return scatter_var(values, index, _out_size)
    if aggr == "mul":
        return scatter_mul(values, index, _out_size)
    if aggr == "median":
        return scatter_quantile(values, index, _out_size, q=0.5, interpolation="lower")

    out_shape = list(values.shape)
    out_shape[axis] = _out_size

    empty_tensor = mx.zeros(out_shape, dtype=values.dtype)

    if aggr == "add":
        return scatter_add(empty_tensor, index, values)
    if aggr == "max":
        return scatter_max(empty_tensor, index, values)
    if aggr == "min":
        return scatter_min(empty_tensor, index, values)

    raise NotImplementedError(f"Aggregation {aggr} not implemented yet.")


def scatter_add(src: mx.array, index: mx.array, values: mx.array):
    """Scatters `values` at `index` within `src`. If duplicate indices are present,
    the sum of the values will be assigned to these index.

    Args:
        src: Source array where the values will be scattered (often an empty array)
        index: Array containing indices that determine the scatter of the 'values'.
        values: Input array containing values to be scattered.

    Returns:
        The resulting array after applying scatter and sum operations on the values at
        duplicate indices
    """
    return src.at[index].add(values)


def scatter_max(src: mx.array, index: mx.array, values: mx.array):
    """Scatters `values` at `index` within `src`. If duplicate indices are present,
    the maximum value is kept at these indices.

    Args:
        src: Source array where the values will be scattered (often an empty array)
        index: Array containing indices that determine the scatter of the 'values'.
        values: Input array containing values to be scattered.

    Returns:
        The resulting array after applying scatter and max operations on the values at
        duplicate indices
    """
    out = _scatter_extremum(src.shape, index, values, "max")
    count = broadcast(degree(index, src.shape[0]), out, 0)
    return mx.where(count > 0, out, src)


def scatter_min(src: mx.array, index: mx.array, values: mx.array):
    """Scatters `values` at `index` within `src`. If duplicate indices are present,
    the minimum value is kept at these indices.

    Args:
        src: Source array where the values will be scattered (often an empty array)
        index: Array containing indices that determine the scatter of the 'values'.
        values: Input array containing values to be scattered.

    Returns:
        The resulting array after applying scatter and min operations on the values at
        duplicate indices
    """
    out = _scatter_extremum(src.shape, index, values, "min")
    count = broadcast(degree(index, src.shape[0]), out, 0)
    return mx.where(count > 0, out, src)


def _scatter_extremum(
    out_shape: Sequence[int], index: mx.array, values: mx.array, reduce: str
) -> mx.array:
    """Scatter-max (``reduce="max"``) or scatter-min of ``values`` into an
    array of shape ``out_shape``, with -inf (+inf for min) in empty groups.

    MLX's gradient of ``.at[].maximum`` gives the full gradient to every
    value equal to the maximum of its group, so ties (e.g. the identical
    messages of duplicate edges) get k times too much in total. As torch's
    ``scatter_reduce`` (used by PyG without torch_scatter), the custom
    gradient instead splits it equally among the tied values.
    """
    floating = mx.issubdtype(values.dtype, mx.floating)
    if reduce == "max":
        fill = -float("inf") if floating else mx.iinfo(values.dtype).min
    else:
        fill = float("inf") if floating else mx.iinfo(values.dtype).max

    def _forward(x: mx.array) -> mx.array:
        out = mx.full(out_shape, fill, dtype=x.dtype).at[index]
        return out.maximum(x) if reduce == "max" else out.minimum(x)

    @mx.custom_function
    def _extremum(x: mx.array) -> mx.array:
        return _forward(x)

    @_extremum.vjp
    def _extremum_vjp(primals, cotangent, output):
        # a single primal is passed as is, several as a tuple
        x = primals if isinstance(primals, mx.array) else primals[0]
        is_extremum = x == output[index]
        num_ties = (
            mx.zeros(out_shape, dtype=mx.int32)
            .at[index]
            .add(is_extremum.astype(mx.int32))
        )
        share = cotangent / mx.maximum(num_ties, 1).astype(cotangent.dtype)
        return (mx.where(is_extremum, share[index], mx.zeros_like(x)),)

    return _extremum(values)


def scatter_mean(
    values: mx.array, index: mx.array, out_size: int, axis: int = 0
) -> mx.array:
    """Computes the mean of values that are scattered along a specified axis, grouped
    by index.

    Args:
        values: Input array containing values to be scattered. These values will
            undergo a scatter and mean operation.
        index: Array containing indices that determine the scatter of the `values`.
        out_size: Size of the output array.
        axis: Axis along which to scatter.

    Returns:
        An array containing mean of `values` grouped by `index`.
    """
    scatt_add = scatter(values, index, out_size, aggr="add", axis=axis)
    out_size = scatt_add.shape[axis]

    degrees = degree(index, out_size)
    degrees = mx.where(degrees < 1, 1, degrees)  # Avoid 0 division
    degrees = broadcast(degrees, scatt_add, axis)  # Match the shapes for division

    return mx.divide(scatt_add, degrees)


def scatter_mul(values: mx.array, index: mx.array, out_size: int) -> mx.array:
    """Computes the product of the values grouped by index (along axis 0).

    Groups without any value are set to 1, as in PyG. MLX has no gradient for
    multiplicative scatters, so this function provides its own: the gradient
    of a value is the product of the other values of its group, which is also
    correct for groups containing zeros.

    Args:
        values: Input array of shape ``[num_values, ...]``.
        index: Array of shape ``[num_values]`` with the group of each value.
        out_size: Number of groups.

    Returns:
        An array of shape ``[out_size, ...]`` with the product of each group.
    """
    out_shape = (out_size, *values.shape[1:])

    def _prod(x: mx.array) -> mx.array:
        return mx.ones(out_shape, dtype=x.dtype).at[index].multiply(x)

    @mx.custom_function
    def _scatter_mul(x: mx.array) -> mx.array:
        return _prod(x)

    @_scatter_mul.vjp
    def _scatter_mul_vjp(primals, cotangent, output):
        # a single primal is passed as is, several as a tuple
        x = primals if isinstance(primals, mx.array) else primals[0]
        is_zero = x == 0
        ones = mx.ones_like(x)
        zeros = mx.zeros_like(x)
        # product of the non-zero values and number of zeros of each group
        prod_nonzero = _prod(mx.where(is_zero, ones, x))[index]
        num_zeros = (
            mx.zeros(out_shape, dtype=mx.int32).at[index].add(is_zero.astype(mx.int32))
        )[index]
        # d(prod)/dx_i = prod_{j != i} x_j
        grad = mx.where(
            is_zero,
            mx.where(num_zeros == 1, prod_nonzero, zeros),
            mx.where(num_zeros == 0, prod_nonzero / mx.where(is_zero, ones, x), zeros),
        )
        return (cotangent[index] * grad,)

    return _scatter_mul(values)


def scatter_var(
    values: mx.array, index: mx.array, out_size: int, semi_grad: bool = False
) -> mx.array:
    r"""Computes the (population) variance of the values grouped by index
    (along axis 0),
    :math:`\mathrm{var}(\mathcal{X}) = \mathrm{mean}(\mathcal{X}^2) -
    \mathrm{mean}(\mathcal{X})^2`, as PyG's ``VarAggregation``. Groups without
    any value are set to 0.

    Args:
        values: Input array of shape ``[num_values, ...]``.
        index: Array of shape ``[num_values]`` with the group of each value.
        out_size: Number of groups.
        semi_grad: If ``True``, no gradient flows through
            :math:`\mathrm{mean}(\mathcal{X}^2)` (semi-gradient, as in PyG).
            Default: ``False``

    Returns:
        An array of shape ``[out_size, ...]`` with the variance of each group.
    """
    mean = scatter_mean(values, index, out_size)
    mean2 = scatter_mean(values * values, index, out_size)
    if semi_grad:
        mean2 = mx.stop_gradient(mean2)
    return mean2 - mean * mean


def scatter_std(
    values: mx.array,
    index: mx.array,
    out_size: int,
    semi_grad: bool = False,
    eps: float = 1e-5,
) -> mx.array:
    """Computes the (population) standard deviation of the values grouped by
    index (along axis 0).

    As in PyG's ``StdAggregation``, the variance is clamped to ``eps`` before
    the square root (so that the gradient stays finite) and standard
    deviations up to ``sqrt(eps)`` are set to 0.

    Args:
        values: Input array of shape ``[num_values, ...]``.
        index: Array of shape ``[num_values]`` with the group of each value.
        out_size: Number of groups.
        semi_grad: See :func:`scatter_var`. Default: ``False``
        eps: Lower bound of the variance. Default: ``1e-5``

    Returns:
        An array of shape ``[out_size, ...]`` with the standard deviation of
        each group.
    """
    var = scatter_var(values, index, out_size, semi_grad=semi_grad)
    out = mx.sqrt(mx.maximum(var, eps))
    return mx.where(out <= math.sqrt(eps), mx.zeros_like(out), out)


def scatter_quantile(
    values: mx.array,
    index: mx.array,
    out_size: int,
    q: Union[float, List[float]] = 0.5,
    interpolation: QuantileInterpolations = "linear",
    fill_value: float = 0.0,
) -> mx.array:
    r"""Computes the feature-wise :math:`q`-th quantile of the values grouped
    by index (along axis 0), as PyG's ``QuantileAggregation``.

    The values of each group are sorted, and the quantile is read at position
    :math:`q \cdot (n - 1)` of its :math:`n` sorted values, interpolating
    between the two neighbouring values when the position is not an integer.
    The index does not need to be sorted. The gradient flows to the selected
    values.

    Args:
        values: Input array of shape ``[num_values, ...]``.
        index: Array of shape ``[num_values]`` with the group of each value.
        out_size: Number of groups.
        q: Quantile(s) in ``[0, 1]``. With several quantiles, the results are
            concatenated along the last axis. Default: ``0.5``
        interpolation: How to read a quantile lying between two values
            ``a <= b``: ``"linear"`` (``a + (b - a) * frac``), ``"lower"``
            (``a``), ``"higher"`` (``b``), ``"nearest"`` or ``"midpoint"``.
            Default: ``"linear"``
        fill_value: Value of groups without any value. Default: ``0.0``

    Returns:
        An array of shape ``[out_size, ...]``, or ``[out_size, ...,
        len(q) * num_features]`` for several quantiles.

    Example:

    .. code-block:: python

        values = mx.array([[1.0], [5.0], [3.0], [2.0]])
        index = mx.array([0, 0, 0, 1])

        scatter_quantile(values, index, 3, q=0.5)
        >>> mx.array([[3.0], [2.0], [0.0]])
    """
    qs = list(q) if isinstance(q, (list, tuple)) else [q]
    if len(qs) == 0:
        raise ValueError("Provide at least one quantile value for `q`.")
    if not all(0.0 <= quantile <= 1.0 for quantile in qs):
        raise ValueError("`q` must be in the range [0, 1].")
    if interpolation not in get_args(QuantileInterpolations):
        raise ValueError(
            f"Invalid interpolation '{interpolation}', available values are "
            f"{get_args(QuantileInterpolations)}"
        )

    num_values = values.shape[0]
    feature_shape = values.shape[1:]
    if num_values == 0:
        out = mx.full((out_size, *feature_shape), fill_value, dtype=values.dtype)
        return out if len(qs) == 1 else mx.concatenate([out] * len(qs), axis=-1)

    x = values.reshape(num_values, -1)

    # Sort each feature column by value, then (stably) by group, so that the
    # values of every group are contiguous and sorted in every column.
    perm = mx.argsort(x, axis=0)
    rank = mx.arange(num_values, dtype=mx.int64).reshape(-1, 1)
    group_key = index.astype(mx.int64)[perm] * num_values + rank
    perm = mx.take_along_axis(perm, mx.argsort(group_key, axis=0), axis=0)
    x = mx.take_along_axis(x, mx.stop_gradient(perm), axis=0)

    count = mx.zeros((out_size,), dtype=mx.int32).at[index].add(1)
    start = mx.minimum(mx.cumsum(count) - count, num_values - 1)
    empty = (count == 0).reshape(-1, 1)
    fill = mx.array(fill_value, dtype=x.dtype)

    outs = []
    for quantile in qs:
        # Position within the group, added to the (integer) start of the
        # group afterwards: a float32 global position would be rounded for
        # more than 2**24 values and read from the wrong group.
        local = mx.maximum(quantile * (count - 1).astype(mx.float32), 0.0)
        low = mx.floor(local)
        frac_point = local - low
        low_idx = mx.minimum(start + low.astype(mx.int32), num_values - 1)
        high_idx = mx.minimum(start + mx.ceil(local).astype(mx.int32), num_values - 1)
        if interpolation == "lower":
            out = x[low_idx]
        elif interpolation == "higher":
            out = x[high_idx]
        elif interpolation == "nearest":
            # round half to even on the global position, as torch.round
            tie_idx = mx.where(low_idx % 2 == 0, low_idx, high_idx)
            nearest = mx.where(frac_point < 0.5, low_idx, high_idx)
            out = x[mx.where(frac_point == 0.5, tie_idx, nearest)]
        elif interpolation == "linear":
            frac = frac_point.reshape(-1, 1).astype(x.dtype)
            out = x[low_idx] + (x[high_idx] - x[low_idx]) * frac
        else:  # midpoint
            out = 0.5 * x[low_idx] + 0.5 * x[high_idx]
        out = mx.where(empty, fill, out)
        outs.append(out.reshape(out_size, *feature_shape))

    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=-1)


def scatter_softmax(
    values: mx.array, index: mx.array, out_size: int, axis: int = 0
) -> mx.array:
    """Computes the softmax of values that are scattered along a specified axis,
    grouped by index.

    Args:
        values: Input array containing values to be scattered. These values will
            undergo a scatter and softmax operation
        index: Array containing indices that determine the scatter of the 'values'.
        out_size: Size of the output array
        axis: Axis along which to scatter

    Returns:
        The resulting array after applying scatter and softmax operations on the
        input 'values'.

    Example:

    .. code-block:: python

        src = mx.array([1., 1., 1., 1.])
        index = mx.array([0, 0, 1, 2])
        num_nodes = src.shape[0]

        scatter_softmax(src, index, num_nodes)
        >>>  mx.array([0.5, 0.5, 1, 1])
    """
    out_shape = list(values.shape)
    out_shape[axis] = out_size

    # Subtracting the per-group maximum keeps exp() from overflowing. The
    # softmax does not depend on that shift, so no gradient needs to flow
    # through it, which skips the backward pass of the scatter-max.
    scatt_max = mx.full(out_shape, -float("inf"), dtype=values.dtype)
    scatt_max = mx.stop_gradient(scatt_max.at[index].maximum(values))
    out = mx.exp(values - scatt_max[index])

    scatt_sum = mx.zeros(out_shape, dtype=values.dtype).at[index].add(out)

    eps = 1e-16
    return out / (scatt_sum[index] + eps)


def degree(
    index: mx.array,
    num_nodes: Optional[int] = None,
    edge_weights: Optional[mx.array] = None,
) -> mx.array:
    """Counts the number of ocurrences of each node in the given `index`.

    Args:
        index: Array with node indices, usually src or dst of an `edge_index`.
        num_nodes: Size of the output degree array. If not provided, the number
            of nodes will be inferred from the `index`.
        edge_weights: Optional edge weights that will be leveraged instead
            of 1 values during the degree compute. Default: ``None``

    Returns:
        Array of length `num_nodes` with the degree of each node.
    """
    if index.ndim != 1:
        raise ValueError(
            f"The `degree` function requires a 1D index array, found {index.ndim}."
        )

    num_nodes = num_nodes if num_nodes is not None else index.max().item() + 1
    if edge_weights is None:
        # int32 count: much faster than float atomics on hub nodes, and exact.
        #  Returned as float32, as before.
        return (
            mx.zeros((num_nodes,), dtype=mx.int32).at[index].add(1).astype(mx.float32)
        )
    return scatter(edge_weights, index, num_nodes, "add")


def invert_sqrt_degree(degree: mx.array) -> mx.array:
    """
    Computes the inverted square root of the degree array.

    Args:
        degree: Array of length num_nodes with the inverted square root degree of
            each node.

    Returns:
        Array of length `num_nodes` with the inverted square root of the degree of
        each node with 'inf' values zeroed out.
    """
    invert_sqrt_degree = degree ** (-0.5)
    invert_sqrt_degree = mx.where(
        invert_sqrt_degree == float("inf"), 0, invert_sqrt_degree
    )
    return invert_sqrt_degree
