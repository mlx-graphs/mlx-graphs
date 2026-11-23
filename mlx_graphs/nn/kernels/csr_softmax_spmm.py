"""Row-wise softmax of sparse logits followed by sparse–dense multiplication."""

from functools import lru_cache
from typing import Optional

import mlx.core as mx

from mlx_graphs.utils import scatter

_SIMD_WIDTH = 32
_THREADGROUP = (128, 1, 1)
_INDEX_TYPES = (mx.int32, mx.uint32, mx.int64)

# logits and grad_logits stay in the caller's edge order. The optional
# permutation maps each edge slot in CSR to its slot in those arrays.
_FORWARD = """
    const uint row = thread_position_in_grid.x / WIDTH;
    const uint head = thread_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint heads = values_shape[1];
    const uint num_edges = col_shape[0];
    const long begin = rowptr[row];
    const long end = rowptr[row + 1];
    const ulong out_offset = (ulong(row) * heads + head) * CHANNELS;
    const ulong stats_offset = (ulong(row) * heads + head) * 2;
    float maximum = -INFINITY;
    float inverse_sum = 0;
    constexpr uint tiles = (CHANNELS + WIDTH - 1) / WIDTH;
    float numerator[tiles] = {};

    if (begin >= 0 && end > begin && end <= num_edges) {
        for (long edge = begin + lane; edge < end; edge += WIDTH) {
            const long index = REORDER ? permutation[edge] : edge;
            if (ulong(index) < num_edges) {
                maximum = max(maximum, logits[ulong(index) * heads + head]);
            }
        }
        maximum = simd_max(maximum);
        float sum = 0;
        for (long edge = begin + lane; edge < end; edge += WIDTH) {
            const long index = REORDER ? permutation[edge] : edge;
            if (ulong(index) < num_edges) {
                sum += exp(logits[ulong(index) * heads + head] - maximum);
            }
        }
        sum = simd_sum(sum);
        inverse_sum = sum == 0 ? 0 : 1.0f / sum;
        for (long tile = begin; tile < end; tile += WIDTH) {
            float weight = 0;
            if (tile + lane < end) {
                const long index = REORDER ? permutation[tile + lane] : tile + lane;
                if (ulong(index) < num_edges) {
                    weight = exp(logits[ulong(index) * heads + head] - maximum)
                           * inverse_sum;
                }
            }
            for (ushort neighbor = 0; neighbor < min(long(WIDTH), end - tile);
                 ++neighbor) {
                const long source = col[tile + neighbor];
                const float neighbor_weight = simd_shuffle(weight, neighbor);
                if (ulong(source) < ulong(values_shape[0])) {
                    const ulong value_offset =
                        (ulong(source) * heads + head) * CHANNELS;
                    for (uint i = 0; i < tiles; ++i) {
                        const uint channel = i * WIDTH + lane;
                        if (channel < CHANNELS) {
                            numerator[i] +=
                                neighbor_weight * values[value_offset + channel];
                        }
                    }
                }
            }
        }
    }

    // Save max and inverse sum separately so large absolute logits do not
    // lose their normalization when rounded to a single log-sum-exp value.
    if (lane == 0) {
        stats[stats_offset] = maximum;
        stats[stats_offset + 1] = inverse_sum;
    }
    for (uint i = 0; i < tiles; ++i) {
        const uint channel = i * WIDTH + lane;
        if (channel < CHANNELS) {
            out[out_offset + channel] = numerator[i];
        }
    }
"""

_BACKWARD = """
    const uint row = thread_position_in_grid.x / WIDTH;
    const uint head = thread_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint heads = values_shape[1];
    const uint num_edges = col_shape[0];
    const long begin = rowptr[row];
    const long end = rowptr[row + 1];
    if (begin < 0 || end <= begin || end > num_edges) {
        return;
    }

    const ulong out_offset = (ulong(row) * heads + head) * CHANNELS;
    const ulong stats_offset = (ulong(row) * heads + head) * 2;
    const float maximum = stats[stats_offset];
    const float inverse_sum = stats[stats_offset + 1];
    constexpr uint tiles = (CHANNELS + WIDTH - 1) / WIDTH;
    float grad_out[tiles] = {};
    float baseline = 0;
    for (uint i = 0; i < tiles; ++i) {
        const uint channel = i * WIDTH + lane;
        if (channel < CHANNELS) {
            grad_out[i] = cotangent[out_offset + channel];
            baseline += grad_out[i] * out[out_offset + channel];
        }
    }
    baseline = simd_sum(baseline);

    for (long edge = begin; edge < end; ++edge) {
        const long source = col[edge];
        const long index = REORDER ? permutation[edge] : edge;
        if (ulong(source) >= ulong(values_shape[0]) || ulong(index) >= num_edges) {
            continue;
        }
        const ulong value_offset = (ulong(source) * heads + head) * CHANNELS;
        const ulong logit_offset = ulong(index) * heads + head;
        const float weight = exp(logits[logit_offset] - maximum) * inverse_sum;
        float dot = 0;
        for (uint i = 0; i < tiles; ++i) {
            const uint channel = i * WIDTH + lane;
            if (channel < CHANNELS) {
                const ulong offset = value_offset + channel;
                dot += grad_out[i] * values[offset];
                atomic_fetch_add_explicit(&grad_values[offset], weight * grad_out[i],
                                          memory_order_relaxed);
            }
        }
        dot = simd_sum(dot);
        if (lane == 0) {
            atomic_store_explicit(&grad_logits[logit_offset], weight * (dot - baseline),
                                  memory_order_relaxed);
        }
    }
"""

_INPUTS = ["logits", "values", "rowptr", "col", "permutation"]


@lru_cache(maxsize=1)
def _metal_kernels():
    forward = mx.fast.metal_kernel(
        name="csr_softmax_spmm",
        input_names=_INPUTS,
        output_names=["out", "stats"],
        source=_FORWARD,
    )
    backward = mx.fast.metal_kernel(
        name="csr_softmax_spmm_backward",
        input_names=[*_INPUTS, "out", "stats", "cotangent"],
        output_names=["grad_logits", "grad_values"],
        source=_BACKWARD,
        atomic_outputs=True,
    )
    return forward, backward


def _parameters(values, rowptr, permutation):
    _, heads, channels = values.shape
    rows = rowptr.size - 1
    return {
        "template": [
            ("WIDTH", _SIMD_WIDTH),
            ("CHANNELS", channels),
            ("REORDER", permutation is not None),
        ],
        "grid": (rows * _SIMD_WIDTH, heads, 1),
        "threadgroup": _THREADGROUP,
    }


def _metal_inputs(logits, values, rowptr, col, permutation):
    # MLX supplies a pointer argument for a one-dimensional array. The
    # kernel only reads the permutation when the REORDER template is set.
    perm = mx.zeros((1,), dtype=mx.int32) if permutation is None else permutation
    return [logits, values, rowptr, col, perm]


@mx.custom_function
def _metal(logits, values, rowptr, col, permutation):
    forward, _ = _metal_kernels()
    rows = rowptr.size - 1
    _, heads, channels = values.shape
    return tuple(
        forward(
            inputs=_metal_inputs(logits, values, rowptr, col, permutation),
            output_shapes=[(rows, heads, channels), (rows, heads, 2)],
            output_dtypes=[mx.float32, mx.float32],
            **_parameters(values, rowptr, permutation),
        )
    )


@_metal.vjp
def _metal_vjp(primals, cotangents, outputs):
    _, backward = _metal_kernels()
    logits, values, rowptr, col, permutation = primals
    grad_logits, grad_values = backward(
        inputs=[
            *_metal_inputs(logits, values, rowptr, col, permutation),
            *outputs,
            cotangents[0],
        ],
        output_shapes=[logits.shape, values.shape],
        output_dtypes=[mx.float32, mx.float32],
        init_value=0,
        **_parameters(values, rowptr, permutation),
    )
    return (
        grad_logits,
        grad_values,
        mx.zeros_like(rowptr),
        mx.zeros_like(col),
        None if permutation is None else mx.zeros_like(permutation),
    )


def _reference(rowptr, col, logits, values, permutation):
    rows = rowptr.size - 1
    # An increment at each row boundary also accounts for empty CSR rows.
    boundaries = mx.zeros((col.size + 1,), dtype=mx.int32).at[rowptr[1:]].add(1)
    row = mx.cumsum(boundaries)[: col.size]
    scores = logits if permutation is None else logits[permutation]
    weights = scatter(scores, row, rows, aggr="softmax")
    return scatter(weights[..., None] * values[col], row, rows, aggr="add")


def supports_metal_csr_softmax_spmm(logits, values, rows):
    return (
        logits.size > 0
        and values.shape[-1] > 0
        and logits.dtype == values.dtype == mx.float32
        and 0 < rows <= (2**31 - 1) // _SIMD_WIDTH
        and mx.default_device().type == mx.gpu
        and mx.metal.is_available()
    )


def csr_softmax_spmm(
    rowptr: mx.array,
    col: mx.array,
    logits: mx.array,
    values: mx.array,
    *,
    permutation: Optional[mx.array] = None,
) -> mx.array:
    """Sum source values using softmax weights over each destination's edges.

    Rows of the CSR structure are destinations and columns are source nodes.
    Logits are shaped ``[edges, heads]`` and source values are shaped
    ``[sources, heads, channels]``. An optional permutation maps each CSR
    edge to its index in the logits. Empty rows produce zero.

    The float32 Metal path keeps only row normalization statistics for its
    custom first-order reverse-mode derivative. Other floating dtypes and
    CPU execution use MLX operations.
    """
    if rowptr.ndim != 1 or rowptr.size == 0 or col.ndim != 1:
        raise ValueError("Expected nonempty 1D row pointers and 1D column indices")
    if values.ndim != 3 or logits.shape != (col.size, values.shape[1]):
        raise ValueError(
            "Expected logits [edges, heads] and values [sources, heads, channels]"
        )
    if permutation is not None and permutation.shape != col.shape:
        raise ValueError("Expected one permutation entry per CSR edge")
    if (
        rowptr.dtype not in _INDEX_TYPES
        or col.dtype not in _INDEX_TYPES
        or (permutation is not None and permutation.dtype not in _INDEX_TYPES)
    ):
        raise ValueError("CSR arrays must use int32, uint32, or int64 indices")
    rows = rowptr.size - 1
    if col.size == 0:
        return mx.zeros((rows, *values.shape[1:]), dtype=mx.result_type(logits, values))
    if supports_metal_csr_softmax_spmm(logits, values, rows):
        return _metal(logits, values, rowptr, col, permutation)[0]
    return _reference(rowptr, col, logits, values, permutation)
