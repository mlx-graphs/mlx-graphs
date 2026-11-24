"""TransformerConv attention scores and their Metal forward and backward kernels.

The score of an edge ``e`` and head ``h`` is the scaled dot product of the
query of its destination node and the key of its source node::

    scores[e, h] = scale * <a[row[e], h, :], b[col[e], h, :]>

The reference, ``scale * (a[row] * b[col]).sum(-1)``, gathers two
``[num_edges, heads, channels]`` arrays forward and scatters two more in the
backward pass. The kernels below read the node features directly instead, so
neither pass allocates anything per edge and channel.
"""

from functools import lru_cache

import mlx.core as mx

# One Apple GPU SIMD group handles the channels of an edge and head, as in the
# GATv2 kernels.
_SIMD_WIDTH = 32
_THREADGROUP = (256, 1, 1)

_HEADER = """
inline long node_index(long index, int num_nodes) {
    return index < 0 ? index + num_nodes : index;
}
"""

_FORWARD = """
    const uint edge = thread_position_in_grid.x / SIMD_WIDTH;
    const uint head = thread_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint heads = a_shape[1];
    const long r = node_index(row[edge], a_shape[0]);
    const long c = node_index(col[edge], b_shape[0]);

    float score = 0.0f;
    if (ulong(r) < ulong(a_shape[0]) && ulong(c) < ulong(b_shape[0])) {
        const ulong a_offset = (ulong(r) * heads + head) * CHANNELS;
        const ulong b_offset = (ulong(c) * heads + head) * CHANNELS;
        for (uint channel = lane; channel < CHANNELS; channel += SIMD_WIDTH) {
            score += a[a_offset + channel] * b[b_offset + channel];
        }
    }
    score = simd_sum(score);
    if (lane == 0) {
        scores[ulong(edge) * heads + head] = scale * score;
    }
"""

# d a[row[e], h] += scale * g[e, h] * b[col[e], h], and symmetrically for b.
# All heads run in one launch; contributions to a node meet with atomics.
_BACKWARD = """
    const uint edge = thread_position_in_grid.x / SIMD_WIDTH;
    const uint head = thread_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint heads = a_shape[1];
    const long r = node_index(row[edge], a_shape[0]);
    const long c = node_index(col[edge], b_shape[0]);
    if (ulong(r) >= ulong(a_shape[0]) || ulong(c) >= ulong(b_shape[0])) {
        return;
    }
    const float g = scale * cotangent[ulong(edge) * heads + head];
    const ulong a_offset = (ulong(r) * heads + head) * CHANNELS;
    const ulong b_offset = (ulong(c) * heads + head) * CHANNELS;
    for (uint channel = lane; channel < CHANNELS; channel += SIMD_WIDTH) {
        const float a_value = a[a_offset + channel];
        const float b_value = b[b_offset + channel];
        atomic_fetch_add_explicit(
            &grad_a[a_offset + channel], g * b_value, memory_order_relaxed);
        atomic_fetch_add_explicit(
            &grad_b[b_offset + channel], g * a_value, memory_order_relaxed);
    }
"""


def _reference_scores(row, col, a, b, scale=1.0):
    """``scale * (a[row] * b[col]).sum(-1)``, with standard MLX operations."""
    return scale * (a[row] * b[col]).sum(axis=-1)


def _use_reference_jvp(function, reference):
    """Use standard MLX operations for forward-mode differentiation."""

    @function.jvp
    def jvp(primals, tangents):
        tangents = [
            mx.zeros_like(primal) if tangent is None else tangent
            for primal, tangent in zip(primals, tangents)
        ]
        return mx.jvp(reference, primals, tangents)[1]


@lru_cache(maxsize=16)
def _metal_scores(scale: float):
    inputs = ["row", "col", "a", "b", "scale"]
    forward = mx.fast.metal_kernel(
        name="transformer_edge_scores",
        input_names=inputs,
        output_names=["scores"],
        source=_FORWARD,
        header=_HEADER,
        ensure_row_contiguous=True,
    )
    backward = mx.fast.metal_kernel(
        name="transformer_edge_scores_backward",
        input_names=[*inputs, "cotangent"],
        output_names=["grad_a", "grad_b"],
        source=_BACKWARD,
        header=_HEADER,
        ensure_row_contiguous=True,
        atomic_outputs=True,
    )

    def reference(row, col, a, b):
        return _reference_scores(
            mx.stop_gradient(row), mx.stop_gradient(col), a, b, scale
        )

    def reference_backward(row, col, a, b, cotangent):
        def scores(a, b):
            return reference(row, col, a, b)

        _, grads = mx.vjp(scores, (a, b), (cotangent,))
        return tuple(grads)

    def launch_shape(row, a):
        _, heads, channels = a.shape
        return (row.shape[0] * _SIMD_WIDTH, heads, 1), heads, channels

    @mx.custom_function
    def score(row, col, a, b):
        grid, heads, channels = launch_shape(row, a)
        return forward(
            # The enclosing custom function provides differentiation rules.
            inputs=[
                mx.stop_gradient(row),
                mx.stop_gradient(col),
                mx.stop_gradient(a),
                mx.stop_gradient(b),
                mx.array(scale, mx.float32),
            ],
            template=[("SIMD_WIDTH", _SIMD_WIDTH), ("CHANNELS", channels)],
            grid=grid,
            threadgroup=_THREADGROUP,
            output_shapes=[(row.shape[0], heads)],
            output_dtypes=[mx.float32],
        )[0]

    @mx.custom_function
    def gradient(row, col, a, b, cotangent):
        grid, _, channels = launch_shape(row, a)
        return tuple(
            backward(
                inputs=[
                    mx.stop_gradient(row),
                    mx.stop_gradient(col),
                    mx.stop_gradient(a),
                    mx.stop_gradient(b),
                    mx.array(scale, mx.float32),
                    mx.stop_gradient(cotangent),
                ],
                template=[("SIMD_WIDTH", _SIMD_WIDTH), ("CHANNELS", channels)],
                grid=grid,
                threadgroup=_THREADGROUP,
                output_shapes=[a.shape, b.shape],
                output_dtypes=[mx.float32] * 2,
                init_value=0,
            )
        )

    @score.vjp
    def score_vjp(primals, cotangent, _):
        row, col = primals[:2]
        grad_a, grad_b = gradient(*primals, cotangent)
        return mx.zeros_like(row), mx.zeros_like(col), grad_a, grad_b

    @gradient.vjp
    def gradient_vjp(primals, cotangents, _):
        _, grads = mx.vjp(reference_backward, primals, cotangents)
        return tuple(grads)

    _use_reference_jvp(score, reference)
    _use_reference_jvp(gradient, reference_backward)
    return score


def _can_use_metal(row, col, a, b) -> bool:
    return (
        isinstance(row, mx.array)
        and isinstance(col, mx.array)
        and row.ndim == col.ndim == 1
        and row.shape == col.shape
        # The kernel launch takes signed 32-bit grid dimensions.
        and 0 < row.shape[0] <= (2**31 - 1) // _SIMD_WIDTH
        and a.ndim == b.ndim == 3
        and a.shape[1:] == b.shape[1:]
        and min(a.shape[1:]) > 0
        and min(a.shape[0], b.shape[0]) > 0
        and row.dtype in (mx.int32, mx.int64, mx.uint32)
        and col.dtype == row.dtype
        and a.dtype == b.dtype == mx.float32
        and mx.default_device().type == mx.gpu
        and mx.metal.is_available()
    )


def transformer_edge_scores(
    row: mx.array, col: mx.array, a: mx.array, b: mx.array, scale: float = 1.0
) -> mx.array:
    """Dot product per edge and per head: ``scale * (a[row] * b[col]).sum(-1)``.

    On the Metal GPU, with float32 features, this runs a kernel that never
    builds the gathered ``a[row]`` and ``b[col]`` arrays
    (``[num_edges, heads, channels]`` each), forward or backward. Otherwise it
    computes the expression above.

    Args:
        row: ``[num_edges]`` index into ``a`` (e.g. destination nodes).
        col: ``[num_edges]`` index into ``b`` (e.g. source nodes).
        a: ``[num_a, heads, channels]`` (or ``[num_a, channels]``).
        b: ``[num_b, heads, channels]`` (or ``[num_b, channels]``).
        scale: factor applied to every score.

    Returns:
        ``[num_edges, heads]`` (or ``[num_edges]``) scores.
    """
    single_head = a.ndim == b.ndim == 2
    a3, b3 = (a[:, None, :], b[:, None, :]) if single_head else (a, b)
    if isinstance(scale, (int, float)) and _can_use_metal(row, col, a3, b3):
        out = _metal_scores(float(scale))(row, col, a3, b3)
        return out[:, 0] if single_head else out
    return _reference_scores(row, col, a, b, scale)
