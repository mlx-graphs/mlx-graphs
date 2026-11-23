"""GATv2 attention scores and their Metal forward and backward kernels."""

from functools import lru_cache
from math import isfinite
from typing import Optional

import mlx.core as mx

# One Apple GPU SIMD group handles the channels of an edge and head. In the
# backward pass each threadgroup reduces its attention-vector contributions
# locally before writing a partial sum.
_SIMD_WIDTH = 32
_THREADGROUP_SIZE = 256
_EDGES_PER_GROUP = 32
_THREADGROUP = (_THREADGROUP_SIZE, 1, 1)

_HEADER = """
inline long node_index(long index, int num_nodes) {
    return index < 0 ? index + num_nodes : index;
}
"""

_FORWARD = """
    const uint edge = thread_position_in_grid.x / SIMD_WIDTH;
    const uint head = thread_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint num_edges = edge_index_shape[1];
    const uint heads = src_feats_shape[1];
    const long src = node_index(edge_index[edge], src_feats_shape[0]);
    const long dst = node_index(edge_index[num_edges + edge], dst_feats_shape[0]);

    float score = 0.0f;
    if (ulong(src) < ulong(src_feats_shape[0]) &&
        ulong(dst) < ulong(dst_feats_shape[0])) {
        for (uint channel = lane; channel < CHANNELS; channel += SIMD_WIDTH) {
            const uint feature = head * CHANNELS + channel;
            const float z = src_feats[src * heads * CHANNELS + feature]
                          + dst_feats[dst * heads * CHANNELS + feature];
            score += att[feature] * (z < 0.0f ? slope * z : z);
        }
    }
    score = simd_sum(score);
    if (lane == 0) {
        scores[ulong(edge) * heads + head] = score;
    }
"""

_BACKWARD = """
    const uint lane = thread_index_in_simdgroup;
    const uint thread_id = thread_index_in_threadgroup;
    const uint simd = thread_id / SIMD_WIDTH;
    const uint simds = THREADGROUP_SIZE / SIMD_WIDTH;
    const uint edges_per_simd = EDGES_PER_GROUP / simds;
    const uint block = threadgroup_position_in_grid.x;
    const uint head = threadgroup_position_in_grid.y;
    const uint heads = src_feats_shape[1];
    const uint num_edges = edge_index_shape[1];
    threadgroup float partials[THREADGROUP_SIZE];

    long src[edges_per_simd];
    long dst[edges_per_simd];
    float grad[edges_per_simd];
    bool valid[edges_per_simd];
    for (uint i = 0; i < edges_per_simd; ++i) {
        const uint edge = block * EDGES_PER_GROUP + i * simds + simd;
        valid[i] = false;
        if (edge < num_edges) {
            src[i] = node_index(edge_index[edge], src_feats_shape[0]);
            dst[i] = node_index(edge_index[num_edges + edge], dst_feats_shape[0]);
            grad[i] = cotangent[ulong(edge) * heads + head];
            valid[i] = ulong(src[i]) < ulong(src_feats_shape[0]) &&
                       ulong(dst[i]) < ulong(dst_feats_shape[0]);
        }
    }

    for (uint tile = 0; tile < CHANNELS; tile += SIMD_WIDTH) {
        const uint channel = tile + lane;
        const uint feature = head * CHANNELS + channel;
        float att_grad = 0.0f;
        if (channel < CHANNELS) {
            for (uint i = 0; i < edges_per_simd; ++i) {
                if (!valid[i]) {
                    continue;
                }
                const long s = src[i] * heads * CHANNELS + feature;
                const long d = dst[i] * heads * CHANNELS + feature;
                const float z = src_feats[s] + dst_feats[d];
                // At z == 0 only the reference's linear term contributes.
                const float dz = (grad[i] * att[feature]) * (z > 0.0f ? 1.0f : slope);
                atomic_fetch_add_explicit(&grad_src[s], dz, memory_order_relaxed);
                atomic_fetch_add_explicit(&grad_dst[d], dz, memory_order_relaxed);
                att_grad += grad[i] * (z < 0.0f ? slope * z : z);
            }
        }
        partials[thread_id] = att_grad;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0 && channel < CHANNELS) {
            for (uint i = 1; i < simds; ++i) {
                att_grad += partials[i * SIMD_WIDTH + lane];
            }
            const ulong i = ulong(block) * heads * CHANNELS + feature;
            atomic_store_explicit(
                &grad_att_partials[i], att_grad, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


def _reference_scores(src_feats, dst_feats, att, edge_index, slope, edge_feats=None):
    src_idx, dst_idx = edge_index
    # Factor the linear term into per-node dot products to keep the reference
    # and its backward pass efficient on devices without the Metal kernel.
    z = src_feats[src_idx] + dst_feats[dst_idx]
    linear = (src_feats * att).sum(-1)[src_idx] + (dst_feats * att).sum(-1)[dst_idx]
    if edge_feats is not None:
        z = z + edge_feats
        linear = linear + (edge_feats * att).sum(-1)
    return slope * linear + (1 - slope) * (mx.maximum(z, 0) * att).sum(-1)


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
def _metal_scores(slope: float):
    inputs = ["src_feats", "dst_feats", "att", "edge_index", "slope"]
    forward = mx.fast.metal_kernel(
        name="gatv2_edge_scores",
        input_names=inputs,
        output_names=["scores"],
        source=_FORWARD,
        header=_HEADER,
        ensure_row_contiguous=True,
    )
    backward = mx.fast.metal_kernel(
        name="gatv2_edge_scores_backward",
        input_names=[*inputs, "cotangent"],
        output_names=["grad_src", "grad_dst", "grad_att_partials"],
        source=_BACKWARD,
        header=_HEADER,
        ensure_row_contiguous=True,
        atomic_outputs=True,
    )

    def reference(src_feats, dst_feats, att, edge_index):
        return _reference_scores(
            src_feats, dst_feats, att, mx.stop_gradient(edge_index), slope
        )

    def reference_backward(src_feats, dst_feats, att, edge_index, cotangent):
        def scores(src, dst, weights):
            return reference(src, dst, weights, edge_index)

        _, grads = mx.vjp(scores, (src_feats, dst_feats, att), (cotangent,))
        return tuple(grads)

    @mx.custom_function
    def score(src_feats, dst_feats, att, edge_index):
        num_edges = edge_index.shape[1]
        _, heads, channels = src_feats.shape
        return forward(
            # The enclosing custom function provides differentiation rules.
            inputs=[
                mx.stop_gradient(src_feats),
                mx.stop_gradient(dst_feats),
                mx.stop_gradient(att),
                mx.stop_gradient(edge_index),
                mx.array(slope, mx.float32),
            ],
            template=[("SIMD_WIDTH", _SIMD_WIDTH), ("CHANNELS", channels)],
            grid=(num_edges * _SIMD_WIDTH, heads, 1),
            threadgroup=_THREADGROUP,
            output_shapes=[(num_edges, heads)],
            output_dtypes=[mx.float32],
        )[0]

    @mx.custom_function
    def gradient(src_feats, dst_feats, att, edge_index, cotangent):
        _, heads, channels = src_feats.shape
        blocks = (edge_index.shape[1] + _EDGES_PER_GROUP - 1) // _EDGES_PER_GROUP
        grad_src, grad_dst, grad_att = backward(
            inputs=[
                mx.stop_gradient(src_feats),
                mx.stop_gradient(dst_feats),
                mx.stop_gradient(att),
                mx.stop_gradient(edge_index),
                mx.array(slope, mx.float32),
                mx.stop_gradient(cotangent),
            ],
            template=[
                ("SIMD_WIDTH", _SIMD_WIDTH),
                ("THREADGROUP_SIZE", _THREADGROUP_SIZE),
                ("EDGES_PER_GROUP", _EDGES_PER_GROUP),
                ("CHANNELS", channels),
            ],
            grid=(blocks * _THREADGROUP_SIZE, heads, 1),
            threadgroup=_THREADGROUP,
            output_shapes=[src_feats.shape, dst_feats.shape, (blocks, heads, channels)],
            output_dtypes=[mx.float32] * 3,
            init_value=0,
        )
        return grad_src, grad_dst, grad_att.sum(axis=0, keepdims=True)

    @score.vjp
    def score_vjp(primals, cotangent, _):
        return (*gradient(*primals, cotangent), mx.zeros_like(primals[-1]))

    @gradient.vjp
    def gradient_vjp(primals, cotangents, _):
        _, grads = mx.vjp(reference_backward, primals, cotangents)
        return tuple(grads)

    _use_reference_jvp(score, reference)
    _use_reference_jvp(gradient, reference_backward)
    return score


def gatv2_edge_scores(
    src_feats: mx.array,
    dst_feats: mx.array,
    att: mx.array,
    edge_index: mx.array,
    slope: float,
    edge_feats: Optional[mx.array] = None,
) -> mx.array:
    """Compute GATv2's dynamic attention logits for each edge and head."""
    if (
        edge_feats is None
        and isinstance(edge_index, mx.array)
        and edge_index.ndim == 2
        and edge_index.shape[0] == 2
        # The kernel launch takes signed 32-bit grid dimensions.
        and 0 < edge_index.shape[1] <= (2**31 - 1) // _SIMD_WIDTH
        and src_feats.ndim == dst_feats.ndim == 3
        and src_feats.shape[1:] == dst_feats.shape[1:]
        and min(src_feats.shape[1:]) > 0
        and att.shape == (1, *src_feats.shape[1:])
        and edge_index.dtype in (mx.int32, mx.int64, mx.uint32)
        and src_feats.dtype == dst_feats.dtype == att.dtype == mx.float32
        and isinstance(slope, (int, float))
        and isfinite(slope)
        and mx.default_device().type == mx.gpu
        and mx.metal.is_available()
    ):
        return _metal_scores(float(slope))(src_feats, dst_feats, att, edge_index)
    return _reference_scores(src_feats, dst_feats, att, edge_index, slope, edge_feats)
