"""A fused per-edge dot product: the worked example of the Custom GPU kernels guide.

    python docs/snippets/edge_dot_kernel.py

For every edge ``(src, dst)``, ``score[e] = <q[dst], k[src]>``, the attention
logit of dot-product attention layers such as ``TransformerConv``. The Metal
kernels compute it, and its gradients, without building the two
``[num_edges, channels]`` arrays that the MLX reference gathers.

Running the script checks the kernel against the reference (values and
gradients, on random graphs, an empty graph and isolated nodes), runs a layer
that uses it, and benchmarks both. The guide includes its sections between
the ``--8<--`` markers, so the page shows the code that was tested.
"""

import math
import time
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.nn import Linear, MessagePassing
from mlx_graphs.utils import softmax


# --8<-- [start:reference]
def edge_dot_reference(q: mx.array, k: mx.array, edge_index: mx.array) -> mx.array:
    """score[e] = <q[dst[e]], k[src[e]]>, with MLX operations."""
    src, dst = edge_index[0], edge_index[1]
    return (q[dst] * k[src]).sum(axis=-1)  # two [num_edges, channels] gathers


# --8<-- [end:reference]


# --8<-- [start:kernels]
SIMD_WIDTH = 32  # threads of an Apple GPU SIMD group
THREADGROUP = (256, 1, 1)

FORWARD = """
    // One SIMD group per edge: each lane sums a strided slice of the channels.
    const uint edge = thread_position_in_grid.x / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint num_edges = edge_index_shape[1];
    const uint channels = q_shape[1];
    const ulong src = edge_index[edge];
    const ulong dst = edge_index[num_edges + edge];
    float acc = 0.0f;
    for (uint c = lane; c < channels; c += 32) {
        acc += q[dst * channels + c] * k[src * channels + c];
    }
    acc = simd_sum(acc);  // reduce the 32 partial sums
    if (lane == 0) {
        score[edge] = acc;
    }
"""

BACKWARD = """
    // d score / d q[dst] = k[src] and d score / d k[src] = q[dst], scaled by
    // the incoming gradient. Several edges share a node: add atomically.
    const uint edge = thread_position_in_grid.x / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint num_edges = edge_index_shape[1];
    const uint channels = q_shape[1];
    const ulong src = edge_index[edge];
    const ulong dst = edge_index[num_edges + edge];
    const float g = cotangent[edge];
    for (uint c = lane; c < channels; c += 32) {
        atomic_fetch_add_explicit(
            &grad_q[dst * channels + c], g * k[src * channels + c],
            memory_order_relaxed);
        atomic_fetch_add_explicit(
            &grad_k[src * channels + c], g * q[dst * channels + c],
            memory_order_relaxed);
    }
"""


@lru_cache(maxsize=1)  # build (and compile) each kernel once
def _kernels():
    forward = mx.fast.metal_kernel(
        name="edge_dot",
        input_names=["q", "k", "edge_index"],
        output_names=["score"],
        source=FORWARD,
        ensure_row_contiguous=True,
    )
    backward = mx.fast.metal_kernel(
        name="edge_dot_backward",
        input_names=["q", "k", "edge_index", "cotangent"],
        output_names=["grad_q", "grad_k"],
        source=BACKWARD,
        ensure_row_contiguous=True,
        atomic_outputs=True,  # outputs are `device atomic<float>*`
    )
    return forward, backward


# --8<-- [end:kernels]


# --8<-- [start:custom_function]
@mx.custom_function
def _edge_dot_metal(q, k, edge_index):
    forward, _ = _kernels()
    num_edges = edge_index.shape[1]
    return forward(
        inputs=[q, k, edge_index],
        grid=(num_edges * SIMD_WIDTH, 1, 1),  # total threads, not threadgroups
        threadgroup=THREADGROUP,
        output_shapes=[(num_edges,)],
        output_dtypes=[mx.float32],
    )[0]


@_edge_dot_metal.vjp
def _edge_dot_metal_vjp(primals, cotangent, output):
    q, k, edge_index = primals
    _, backward = _kernels()
    num_edges = edge_index.shape[1]
    grad_q, grad_k = backward(
        inputs=[q, k, edge_index, cotangent],
        grid=(num_edges * SIMD_WIDTH, 1, 1),
        threadgroup=THREADGROUP,
        output_shapes=[q.shape, k.shape],
        output_dtypes=[mx.float32, mx.float32],
        init_value=0,  # the atomic adds accumulate into zeros
    )
    # No gradient for the integer edge index.
    return grad_q, grad_k, mx.zeros_like(edge_index)


# --8<-- [end:custom_function]


# --8<-- [start:dispatch]
def edge_dot(q: mx.array, k: mx.array, edge_index: mx.array) -> mx.array:
    """The Metal kernel when it applies, the MLX reference otherwise."""
    if (
        edge_index.ndim == 2
        and 0 < edge_index.shape[1] <= (2**31 - 1) // SIMD_WIDTH
        and edge_index.dtype == mx.int32
        and q.dtype == k.dtype == mx.float32
        and q.ndim == k.ndim == 2
        and q.shape[1] == k.shape[1]
        and mx.default_device().type == mx.gpu
        and mx.metal.is_available()
    ):
        return _edge_dot_metal(q, k, edge_index)
    return edge_dot_reference(q, k, edge_index)


# --8<-- [end:dispatch]


# --8<-- [start:layer]
class DotAttentionConv(MessagePassing):
    """Single-head dot-product graph attention, scored with `edge_dot`."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="add")
        self.query = Linear(in_dim, out_dim)
        self.key = Linear(in_dim, out_dim)
        self.value = Linear(in_dim, out_dim)
        self.scale = 1 / math.sqrt(out_dim)

    def __call__(self, edge_index: mx.array, node_features: mx.array) -> mx.array:
        q = self.query(node_features)
        k = self.key(node_features)
        v = self.value(node_features)
        scores = edge_dot(q, k, edge_index) * self.scale
        # softmax over the incoming edges of each destination node
        weights = softmax(scores, edge_index[1], num_nodes=node_features.shape[0])
        # the default message is edge_weights * source features
        return self.propagate(
            edge_index, v, message_kwargs={"edge_weights": weights[:, None]}
        )


# --8<-- [end:layer]


# --8<-- [start:test]
def random_graph(num_nodes, num_edges, channels, seed=0):
    mx.random.seed(seed)
    edge_index = mx.random.randint(0, num_nodes, (2, num_edges)).astype(mx.int32)
    q = mx.random.normal((num_nodes, channels))
    k = mx.random.normal((num_nodes, channels))
    return q, k, edge_index


def check(q, k, edge_index, atol=1e-4, rtol=1e-4):
    """Values and gradients of the kernel against the reference."""
    weights = mx.random.normal((edge_index.shape[1],))  # a non-trivial cotangent

    def loss(fn):
        return lambda q, k: (fn(q, k, edge_index) * weights).sum()

    value, grads = mx.value_and_grad(loss(edge_dot), argnums=(0, 1))(q, k)
    ref_value, ref_grads = mx.value_and_grad(loss(edge_dot_reference), argnums=(0, 1))(
        q, k
    )
    assert mx.allclose(
        edge_dot(q, k, edge_index),
        edge_dot_reference(q, k, edge_index),
        atol=atol,
        rtol=rtol,
    )
    assert mx.allclose(value, ref_value, atol=atol, rtol=rtol)
    for grad, ref_grad in zip(grads, ref_grads):
        assert mx.allclose(grad, ref_grad, atol=atol, rtol=rtol)


def run_tests():
    for num_nodes, num_edges, channels in [
        (50, 300, 16),
        (1000, 20000, 64),
        (7, 5, 100),
    ]:
        check(*random_graph(num_nodes, num_edges, channels))
    # isolated nodes: edges only between the first 10 of 100 nodes
    q, k, _ = random_graph(100, 1, 32)
    edge_index = mx.random.randint(0, 10, (2, 500)).astype(mx.int32)
    check(q, k, edge_index)
    # an empty graph falls back to the reference, which returns no scores
    q, k, _ = random_graph(10, 1, 8)
    empty = mx.zeros((2, 0), dtype=mx.int32)
    assert edge_dot(q, k, empty).shape == (0,)
    # the layer runs and is differentiable
    layer = DotAttentionConv(16, 32)
    x = mx.random.normal((50, 16))
    edge_index = mx.random.randint(0, 50, (2, 400)).astype(mx.int32)
    out = layer(edge_index, x)
    grads = nn.value_and_grad(layer, lambda m: m(edge_index, x).sum())(layer)[1]
    mx.eval(out, grads)
    assert out.shape == (50, 32)
    print("kernel matches the reference: values and gradients (atol=rtol=1e-4)")


# --8<-- [end:test]


# --8<-- [start:benchmark]
def timeit(fn, *args, warmup=5, iters=20):
    for _ in range(warmup):  # compiles the kernel, warms the caches
        mx.eval(fn(*args))
    start = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(*args))  # MLX is lazy: evaluate to time the work
    return (time.perf_counter() - start) / iters * 1e3  # ms


def benchmark(num_nodes=100_000, num_edges=2_000_000, channels=64):
    q, k, edge_index = random_graph(num_nodes, num_edges, channels)
    mx.eval(q, k, edge_index)

    def forward_backward(fn):
        grad = mx.grad(lambda q, k: fn(q, k, edge_index).sum(), argnums=(0, 1))
        return lambda q, k: grad(q, k)

    print(f"{num_nodes:,} nodes, {num_edges:,} edges, {channels} channels")
    for name, fn in [
        ("forward", lambda f: lambda q, k: f(q, k, edge_index)),
        ("forward + backward", forward_backward),
    ]:
        kernel = timeit(fn(edge_dot), q, k)
        reference = timeit(fn(edge_dot_reference), q, k)
        print(
            f"  {name:<20} kernel {kernel:6.2f} ms  reference {reference:6.2f} ms"
            f"  speedup {reference / kernel:4.1f}x"
        )


# --8<-- [end:benchmark]


if __name__ == "__main__":
    gpu = mx.device_info().get("device_name", "") if mx.metal.is_available() else ""
    print(f"MLX {mx.__version__}, Metal available: {mx.metal.is_available()}, {gpu}")
    run_tests()
    benchmark()
    benchmark(num_nodes=10_000, num_edges=100_000, channels=64)
