# Custom GPU kernels

This guide is for you if you wrote a layer and one of its operations on edges is slow. It shows how to write that operation as a Metal kernel with [`mx.fast.metal_kernel`](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), give it gradients with [`mx.custom_function`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html), and plug it into a [`MessagePassing`][mlx_graphs.nn.MessagePassing] layer with a fallback. The worked example is a real fused graph operation; all the code on this page comes from [`docs/snippets/edge_dot_kernel.py`](https://github.com/mlx-graphs/mlx-graphs/blob/main/docs/snippets/edge_dot_kernel.py), which runs its checks and its benchmark with `python docs/snippets/edge_dot_kernel.py`. The test suite runs the same checks (`tests/docs/test_edge_dot_kernel.py`).

The API used here is the one of MLX 0.32. mlx-graphs ships two such kernels, which are longer versions of the same recipe: [`gatv2_edge_scores.py`](https://github.com/mlx-graphs/mlx-graphs/blob/main/mlx_graphs/nn/kernels/gatv2_edge_scores.py) (GATv2's attention logits) and [`csr_softmax_spmm.py`](https://github.com/mlx-graphs/mlx-graphs/blob/main/mlx_graphs/nn/kernels/csr_softmax_spmm.py) (softmax over each node's edges fused with the weighted sum).

## When a custom kernel is worth it

A message passing layer written with MLX operations gathers the features of both ends of every edge, computes one message per edge, then scatters the messages to their destinations:

- the gathers build arrays of shape `[num_edges, features]`, written to and read back from memory, often several of them; on a graph with 2 million edges and 64 features, each one is 512 MB in float32;
- the scatter adds the messages of all edges into the rows of their destinations, with atomic additions on the GPU, which collide on high-degree nodes;
- the backward pass does the same in reverse.

Two cases are already covered without a kernel of your own:

- **The message is a weighted sum of neighbor features** (`edge_weight * x_j`, mean, sum): decorate the layer with [`@fused_message_passing`][mlx_graphs.nn.fused_message_passing], which aggregates with one sparse-dense matrix product and never builds the message array. See [Writing your own layers](writing-layers.md#speed).
- **The attention of GAT, GATv2 and TransformerConv**: the built-in layers already use the two kernels above. GATv2 switches to `mlx_cluster.gatv2_scores` on graphs with at least 1M edges (see [GATv2 attention scores](../../api/nn/index.md#gatv2-attention-scores)).

Write a kernel when the expensive part is something else, typically **a per-edge quantity computed from the features of both ends** (a score, a distance, a gate) that the layer needs before its weighted sum. The kernel computes it directly from the node features, so the `[num_edges, features]` intermediates never exist.

## The example: a per-edge dot product

Dot-product attention (TransformerConv, graph transformers) scores every edge `(src, dst)` with

$$
\text{score}_e = \langle q_{\text{dst}(e)},\; k_{\text{src}(e)} \rangle .
$$

The reference implementation is two gathers and a reduction:

```python
--8<-- "edge_dot_kernel.py:reference"
```

It is correct on any device and differentiable by MLX, so it serves as the fallback and as the ground truth for the tests.

## The Metal kernels

`mx.fast.metal_kernel` takes the *body* of a Metal function; MLX generates the signature from `input_names` and `output_names`, and gives each input `x` a `x_shape` array (and `x_strides`, `x_ndim` when the body uses them). Inside the body, `thread_position_in_grid`, `thread_index_in_simdgroup` and the other Metal attributes are available.

The forward kernel gives each edge one SIMD group of 32 threads: each thread multiplies a strided slice of the channels, and `simd_sum` adds the 32 partial sums. The backward kernel computes `grad_q[dst] += g * k[src]` and `grad_k[src] += g * q[dst]` for every edge. Several edges share a node, so it adds with `atomic_fetch_add_explicit`, which requires `atomic_outputs=True`:

```python
--8<-- "edge_dot_kernel.py:kernels"
```

`metal_kernel` returns a callable. Every time you make one, MLX creates a new Metal library, which it may compile: build each kernel once (`lru_cache` here) and call it many times.

## Gradients with `mx.custom_function`

MLX cannot differentiate through a Metal kernel. Wrap the forward call in `mx.custom_function` and define its vector-Jacobian product with the backward kernel. The VJP receives the inputs (`primals`), the cotangent of the output and the output itself, and returns one gradient per input:

```python
--8<-- "edge_dot_kernel.py:custom_function"
```

Things to know about the kernel call:

- `grid` is the **total number of threads** along each axis (Metal's `dispatchThreads`), not the number of threadgroups: here `num_edges * 32`.
- `output_shapes` and `output_dtypes` are required; outputs are not initialized unless you pass `init_value`, which the backward kernel needs because it accumulates.
- `template` passes compile-time constants (`[("CHANNELS", 64)]`); each new value compiles a new specialization. The example reads the sizes from `q_shape` instead.

The gradient of the backward kernel itself (second-order gradients) is not defined here; the repository's GATv2 kernel shows how to delegate it to the reference with `mx.vjp`.

## Plugging it into a layer

Dispatch to the kernel only when it applies, and use the reference otherwise: on the CPU device, without Metal, for empty graphs, other dtypes or sizes the grid cannot hold:

```python
--8<-- "edge_dot_kernel.py:dispatch"
```

A layer then calls `edge_dot` like any MLX operation, normalizes the scores over each node's incoming edges with [`softmax`][mlx_graphs.utils.softmax], and lets `propagate` do the weighted sum:

```python
--8<-- "edge_dot_kernel.py:layer"
```

## Testing

Compare values **and** gradients with the reference, with a random cotangent (a plain `.sum()` would hide errors that cancel out), on random graphs of a few sizes, on nodes without edges and on an empty graph:

```python
--8<-- "edge_dot_kernel.py:test"
```

The tolerance is `1e-4`: the kernel adds the channels in a different order than MLX, and the atomic additions of the backward kernel run in an order that changes from run to run, so results differ in the last bits and are not bitwise deterministic.

## Benchmarking

MLX evaluates lazily: time `mx.eval(...)`, after a few warm-up calls that compile the kernel:

```python
--8<-- "edge_dot_kernel.py:benchmark"
```

On an Apple M5 Pro (MLX 0.32.4), `python docs/snippets/edge_dot_kernel.py` prints:

```text
MLX 0.32.4, Metal available: True, Apple M5 Pro
kernel matches the reference: values and gradients (atol=rtol=1e-4)
100,000 nodes, 2,000,000 edges, 64 channels
  forward              kernel   2.72 ms  reference  15.46 ms  speedup  5.7x
  forward + backward   kernel  10.39 ms  reference  24.79 ms  speedup  2.4x
10,000 nodes, 100,000 edges, 64 channels
  forward              kernel   0.26 ms  reference   0.86 ms  speedup  3.3x
  forward + backward   kernel   0.41 ms  reference   1.20 ms  speedup  2.9x
```

Times vary between runs: over our runs on this machine, the forward speedup on 2 million edges was 5.5x to 5.7x and the forward + backward one 2.4x to 2.8x.

How to read it:

- The forward pass gains the most: the reference writes and reads two `[2M, 64]` arrays (1 GB), the kernel only reads the node features.
- With the backward pass the gain is smaller: both versions scatter-add into the node gradients, and the kernel's atomic additions collide on the same rows as the reference's.
- On small graphs, launch and Python overhead dominate both versions; measure at the sizes you train on.

## Pitfalls

- **Index dtypes**: the kernel reads `int32` indices. Check the dtype in the dispatch (or cast once outside the training loop); a Metal kernel reinterprets the bytes it is given, so `int64` indices would be read as garbage. Compute offsets in `ulong` (`src * channels` overflows 32 bits past 2^31 elements).
- **Sorted or CSR layouts**: kernels that walk the edges of each node (like `csr_softmax_spmm`) need the edges sorted by destination with a row pointer; convert once and cache it, as the fused message passing does.
- **Threadgroup sizes**: a threadgroup holds at most 1024 threads; keep it a multiple of the SIMD width (32), and make sure every lane of a SIMD group reaches `simd_sum` (no early `return` for part of a group).
- **Atomics**: only for 32-bit types; they make floating-point sums non-deterministic. Reduce in registers or threadgroup memory first when many threads write the same address (the GATv2 kernel reduces each threadgroup's share before writing).
- **Contiguity**: with `ensure_row_contiguous=True` (the default) MLX copies strided inputs before the call, and the body can index them as flat row-major arrays. With `False`, use `x_strides` and `elem_to_loc`.
- **Caching**: build the kernel objects once (`lru_cache`) instead of in every call; with `template`, each new value is a new specialization to compile, so keep their number small.
- **Importable without a GPU**: documentation tools and some tests import modules with `mlx` replaced by a mock. Build kernels lazily, inside functions, and avoid annotations evaluated at import time on MLX types, such as `mx.array | None`: that one broke the import of `csr_softmax_spmm` under a mock until it was written `Optional[mx.array]` (#214).

## Heavier kernels

`mx.fast.metal_kernel` suits element-wise and per-edge work written in one Metal function. For kernels that need several passes, shared buffers or their own memory planning (sparse matrix products, random walks, graph partitioning), write an MLX C++ extension: [mlx-cluster](https://github.com/mlx-graphs/mlx_cluster), which provides the SpMM of the fused message passing and the random walks of Node2Vec and MetaPath2Vec, is one, built on [MLX's custom extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html).
