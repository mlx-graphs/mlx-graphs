# Creating message passing layers

Every graph layer of mlx-graphs is a subclass of [`MessagePassing`][mlx_graphs.nn.MessagePassing], the counterpart of PyG's `MessagePassing`. A layer computes one message per edge, aggregates the messages that reach each node and updates the node embeddings. This page shows how to write such a layer, how to choose its aggregation, how to write attention layers, how to make a layer fast, and how to check it against PyG.

## The `MessagePassing` contract

A layer chooses its aggregation in `__init__` and calls `self.propagate` in `__call__`. `propagate(edge_index, node_features, message_kwargs=None, aggregate_kwargs=None, update_kwargs=None)` runs three methods, and each one can be overridden:

| Method                                                  | What it receives and returns                                                                                                                                                                                                                                                                                                                             |
|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `message(src_features, dst_features, **message_kwargs)` | The features of the two ends of every edge: `src_features = node_features[edge_index[0]]` (PyG's `x_j`) and `dst_features = node_features[edge_index[1]]` (PyG's `x_i`), plus the entries of `message_kwargs`. Returns one message per edge, `[num_edges, ...]`. By default, the source features, multiplied by `edge_weights` if that keyword is given. |
| `aggregate(messages, indices, **aggregate_kwargs)`      | The messages and their destination nodes (`indices = edge_index[1]`). Returns one row per destination node. By default, it reduces the messages with the layer's aggregation.                                                                                                                                                                            |
| `update_nodes(aggregated, **update_kwargs)`             | The aggregated messages, plus the entries of `update_kwargs`. Returns the output of the layer. By default, the aggregated messages unchanged.                                                                                                                                                                                                            |

Messages flow from `edge_index[0]` (the source) to `edge_index[1]` (the destination), as with PyG's default `flow="source_to_target"`. An undirected graph lists every edge in both directions. Nodes without incoming edges get zeros with the `"add"`, `"mean"`, `"max"` and `"min"` aggregations.

`propagate` also sets `self.num_nodes` to the number of destination nodes, read from the shape of the features. The default `aggregate` uses it as the size of its output, so the output always has one row per node, even for nodes without edges.

**Edges first**

The built-in layers take the graph first: `conv(edge_index, node_features, ...)`, where PyG writes `conv(x, edge_index)`. `propagate` takes its arguments in the same order. Write your own `__call__` the same way, so that your layers work in the models and wrappers that call layers like this (e.g. [`to_hetero()`][mlx_graphs.nn.to_hetero]).

The smallest layer passes the features through `propagate` with the default `message`:

```python
import mlx.core as mx

from mlx_graphs.nn import MessagePassing

class NeighborAggr(MessagePassing):
    def __call__(self, edge_index, node_features):
        return self.propagate(edge_index, node_features)

# edges 0 -> 1, 1 -> 2, 2 -> 1 and 3 -> 1; node 0 has no incoming edge
edge_index = mx.array([[0, 1, 2, 3], [1, 2, 1, 1]])
x = mx.array([[1.0], [2.0], [3.0], [4.0]])

print(NeighborAggr(aggr="mean")(edge_index, x))
```

```text
array([[0],
       [2.66667],
       [2],
       [0]], dtype=float32)
```

Node 1 receives the features of nodes 0, 2 and 3, and averages them.

**Bipartite graphs**

`node_features` can also be a `(src_features, dst_features)` tuple, for messages between two sets of nodes (e.g. `author -> paper` in a heterogeneous graph). `edge_index[0]` then indexes `src_features`, `edge_index[1]` indexes `dst_features`, and the output has one row per destination node:

```python
authors = mx.array([[1.0], [2.0], [3.0]])
papers = mx.zeros((2, 1))
writes = mx.array([[0, 1, 2], [0, 0, 1]])  # author -> paper

print(NeighborAggr(aggr="mean")(writes, (authors, papers)))
```

```text
array([[1.5],
       [3]], dtype=float32)
```

## A layer with edge features, step by step

This section builds a layer that uses the features of both ends of an edge and the edge's own features:

$$\mathbf{h}_i = \mathbf{W}_r \mathbf{x}_i +
\mathrm{mean}_{j \to i} \, \mathrm{ReLU} \left(
\mathbf{W}_s \mathbf{x}_j + \mathbf{W}_d \mathbf{x}_i +
\mathbf{W}_e \mathbf{e}_{j i} \right)$$

**1. Decide what goes where.** The sum inside the ReLU is the message of edge $j \to i$. The mean is the aggregation. Adding the root term $\mathbf{W}_r \mathbf{x}_i$ is the update.

**2. Project the nodes once, not every edge.** The linear maps act on one node at a time, so they can be applied to the `[num_nodes, features]` arrays before `propagate` gathers them per edge. That computes `num_nodes` projections instead of `num_edges`, which is how [`GATConv`][mlx_graphs.nn.GATConv] and [`SAGEConv`][mlx_graphs.nn.SAGEConv] work. The projected source and destination features go to `propagate` as a `(src, dst)` tuple, the projected edge features through `message_kwargs` and the root term through `update_kwargs`.

**3. Write the layer.**

```python
import mlx.nn as nn

from mlx_graphs.nn import Linear

class EdgeFeatureConv(MessagePassing):
    # update_nodes adds a root term of size out_features_dim, so a list of
    # aggregations whose outputs are concatenated cannot work (see below)
    supports_concat_aggr = False

    def __init__(self, node_features_dim, edge_features_dim, out_features_dim, **kwargs):
        kwargs.setdefault("aggr", "mean")
        super().__init__(**kwargs)
        self.lin_src = Linear(node_features_dim, out_features_dim)
        self.lin_dst = Linear(node_features_dim, out_features_dim, bias=False)
        self.lin_edge = Linear(edge_features_dim, out_features_dim, bias=False)
        self.lin_root = Linear(node_features_dim, out_features_dim, bias=False)

    def __call__(self, edge_index, node_features, edge_features):
        if isinstance(node_features, tuple):  # bipartite (src, dst)
            src, dst = node_features
        else:
            src = dst = node_features
        return self.propagate(
            edge_index,
            (self.lin_src(src), self.lin_dst(dst)),
            message_kwargs={"edge_features": self.lin_edge(edge_features)},
            update_kwargs={"root": self.lin_root(dst)},
        )

    def message(self, src_features, dst_features, edge_features=None, **kwargs):
        return nn.relu(src_features + dst_features + edge_features)

    def update_nodes(self, aggregated, root=None, **kwargs):
        return aggregated + root
```

`message` and `update_nodes` take their extra arguments by name, as given in `message_kwargs` and `update_kwargs`. Keep a `**kwargs` in their signature: `propagate` passes every entry of the dictionaries.

**4. Run it.** A small graph with a duplicate edge, a self-loop and an isolated node (node 5), which are the cases where layers usually go wrong:

```python
mx.random.seed(0)
edge_index = mx.array([[0, 1, 1, 2, 3, 3, 4, 2], [1, 0, 2, 1, 4, 4, 3, 2]])
x = mx.random.normal((6, 8))
edge_features = mx.random.normal((edge_index.shape[1], 3))

conv = EdgeFeatureConv(8, 3, 16)
out = conv(edge_index, x, edge_features)
print(out.shape)
```

```text
(6, 16)
```

**5. Check it against the formula**, written with [`scatter()`][mlx_graphs.utils.scatter.scatter]:

```python
from mlx_graphs.utils import scatter

src, dst = edge_index
messages = nn.relu(
    conv.lin_src(x)[src] + conv.lin_dst(x)[dst] + conv.lin_edge(edge_features)
)
expected = scatter(messages, dst, x.shape[0], "mean") + conv.lin_root(x)
print(mx.allclose(out, expected, atol=1e-5).item())
# node 5 has no incoming edge: only its root term is left
print(mx.allclose(out[5], conv.lin_root(x)[5]).item())
```

```text
True
True
```

**6. Train it.** The layer is an `mlx.nn.Module`, so its parameters get gradients like any other module's:

```python
from mlx.utils import tree_flatten

def loss_fn(model):
    return model(edge_index, x, edge_features).square().mean()

loss, grads = nn.value_and_grad(conv, loss_fn)(conv)
print([name for name, _ in tree_flatten(grads)])
```

```text
['lin_src.weight', 'lin_src.bias', 'lin_dst.weight', 'lin_edge.weight', 'lin_root.weight']
```

**7. Use it on a bipartite graph.** `__call__` already accepts a `(src, dst)` tuple:

```python
x_src, x_dst = mx.random.normal((4, 8)), mx.random.normal((2, 8))
bipartite = mx.array([[0, 1, 2, 3], [0, 0, 1, 1]])
print(conv(bipartite, (x_src, x_dst), mx.random.normal((4, 3))).shape)
```

```text
(2, 16)
```

Because `__init__` passes `**kwargs` to `MessagePassing`, the aggregation can be changed when building the layer, e.g. `EdgeFeatureConv(8, 3, 16, aggr="max")`.

## Aggregations

The `aggr` argument of `MessagePassing.__init__` accepts the same values as PyG's:

| `aggr`                                                                                  | What the layer does with it                                                                                                                                                                                                                                        |
|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `"add"` (or `"sum"`), `"mean"`, `"max"`, `"min"`, `"mul"`, `"var"`, `"std"`, `"median"` | Reduces with [`scatter()`][mlx_graphs.utils.scatter.scatter]. The first four can run on the fused kernels (see [Speed](#speed)).                                                                                                                                                                      |
| Any other name: `"softmax"`, `"powermean"`, `"lstm"`, `"set_transformer"`...            | Builds the aggregation module of that name with [`aggregation_resolver()`][mlx_graphs.nn.aggr.aggregation_resolver], stored in `self.aggr_module`. As in PyG, `"softmax"` is [`SoftmaxAggregation`][mlx_graphs.nn.aggr.SoftmaxAggregation], which gives one output per node.                                                                                               |
| A name with `aggr_kwargs`                                                               | Builds the aggregation module of that name with these arguments (even for `"mean"` or `"max"`).                                                                                                                                                                    |
| An [`Aggregation`][mlx_graphs.nn.aggr.Aggregation] instance                                                                | Uses it as `self.aggr_module`. Its parameters (e.g. a learnable temperature) are trained with the layer.                                                                                                                                                           |
| A list of names and/or aggregations                                                     | Combines them with [`MultiAggregation`][mlx_graphs.nn.aggr.MultiAggregation]: concatenated along the last axis by default, or combined element-wise with `aggr_kwargs={"mode": "sum"}` (also `"mean"`, `"max"`, `"min"`, `"logsumexp"`, `"std"`, `"var"`, or `"proj"`, a linear layer after the concatenation). |

With the `NeighborAggr` layer of the first section:

```python
from mlx_graphs.nn.aggr import SoftmaxAggregation

edge_index = mx.array([[0, 1, 2, 3], [1, 2, 1, 1]])
x = mx.array([[1.0], [2.0], [3.0], [4.0]])

print(NeighborAggr(aggr="max")(edge_index, x)[1])
print(NeighborAggr(aggr=["mean", "max"])(edge_index, x)[1])
print(NeighborAggr(aggr=["mean", "max"], aggr_kwargs={"mode": "sum"})(edge_index, x)[1])

conv = NeighborAggr(aggr=SoftmaxAggregation(learn=True))
print(conv.aggr_module)
print([name for name, _ in tree_flatten(conv.trainable_parameters())])
```

```text
array([4], dtype=float32)
array([2.66667, 4], dtype=float32)
array([6.66667], dtype=float32)
SoftmaxAggregation(learn=True)
['aggr_module.t']
```

**``supports_concat_aggr``**

A list of aggregations concatenates their outputs, so the aggregated features are `len(aggr)` times wider. Most layers project or combine the aggregated features with arrays sized for one aggregation, and would fail with a shape error, or for multi-head attention return wrong rows. Such layers set the class attribute `supports_concat_aggr = False`, as `EdgeFeatureConv` does above: `MessagePassing.__init__` then refuses a concatenating list with a clear error, and still accepts a list combined with `aggr_kwargs={"mode": ...}`:

```python
try:
    EdgeFeatureConv(8, 3, 16, aggr=["mean", "max"])
except ValueError as error:
    print(str(error).split(":")[0])

conv = EdgeFeatureConv(8, 3, 16, aggr=["mean", "max"], aggr_kwargs={"mode": "mean"})
print(conv.aggr_module)
```

```text
EdgeFeatureConv does not support a list of aggregations whose outputs are concatenated
MultiAggregation(mode=mean
  (aggrs.0): MeanAggregation()
  (aggrs.1): MaxAggregation()
)
```

The default is `True`, for layers whose output is the aggregation itself (such as `NeighborAggr`). A layer can also support concatenation by sizing its next projection with `self.aggr_module.get_out_features_dim(dim)`, which is what [`SAGEConv`][mlx_graphs.nn.SAGEConv] does (`"aggr_module" in self` tells whether the layer has an aggregation module).

**``aggregation_resolver``**

[`aggregation_resolver()`][mlx_graphs.nn.aggr.aggregation_resolver] turns a name, a list or a module into an [`Aggregation`][mlx_graphs.nn.aggr.Aggregation], as PyG's function of the same name. Names are case-insensitive and can be short names or class names, and PyG's `in_channels` / `out_channels` are accepted for `node_features_dim` / `out_features_dim`:

```python
from mlx_graphs.nn.aggr import aggregation_resolver

print(aggregation_resolver("SoftmaxAggregation", learn=True))
print(aggregation_resolver("lstm", in_channels=8, out_channels=8))
print(aggregation_resolver(["mean", "std"], mode="cat"))
```

```text
SoftmaxAggregation(learn=True)
LSTMAggregation(
  (lstm): LSTM(input_dims=8, hidden_size=8, bias=True)
)
MultiAggregation(mode=cat
  (aggrs.0): MeanAggregation()
  (aggrs.1): StdAggregation(semi_grad=False)
)
```

Aggregations also work on their own, e.g. as a readout over the nodes of each graph of a batch: `aggr(x, index, dim_size)` reduces the rows of `x` that share an `index`:

```python
from mlx_graphs.nn.aggr import SumAggregation

print(SumAggregation()(x, mx.array([0, 0, 1, 1]), dim_size=2))
```

```text
array([[3],
       [7]], dtype=float32)
```

## Attention layers

An attention layer weighs each message by a coefficient that is normalized over the incoming edges of the destination node. That normalization needs the scores of all the incoming edges of a node at once, so it can't be done inside `message`, which sees one edge at a time. Compute the scores before `propagate`, normalize them with `scatter(scores, dst, num_nodes, "softmax")` (or [`softmax()`][mlx_graphs.utils.softmax], its PyG-named version), and pass the coefficients to `propagate` as `edge_weights`. Unlike the other reductions of `scatter`, `"softmax"` returns one value per edge.

This is how [`GATConv`][mlx_graphs.nn.GATConv] works. Its scores are a sum of a term of the source node and a term of the destination node, so they are computed per node and only gathered per edge as `[num_edges, heads]` scalars. A simplified GAT layer (no self-loops, dropout or bias):

```python
from mlx_graphs.nn import fused_message_passing
from mlx_graphs.nn.linear import glorot

@fused_message_passing
class SimpleGATConv(MessagePassing):
    supports_concat_aggr = False

    def __init__(self, node_features_dim, out_features_dim, heads=1, negative_slope=0.2):
        super().__init__(aggr="add")
        self.heads, self.out_features_dim = heads, out_features_dim
        self.negative_slope = negative_slope
        self.lin = Linear(node_features_dim, heads * out_features_dim, bias=False)
        self.att_src = glorot((1, heads, out_features_dim))
        self.att_dst = glorot((1, heads, out_features_dim))

    def scores(self, edge_index, node_features):
        """Attention coefficients [num_edges, heads] and projected features."""
        H, C = self.heads, self.out_features_dim
        h = self.lin(node_features).reshape(-1, H, C)  # [num_nodes, H, C]
        src, dst = edge_index
        a_src = (h * self.att_src).sum(-1)  # [num_nodes, H]
        a_dst = (h * self.att_dst).sum(-1)
        logits = nn.leaky_relu(a_src[src] + a_dst[dst], self.negative_slope)
        return scatter(logits, dst, node_features.shape[0], "softmax"), h

    def __call__(self, edge_index, node_features):
        alpha, h = self.scores(edge_index, node_features)
        # one weight per edge and head: aggregated as one SpMM per head
        out = self.propagate(edge_index, h, message_kwargs={"edge_weights": alpha})
        return out.reshape(node_features.shape[0], -1)  # concatenate the heads

mx.random.seed(0)
edge_index = mx.array([[0, 1, 1, 2, 3, 3, 4, 2], [1, 0, 2, 1, 4, 4, 3, 2]])
x = mx.random.normal((6, 8))

conv = SimpleGATConv(8, 16, heads=4)
print(conv(edge_index, x).shape)

alpha, _ = conv.scores(edge_index, x)
# the coefficients of the incoming edges of each node sum to 1
print(scatter(alpha, edge_index[1], 6, "add")[:, 0])
```

```text
(6, 64)
array([1, 1, 1, 1, 1, 0], dtype=float32)
```

The default `message` multiplies features of shape `[num_edges, heads, channels]` by `edge_weights` of shape `[num_edges, heads]`, one weight per head, and so does the fused kernel. Node 5 has no incoming edge: its coefficients sum to 0 and its output is 0.

[`softmax()`][mlx_graphs.utils.softmax] gives the same coefficients:

```python
from mlx_graphs.utils import softmax

H, C = conv.heads, conv.out_features_dim
h = conv.lin(x).reshape(-1, H, C)
src, dst = edge_index
logits = nn.leaky_relu(
    (h * conv.att_src).sum(-1)[src] + (h * conv.att_dst).sum(-1)[dst], 0.2
)
print(mx.allclose(softmax(logits, dst, num_nodes=6), alpha).item())
```

```text
True
```

Two details of [`GATConv`][mlx_graphs.nn.GATConv] are worth copying. Edges can be excluded by setting their logits to `-inf` before the softmax, and dropout is applied to the coefficients (`nn.Dropout` on `alpha`), after the softmax.

## Speed

**Fused message passing**

Gather-scatter, the default path of `propagate`, builds a `[num_edges, features]` array of messages and reduces it. On large graphs, that array costs more time and memory than the rest of the layer. When the messages are a weighted sum of neighbour features, the class decorator [`fused_message_passing()`][mlx_graphs.nn.fused_message_passing] replaces `message` + `aggregate` by a single sparse x dense matmul (SpMM) from `mlx-cluster` (`message_and_aggregate`), and the message array is never built. The built-in GCN, GraphSAGE, GIN, GAT and GATv2 layers, among others, use it.

The decorator only enables the fused path. `propagate` decides at every call, and falls back to gather-scatter, with the same result, when:

- `mlx_cluster` is not installed, or fused message passing is turned off ([`set_fused_message_passing()`][mlx_graphs.nn.set_fused_message_passing], [`fused_message_passing_mode()`][mlx_graphs.nn.fused_message_passing_mode]);
- `aggr` is not `"add"`, `"mean"`, `"max"` or `"min"`, e.g. a list or an aggregation module (`"max"` and `"min"` also need a recent `mlx-cluster`);
- an entry of `message_kwargs` other than `edge_weights`, or of `aggregate_kwargs`, is not `None` (e.g. `edge_features`): the messages are then no longer a weighted sum of neighbour features;
- the source features are not floating point or have fewer than two dimensions, or `edge_weights` is neither one scalar per edge nor `[num_edges, heads]` for `[num_nodes, heads, channels]` features;
- the layer subclasses a decorated layer and overrides `message()` or `aggregate()` without also overriding `message_and_aggregate()`.

The two paths give the same output, which [`fused_message_passing_mode()`][mlx_graphs.nn.fused_message_passing_mode] makes easy to check:

```python
from mlx_graphs.nn import fused_message_passing_mode, is_fused_message_passing_enabled

@fused_message_passing
class WeightedSum(MessagePassing):
    def __call__(self, edge_index, node_features, edge_weights=None):
        return self.propagate(
            edge_index, node_features, message_kwargs={"edge_weights": edge_weights}
        )

conv = WeightedSum(aggr="add")
weights = mx.random.uniform(shape=(edge_index.shape[1],))
print(is_fused_message_passing_enabled())

fused = conv(edge_index, x, weights)
with fused_message_passing_mode(False):
    reference = conv(edge_index, x, weights)
print(mx.allclose(fused, reference, atol=1e-5).item())
```

```text
True
True
```

**``allow_message_override``**

The fused kernel ignores `message` and `aggregate`, so decorating a class that overrides them raises an error when the class is defined:

```python
try:

    @fused_message_passing
    class Doubled(MessagePassing):
        def message(self, src_features, dst_features, **kwargs):
            return 2 * src_features

except TypeError as error:
    print(error)
```

```text
Doubled overrides message(), which the fused SpMM kernel would ignore. Implement message_and_aggregate(), or pass allow_message_override=True if the override reduces to the default when only edge_weights are given.
```

Layers whose messages are not a weighted sum of neighbour features, such as `EdgeFeatureConv` or [`EdgeConv`][mlx_graphs.nn.EdgeConv] (an MLP per edge), have no fused path and are left undecorated. Some layers override `message` only for an optional input, and fall back to the default message without it. They are decorated with `allow_message_override=True`: the fused path runs when the optional input is `None` and gather-scatter runs otherwise. This is how [`GINConv`][mlx_graphs.nn.GINConv] handles its optional edge features:

```python
@fused_message_passing(allow_message_override=True)
class MaybeEdgeConv(MessagePassing):
    def __call__(self, edge_index, node_features, edge_features=None):
        return self.propagate(
            edge_index, node_features, message_kwargs={"edge_features": edge_features}
        )

    def message(self, src_features, dst_features, edge_features=None, **kwargs):
        if edge_features is None:  # the default message: fused
            return super().message(src_features, dst_features, **kwargs)
        return nn.relu(src_features + edge_features)  # gather-scatter

conv = MaybeEdgeConv()
print(conv(edge_index, x).shape)
print(conv(edge_index, x, mx.random.normal((edge_index.shape[1], 8))).shape)
```

```text
(6, 8)
(6, 8)
```

A layer can also provide its own fused implementation by overriding `message_and_aggregate(edge_index, node_features, edge_weights=None, **kwargs)`.

**``format``**

`@fused_message_passing(format=...)` picks the kernel:

- `"coo"` works on `edge_index` as it is.
- `"csr"` converts the graph to CSR once and uses a faster, deterministic kernel without atomics. The conversion (and its transpose, for the backward pass) is cached per `edge_index` array, in a cache of 16 graphs.
- `"auto"` (the default) starts with `"coo"` and switches to `"csr"` when the conversion pays off: right away for graphs with at least a million edges, and after a few calls with the same `edge_index` array otherwise (e.g. full-batch training). Mini-batches seen once stay on COO.

[`set_fused_message_passing()`][mlx_graphs.nn.set_fused_message_passing] `(True, format="csr")` overrides the format of every layer.

**``clear_csr_cache``**

Cached conversions are found by the identity of the `edge_index` array, not by its content. Modifying an `edge_index` in place between calls leaves a stale conversion in the cache. Build a new array instead, or call [`clear_csr_cache()`][mlx_graphs.nn.clear_csr_cache]:

```python
from mlx_graphs.nn import clear_csr_cache

@fused_message_passing(format="csr")
class CSRSum(MessagePassing):
    def __call__(self, edge_index, node_features):
        return self.propagate(edge_index, node_features)

conv = CSRSum()
graph = mx.array([[0, 1], [1, 2]])
features = mx.array([[1.0], [10.0], [100.0]])
print(conv(graph, features)[:, 0])

graph[0, 0] = 2  # in place: the edge 0 -> 1 becomes 2 -> 1
print(conv(graph, features)[:, 0])  # stale: still the old graph

clear_csr_cache()
print(conv(graph, features)[:, 0])
```

```text
array([0, 1, 10], dtype=float32)
array([0, 1, 10], dtype=float32)
array([0, 100, 10], dtype=float32)
```

**Staying compatible with ``mx.compile``**

`mx.compile` traces a function once and replays the trace. While tracing, arrays have no values, so anything that reads a value on the host fails: `.item()`, `float(array)`, `np.array(array)`, or a Python `if` on an array. Shapes are known, so sizes read from shapes are fine.

The usual culprit is a number of nodes computed as `index.max() + 1`. [`degree()`][mlx_graphs.utils.scatter.degree], [`softmax()`][mlx_graphs.utils.softmax] and the aggregation modules do that (with `.item()`) when they are not given a size. Pass the size you already know: `num_nodes` to `degree` and `softmax`, `out_size` to `scatter` (whose default is the number of values, not of nodes) and `dim_size` to an aggregation. `propagate` and the default `aggregate` already use `self.num_nodes`, read from the shape of the features.

```python
from mlx_graphs.utils import degree

dst = edge_index[1]
try:
    mx.compile(lambda index: degree(index))(dst)
except ValueError as error:
    print(error)

print(mx.compile(lambda index: degree(index, 6))(dst))
```

```text
[eval] Attempting to eval an array during function transformations like compile or vmap is not allowed.
array([1, 2, 2, 1, 2, 0], dtype=float32)
```

The layers of this page follow these rules and compile:

```python
conv = EdgeFeatureConv(8, 3, 16)
edge_features = mx.random.normal((edge_index.shape[1], 3))
compiled = mx.compile(lambda e, h, f: conv(e, h, f))
eager = conv(edge_index, x, edge_features)
print(mx.allclose(compiled(edge_index, x, edge_features), eager, atol=1e-5).item())

gat = SimpleGATConv(8, 16, heads=4)
compiled = mx.compile(lambda e, h: gat(e, h))
print(mx.allclose(compiled(edge_index, x), gat(edge_index, x), atol=1e-5).item())
```

```text
True
True
```

The CSR conversion of the fused kernels also reads the graph on the host. When `edge_index` is an input of the compiled function, fused layers (`format="auto"` or `"csr"`) fall back to the COO kernel. To keep the CSR kernel under `mx.compile`, call the layer once eagerly, which caches the conversion, and let the compiled function capture that same `edge_index` array instead of taking it as an input:

```python
conv = CSRSum()
graph = mx.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
features = mx.random.normal((5, 4))

eager = conv(graph, features)  # converts and caches graph
compiled = mx.compile(lambda h: conv(graph, h))
print(mx.allclose(compiled(features), eager).item())
```

```text
True
```

Layers that need the sizes of the graphs of a batch on the host (the LSTM, GRU, sort and set transformer aggregations, `TopKPooling` / `SAGPooling`, and `GPSConv` / `SGFormer` / `Polynormer` without a precomputed `layout`) work with `nn.value_and_grad` but not inside `mx.compile`.

## Testing a layer against PyG

Most layers of mlx-graphs have a test that builds the same layer in PyG, copies PyG's weights into the mlx-graphs layer and compares the outputs (and, for many of them, the gradients) on the same random graph. They are in `tests/nn/conv/test_*.py`. For example `test_gat_conv.py` compares the output of `GATConv` and the gradients with respect to the node and edge features, and `test_gine_conv.py` compares `GINEConv` on homogeneous and bipartite graphs. `tests/pyg_copy.py` has helpers to convert a torch `state_dict` to mlx-graphs names (`torch_state`) and load it with shape checks (`load_torch`). The tests need `torch` and `torch_geometric` (`pip install -e '.[test]'`).

The pattern, for `EdgeFeatureConv`:

1.  Write the same layer in PyG.
2.  Copy the weights. `torch.nn.Linear` and `mlx.nn.Linear` store their weights with the same `[out_features, in_features]` shape.
3.  Run both on a graph with duplicate edges, self-loops and isolated nodes.
4.  Compare the outputs and the input gradients.

```python
import numpy as np
import torch
from torch_geometric.nn import MessagePassing as PyGMessagePassing

class PyGEdgeFeatureConv(PyGMessagePassing):
    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__(aggr="mean")
        self.lin_src = torch.nn.Linear(in_dim, out_dim)
        self.lin_dst = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.lin_edge = torch.nn.Linear(edge_dim, out_dim, bias=False)
        self.lin_root = torch.nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(
            edge_index,
            h_src=self.lin_src(x),
            h_dst=self.lin_dst(x),
            edge_attr=self.lin_edge(edge_attr),
        )
        return out + self.lin_root(x)

    def message(self, h_src_j, h_dst_i, edge_attr):
        return torch.relu(h_src_j + h_dst_i + edge_attr)

# a random graph with duplicate edges, self-loops and isolated nodes
rng = np.random.default_rng(0)
edges = rng.integers(0, 25, (2, 100))
edges = np.concatenate([edges, edges[:, :10], [[0, 1], [0, 1]]], axis=1)
x_np = rng.normal(size=(30, 8)).astype(np.float32)
e_np = rng.normal(size=(edges.shape[1], 3)).astype(np.float32)

torch.manual_seed(0)
ref = PyGEdgeFeatureConv(8, 3, 16)
conv = EdgeFeatureConv(8, 3, 16)
for name in ("lin_src", "lin_dst", "lin_edge", "lin_root"):
    for param, value in getattr(ref, name).named_parameters():
        setattr(getattr(conv, name), param, mx.array(value.detach().numpy()))

x_torch = torch.tensor(x_np, requires_grad=True)
expected = ref(x_torch, torch.tensor(edges), torch.tensor(e_np))
expected.square().sum().backward()

def loss_fn(features):
    return conv(mx.array(edges), features, mx.array(e_np)).square().sum()

out = conv(mx.array(edges), mx.array(x_np), mx.array(e_np))
grad = mx.grad(loss_fn)(mx.array(x_np))

# MLX's float32 GPU matmul is accurate to ~1e-3 relative: a wrong formula
# is off by O(1), so a loose tolerance still catches it
print(np.allclose(np.array(out), expected.detach().numpy(), rtol=2e-2, atol=2e-2))
grad_error = np.linalg.norm(np.array(grad) - x_torch.grad.numpy())
print(grad_error <= 2e-2 * np.linalg.norm(x_torch.grad.numpy()))
```

```text
True
True
```

A few things make these tests catch bugs:

- **Edge cases in the graph.** Duplicate edges, self-loops and nodes without incoming edges are where aggregations and normalizations differ. `test_zero_edges.py` also runs the layers that project per-edge arrays on graphs without edges.
- **Gradients.** A forward pass can match while the backward pass doesn't (e.g. infinite derivatives of norms at zero, or the gradient of max among tied messages). Compare gradients with respect to the inputs, and check that every parameter gets a finite gradient.
- **Tolerances.** MLX's float32 matmul on the GPU has a relative error of about 1e-3, where torch on the CPU has about 1e-7. The tests use a tolerance of 2e-2, which a wrong formula exceeds by far.
- **Both paths.** For a layer decorated with `@fused_message_passing`, compare the fused and gather-scatter outputs with [`fused_message_passing_mode()`][mlx_graphs.nn.fused_message_passing_mode] (see `tests/nn/test_fused_message_passing.py`).

See [Validation](../../validation.md) for the layer-level, sampler and training checks against PyG.
