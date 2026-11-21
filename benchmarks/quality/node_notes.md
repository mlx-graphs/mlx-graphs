## Setup

- Data: Planetoid Cora / CiteSeer / PubMed loaded through PyG
  (`benchmarks/pyg_comparison/worker.py:load_node_dataset`), public split,
  identical arrays in both frameworks. NNConv / CGConv get two synthetic,
  deterministic edge features (`1/sqrt(1 + in-degree)` of both ends).
- Models (`node_models.py`): the same architecture in both frameworks, each
  framework with its own default initialization. Typical hyperparameters of
  the papers / PyG examples: hidden 64 (16 for GCN, ARMA, AGNN, NNConv),
  dropout 0.5, Adam lr 0.01, weight decay 5e-4 (torch's L2 form, reproduced
  in MLX by adding `wd * p` to the gradients), 200 epochs, evaluation every
  epoch. GCNII as in PyG's `gcn2_cora.py` / the paper (64 layers on Cora,
  32 x 256 on CiteSeer, 16 x 256 on PubMed, weight decay 0.01 on the
  convolutions, 500 epochs, evaluation every 5 epochs). SSGC as in the paper
  (one SSGConv, K = 16, alpha 0.05, lr 0.2). FAGCN: 4 FAConv, eps 0.3.
  DeeperGCN: 7 res+ `DeepGCNLayer`s of GENConv (softmax, learnable t,
  LayerNorm). The norms are compared in a 4-layer GCN (a 2-layer one barely
  uses them), against the same GCN without norm.
- Feature transforms are applied by each framework's own implementation
  (mlx-graphs' `NormalizeFeatures`, `SIGN`, `AddLaplacianEigenvectorPE`,
  `LocalDegreeProfile` vs PyG's).
- `LabelPropagation` is deterministic (one run). `CorrectAndSmooth` runs on
  each framework's own MLP; fed the *same* soft predictions on Cora, the two
  implementations agree to 3e-5 (both settings tried), so any gap there is
  the MLP's.
- PyG runs on the CPU, one thread per worker (several workers in
  parallel), except GCNII and DeeperGCN on MPS (minutes per seed on the
  CPU); the device does not change what is compared.
- Single layers were also checked with copied weights on the real Cora
  graph (forward and input gradients): PairNorm, GraphNorm, GENConv (with
  and without MessageNorm) agree to GPU matmul precision; CorrectAndSmooth
  agrees to 3e-5; LabelPropagation gives identical accuracies.

## Discrepancies found, and what was done

Bugs fixed in mlx-graphs (with regression tests; the full suite passes):

1. **SAGEConv with a non-linear aggregation** (`aggr="max"`, `"min"`,
   `"std"`, lists, modules) and more input than output features projected
   the features *before* aggregating (DGL's trick, valid only for sum /
   mean): it computed `max(W x)` instead of PyG's `W max(x)`. Max error vs
   PyG with copied weights: 1.7. Now projects after aggregating unless the
   aggregation is `"add"` / `"mean"`.
2. **Lists of aggregations** (`aggr=["mean", "max"]`): SAGEConv crashed
   (its neighbor projection ignored the concatenated size). Now sized with
   `aggr_module.get_out_features_dim`, as PyG's `get_out_channels`.
   GENConv did not accept a list at all; it now concatenates and projects
   back with `lin_aggr_out`, as PyG. These are the only two PyG layers
   that size for a concatenating list; every other layer with a projection
   after the aggregation crashed with a shape error in the forward pass
   (GCNConv, GraphConv, GIN, ...), and GATConv / GATv2Conv silently
   returned twice as many rows (as PyG does). They now refuse it when built
   with a clear `ValueError` (`MessagePassing.supports_concat_aggr`),
   pointing to size-preserving combinations (`aggr_kwargs={"mode": "sum"}`);
   GeneralConv and RGATConv, which reduce with `scatter` directly, refuse
   any aggregation module, as RGCNConv already did. Layers without a
   projection after aggregating (SimpleConv, LGConv, AGNNConv, EdgeConv,
   custom subclasses) keep concatenating.
3. **NaN gradients from L2 norms at zero**: AGNNConv (cosine similarity),
   MessageNorm and GENConv(msg_norm=True) used `mx.linalg.norm` /
   `sqrt(sum(x**2))`, whose derivative is infinite at zero; a node whose
   features are all zero (common after ReLU + dropout) produced NaN
   gradients that spread to all parameters. torch's norm has a zero
   gradient there. Seen as an AGNN on PubMed going NaN at epoch 145, and
   GENConv + MessageNorm reaching 66% instead of 79% on Cora (one seed,
   before the fix). New `mlx_graphs.utils.array_ops.p_norm`, also used by
   GeneralConv(l2_normalize=True) and PairNorm(scale_individually=True).
   The `agnn` and `gen_msgnorm` rows were re-run with the fix.
4. **MixHopConv** kept its per-power linear layers in a dict keyed `"0"`,
   `"1"`, ..., which MLX reads back as list indices: `load_weights` raised
   `KeyError` (and so did any `tree_unflatten`-based code, e.g. this
   harness' weight decay at first). Now a list indexed by power with
   `Identity` for the unused ones, as PyG (same parameter names for the
   default powers).
5. **AddLaplacianEigenvectorPE** failed on Cora (`ArpackNoConvergence`):
   the sparse eigenproblem was solved in float64 on float32-rounded values,
   so ARPACK had to separate the 78 zero eigenvalues of Cora's components
   at float64 precision. Now float32, as PyG (which converges).
6. **DirGNNConv(GCNConv(...))**, PyG's usual way to build it, raised because
   GCNConv did not store its feature sizes; it now does.
7. **DirGNNConv(SAGEConv)** trained 2-3 points below PyG on Cora and
   CiteSeer, with twice its training loss. PyG's DirGNNConv turns off the
   wrapped layer's root weight, but mlx-graphs' SAGEConv had no
   `root_weight`, so both copies kept a root projection (three root terms
   instead of one). SAGEConv now has PyG's `root_weight` (the bias moves to
   the neighbor term without it); the rerun rows pass (+0.9 / +0.7 / +0.5).

Differences that are not bugs:

- mlx-graphs' `GCNConv` defaulted to `add_self_loops=False` (PyG: `True`)
  until 0.0.11; it now defaults to `True`, as in PyG. The harness sets it
  explicitly, so the results do not depend on the default.
- Initializations differ in places (each framework's own defaults); no gap
  in the table points to an initialization problem.
- Single-epoch training losses are noisy (dropout, occasional loss spikes
  with sum aggregations such as LEConv / GatedGraphConv, in both
  frameworks), hence the loss columns average epochs 25-75 and the last 10
  epochs.
- PNAConv on raw features (no normalization, amplification scalers) trains
  with large loss spikes in both frameworks (PyG seed 0 on PubMed jumps to
  3.0, mlx-graphs seed 3 to 4.9 near the end, hence the different final
  losses at equal accuracy). With copied weights on PubMed, PNAConv's
  output and input gradients agree with PyG's to 1e-4 / 1e-8 (relative
  ~1e-3).

## Not checked

- Mini-batch / sampled training (only full-batch); larger graphs (Amazon,
  ogbn): out of the memory budget agreed for this check.
- Attention variants of GeneralConv on CiteSeer / PubMed (Cora only), and
  MLX's `mx.compile`.
