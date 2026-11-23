.. _validation:

Validation against PyG
======================

mlx-graphs follows `PyTorch Geometric <https://pyg.org>`_ (PyG), and every
layer, sampler and model added for PyG parity was checked against PyG: layers
by copying PyG's weights and comparing outputs and gradients, samplers
statistically, and models by training them in both frameworks on the same
data. This page describes how, summarizes the results, lists the bugs these
checks found, and shows how to rerun them.

The full result tables are in the repository, under
`benchmarks/quality <https://github.com/mlx-graphs/mlx-graphs/tree/main/benchmarks/quality>`_.
The speed comparison is on the :doc:`benchmarks` page.

.. contents::
    :local:


Method
------

The checks work at three levels:

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Level
     - What is compared
     - Where
   * - Layers
     - The same layer is built in both frameworks and PyG's weights are copied
       into the mlx-graphs one. Both run on the same random graph, with
       duplicate edges, self-loops and isolated nodes, and the outputs (and
       for many layers the gradients) must agree.
     - The unit tests, ``tests/``
   * - Samplers and loaders
     - Exact comparisons with PyG's pure-Python code paths where they exist,
       and otherwise statistical properties: coverage of every node once per
       epoch, uniform neighbour sampling, unbiased normalizations. Then
       mini-batch training against full-batch training.
     - ``benchmarks/quality/sampler_quality.py``
   * - Training quality
     - The same model is trained in both frameworks, with the same data,
       splits and hyperparameters, over several seeds. The test metric is
       read at the epoch of best validation metric.
     - ``benchmarks/quality/node_quality.py``,
       ``benchmarks/quality/graph_quality.py``, and the TGN, GPS and
       ``to_hetero`` comparisons

A training configuration **passes** when the gap between the means is within
1 point, or within 2 standard deviations of PyG's results:
:math:`|\mathrm{gap}| \le 1` or :math:`|\mathrm{gap}| \le 2\,\sigma_{\mathrm{PyG}}`.
The node classification table also marks *pass (noise)* a gap within 2
standard errors of the difference of the means.

.. rubric:: Keeping the comparison fair

- **Same model and data.** Both frameworks train the same architecture with
  the same hyperparameters, on identical arrays: the data is loaded once and
  handed to both, and the splits are the same. Each framework keeps its own
  default initialization, and mlx-graphs layers are initialized as in PyG.
- **Adam's bias correction.** MLX's ``Adam`` is created with
  ``bias_correction=True``, which is the update rule of ``torch.optim.Adam``.
  MLX's default has no bias correction and takes much larger first steps: it
  lowered GAT's accuracy on Cora by about 4 points in the speed benchmarks,
  and gave GPS on Peptides-func a clearly higher training loss.
- **L2 penalty vs weight decay.** ``torch.optim.Adam(weight_decay=...)`` adds
  ``weight_decay * p`` to the gradients (an L2 penalty), which is not the
  decoupled weight decay of ``AdamW``. The harnesses reproduce it in MLX by
  adding the same term to the gradients. Where the PyG script uses
  ``AdamW`` (the ``to_hetero`` comparison), both frameworks use ``AdamW``.
- **Devices.** mlx-graphs runs on the GPU and PyG on the CPU (with one or
  two torch threads per run), except for the slow deep models of the node
  classification harness (GCNII, DeeperGCN), for which PyG runs on MPS. The
  device changes the speed, not what is compared.
- **One process per run.** Every configuration and seed runs in its own
  process under a memory cap.

.. rubric:: Numerical tolerance

MLX's float32 matmul on the Mac's GPU has a relative error of about 1e-3,
where NumPy and torch on the CPU have about 1e-7. This is MLX behaviour, not
mlx-graphs', and it is the source of the ~1e-3 differences of the forward
passes. It can be larger for gradients that go through a cancellation:
EdgeConv's first layer, whose inputs are differences ``x_j - x_i``, has
gradient differences of up to about 10 %. The layer tests therefore use a
tolerance of 2e-2: a wrong formula is off by far more. It did not change any
training result.


Results
-------

Layers
~~~~~~

108 of the 151 test files compare mlx-graphs with PyG or use PyG as a
reference:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Tests
     - Files using PyG
     - Examples
   * - Convolution layers (``tests/nn/conv``)
     - 39 of 44
     - ``test_gat_conv.py``, ``test_gine_conv.py``, ``test_pna_conv.py``
   * - Models (``tests/nn/models``)
     - 7 of 8
     - autoencoders, KGE, LightGCN, label propagation, TGN
   * - Norms (``tests/nn/norm``)
     - 8 of 9
     - ``GraphNorm``, ``PairNorm``, ``LayerNorm``, ``MessageNorm``
   * - Pooling (``tests/nn/pooling``)
     - 5 of 6
     - TopK pooling, ``Set2Set``, global pooling
   * - Aggregations (``tests/nn/aggr``)
     - 2 of 2
     - the aggregation modules, including the learnable ones
   * - Message passing, ``to_hetero``, graph transformers (``tests/nn``)
     - 3 of 5
     - ``test_to_hetero.py``, ``test_message_passing_aggr.py``
   * - Transforms, utilities, data, loaders, metrics
     - 27 of 52
     - ``test_scatter_reductions.py``, ``test_pyg_reference.py``
   * - Datasets (slow tests, ``pytest --datasets``)
     - 17 of 23
     - a dataset's arrays against PyG's version of it
   * - Other (``tests/algorithms``, ``tests/test_hetero_linear.py``)
     - 0 of 2
     - (mlx-graphs only)

12 of these files also compare gradients with torch's autograd, among them
the GAT, GATv2, AGNN, GEN, MF and DirGNN convolutions, the aggregations, the
scatter reductions and ``TGNMemory``. The ``to_hetero`` tests copy PyG's
weights into converted models, including on the full DBLP and IMDB graphs
(slow tests). ``tests/pyg_copy.py`` holds the helpers that convert PyG
state dicts. :ref:`message_passing_guide` shows how to write such a test.

Some layers were also checked with copied weights on full datasets:

- On Cora: PairNorm, GraphNorm and GENConv (with and without MessageNorm)
  agree with PyG to GPU matmul precision, forward and input gradients;
  CorrectAndSmooth agrees to 3e-5; LabelPropagation gives identical
  accuracies.
- On PubMed: PNAConv's output and input gradients agree with PyG's to 1e-4
  and 1e-8 (relative ~1e-3).
- On DBLP: the HGT model's forward pass matches PyG's on the full graph.
- On Peptides-func: the full GPS model gives the same train-mode loss
  (0.70363 vs 0.70362 on 16 graphs) and gradients within ~1 %.

Node classification
~~~~~~~~~~~~~~~~~~~

38 model configurations on the public splits of Cora, CiteSeer and PubMed,
5 seeds each
(one deterministic run for ``LabelPropagation``): **99 configurations, 96
pass**. 81 of the 99 gaps are within 1 point.

.. list-table::
   :header-rows: 1
   :widths: 30 40 15 15

   * - Family
     - Models
     - Configs
     - Pass
   * - GCN and propagation
     - GCN, GCNII (64 layers), SSGConv, LGConv, ARMAConv, ClusterGCNConv,
       MixHopConv, SIGN, MaskLabel, NormalizeFeatures and Laplacian PE
       pipelines
     - 29
     - 29
   * - Other convolutions
     - GatedGraphConv, ResGatedGraphConv, NNConv, CGConv, MFConv, LEConv,
       FiLMConv, GeneralConv
     - 24
     - 24
   * - Attention
     - AGNNConv, FAConv (FAGCN), GeneralConv with attention
     - 7
     - 7
   * - Aggregations
     - GENConv (softmax), lists of aggregations in SAGEConv and GENConv,
       PNAConv, PNA, DeeperGCN
     - 15
     - 13
   * - Directed graphs
     - DirGNNConv with GCNConv and SAGEConv
     - 6
     - 5
   * - Norms (4-layer GCN)
     - no norm, GraphNorm, PairNorm, DiffGroupNorm, and MessageNorm in GENConv
     - 10
     - 10
   * - Label propagation and non-GNN models
     - LabelPropagation, CorrectAndSmooth, LINKX
     - 8
     - 8

A few rows (test accuracy in %, mean ± std over seeds):

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Model
     - Dataset
     - PyG
     - mlx-graphs
     - Gap
   * - GCNConv x2 (reference)
     - Cora
     - 80.5 ± 0.6
     - 80.9 ± 0.5
     - +0.4
   * - GCNII: 64 GCN2Conv layers
     - Cora
     - 82.7 ± 1.4
     - 83.0 ± 1.0
     - +0.3
   * - DeeperGCN: 7 GENConv layers
     - Cora
     - 75.2 ± 0.9
     - 75.4 ± 1.3
     - +0.2
   * - SAGEConv(aggr=["mean", "max"]) x2
     - PubMed
     - 78.1 ± 0.7
     - 78.3 ± 0.4
     - +0.1
   * - LabelPropagation (deterministic)
     - CiteSeer
     - 49.9
     - 49.9
     - +0.0
   * - DirGNNConv(GCNConv) x2
     - CiteSeer
     - 66.9 ± 0.7
     - 68.7 ± 1.4
     - +1.8, mismatch
   * - PNAConv x2
     - Cora
     - 75.0 ± 2.0
     - 70.4 ± 1.8
     - -4.6, mismatch
   * - PNAConv x2
     - CiteSeer
     - 52.0 ± 2.8
     - 58.8 ± 2.1
     - +6.8, mismatch

Full table, with the training losses of both frameworks:
`node_results.md <https://github.com/mlx-graphs/mlx-graphs/blob/main/benchmarks/quality/node_results.md>`_
(setup and notes:
`node_notes.md <https://github.com/mlx-graphs/mlx-graphs/blob/main/benchmarks/quality/node_notes.md>`_).

Graph-level, heterogeneous, link and knowledge graph models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**34 configurations, all pass**, most of them set up after PyG's example
scripts:

.. list-table::
   :header-rows: 1
   :widths: 30 45 12 13

   * - Task
     - Models and datasets
     - Configs
     - Pass
   * - Graph classification
     - GINE, PNA, TopK and SAG pooling, DGCNN, MultiAggregation, GraphNorm,
       Set2Set, DeepSets on MUTAG (10 seeds) and PROTEINS (5 seeds)
     - 17
     - 17
   * - Hierarchical pooling
     - DiffPool, MinCutPool, DMoN, ASAPooling, EdgePooling, GraphUNet on
       PROTEINS (5 seeds, ``pooling_quality.py``)
     - 6
     - 6
   * - Heterogeneous graphs
     - HAN on IMDB, HGT on DBLP, RGAT and FastRGCN on DBLP as a homogeneous
       graph
     - 4
     - 4
   * - Link prediction
     - GAE and VGAE on Cora and CiteSeer (and GAE with
       ``train_test_split_edges``), LightGCN on MovieLens100K
     - 6
     - 6
   * - Self-supervised
     - DeepGraphInfomax on Cora
     - 1
     - 1
   * - Knowledge graph embeddings
     - TransE, DistMult, ComplEx, RotatE on FB15k-237
     - 4
     - 4
   * - Point clouds
     - EdgeCNN and DynamicEdgeConv on synthetic point clouds
     - 2
     - 2

For example, HGT on DBLP reaches 71.8 ± 1.8 % in PyG and 73.0 ± 2.5 % in
mlx-graphs, GAE on Cora a test AUC of 90.7 % in both, and LightGCN a
recall@20 of 27.4 % in both. MUTAG has only 19-20 test graphs per split, so
its accuracies move in steps of 5 points and its gaps are larger (up to 6
points), within 1.5 paired standard errors. Full table:
`graph_results.md <https://github.com/mlx-graphs/mlx-graphs/blob/main/benchmarks/quality/graph_results.md>`_.

Samplers and loaders
~~~~~~~~~~~~~~~~~~~~

PyG's own neighbour samplers, GraphSAINT, ShaDow, ``ClusterData`` and
``knn`` / ``radius`` need compiled extensions (``pyg-lib``,
``torch_sparse``, ``torch_cluster``), so the samplers are checked against
references that don't:

- **Exactly, in the test suite**: ``RandomNodeLoader``, ``ClusterLoader``,
  ``DynamicBatchSampler`` and ``ImbalancedSampler`` against PyG's
  pure-Python classes; ShaDow subgraphs and full-fan-out ``NeighborLoader``
  against PyG's ``k_hop_subgraph``; ``knn`` / ``radius`` against SciPy's
  KD-tree; ``KNNGraph`` / ``RadiusGraph`` against PyG's transforms.
- **Statistically**: every seed node or item covered exactly once per epoch
  (``NeighborLoader``, ``LinkNeighborLoader``, ``ShaDowKHopSampler``,
  ``RandomNodeLoader``, ``ClusterLoader``); uniform neighbour sampling
  (chi-squared tests on the highest-degree nodes of Cora); class balance of
  ``ImbalancedSampler``; batch budgets of ``DynamicBatchSampler``; and the
  bias of GraphSAINT's normalizations.
- **By training**: a 2-layer GraphSAGE trained with each mini-batch method
  against full-batch training.

Every mini-batch method lands within one standard deviation of full-batch
training (test accuracy in %, 5 seeds):

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Training
     - Cora
     - PubMed
   * - Full batch, PyG
     - 80.3 ± 0.7
     - 76.9 ± 0.8
   * - Full batch, mlx-graphs
     - 80.2 ± 0.7
     - 77.3 ± 0.8
   * - ``NeighborLoader`` [10, 10]
     - 80.3 ± 0.9
     - 76.6 ± 0.4
   * - ``GraphSAINTRandomWalkSampler``
     - 80.3 ± 1.1
     - 77.0 ± 0.9
   * - ``ClusterLoader`` (20 parts, 5 per batch)
     - 79.4 ± 1.4
     - 76.9 ± 0.6
   * - ``ShaDowKHopSampler`` (depth 2, 10)
     - 80.2 ± 0.8
     - 76.9 ± 0.5

For link prediction, ``LinkNeighborLoader`` reaches a test AUC of
84.3 ± 1.4 % on Cora (full batch: 84.4 ± 3.2 % in mlx-graphs, 83.3 ± 1.6 %
in PyG) and 89.2 ± 0.7 % on PubMed (full batch: 90.1 % in both). Full
results:
`sampler_results.md <https://github.com/mlx-graphs/mlx-graphs/blob/main/benchmarks/quality/sampler_results.md>`_.

Temporal graph networks
~~~~~~~~~~~~~~~~~~~~~~~

TGN link prediction on JODIE Wikipedia, ``examples/tgn_jodie.py`` against
PyG's ``examples/tgn.py``, with the same hyperparameters, 8 epochs, seeds 0
and 1:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Run
     - Test AP, seeds 0 / 1
     - Test AUC, seeds 0 / 1
   * - mlx-graphs
     - 0.957 / 0.955
     - 0.953 / 0.951
   * - PyG
     - 0.950 / 0.949
     - 0.946 / 0.948
   * - PyG with the torch-scatter ``scatter_argmax``
     - 0.952 / 0.952
     - 0.949 / 0.952

Without torch-scatter, PyG's fallback ``scatter_argmax`` gives empty groups
the index ``dim_size - 1``, so its ``LastAggregator`` copies an unrelated
message to the nodes that have none instead of zeros. mlx-graphs follows the
torch-scatter semantics, so the comparison also runs PyG with that function
patched. Results:
`results.json <https://github.com/mlx-graphs/mlx-graphs/blob/main/benchmarks/tgn_comparison/results.json>`_.
The :doc:`TGN tutorial <tutorials/notebooks/temporal_graph_networks>`
compares the two in more detail.

GPS on Peptides-func
~~~~~~~~~~~~~~~~~~~~

``GPSConv`` (``examples/graph_gps_peptides.py``) against a PyG twin of the
same model: 4 GPS layers (GINE and 4-head attention), the same data with
random-walk encodings, Adam (lr 1e-3, L2 1e-5), 10 epochs, seeds 0 and 1.
Average precision over the 10 tasks:

.. list-table::
   :header-rows: 1
   :widths: 30 25 20 25

   * - Run
     - Final train loss
     - Best val AP
     - Test AP
   * - PyG, seed 0
     - 0.2522
     - 0.4044
     - 0.3881
   * - PyG, seed 1
     - 0.2613
     - 0.4194
     - 0.4079
   * - mlx-graphs, seed 0
     - 0.2588
     - 0.4447
     - 0.4288
   * - mlx-graphs, seed 1
     - 0.2553
     - 0.4437
     - 0.4268

The full-model gradient check (above) gives the same loss and gradients
within ~1 %. Details:
`benchmarks/gps_peptides <https://github.com/mlx-graphs/mlx-graphs/blob/main/benchmarks/gps_peptides/README.md>`_.

``to_hetero`` on DBLP
~~~~~~~~~~~~~~~~~~~~~

``benchmarks/to_hetero/compare_dblp.py`` trains ``to_hetero(GraphSAGE)`` and
``to_hetero(GAT)`` for author classification on DBLP, full batch, in both
frameworks with the same data, architecture and optimizer (AdamW, lr 0.005,
weight decay 0.001), and reports the test accuracy at the best validation
epoch over several seeds. Mini-batch training (``NeighborLoader``,
``HGTLoader``) runs in mlx-graphs only, since PyG's heterogeneous samplers
need ``pyg-lib`` or ``torch-sparse``. Its results are not committed to the
repository. The models it converts are also checked with copied weights in
``tests/nn/test_to_hetero.py``, including on the full DBLP and IMDB graphs.

Remaining mismatches
~~~~~~~~~~~~~~~~~~~~

Three node classification configurations don't pass.

**PNAConv on Cora (-4.6) and CiteSeer (+6.8).** The gaps have opposite
signs. PNAConv on raw features, with amplification scalers, trains with large
loss spikes in both frameworks: PyG's seed 0 on PubMed jumps to a loss of
3.0, and mlx-graphs' seed 3 to 4.9 near the end. With copied weights on
PubMed, the layer's output and input gradients agree with PyG's to 1e-4 and
1e-8 (relative ~1e-3): the layer computes the same function, which points to
the unstable training rather than to the layer. The PubMed row passes
(-0.2), and so do the PNA model on Cora and CiteSeer (+1.7, +3.1) and PNA on
MUTAG and PROTEINS.

**DirGNNConv(GCNConv) on CiteSeer (+1.8).** mlx-graphs is ahead: 68.7 ±
1.4 % against PyG's 66.9 ± 0.7 %, beyond 1 point and twice PyG's small
standard deviation. The training losses are close (0.006 / 0.008 over epochs
25-75, 0.011 / 0.008 at the end, PyG / mlx-graphs), the Cora (+0.6) and
PubMed (-0.4) rows pass, and with copied weights the layer gives the same
outputs as PyG's ``DirGNNConv`` (``tests/nn/conv/test_dir_gnn_conv.py``). No
cause has been identified.

Bugs found and fixed
~~~~~~~~~~~~~~~~~~~~

Found by the training and sampler comparisons, each fixed with a regression
test:

- ``SAGEConv`` with a non-linear aggregation (``"max"``, ``"min"``,
  ``"std"``, lists, modules) and more input than output features projected
  the features before aggregating (``max(W x)`` instead of PyG's ``W
  max(x)``), a trick only valid for sum and mean. Max error against PyG with
  copied weights: 1.7.
- Lists of aggregations: ``SAGEConv`` crashed, and ``GENConv`` did not
  accept them. Both now size their projections like PyG's. The other layers
  with a projection after the aggregation crashed with a shape error, or for
  ``GATConv`` / ``GATv2Conv`` returned twice as many rows; they now refuse
  concatenating lists with a clear error (``supports_concat_aggr``).
- NaN gradients from L2 norms at zero in ``AGNNConv``, ``MessageNorm`` and
  ``GENConv(msg_norm=True)``: a node whose features are all zero (common
  after ReLU and dropout) made every gradient NaN. Seen as AGNN on PubMed
  going NaN at epoch 145, and GENConv with MessageNorm reaching 66 % instead
  of 79 % on Cora (one seed).
- ``MixHopConv`` stored its per-power layers in a dict that ``load_weights``
  could not load.
- ``AddLaplacianEigenvectorPE`` did not converge on Cora (float64 eigensolver
  on float32 values).
- ``DirGNNConv(GCNConv(...))``, PyG's usual way to build it, raised an error.
- ``DirGNNConv(SAGEConv)`` trained 2-3 points below PyG with twice its
  training loss: ``SAGEConv`` had no ``root_weight``, so both directions kept
  a root projection. ``SAGEConv`` now has PyG's ``root_weight``, and the rows
  pass (+0.9 / +0.7 / +0.5).
- ``TUDataset`` dropped edge labels (e.g. MUTAG's bond types) and did not
  reorder edge features with the edges. MUTAG, PROTEINS and NCI1 now equal
  PyG's arrays.
- ``IMDB`` stored its labels under ``"movies"`` instead of the ``"movie"``
  node type.
- ``HeteroDictLinear`` (used by ``HGTConv``) initialized with Glorot
  weights instead of ``torch.nn.Linear``'s: HGT on DBLP reached 76.9 ± 2.3 %
  against PyG's 71.8 ± 1.8 %. With PyG's initialization it reaches
  73.0 ± 2.5 %.
- ``GINEConv``'s edge projection was also initialized with Glorot weights
  (GINE on MUTAG: 80.0 ± 10.5 % before the fix, 82.0 ± 7.1 % after,
  78.0 ± 13.6 % in PyG).
- GraphSAINT's ``edge_norm`` used the count of the edge's source node, as
  PyG does, instead of its target (the paper's definition). On Cora, the
  median relative error of the normalized aggregation was about 50 % with
  the source count; with the target count it is sampling noise that shrinks
  with ``sample_coverage``.

Other differences from PyG fixed in the same release (see the
`changelog <https://github.com/mlx-graphs/mlx-graphs/blob/main/CHANGELOG.md>`_):

- ``LayerNormalization(mode="graph")`` gave wrong shapes with batch indices,
  and ``mode="node"`` ignored its own ``weight`` and ``bias``. Both now match
  PyG's ``LayerNorm``.
- Max and min aggregations gave the full gradient to every tied value (e.g.
  the identical messages of duplicate edges). They now split it equally, as
  PyG does.
- ``MultiAggregation(mode="std")`` had a NaN gradient for sets of a single
  element.
- ``GIN`` and ``EdgeCNN`` did not apply the model's ``act`` and ``norm`` to
  the MLPs inside their convolutions, as PyG does.

Where mlx-graphs differs from PyG on purpose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **GraphSAINT normalization**: target count in ``edge_norm`` (see above).
- **GraphSAINT edge sampler**: mlx-graphs samples edges with the paper's
  probabilities, :math:`1/\deg(u) + 1/\deg(v)`. PyG's
  ``GraphSAINTEdgeSampler`` computes the inverse weighting, which favours
  edges between hubs. At the same number of edges, the paper's
  probabilities give a 3.5-4.4x smaller standard deviation of the estimated
  aggregation.
- **DynamicBatchSampler** with ``skip_too_big=True`` yields every graph once;
  PyG re-yields the graphs that follow a skipped one. Without
  ``skip_too_big``, the batches are the same as PyG's.
- **TGN's** ``LastAggregator`` gives zeros to nodes without messages, as PyG
  does with torch-scatter (see above).
- **TopK and SAG pooling** break ties between equal scores differently. Both
  are valid, and training quality is the same.


Reproducing
-----------

Install the test and benchmark dependencies (some harnesses also use
scikit-learn), then run:

.. code-block:: bash

   pip install -e '.[test,benchmarks]'

   # layer-level parity (add --datasets for the slow tests that download data)
   pytest tests

   # node classification (--configs gcn arma ... for a subset, --summarize for the table only)
   python benchmarks/quality/node_quality.py

   # graph-level, heterogeneous, link and KG models (--groups tu kge ... for a subset)
   python benchmarks/quality/graph_quality.py

   # hierarchical and dense pooling on PROTEINS (--models diffpool asap ... for a subset)
   python benchmarks/quality/pooling_quality.py

   # samplers and loaders (about 15 minutes)
   python benchmarks/run_capped.py --gb 6 -- python benchmarks/quality/sampler_quality.py \
       --data-dir .mlx_graphs_data --seeds 0 1 2 3 4

   # TGN on JODIE Wikipedia
   python benchmarks/tgn_comparison/compare.py --data <data_dir> --epochs 8

   # GPS on Peptides-func: export the data, train both, check the gradients
   python benchmarks/gps_peptides/export_npz.py <data_dir>
   python examples/graph_gps_peptides.py --data_dir <data_dir> --epochs 10 --seed 0
   python benchmarks/gps_peptides/gps_pyg.py --data_dir <data_dir> --epochs 10 --seed 0
   python benchmarks/gps_peptides/grad_parity.py <data_dir>

   # to_hetero on DBLP
   python benchmarks/to_hetero/compare_dblp.py --data_dir .mlx_graphs_data

.. note::

    The harnesses run every configuration in its own process under a hard
    memory cap, and kill it above the cap: 6 GB by default for the node and
    graph harnesses (``--memory-gb``), 4 GB for the TGN and ``to_hetero``
    comparisons. The memory is the process' physical footprint, as shown by
    Activity Monitor, which, unlike RSS, includes the GPU (Metal)
    allocations of MLX and PyTorch MPS. Run any other script the same
    way with ``python benchmarks/run_capped.py --gb <cap> -- <command>``.
    Mini-batches of varying shapes make MLX's buffer cache grow: the
    sampler harness caps it with ``mx.set_cache_limit``.

The node and graph harnesses append one line per run to
``node_results.jsonl`` / ``graph_results.jsonl`` and skip the runs already
there, so an interrupted run resumes where it stopped.


Not covered
-----------

- **Training quality on large graphs** (Amazon, ogbn): out of the memory
  budget of these checks. They are part of the speed :doc:`benchmarks`,
  which also record accuracy.
- **Mini-batch training in the node classification harness**, which trains
  full batch only. Mini-batch training is checked for GraphSAGE by the
  sampler harness, and for ``to_hetero`` models in mlx-graphs only.
- **PyG's compiled samplers and operations**: PyG's neighbour samplers,
  GraphSAINT, ShaDow, ``ClusterData`` (METIS), ``DynamicEdgeConv`` and
  ``knn_graph`` need ``pyg-lib``, ``torch_sparse`` or ``torch_cluster``,
  which were not installed. The samplers were checked against references
  instead, and ``ClusterLoader`` was trained with the dependency-free
  balanced partitioner, not METIS.
- **NCI1** graph classification, which did not fit in the time budget.
- **Converged knowledge graph embeddings**: the KGE models trained for 20-40
  epochs instead of PyG's 500, so the comparison covers the early training
  trajectories, not converged models.
- **Attention variants of GeneralConv** on CiteSeer and PubMed (Cora only).
- **Compilation**: ``mx.compile`` for mlx-graphs and ``torch.compile`` for
  PyG.
