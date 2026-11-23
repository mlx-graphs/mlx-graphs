Changelog
=========

Unreleased
----------

Documentation
~~~~~~~~~~~~~

- A cybersecurity tutorial with four cases: host intrusion detection on DARPA
  TC E3 CADETS provenance graphs (Orthrus, Velox), anti-money laundering on
  Elliptic, network intrusion detection on NF-UNSW-NB15-v3 (E-GraphSAGE) and
  malware classification on MalNet-Tiny (GIN, GraphGPS).
- A Graph RAG tutorial that runs the released G-reasoner checkpoint with a
  custom Metal kernel and uses it as the retriever of a local LLM.
- The G-reasoner and QM9 notebooks are now listed with the other tutorials.
  The Examples page only lists the scripts of the ``examples/`` folder.

0.2.0: PyG parity
------------------

This release brings mlx-graphs close to PyTorch Geometric in features and API:
about 235 new public classes and functions, samplers and loaders for mini-batch
training, heterogeneous and temporal graphs, graph transformers, and 18 new
datasets. Every addition was checked against PyG: layers with copied weights
(outputs and gradients), samplers statistically, and models by training them in
both frameworks on the same data (``benchmarks/quality/``).

Highlights
~~~~~~~~~~

- **Samplers and loaders**: ``LinkNeighborLoader``, ``HGTLoader``, ``ClusterData`` /
  ``ClusterLoader``, the GraphSAINT samplers, ``ShaDowKHopSampler``,
  ``RandomNodeLoader``, ``ImbalancedSampler``, ``DynamicBatchSampler``,
  ``DataListLoader``, ``DenseDataLoader``; ``Dataloader`` gains ``drop_last``,
  ``follow_batch``, ``sampler``, ``batch_sampler``, ``transform`` and ``collate_fn``.
- **Heterogeneous graphs**: ``to_hetero`` and ``to_hetero_with_bases`` turn a
  homogeneous model into a heterogeneous one, as in PyG (without ``torch.fx``:
  see the docstring for the supported model patterns); ``HANConv``, ``HGTConv``,
  ``RGATConv``, ``FastRGCNConv``; the ``OGBMAG`` dataset.
- **Temporal graphs**: ``TemporalData``, ``TemporalDataLoader``, ``TGNMemory`` and
  its building blocks, ``LastNeighborLoader``, ``JODIEDataset``, and the new
  ``mlx_graphs.metrics`` package (link prediction ranking metrics, AUC, AP).
- **Graph transformers**: ``GPSConv``, ``SGFormer``, ``Polynormer``,
  ``MultiheadAttention``, ``PerformerAttention``, positional encodings
  (``LapPEEncoder``, ``RWSEEncoder``, ``PositionalEncoding``, ``TemporalEncoding``),
  ``LRGBDataset`` and ``GNNBenchmarkDataset``.
- **Aggregations**: the ``mlx_graphs.nn.aggr`` module (sum, mean, max, min, mul,
  var, std, median, quantile, softmax, power mean, LSTM, GRU, set transformer,
  graph multiset transformer, sort, MLP, LCM, deep sets, equilibrium, degree
  scaler, ...). ``MessagePassing(aggr=...)`` accepts a name, a list of
  aggregations (concatenated or combined with ``aggr_kwargs={"mode": ...}``) or an
  ``Aggregation`` module.
- **PyG names**: ``GraphData`` has the ``x``, ``edge_attr`` and ``y`` aliases,
  ``GraphDataBatch`` has ``batch`` and ``ptr``, and the norm layers are also exported
  under their PyG names (``GraphNorm``, ``PairNorm``, ...).
- **Documentation**:

  - new tutorials, each compared with PyG on accuracy and runtime: mini-batch
    training on large graphs, link prediction, temporal graph networks,
    heterogeneous graphs with ``to_hetero``, and graph transformers with GPS;
    node2vec is now a single tutorial;
  - new pages: "Creating message passing layers" (user guide) and
    "Validation against PyG"; an updated "Coming from PyTorch Geometric";
    an overview figure at the top of every tutorial.

New API
~~~~~~~

- ``mlx_graphs.nn``:

  - Message passing layers:

    - ``AGNNConv``, ``ARMAConv``, ``CGConv``, ``ClusterGCNConv``, ``DirGNNConv``,
      ``DynamicEdgeConv``, ``ECConv``, ``FAConv``, ``FastRGCNConv``, ``FiLMConv``;
    - ``GCN2Conv``, ``GENConv``, ``GINEConv``, ``GPSConv``, ``GatedGraphConv``,
      ``GeneralConv``, ``HANConv``, ``HGTConv``, ``LEConv``, ``LGConv``, ``MFConv``;
    - ``MixHopConv``, ``NNConv``, ``PNAConv``, ``RGATConv``, ``ResGatedGraphConv``,
      ``SSGConv``.

  - Models:

    - ``PNA``, ``EdgeCNN``, ``DeepGCNLayer``, ``LINKX``, ``LightGCN``, ``LabelPropagation``,
      ``CorrectAndSmooth``, ``MaskLabel``, ``RECT_L``;
    - graph autoencoders: ``GAE``, ``VGAE``, ``ARGA``, ``ARGVA``, ``InnerProductDecoder``;
    - ``DeepGraphInfomax``;
    - knowledge graph embeddings: ``KGEModel``, ``TransE``, ``DistMult``, ``ComplEx``,
      ``RotatE``, ``KGTripletLoader``;
    - ``SGFormer``, ``Polynormer``, ``TGNMemory``.

  - Norms: ``BatchNorm``, ``InstanceNorm``, ``LayerNorm``, ``GraphNorm``, ``PairNorm``,
    ``DiffGroupNorm``, ``MessageNorm``, ``MeanSubtractionNorm``, ``GraphSizeNorm``,
    ``HeteroBatchNorm``, ``HeteroLayerNorm``.
  - Pooling and graph construction: ``TopKPooling``, ``SAGPooling``,
    ``global_sort_pool``, ``knn``, ``knn_graph``, ``radius``, ``radius_graph``.

- ``mlx_graphs.transforms``: ``AddLaplacianEigenvectorPE``, ``AddRemainingSelfLoops``,
  ``FeaturePropagation``, ``IndexToMask``, ``KNNGraph``, ``LineGraph``,
  ``LocalDegreeProfile``, ``MaskToIndex``, ``NormalizeFeatures``, ``RadiusGraph``,
  ``RemoveDuplicatedEdges``, ``RemoveTrainingClasses``, ``SIGN``,
  ``SVDFeatureReduction``, ``TargetIndegree``, ``ToDense``, ``TwoHop``, ``VirtualNode``.
- ``mlx_graphs.utils``:

  - graph generators;
  - subgraph, mask and self-loop helpers;
  - negative sampling for batches;
  - ``get_ppr``;
  - ``assortativity``, ``normalized_cut``;
  - scipy conversions;
  - the ``scatter`` reductions ``var``, ``std``, ``mul``, ``median`` and ``quantile``.

- ``mlx_graphs.datasets``:

  - citation and web graphs: ``CitationFull``, ``CoraFull``, ``WikiCS``, ``WebKB``,
    ``WikipediaNetwork``, ``Actor``, ``HeterophilousGraphDataset``;
  - knowledge graphs: ``FB15k_237``, ``WordNet18``, ``WordNet18RR``;
  - synthetic graphs: ``FakeDataset``, ``FakeHeteroDataset``;
  - ``JODIEDataset``, ``LRGBDataset``, ``GNNBenchmarkDataset``, ``OGBMAG``;
  - a torch-free reader for ``.pt`` files (``load_torch_file``), which uses a
    restricted unpickler.

- ``Dataset``, ``GraphData`` and ``HeteroGraphData`` gain PyG conveniences, among
  them ``num_classes`` (inferred as in PyG), ``shuffle``, ``to_homogeneous`` and
  the feature counts.

Changes that can affect existing code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Self-loops now default to PyG's behaviour.**

  - ``GCNConv`` defaults to ``add_self_loops=True`` (it was ``False``).
  - ``GATConv`` and ``GATv2Conv`` gain ``add_self_loops=True`` and
    ``fill_value="mean"``.
  - To keep the previous results, pass ``add_self_loops=False``.

- **``GIN``**: the ``act`` and ``norm`` arguments now also apply to the MLP inside
  every ``GINConv``, as in PyG. Before, that MLP always used ReLU and no
  normalization.

  - Models with the defaults (``act="relu"``, ``norm=None``) are unchanged.
  - With ``norm="batch"`` or ``"layer"``, the MLPs gain norm layers, so weights
    saved before this change will not load as-is.

- **``GAT``**: ``dropout`` now also applies to the attention coefficients of
  every ``GATConv`` / ``GATv2Conv``, as in PyG's ``GAT``. Before, it only
  applied between layers. Models with ``dropout=0`` are unchanged.
- **``LayerNormalization``**:

  - ``mode="graph"`` (the default) was broken and now matches PyG's ``LayerNorm``.
  - ``mode="node"`` now uses the layer's own ``weight`` and ``bias``. Its unused
    ``layernorm.*`` parameters are gone, so older checkpoints do not load
    strictly.

- **Index dtypes are int32 throughout.** This covers:

  - the ``n_id``, ``e_id`` and ``edge_index`` returned by ``NeighborLoader``;
  - ``GraphDataBatch.batch_indices``, ``batch`` and ``ptr``;
  - the ``follow_batch`` vectors.

  Indexing with them works as before. int64 is used only for graphs with more
  than 2^31 - 1 nodes or edges.

- **``GraphDataBatch`` indexes like a Python list of graphs.**

  - Slices are clipped instead of raising: ``batch[:101]`` on a smaller batch, or
    ``batch[3:2]``, which gives ``[]``.
  - Negative steps work.
  - ``batch[num_graphs]`` raises ``IndexError``.
  - Index arrays, lists and boolean masks are accepted.

- **``MessagePassing(aggr="softmax")``** is now the softmax aggregation
  (``SoftmaxAggregation``), as in PyG and as ``aggr=["softmax", ...]`` already was.
  Plain string aggregations given with ``aggr_kwargs`` now use them instead of
  ignoring them.

- **Max and min aggregations split the gradient equally among tied values**, as
  PyG does (e.g. with duplicate edges). Before, every tied value got the full
  gradient. The backward pass of the fused max/min kernels is slower as a
  result; forward passes are unchanged.

- **Initialization**: ``HeteroDictLinear`` now initializes like
  ``torch.nn.Linear``, as in PyG, instead of Glorot.

Fixes to previously released code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``SAGEConv`` with a max, min, std or list aggregation and more input than
  output features projected before aggregating (``max(Wx)`` instead of
  ``W max(x)``).
- ``SAGEConv`` gains PyG's ``root_weight``. Without it, ``DirGNNConv(SAGEConv)``
  kept a root projection in both directions and trained 2-3 points below PyG.
- ``TUDataset`` dropped edge labels.
- ``IMDB`` keyed its labels by ``"movies"`` instead of the ``"movie"`` node type.
- ``NeighborLoader`` with out-of-range seed nodes read past the sampler's
  buffers. It now raises ``IndexError``.
- Batching graphs whose first graph has no nodes broke the edge offsets.
- Unbatching turned int32 ``edge_index`` into int64.
- QM7b downloads failed: the old http URL returns 400. It now uses PyG's
  https URL.
- Tar extraction is now protected against path traversal and symlink escapes.

Tips when porting PyG code
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``mlx.optimizers.Adam`` has no bias correction by default, unlike
  ``torch.optim.Adam``. Use ``Adam(learning_rate=..., bias_correction=True)`` to
  train the same way.
- Batches of graphs of varying sizes create arrays of many different shapes.
  To bound memory, cap MLX's buffer cache with ``mx.set_cache_limit(...)``.

Known limitations
~~~~~~~~~~~~~~~~~

- ``to_hetero``:

  - It does not detect operations that mix rows across node types inside the
    model's ``__call__`` (e.g. ``x.mean(0)``, sorting). PyG traces these per node
    type.
  - Copies without a ``reset_parameters()`` method are re-initialized by
    shuffling their weights.

- GPS, SGFormer, Polynormer and the LSTM, GRU, set transformer and sort
  aggregations read the batch layout on the host. They work with
  ``nn.value_and_grad`` but not with ``mx.compile``, unless the precomputed
  ``layout=`` argument is given (GPS, SGFormer, Polynormer).
- ``TGNMemory``'s memory is not saved by ``save_weights``. Use ``memory_state()`` and
  ``load_memory_state()``.
- MLX 0.32.4 crashes when a compiled activation is applied to an empty matmul
  result. ``CGConv``, ``NNConv`` and ``EdgeConv`` are guarded against it. Most layers
  still fail on graphs with zero nodes.
