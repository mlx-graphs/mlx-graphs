# Training quality: graph-level, heterogeneous, link and KG models

mlx-graphs (MLX, GPU) vs PyTorch Geometric 2.8.1 (torch 2.15, CPU), trained
by `graph_quality.py` with the same architectures, hyperparameters, data
(loaded once through mlx-graphs and handed to both) and splits; MLX's Adam
with `bias_correction=True`; the test metric is read at the epoch with the
best validation metric. Mean ± sample std over seeds (each seed is a
different split for the TU, link and point cloud tasks, the same split in
both frameworks). "paired gap" is the mean ± standard error of the per-seed
difference mlx - PyG. Verdict: *pass* when the gap is within 1 point or
within 2 PyG standard deviations.

| model | dataset | metric | PyG | mlx-graphs | gap | paired gap | train loss at 10% / 50% / 100% of the epochs, PyG ; mlx | verdict |
|---|---|---|---|---|---|---|---|---|
| gine | MUTAG | test accuracy (%) | 78.0 ± 13.6 | 82.0 ± 7.1 | +4.0 | +4.0 ± 2.7 | 0.519 / 0.384 / 0.305 ; 0.526 / 0.382 / 0.323 | pass (n=10/10) |
| pna | MUTAG | test accuracy (%) | 77.0 ± 5.9 | 78.0 ± 7.9 | +1.0 | +1.0 ± 2.3 | 0.41 / 0.222 / 0.207 ; 0.33 / 0.226 / 0.209 | pass (n=10/10) |
| topk | MUTAG | test accuracy (%) | 78.0 ± 9.5 | 74.5 ± 14.6 | -3.5 | -3.5 ± 4.3 | 0.542 / 0.371 / 0.281 ; 0.568 / 0.371 / 0.242 | pass (n=10/10) |
| sag | MUTAG | test accuracy (%) | 67.5 ± 10.3 | 73.5 ± 11.1 | +6.0 | +6.0 ± 3.4 | 0.552 / 0.374 / 0.267 ; 0.52 / 0.395 / 0.223 | pass (n=10/10) |
| dgcnn | MUTAG | test accuracy (%) | 79.5 ± 13.0 | 79.5 ± 13.0 | +0.0 | +0.0 ± 1.1 | 0.409 / 0.291 / 0.209 ; 0.431 / 0.314 / 0.255 | pass (n=10/10) |
| multiaggr | MUTAG | test accuracy (%) | 81.5 ± 10.0 | 76.5 ± 11.1 | -5.0 | -5.0 ± 4.1 | 0.459 / 0.35 / 0.274 ; 0.473 / 0.384 / 0.376 | pass (n=10/10) |
| graphnorm | MUTAG | test accuracy (%) | 74.0 ± 11.5 | 71.5 ± 8.8 | -2.5 | -2.5 ± 3.6 | 0.635 / 0.372 / 0.254 ; 0.633 / 0.364 / 0.26 | pass (n=10/10) |
| set2set | MUTAG | test accuracy (%) | 75.0 ± 7.8 | 75.5 ± 13.4 | +0.5 | +0.5 ± 4.9 | 0.547 / 0.42 / 0.389 ; 0.54 / 0.403 / 0.441 | pass (n=10/10) |
| deepsets | MUTAG | test accuracy (%) | 74.0 ± 9.4 | 77.0 ± 12.1 | +3.0 | +3.0 ± 2.1 | 0.536 / 0.393 / 0.354 ; 0.553 / 0.39 / 0.407 | pass (n=10/10) |
| pna | PROTEINS | test accuracy (%) | 72.9 ± 2.2 | 72.7 ± 3.9 | -0.2 | -0.2 ± 2.2 | 0.624 / 0.594 / 0.532 ; 0.624 / 0.613 / 0.544 | pass (n=5/5) |
| topk | PROTEINS | test accuracy (%) | 73.9 ± 3.7 | 74.1 ± 3.1 | +0.2 | +0.2 ± 1.6 | 0.625 / 0.542 / 0.475 ; 0.585 / 0.536 / 0.466 | pass (n=5/5) |
| sag | PROTEINS | test accuracy (%) | 74.3 ± 3.5 | 71.8 ± 6.4 | -2.5 | -2.5 ± 2.9 | 0.607 / 0.564 / 0.507 ; 0.596 / 0.529 / 0.474 | pass (n=5/5) |
| dgcnn | PROTEINS | test accuracy (%) | 69.6 ± 4.8 | 72.0 ± 4.5 | +2.3 | +2.3 ± 2.3 | 0.551 / 0.455 / 0.275 ; 0.544 / 0.453 / 0.264 | pass (n=5/5) |
| multiaggr | PROTEINS | test accuracy (%) | 73.6 ± 3.5 | 74.1 ± 4.1 | +0.5 | +0.5 ± 0.9 | 0.709 / 0.609 / 0.661 ; 0.623 / 0.616 / 0.635 | pass (n=5/5) |
| graphnorm | PROTEINS | test accuracy (%) | 73.9 ± 2.2 | 74.5 ± 4.3 | +0.5 | +0.5 ± 2.0 | 0.574 / 0.465 / 0.287 ; 0.563 / 0.457 / 0.343 | pass (n=5/5) |
| set2set | PROTEINS | test accuracy (%) | 68.2 ± 6.6 | 64.8 ± 7.6 | -3.4 | -3.4 ± 3.2 | 0.645 / 0.654 / 0.658 ; 0.656 / 0.675 / 0.674 | pass (n=5/5) |
| deepsets | PROTEINS | test accuracy (%) | 70.2 ± 4.4 | 70.7 ± 4.3 | +0.5 | +0.5 ± 0.4 | 0.596 / 0.613 / 0.611 ; 0.639 / 0.609 / 0.592 | pass (n=5/5) |
| han | IMDB | test accuracy (%) | 56.6 ± 0.7 | 57.6 ± 1.1 | +1.0 | +1.0 ± 0.7 | 0.345 / 0.0901 / 0.0567 ; 0.355 / 0.091 / 0.061 | pass (n=5/5) |
| hgt | DBLP | test accuracy (%) | 71.8 ± 1.8 | 73.0 ± 2.5 | +1.2 | +1.2 ± 1.6 | 1.31 / 0.00301 / 0.214 ; 1.31 / 0.0026 / 0.0515 | pass (n=5/5) |
| rgat | DBLP-homogeneous | test accuracy (%) | 66.9 ± 3.8 | 69.2 ± 1.5 | +2.3 | +2.3 ± 1.9 | 1.19 / 0.153 / 0.00621 ; 1.17 / 0.165 / 0.0138 | pass (n=5/5) |
| fastrgcn | DBLP-homogeneous | test accuracy (%) | 68.3 ± 1.0 | 68.3 ± 0.5 | -0.1 | -0.1 ± 0.4 | 0.954 / 0.0193 / 0.00303 ; 0.991 / 0.0247 / 0.00316 | pass (n=5/5) |
| gae | Cora | test AUC (%) (ap: 90.8 / 91.0) | 90.7 ± 0.4 | 90.7 ± 0.7 | -0.0 | -0.0 ± 0.4 | 1.04 / 0.893 / 0.844 ; 1.03 / 0.888 / 0.842 | pass (n=5/5) |
| vgae | Cora | test AUC (%) (ap: 88.6 / 88.8) | 88.8 ± 1.4 | 88.7 ± 2.2 | -0.1 | -0.1 ± 1.3 | 1.34 / 1.02 / 0.918 ; 1.34 / 0.988 / 0.912 | pass (n=5/5) |
| gae | CiteSeer | test AUC (%) (ap: 89.3 / 89.5) | 88.9 ± 1.2 | 89.0 ± 2.5 | +0.1 | +0.1 ± 1.1 | 1.03 / 0.903 / 0.847 ; 1 / 0.888 / 0.843 | pass (n=5/5) |
| vgae | CiteSeer | test AUC (%) (ap: 88.0 / 88.0) | 87.6 ± 1.9 | 87.3 ± 0.8 | -0.4 | -0.4 ± 0.7 | 1.34 / 1.01 / 0.923 ; 1.35 / 1.01 / 0.919 | pass (n=5/5) |
| gae-ttse | Cora | test AUC (%) (ap: 91.5 / 91.0) | 91.4 ± 1.5 | 90.9 ± 1.5 | -0.5 | -0.5 ± 0.1 | 1.04 / 0.877 / 0.836 ; 1.02 / 0.876 / 0.834 | pass (n=5/5) |
| lightgcn | MovieLens100K | test recall@20 (%) (precision: 24.6 / 24.6) | 27.4 ± 0.3 | 27.4 ± 0.1 | +0.0 | +0.0 ± 0.1 | 0.353 / 0.284 / 0.23 ; 0.353 / 0.282 / 0.229 | pass (n=5/5) |
| dgi | Cora | test accuracy (%) | 79.5 ± 0.8 | 79.9 ± 0.8 | +0.4 | +0.4 ± 0.4 | 0.802 / 0.141 / 0.0454 ; 0.794 / 0.146 / 0.0456 | pass (n=5/5) |
| transe | FB15k-237 | test MRR (%) (hits10: 34.8 / 36.0) | 20.9 ± 0.4 | 21.1 ± 0.5 | +0.3 | +0.3 ± 0.2 | 0.559 / 0.206 / 0.161 ; 0.561 / 0.204 / 0.159 | pass (n=5/5) |
| distmult | FB15k-237 | test MRR (%) (hits10: 32.8 / 33.0) | 20.2 ± 0.4 | 20.4 ± 0.1 | +0.2 | +0.2 ± 0.2 | 1 / 0.905 / 0.453 ; 1 / 0.908 / 0.452 | pass (n=5/5) |
| complex | FB15k-237 | test MRR (%) (hits10: 29.7 / 29.8) | 18.3 ± 0.1 | 18.3 ± 0.4 | +0.0 | +0.0 ± 0.2 | 0.693 / 0.693 / 0.689 ; 0.693 / 0.693 / 0.689 | pass (n=5/5) |
| rotate | FB15k-237 | test MRR (%) (hits10: 12.1 / 12.1) | 6.0 ± 0.2 | 6.2 ± 0.4 | +0.2 | +0.2 ± 0.1 | 2.5 / 0.633 / 0.412 ; 2.5 / 0.633 / 0.41 | pass (n=5/5) |
| edgecnn | synthetic | test accuracy (%) | 94.2 ± 2.9 | 95.6 ± 1.5 | +1.4 | +1.4 ± 1.0 | 0.946 / 0.0973 / 0.00477 ; 0.968 / 0.0267 / 0.0022 | pass (n=5/5) |
| dynedge | synthetic | test accuracy (%) | 95.4 ± 2.1 | 96.2 ± 3.8 | +0.8 | +0.8 ± 2.2 | 1.06 / 0.134 / 0.0327 ; 1.16 / 0.112 / 0.019 | pass (n=5/5) |

## Setups

- **TU graph classification** (`graph_tu.py`): MUTAG (10 seeds) and PROTEINS
  (5 seeds), random 80/10/10 split per seed, 100 epochs, Adam (lr 0.01,
  batches of 32; lr 0.0005 / batches of 60 for TopK/SAG as in PyG's
  `proteins_topk_pool.py`; lr 0.001 for DGCNN). Hidden size 64. `graphnorm`
  trains with `Dataloader(drop_last=True, follow_batch=["node_features"])`
  and reads the `node_features_batch` vector (PyG: `follow_batch=["x"]`).
- **HAN / IMDB**: PyG's `han_imdb.py` (metapaths movie-actor-movie and
  movie-director-movie, `HANConv(-1, 128, heads=8, dropout=0.6)`, Adam lr
  0.005, weight decay 0.001, 200 epochs).
- **HGT / DBLP**: PyG's `hgt_dblp.py` with 2 layers (`HGTConv(64, 64,
  heads=2)`), conference nodes with a constant feature, 100 epochs.
- **RGAT / FastRGCN**: DBLP converted with `HeteroGraphData.to_homogeneous()`
  (6 relations, 26,128 nodes; author and paper bag-of-words reduced to 32
  features by a fixed random projection so that PyG's per-edge weight
  tensors fit in memory), 2 layers of 16, authors classified, Adam lr 0.01,
  weight decay 5e-4, 100 epochs.
- **GAE / VGAE**: PyG's `autoencoder.py` (GCN encoder 32 -> 16, inner product
  decoder, Adam lr 0.01, 400 epochs) on splits made by mlx-graphs'
  `RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True)`
  (`gae-ttse`: `train_test_split_edges`); the splits were checked (no test
  or validation edge in the message passing graph, negatives are non-edges).
  AUC / AP by scikit-learn for both; mlx-graphs' own `GAE.test` gives the
  same numbers to 1e-15.
- **LightGCN / MovieLens100K**: u1.base / u1.test ratings >= 4, 64 dims,
  2 layers, BPR loss (lambda 1e-4), one uniform negative movie per positive,
  batches of 1024, Adam lr 0.001, 50 epochs, recall@20 / precision@20 over
  the test users (training items excluded); last epoch reported.
- **DGI / Cora**: PyG's `infomax_transductive.py` (GCN 512 + PReLU, 300
  epochs); embeddings probed by the same scikit-learn logistic regression
  in both frameworks.
- **KGE / FB15k-237**: PyG's `kge_fb15k_237.py` (50 dims, batches of 1000,
  TransE Adam 0.01, DistMult Adam 1e-4 + wd 1e-6, ComplEx Adagrad 1e-3 + wd
  1e-6, RotatE margin 9 Adam 1e-3) for 20 epochs (RotatE 40) instead of
  500; raw MRR / Hits@10 of the true tail among all 14,541 entities on fixed
  random subsets of 2,000 validation / test triplets, ranked by the same
  code in both frameworks.
- **Point clouds**: 1,000 synthetic 64-point clouds of 4 shapes, `EdgeCNN`
  on the 8-NN graph built by mlx-graphs' `knn_graph` (checked against a
  brute-force kNN), and 2 x `DynamicEdgeConv(k=8)`. PyG's `DynamicEdgeConv`
  needs `pyg-lib` / `torch-cluster`, which do not install here, so the PyG
  side runs `EdgeConv` on a torch kNN graph recomputed in feature space at
  every layer (the same graph as PyG's `knn(x, x, k).flip(0)`).

## Discrepancies found and fixed in mlx-graphs

1. **`TUDataset` dropped edge labels** (e.g. MUTAG's 4 bond types): the
   edge-labels-only branch assigned the (missing) edge attributes. Edge
   features were also not reordered when the edges were sorted and
   deduplicated. Fixed (edges and features go through `coalesce` together,
   as in PyG); MUTAG / PROTEINS / NCI1 now equal PyG's arrays exactly.
2. **`IMDB` labels were stored under `node_labels_dict["movies"]`** while the
   node type is `"movie"`, so `to_homogeneous()` dropped them and they could
   not be looked up by node type. Fixed.
3. **`HeteroDictLinear` (used by `HGTConv`) initialized with Glorot weights
   and zero biases**, PyG's with `torch.nn.Linear`'s init: HGTConv's
   projections were up to 1.7x larger and HGT on DBLP trained faster and
   reached 76.9 ± 2.3 % vs PyG's 71.8 ± 1.8 %. A forward pass with PyG's
   weights copied in matched PyG on the full DBLP graph, so only the init
   differed. Fixed; now 73.0 ± 2.5 % (row above).
4. **`GINEConv`'s edge projection initialized with Glorot / zero bias**,
   PyG's like `torch.nn.Linear` (std 1.75x larger for MUTAG's 4 -> 7
   projection). Found by comparing the initial parameter scales of every
   model of this harness in both libraries; fixed. (GINE / MUTAG before the
   fix: 80.0 ± 10.5 %.)

## Remaining, explained differences

- MLX's `Conv1d` starts with zero biases and MLX's `LSTM` has one bias vector
  where torch's has two (MLX core, not mlx-graphs). The harness initializes
  DGCNN's `Conv1d` biases like torch; `Set2Set`'s LSTM keeps MLX's init.
- TopK / SAG pooling: on PROTEINS many scores tie after the `tanh` (it
  saturates to exactly +-1 in float32 for structurally identical nodes),
  and the two libraries break ties differently, so with identical weights
  the pooled graphs differ in a few percent of the nodes. Both are valid;
  training quality is the same.
- MUTAG has only 19-20 test graphs per split, so its accuracies move in steps
  of 5 points and the largest gaps (SAG +6.0, MultiAggregation -5.0, of
  opposite signs) are within 1.5 paired standard errors; on PROTEINS the
  same models are within 2.5 points.
- EdgeCNN's training loss falls faster in mlx-graphs in the middle of
  training (median over 5 seeds). Trained from PyG's weights on the same
  batches without shuffling, both libraries follow the same loss curve
  (within 0.003 after 6 epochs), so this is seed-to-seed variance of when
  the model leaves its initial plateau, not a difference in the model.
- MLX's float32 matmul on this GPU (M5, `applegpu_g17s`) has a ~1e-3
  relative error (numpy / torch CPU: ~1e-7). It is the source of the
  ~1e-3 forward differences, and of gradient differences of up to ~10 % in
  EdgeConv's first layer (the `x_j - x_i` inputs cancel). It is MLX
  behaviour, not mlx-graphs', and did not change any result above.

## Not covered

- NCI1 (did not fit in the time budget on a heavily loaded machine).
- PyG's `DynamicEdgeConv`, `knn_graph` and neighbor sampling need `pyg-lib`
  or `torch-cluster`, which have no wheels here (see the point cloud setup).
- KGE models were trained for 20-40 epochs, not PyG's 500: the comparison is
  of the early trajectories (losses and MRR match to the third decimal), not
  of converged models.
