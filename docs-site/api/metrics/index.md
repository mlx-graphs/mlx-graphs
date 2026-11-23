# Metrics { #mlx_graphs.metrics }

Link prediction and ranking metrics (`mlx_graphs.metrics`, as
`torch_geometric.metrics`). They are computed on the host with NumPy from
MLX or NumPy arrays, accumulate over mini-batches with `update` and return a
scalar `mx.array` from `compute`.

## Functions

<!-- api mlx_graphs.metrics -->
roc_auc_score
average_precision_score
<!-- /api -->

## Link prediction metrics

<!-- api mlx_graphs.metrics -->
LinkPredMetric
LinkPredMetricCollection
LinkPredMetricData
LinkPredPrecision
LinkPredRecall
LinkPredF1
LinkPredMAP
LinkPredNDCG
LinkPredMRR
LinkPredHitRatio
LinkPredCoverage
LinkPredDiversity
LinkPredPersonalization
LinkPredAveragePopularity
<!-- /api -->
