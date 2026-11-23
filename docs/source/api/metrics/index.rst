.. _metrics:

Metrics
=======

Link prediction and ranking metrics (``mlx_graphs.metrics``, as
``torch_geometric.metrics``). They are computed on the host with NumPy from
MLX or NumPy arrays, accumulate over mini-batches with ``update`` and return a
scalar ``mx.array`` from ``compute``.

.. currentmodule:: mlx_graphs.metrics


.. autosummary::
	:toctree: _autosummary

	roc_auc_score
	average_precision_score
	tgb_mrr_hits
	tgb_ndcg


.. autosummary::
	:toctree: _autosummary
	:template: custom-class-template.rst

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
	TGBEvaluator
