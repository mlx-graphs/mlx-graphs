# Explainability { #mlx_graphs.explain }

Explain the predictions of a GNN: which nodes, edges and node features drove
them (`mlx_graphs.explain`, as `torch_geometric.explain`). An `Explainer`
wraps a trained model and an algorithm, and returns an `Explanation` that
holds a node mask and an edge mask.

```python
from mlx_graphs.explain import Explainer
from mlx_graphs.explain.algorithm import GNNExplainer

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=dict(
        mode="multiclass_classification", task_level="node", return_type="raw"
    ),
)
explanation = explainer(edge_index, node_features, index=10)
explanation.threshold("topk_hard", 10).get_explanation_subgraph()
```

The model is called as `model(edge_index, node_features, ...)`. To explain
edges, its `__call__` must accept an `edge_weights` keyword and pass it to
its message passing layers. The plots need `pip install "mlx-graphs[viz]"`.
A full run on BA-Shapes is in
[examples/explain_ba_shapes.py](https://github.com/mlx-graphs/mlx-graphs/blob/main/examples/explain_ba_shapes.py).

## Explainer

<!-- api mlx_graphs.explain -->
Explainer
Explanation
ExplainerConfig
ModelConfig
ThresholdConfig
<!-- /api -->

## Algorithms

PyG runs gradient methods through Captum (`CaptumExplainer`); MLX has no
Captum, so `Saliency` and `IntegratedGradients` are implemented natively.

<!-- api mlx_graphs.explain.algorithm -->
ExplainerAlgorithm
GNNExplainer
Saliency
IntegratedGradients
<!-- /api -->

## Metrics

<!-- api mlx_graphs.explain.metric -->
fidelity
characterization_score
groundtruth_auc
<!-- /api -->

## Datasets

Synthetic benchmarks with ground-truth explanations.

<!-- api mlx_graphs.explain.datasets -->
ba_shapes
<!-- /api -->

## Visualization

<!-- api mlx_graphs.explain.visualization -->
visualize_graph
visualize_feature_importance
<!-- /api -->
