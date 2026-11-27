"""The :class:`Explainer` entry point, as ``torch_geometric.explain.Explainer``."""

from typing import Any, Callable, Optional, Union

import mlx.core as mx

from mlx_graphs.explain.algorithm.base import ExplainerAlgorithm, Index, as_2d, forward
from mlx_graphs.explain.config import ExplainerConfig, ModelConfig, ThresholdConfig
from mlx_graphs.explain.explanation import Explanation


class Explainer:
    """Explains predictions of an mlx-graphs model.

    The model is called as ``model(edge_index, node_features, **kwargs)``
    (mlx-graphs argument order). To explain edges, it must also accept an
    ``edge_weights`` keyword and pass it to its message passing layers.

    Args:
        model: the model to explain.
        algorithm: the explanation algorithm, e.g. :class:`GNNExplainer`.
        explanation_type: ``"model"`` or ``"phenomenon"``.
        model_config: a :class:`ModelConfig` or a dict of its arguments.
        node_mask_type: ``None``, ``"object"``, ``"attributes"`` or
            ``"common_attributes"``.
        edge_mask_type: ``None`` or ``"object"``.
        threshold_config: optional :class:`ThresholdConfig` or dict
            (``threshold_type``, ``value``) applied to every explanation.

    Example:

    .. code-block:: python

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
    """

    def __init__(
        self,
        model: Callable,
        algorithm: ExplainerAlgorithm,
        explanation_type: str,
        model_config: Union[ModelConfig, dict],
        node_mask_type: Optional[str] = None,
        edge_mask_type: Optional[str] = None,
        threshold_config: Optional[Union[ThresholdConfig, dict]] = None,
    ):
        self.model = model
        self.algorithm = algorithm
        self.explainer_config = ExplainerConfig(
            explanation_type=explanation_type,
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
        )
        self.model_config = (
            model_config
            if isinstance(model_config, ModelConfig)
            else ModelConfig(**model_config)
        )
        if isinstance(threshold_config, dict):
            threshold_config = ThresholdConfig(**threshold_config)
        self.threshold_config = threshold_config
        self.algorithm.connect(self.explainer_config, self.model_config)

    def get_prediction(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        edge_weights: Optional[mx.array] = None,
        **kwargs: Any,
    ) -> mx.array:
        """Runs the model in eval mode and returns its output."""
        was_training = getattr(self.model, "training", False)
        if hasattr(self.model, "eval"):
            self.model.eval()
        out = forward(self.model, edge_index, node_features, edge_weights, **kwargs)
        if was_training and hasattr(self.model, "train"):
            self.model.train()
        mx.eval(out)
        return out

    def get_target(self, prediction: mx.array) -> mx.array:
        """Turns a model output into the target being explained."""
        mode, rt = self.model_config.mode, self.model_config.return_type
        if self.model_config.task_level == "graph":
            prediction = as_2d(prediction)
        if mode == "multiclass_classification":
            return mx.argmax(prediction, axis=-1)
        if mode == "binary_classification":
            threshold = 0.0 if rt == "raw" else 0.5
            return (prediction > threshold).astype(mx.int32).reshape(-1)
        return prediction

    def __call__(
        self,
        edge_index: mx.array,
        node_features: mx.array,
        *,
        target: Optional[mx.array] = None,
        index: Index = None,
        **kwargs: Any,
    ) -> Explanation:
        """Explains the prediction(s) of the model.

        Args:
            edge_index: graph connectivity, shape ``[2, E]``.
            node_features: node features, shape ``[N, F]``.
            target: ground-truth target; required for ``"phenomenon"``
                explanations, ignored for ``"model"``.
            index: node index (or indices) to explain, for node-level tasks.
            **kwargs: extra arguments forwarded to the model.
        """
        prediction = self.get_prediction(edge_index, node_features, **kwargs)
        if self.explainer_config.explanation_type == "phenomenon":
            if target is None:
                raise ValueError("`target` is required for 'phenomenon' explanations")
        else:
            if target is not None:
                raise ValueError(
                    "`target` must be None for 'model' explanations "
                    "(the model's own prediction is explained)"
                )
            target = self.get_target(prediction)

        if hasattr(self.model, "eval"):
            self.model.eval()
        explanation = self.algorithm(
            self.model, edge_index, node_features, target=target, index=index, **kwargs
        )
        explanation.prediction = prediction
        if self.threshold_config is not None:
            explanation = explanation.threshold(self.threshold_config)
        return explanation

    def __repr__(self) -> str:
        return (
            f"Explainer(algorithm={self.algorithm}, "
            f"explanation_type={self.explainer_config.explanation_type!r}, "
            f"node_mask_type={self.explainer_config.node_mask_type!r}, "
            f"edge_mask_type={self.explainer_config.edge_mask_type!r})"
        )
