"""Base class and helpers shared by all explanation algorithms."""

from typing import Any, Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_graphs.explain.config import ExplainerConfig, ModelConfig
from mlx_graphs.explain.explanation import Explanation

Index = Optional[Union[int, mx.array]]


def forward(
    model: Callable,
    edge_index: mx.array,
    node_features: mx.array,
    edge_weights: Optional[mx.array] = None,
    **kwargs: Any,
) -> mx.array:
    """Calls ``model(edge_index, node_features, ...)`` (mlx-graphs convention).

    Edge masks are passed through the ``edge_weights`` keyword, so models
    explained with an edge mask must accept it and forward it to their layers
    (e.g. :class:`mlx_graphs.nn.GCNConv`).
    """
    if edge_weights is None:
        return model(edge_index, node_features, **kwargs)
    try:
        return model(edge_index, node_features, edge_weights=edge_weights, **kwargs)
    except TypeError as e:
        if "edge_weights" not in str(e):
            raise
        raise TypeError(
            "Explaining edges requires a model whose __call__ accepts an "
            "`edge_weights` keyword and passes it to its message passing layers, "
            "e.g. `def __call__(self, edge_index, node_features, edge_weights=None)`"
        ) from e


def as_2d(out: mx.array) -> mx.array:
    """Graph-level models may return ``[C]`` for a single graph."""
    return out.reshape(1, -1) if out.ndim == 1 else out


def select(out: mx.array, index: Index, model_config: ModelConfig) -> mx.array:
    """Keeps the rows of the model output that are being explained."""
    out = as_2d(out) if model_config.task_level == "graph" else out
    if model_config.task_level == "node" and index is not None:
        idx = mx.array(index).reshape(-1)
        return out[idx]
    return out


def select_target(target: mx.array, index: Index, model_config: ModelConfig):
    """Keeps the targets of the explained nodes (node-level) or graphs."""
    if model_config.task_level == "node" and index is not None:
        return target[mx.array(index).reshape(-1)]
    return target.reshape(-1) if target.ndim == 0 else target


def to_log_probs(out: mx.array, model_config: ModelConfig) -> mx.array:
    if model_config.return_type == "raw":
        return nn.log_softmax(out, axis=-1)
    if model_config.return_type == "probs":
        return mx.log(mx.clip(out, 1e-15, 1.0))
    return out


def explanation_loss(
    out: mx.array, target: mx.array, model_config: ModelConfig
) -> mx.array:
    """Loss between (selected) model outputs and (selected) targets."""
    if model_config.mode == "multiclass_classification":
        log_probs = to_log_probs(out, model_config)
        return nn.losses.nll_loss(log_probs, target.reshape(-1), reduction="mean")
    if model_config.mode == "binary_classification":
        out = out.reshape(target.shape).astype(mx.float32)
        target = target.astype(mx.float32)
        return nn.losses.binary_cross_entropy(
            out, target, with_logits=model_config.return_type == "raw", reduction="mean"
        )
    return nn.losses.mse_loss(out.reshape(target.shape), target, reduction="mean")


def target_score(out: mx.array, target: mx.array, model_config: ModelConfig):
    """Scalar whose gradient gradient-based explainers attribute.

    For classification this is the (summed) output of the target class, for
    regression the (summed) output itself.
    """
    if model_config.mode == "multiclass_classification":
        target = target.reshape(-1).astype(mx.int32)
        return mx.take_along_axis(out, target[:, None], axis=-1).sum()
    if model_config.mode == "binary_classification":
        out = out.reshape(target.shape)
        # push towards the predicted class: +out for class 1, -out for class 0
        sign = 2 * target.astype(out.dtype) - 1
        return (sign * out).sum()
    return out.sum()


def init_mask_shape(
    mask_type: Optional[str], num_nodes: int, num_features: int
) -> Optional[tuple]:
    if mask_type is None:
        return None
    return {
        "object": (num_nodes, 1),
        "attributes": (num_nodes, num_features),
        "common_attributes": (1, num_features),
    }[mask_type]


def reduce_node_attribution(attr: mx.array, mask_type: Optional[str]):
    """Turns a ``[N, F]`` attribution into the requested node mask shape."""
    if mask_type is None:
        return None
    if mask_type == "object":
        return attr.sum(axis=-1, keepdims=True)
    if mask_type == "common_attributes":
        return attr.sum(axis=0, keepdims=True)
    return attr


class ExplainerAlgorithm:
    """Base class of explanation algorithms (as in PyG).

    Subclasses implement :meth:`forward`, and may override :meth:`supports`
    to reject configurations they cannot handle.
    """

    explainer_config: ExplainerConfig
    model_config: ModelConfig

    def connect(self, explainer_config: ExplainerConfig, model_config: ModelConfig):
        self.explainer_config = explainer_config
        self.model_config = model_config
        if not self.supports():
            raise ValueError(
                f"{type(self).__name__} does not support this configuration: "
                f"{explainer_config}, {model_config}"
            )

    def supports(self) -> bool:
        return True

    def forward(
        self,
        model: Callable,
        edge_index: mx.array,
        node_features: mx.array,
        *,
        target: mx.array,
        index: Index = None,
        **kwargs: Any,
    ) -> Explanation:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Explanation:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
