"""Gradient-based attribution methods, implemented natively with ``mx.grad``.

PyG runs these through Captum (``CaptumExplainer``); MLX has no Captum, so
they live here as first-class algorithms.
"""

from typing import Any, Callable, Optional

import mlx.core as mx

from mlx_graphs.explain.algorithm.base import (
    ExplainerAlgorithm,
    Index,
    forward,
    reduce_node_attribution,
    select,
    select_target,
    target_score,
)
from mlx_graphs.explain.explanation import Explanation


class _GradientAlgorithm(ExplainerAlgorithm):
    def supports(self) -> bool:
        if self.explainer_config.explanation_type != "model":
            # gradients explain the model's output, not a ground-truth phenomenon
            return False
        return True

    def _score_fn(self, model, edge_index, target, index, kwargs) -> Callable:
        """f(node_features, edge_weights) -> scalar score of the target."""
        target_sel = select_target(target, index, self.model_config)

        def fn(node_features: mx.array, edge_weights: Optional[mx.array]):
            out = forward(model, edge_index, node_features, edge_weights, **kwargs)
            out = select(out, index, self.model_config)
            return target_score(out, target_sel, self.model_config)

        return fn

    def _explanation(
        self, node_attr, edge_attr, edge_index, node_features, target, index
    ):
        return Explanation(
            node_mask=reduce_node_attribution(
                node_attr, self.explainer_config.node_mask_type
            ),
            edge_mask=edge_attr,
            edge_index=edge_index,
            node_features=node_features,
            target=target,
            index=index,
        )


class Saliency(_GradientAlgorithm):
    """Absolute gradient of the target score with respect to the inputs.

    From `"Deep Inside Convolutional Networks: Visualising Image Classification
    Models and Saliency Maps" <https://arxiv.org/abs/1312.6034>`_.

    Node masks are ``|d score / d x|``; edge masks are the absolute gradient
    with respect to a weight of 1 on every edge.
    """

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
        fn = self._score_fn(model, edge_index, target, index, kwargs)
        use_edges = self.explainer_config.edge_mask_type is not None
        edge_weights = mx.ones((edge_index.shape[1],)) if use_edges else None

        argnums = (0, 1) if use_edges else 0
        grads = mx.grad(fn, argnums=argnums)(node_features, edge_weights)
        node_grad, edge_grad = grads if use_edges else (grads, None)

        node_attr = mx.abs(node_grad)
        edge_attr = mx.abs(edge_grad) if use_edges else None
        mx.eval(node_attr, edge_attr)
        return self._explanation(
            node_attr, edge_attr, edge_index, node_features, target, index
        )


class IntegratedGradients(_GradientAlgorithm):
    """Integrated Gradients from `"Axiomatic Attribution for Deep Networks"
    <https://arxiv.org/abs/1703.01365>`_.

    Gradients are averaged along a straight path from a baseline (all-zero
    features, and all-zero edge weights when edges are explained) to the input,
    then multiplied by ``input - baseline``. Attributions are signed and,
    summed over all inputs, approximate ``score(input) - score(baseline)``.

    Args:
        n_steps: number of points on the path (Riemann midpoint rule).
        baseline: optional node feature baseline, broadcastable to the
            features. Default: zeros.
        absolute: whether to return absolute attributions. Default: ``False``.
    """

    def __init__(
        self,
        n_steps: int = 50,
        baseline: Optional[mx.array] = None,
        absolute: bool = False,
    ):
        if n_steps < 1:
            raise ValueError("`n_steps` must be at least 1")
        self.n_steps = n_steps
        self.baseline = baseline
        self.absolute = absolute

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
        fn = self._score_fn(model, edge_index, target, index, kwargs)
        use_edges = self.explainer_config.edge_mask_type is not None

        x_base = (
            mx.zeros_like(node_features)
            if self.baseline is None
            else mx.broadcast_to(self.baseline, node_features.shape)
        )
        x_delta = node_features - x_base
        num_edges = edge_index.shape[1]
        ew_input = mx.ones((num_edges,))  # baseline for edges is 0

        argnums = (0, 1) if use_edges else 0
        grad_fn = mx.grad(fn, argnums=argnums)

        node_total = mx.zeros_like(node_features)
        edge_total = mx.zeros((num_edges,)) if use_edges else None
        for step in range(self.n_steps):
            alpha = (step + 0.5) / self.n_steps
            x = x_base + alpha * x_delta
            ew = alpha * ew_input if use_edges else None
            grads = grad_fn(x, ew)
            if use_edges:
                node_total = node_total + grads[0]
                edge_total = edge_total + grads[1]
            else:
                node_total = node_total + grads
            mx.eval(node_total, edge_total)

        node_attr = x_delta * node_total / self.n_steps
        edge_attr = ew_input * edge_total / self.n_steps if use_edges else None
        if self.absolute:
            node_attr = mx.abs(node_attr)
            edge_attr = mx.abs(edge_attr) if use_edges else None
        mx.eval(node_attr, edge_attr)
        return self._explanation(
            node_attr, edge_attr, edge_index, node_features, target, index
        )

    def __repr__(self) -> str:
        return f"IntegratedGradients(n_steps={self.n_steps})"
