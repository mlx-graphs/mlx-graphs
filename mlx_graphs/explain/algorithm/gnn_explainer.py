"""GNNExplainer, following ``torch_geometric.explain.algorithm.GNNExplainer``."""

import math
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.optimizers as optim

from mlx_graphs.explain.algorithm.base import (
    ExplainerAlgorithm,
    Index,
    explanation_loss,
    forward,
    init_mask_shape,
    select,
    select_target,
)
from mlx_graphs.explain.explanation import Explanation

EPS = 1e-15

DEFAULT_COEFFS = {
    "edge_size": 0.005,
    "edge_reduction": "sum",
    "node_feat_size": 1.0,
    "node_feat_reduction": "mean",
    "edge_ent": 1.0,
    "node_feat_ent": 0.1,
}


def _reduce(x: mx.array, keep: mx.array, how: str) -> mx.array:
    total = (x * keep).sum()
    return total if how == "sum" else total / mx.maximum(keep.sum(), 1)


def _entropy(m: mx.array, keep: mx.array) -> mx.array:
    ent = -m * mx.log(m + EPS) - (1 - m) * mx.log(1 - m + EPS)
    return (ent * keep).sum() / mx.maximum(keep.sum(), 1)


class GNNExplainer(ExplainerAlgorithm):
    """GNNExplainer from `"GNNExplainer: Generating Explanations for Graph
    Neural Networks" <https://arxiv.org/abs/1903.03894>`_.

    Learns soft masks over edges and/or node features that keep the model's
    prediction while being small and close to binary. Masks are returned after
    a sigmoid, in ``[0, 1]``. For node-level tasks, entries that cannot affect
    the explained node (outside its receptive field) are set to 0.

    Edge masks are passed to the model as ``edge_weights``, so the model must
    accept that keyword (see :func:`~mlx_graphs.explain.algorithm.base.forward`).

    Args:
        epochs: number of optimization steps. Default: 100
        lr: Adam learning rate. Default: 0.01
        seed: optional seed for the mask initialization.
        **coeffs: overrides of the regularization coefficients
            (``edge_size``, ``edge_reduction``, ``node_feat_size``,
            ``node_feat_reduction``, ``edge_ent``, ``node_feat_ent``).
    """

    def __init__(
        self, epochs: int = 100, lr: float = 0.01, seed: Optional[int] = None, **coeffs
    ):
        unknown = set(coeffs) - set(DEFAULT_COEFFS)
        if unknown:
            raise ValueError(f"Unknown coefficients: {sorted(unknown)}")
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.coeffs = {**DEFAULT_COEFFS, **coeffs}

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
        if self.seed is not None:
            mx.random.seed(self.seed)

        num_nodes, num_features = node_features.shape
        num_edges = edge_index.shape[1]
        node_type = self.explainer_config.node_mask_type
        use_edges = self.explainer_config.edge_mask_type is not None

        params = {}
        shape = init_mask_shape(node_type, num_nodes, num_features)
        if shape is not None:
            params["node_mask"] = 0.1 * mx.random.normal(shape)
        if use_edges:
            std = math.sqrt(2.0) * math.sqrt(2.0 / (2 * num_nodes))  # as in PyG
            params["edge_mask"] = std * mx.random.normal((num_edges,))

        target_sel = select_target(target, index, self.model_config)

        def prediction_loss(p):
            x = node_features
            if "node_mask" in p:
                x = x * mx.sigmoid(p["node_mask"])
            ew = mx.sigmoid(p["edge_mask"]) if "edge_mask" in p else None
            out = forward(model, edge_index, x, ew, **kwargs)
            out = select(out, index, self.model_config)
            return explanation_loss(out, target_sel, self.model_config)

        # Entries whose gradient is zero cannot influence the explained output
        # (outside the receptive field of the node): exclude them, as PyG does.
        if self.model_config.task_level == "node":
            first = mx.grad(prediction_loss)(params)
            hard = {k: (g != 0).astype(mx.float32) for k, g in first.items()}
        else:
            hard = {k: mx.ones(v.shape) for k, v in params.items()}

        def loss_fn(p):
            return prediction_loss(p) + self._regularization(p, hard)

        optimizer = optim.Adam(learning_rate=self.lr)
        loss_and_grad = mx.value_and_grad(loss_fn)
        loss = mx.array(float("nan"))
        for _ in range(self.epochs):
            loss, grads = loss_and_grad(params)
            params = optimizer.apply_gradients(grads, params)
            mx.eval(params, optimizer.state, loss)

        node_mask = edge_mask = None
        if "node_mask" in params:
            node_mask = self._post(params["node_mask"], hard["node_mask"])
        if "edge_mask" in params:
            edge_mask = self._post(params["edge_mask"], hard["edge_mask"])

        return Explanation(
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_index=edge_index,
            node_features=node_features,
            target=target,
            index=index,
            extra={"final_loss": float(loss)},
        )

    def _regularization(self, p, hard) -> mx.array:
        c = self.coeffs
        reg = mx.array(0.0)
        if "edge_mask" in p:
            m, h = mx.sigmoid(p["edge_mask"]), hard["edge_mask"]
            reg = reg + c["edge_size"] * _reduce(m, h, c["edge_reduction"])
            reg = reg + c["edge_ent"] * _entropy(m, h)
        if "node_mask" in p:
            m, h = mx.sigmoid(p["node_mask"]), hard["node_mask"]
            reg = reg + c["node_feat_size"] * _reduce(m, h, c["node_feat_reduction"])
            reg = reg + c["node_feat_ent"] * _entropy(m, h)
        return reg

    @staticmethod
    def _post(raw: mx.array, hard: mx.array) -> mx.array:
        return mx.sigmoid(raw) * hard

    def __repr__(self) -> str:
        return f"GNNExplainer(epochs={self.epochs}, lr={self.lr})"
