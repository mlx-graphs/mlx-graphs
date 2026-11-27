"""Configuration objects, mirroring ``torch_geometric.explain.config``."""

from dataclasses import dataclass
from typing import Optional

EXPLANATION_TYPES = ("model", "phenomenon")
MASK_TYPES = ("object", "attributes", "common_attributes")
MODEL_MODES = ("binary_classification", "multiclass_classification", "regression")
TASK_LEVELS = ("node", "graph")
RETURN_TYPES = ("raw", "probs", "log_probs")
THRESHOLD_TYPES = ("hard", "topk", "topk_hard")


def _check(name: str, value, allowed, optional: bool = False):
    if optional and value is None:
        return
    if value not in allowed:
        raise ValueError(f"`{name}` must be one of {allowed} (got {value!r})")


@dataclass
class ExplainerConfig:
    """What kind of explanation to compute.

    Args:
        explanation_type: ``"model"`` explains the model's own prediction,
            ``"phenomenon"`` explains why the model gives the true ``target``.
        node_mask_type: ``None`` (no node mask), ``"object"`` (one value per
            node), ``"attributes"`` (one value per node feature) or
            ``"common_attributes"`` (one value per feature, shared by all nodes).
        edge_mask_type: ``None`` or ``"object"`` (one value per edge).
    """

    explanation_type: str = "model"
    node_mask_type: Optional[str] = None
    edge_mask_type: Optional[str] = None

    def __post_init__(self):
        _check("explanation_type", self.explanation_type, EXPLANATION_TYPES)
        _check("node_mask_type", self.node_mask_type, MASK_TYPES, optional=True)
        _check("edge_mask_type", self.edge_mask_type, ("object",), optional=True)
        if self.node_mask_type is None and self.edge_mask_type is None:
            raise ValueError(
                "At least one of `node_mask_type` and `edge_mask_type` must be set"
            )


@dataclass
class ModelConfig:
    """What the explained model does.

    Args:
        mode: ``"binary_classification"``, ``"multiclass_classification"`` or
            ``"regression"``.
        task_level: ``"node"`` or ``"graph"``.
        return_type: what the model outputs: ``"raw"`` (logits), ``"probs"`` or
            ``"log_probs"``. Ignored for regression.
    """

    mode: str
    task_level: str = "node"
    return_type: str = "raw"

    def __post_init__(self):
        _check("mode", self.mode, MODEL_MODES)
        _check("task_level", self.task_level, TASK_LEVELS)
        _check("return_type", self.return_type, RETURN_TYPES)
        if self.mode == "binary_classification" and self.return_type == "log_probs":
            raise ValueError(
                "Binary classification supports return_type 'raw' or 'probs'"
            )


@dataclass
class ThresholdConfig:
    """How to post-process the masks of an explanation.

    Args:
        threshold_type: ``"hard"`` keeps values above ``value`` (others set to 0),
            ``"topk"`` keeps the ``value`` largest entries, ``"topk_hard"`` sets
            the ``value`` largest entries to 1 and the rest to 0.
        value: the threshold (``"hard"``) or number of entries to keep.
    """

    threshold_type: str
    value: float

    def __post_init__(self):
        _check("threshold_type", self.threshold_type, THRESHOLD_TYPES)
        if self.threshold_type in ("topk", "topk_hard"):
            if int(self.value) != self.value or self.value <= 0:
                raise ValueError("`value` must be a positive integer for top-k")
            self.value = int(self.value)
