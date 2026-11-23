"""Helpers to copy the weights of PyG / PyTorch modules into mlx-graphs ones."""

import re
from typing import Callable, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten


def torch_state(
    module,
    renames: Sequence[Tuple[str, str]] = (),
    rename_fn: Optional[Callable[[str], str]] = None,
    drop: Sequence[str] = (),
) -> dict:
    """Converts the state dict of a torch module to mlx-graphs names.

    ``torch.nn.MultiheadAttention``'s packed ``in_proj_weight`` /
    ``in_proj_bias`` are split into ``query_proj``, ``key_proj`` and
    ``value_proj``, ``num_batches_tracked`` is dropped, and the regex
    ``renames`` are then applied in order (then ``rename_fn``). Keys matching
    a regex of ``drop`` (torch names) are skipped.
    """
    out = {}
    for key, value in module.state_dict().items():
        value = value.detach().cpu().numpy()
        if key.endswith("num_batches_tracked") or any(re.search(p, key) for p in drop):
            continue
        if key.endswith("in_proj_weight") or key.endswith("in_proj_bias"):
            prefix, name = key.rsplit(".", 1) if "." in key else ("", key)
            kind = "weight" if name.endswith("weight") else "bias"
            for proj, part in zip(
                ("query_proj", "key_proj", "value_proj"), np.split(value, 3, axis=0)
            ):
                out[f"{prefix}.{proj}.{kind}".lstrip(".")] = part
            continue
        out[key] = value
    renamed = {}
    for key, value in out.items():
        for pattern, repl in renames:
            key = re.sub(pattern, repl, key)
        if rename_fn is not None:
            key = rename_fn(key)
        renamed[key] = value
    return renamed


def load_torch(mlx_module: nn.Module, state: dict, strict: bool = True):
    """Loads a converted state dict, checking that every parameter of
    ``mlx_module`` is set with the right shape."""
    expected = dict(tree_flatten(mlx_module.parameters()))
    if strict:
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        assert not missing and not extra, f"missing={missing} extra={extra}"
    for key, value in state.items():
        assert expected[key].shape == value.shape, (
            key,
            expected[key].shape,
            value.shape,
        )
    mlx_module.load_weights([(k, mx.array(v)) for k, v in state.items()])
    mx.eval(mlx_module.parameters())


def lstm_state(prefix: str, lstm) -> dict:
    """Converts a single-layer ``torch.nn.LSTM`` to ``mlx.nn.LSTM`` weights."""
    sd = {k: v.detach().numpy() for k, v in lstm.state_dict().items()}
    out = {f"{prefix}.Wx": sd["weight_ih_l0"], f"{prefix}.Wh": sd["weight_hh_l0"]}
    if "bias_ih_l0" in sd:
        out[f"{prefix}.bias"] = sd["bias_ih_l0"] + sd["bias_hh_l0"]
    return out


def gru_state(prefix: str, gru, suffix: str = "_l0") -> dict:
    """Converts a single-layer ``torch.nn.GRU`` (or a ``GRUCell`` with
    ``suffix=""``) to ``mlx.nn.GRU`` weights."""
    sd = {k: v.detach().numpy() for k, v in gru.state_dict().items()}
    out = {
        f"{prefix}.Wx": sd[f"weight_ih{suffix}"],
        f"{prefix}.Wh": sd[f"weight_hh{suffix}"],
    }
    if f"bias_ih{suffix}" in sd:
        b_ih, b_hh = sd[f"bias_ih{suffix}"], sd[f"bias_hh{suffix}"]
        hidden = b_hh.shape[0] // 3
        out[f"{prefix}.b"] = b_ih + np.concatenate(
            [b_hh[: 2 * hidden], 0 * b_hh[:hidden]]
        )
        out[f"{prefix}.bhn"] = b_hh[2 * hidden :]
    return out


def torch_grads(
    module,
    renames: Sequence[Tuple[str, str]] = (),
    drop: Sequence[str] = (),
) -> dict:
    """The gradients of the parameters of a torch module (after
    ``backward()``), with the names converted as in :func:`torch_state`."""
    out = {}
    for key, param in module.named_parameters():
        if param.grad is None or any(re.search(p, key) for p in drop):
            continue
        for pattern, repl in renames:
            key = re.sub(pattern, repl, key)
        out[key] = param.grad.detach().cpu().numpy()
    return out


def assert_grads_close(mlx_grads, torch_grads: dict, rtol: float = 2e-2):
    """Checks the gradients of an mlx module (a tree, e.g. from
    ``nn.value_and_grad``) against :func:`torch_grads`, parameter by
    parameter, in relative norm."""
    flat = dict(tree_flatten(mlx_grads))
    assert torch_grads, "no torch gradients"
    for key, expected in torch_grads.items():
        assert key in flat, f"missing gradient {key}"
        got = np.array(flat[key])
        assert got.shape == expected.shape, (key, got.shape, expected.shape)
        diff = np.linalg.norm(got - expected)
        scale = max(np.linalg.norm(expected), 1e-6)
        assert diff <= rtol * scale, f"{key}: relative error {diff / scale:.2e}"
