"""Layers that run a projection on per-edge arrays must not crash on graphs
without edges.

MLX 0.32 can crash (segfault) when a compiled function such as
``nn.softplus`` or ``nn.relu`` runs on the output of an empty matmul (e.g. a
``Linear`` without bias on zero edges). Each case runs in a subprocess, so
that a crash fails the test instead of the whole test session.
"""

import os
import pathlib
import subprocess
import sys

import pytest

import mlx_graphs

CASES = {
    "CGConv": "CGConv(6, 3, bias=False)(ei, x, mx.zeros((0, 3)))",
    "CGConv without edge features": "CGConv(6, 0, bias=False)(ei, x)",
    "NNConv": (
        "NNConv(6, 8, nn.Sequential(nn.Linear(3, 16, bias=False), nn.ReLU(), "
        "nn.Linear(16, 48)))(ei, x, mx.zeros((0, 3)))"
    ),
    "EdgeConv": (
        "EdgeConv(nn.Sequential(nn.Linear(12, 16, bias=False), nn.ReLU()))(ei, x)"
    ),
}

SCRIPT = """
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_graphs.nn import CGConv, EdgeConv, NNConv

ei = mx.zeros((2, 0), dtype=mx.int32)
x = mx.random.normal((5, 6))
out = {expr}
mx.eval(out)
assert out.shape[0] == 5 and np.isfinite(np.array(out)).all()
"""


@pytest.mark.parametrize("name", list(CASES))
def test_zero_edges_do_not_crash(name):
    # the mlx_graphs imported by this test session, not another install
    root = str(pathlib.Path(mlx_graphs.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([root, env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(expr=CASES[name])],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
    )
    assert result.returncode == 0, (result.returncode, result.stderr[-2000:])
