"""Platform detection and the switches that force the portable fallbacks."""

import json
import os
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs import backend
from mlx_graphs.nn.kernels import csr_softmax_spmm, gatv2_edge_scores

_PROBE = """
import json
import mlx.core as mx
from mlx_graphs import backend, sampling
from mlx_graphs.loaders import NeighborLoader
from mlx_graphs.algorithms import Node2Vec, MetaPath2Vec
from mlx_graphs.nn import message_passing

print(json.dumps({
    "has_mlx_cluster": backend.HAS_MLX_CLUSTER,
    "sampling_has_mlx_cluster": sampling.HAS_MLX_CLUSTER,
    "fused_kernels": message_passing.HAS_FUSED_KERNELS,
    "fused_enabled": message_passing.is_fused_message_passing_enabled(),
    "metal_kernels": backend.metal_kernels_available(),
}))
"""


def _probe(**env):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, **env, "PYTHONPATH": root}
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_disable_switches_force_the_fallbacks():
    state = _probe(
        MLX_GRAPHS_DISABLE_MLX_CLUSTER="1", MLX_GRAPHS_DISABLE_METAL_KERNELS="1"
    )
    assert state == {
        "has_mlx_cluster": False,
        "sampling_has_mlx_cluster": False,
        "fused_kernels": False,
        "fused_enabled": False,
        "metal_kernels": False,
    }


def test_switches_set_to_zero_change_nothing():
    state = _probe(
        MLX_GRAPHS_DISABLE_MLX_CLUSTER="0", MLX_GRAPHS_DISABLE_METAL_KERNELS="0"
    )
    try:
        import mlx_cluster  # noqa: F401

        installed = True
    except ImportError:
        installed = False
    assert state["has_mlx_cluster"] == installed
    assert state["metal_kernels"] == mx.metal.is_available()


@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")
def test_disabled_metal_kernels_give_the_same_results(monkeypatch):
    """With Metal kernels disabled, the custom-kernel ops run their MLX
    reference implementation, which gives the same values and gradients."""
    rng = np.random.default_rng(0)
    degrees = [3, 0, 7, 1, 12]
    rowptr = mx.array([0, *np.cumsum(degrees).tolist()], dtype=mx.int64)
    edges = int(sum(degrees))
    col = mx.array(rng.integers(0, 9, edges), dtype=mx.int64)
    logits = mx.array(rng.normal(size=(edges, 2)), dtype=mx.float32)
    values = mx.array(rng.normal(size=(9, 2, 16)), dtype=mx.float32)
    src = mx.array(rng.normal(size=(9, 2, 16)), dtype=mx.float32)
    att = mx.array(rng.normal(size=(1, 2, 16)), dtype=mx.float32)
    edge_index = mx.array(rng.integers(0, 9, (2, 40)), dtype=mx.int32)

    def run():
        softmax = mx.vjp(
            lambda lg, v: csr_softmax_spmm.csr_softmax_spmm(rowptr, col, lg, v),
            [logits, values],
            [mx.ones((5, 2, 16))],
        )
        scores = mx.vjp(
            lambda s, d, a: gatv2_edge_scores.gatv2_edge_scores(
                s, d, a, edge_index, 0.2
            ),
            [src, src * 0.5, att],
            [mx.ones((40, 2))],
        )
        return softmax, scores

    with mx.stream(mx.gpu):
        monkeypatch.setattr(backend, "_METAL_KERNELS_DISABLED", False)
        assert backend.use_metal_kernels()
        fast = run()
        monkeypatch.setattr(backend, "_METAL_KERNELS_DISABLED", True)
        assert not backend.use_metal_kernels()

        def unexpected(*args, **kwargs):
            pytest.fail("A Metal kernel ran although Metal kernels are disabled")

        monkeypatch.setattr(csr_softmax_spmm, "_metal", unexpected)
        monkeypatch.setattr(gatv2_edge_scores, "_metal_scores", unexpected)
        fallback = run()

    for (out_a, grads_a), (out_b, grads_b) in zip(fast, fallback):
        for a, b in zip([*out_a, *grads_a], [*out_b, *grads_b]):
            np.testing.assert_allclose(np.array(a), np.array(b), rtol=1e-5, atol=1e-5)


def test_metal_kernels_never_run_on_the_cpu():
    with mx.stream(mx.cpu):
        assert not backend.use_metal_kernels()
