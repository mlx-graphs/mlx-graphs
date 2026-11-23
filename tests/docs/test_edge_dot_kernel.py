"""The example of the Custom GPU kernels guide: docs/snippets/edge_dot_kernel.py.

The kernel must match its MLX reference in values and gradients. Without Metal,
`edge_dot` falls back to the reference and the checks still run.
"""

import importlib.util
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

SNIPPET = (
    Path(__file__).resolve().parents[2] / "docs" / "snippets" / "edge_dot_kernel.py"
)


@pytest.fixture(scope="module")
def guide():
    spec = importlib.util.spec_from_file_location("edge_dot_kernel", SNIPPET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kernel_matches_reference(guide):
    guide.run_tests()


@pytest.mark.parametrize(
    "num_nodes,num_edges,channels", [(30, 200, 7), (500, 5000, 33)]
)
def test_values_and_gradients(guide, num_nodes, num_edges, channels):
    guide.check(*guide.random_graph(num_nodes, num_edges, channels))
