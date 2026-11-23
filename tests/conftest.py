import os

import mlx.core as mx
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--datasets", action="store_true", default=False, help="run slow dataset tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    # MLX_GRAPHS_TEST_DEVICE=cpu runs the whole suite on the CPU, as on a
    # machine without a GPU backend (e.g. Linux with mlx[cpu]).
    device = os.environ.get("MLX_GRAPHS_TEST_DEVICE")
    if device:
        mx.set_default_device(getattr(mx, device))


def pytest_collection_modifyitems(config, items):
    if config.getoption("--datasets"):
        # --datasets given in cli: do not skip slow tests related to datasets
        return
    skip_slow = pytest.mark.skip(reason="need --datasets option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
