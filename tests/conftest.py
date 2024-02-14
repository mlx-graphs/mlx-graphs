import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--datasets", action="store_true", default=False, help="run slow dataset tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--datasets"):
        # --datasets given in cli: do not skip slow tests related to datasets
        return
    skip_slow = pytest.mark.skip(reason="need --datasets option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
