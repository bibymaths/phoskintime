"""Shared pytest configuration."""


import pytest as _pytest_env_guard


@_pytest_env_guard.fixture(autouse=True)
def _restore_phoskintime_env():
    """Restore PHOSKINTIME_* environment variables after every test: the runners export the selected config path
    (PHOSKINTIME_NETWORKMODEL_CONFIG, PHOSKINTIME_ODE_CONFIG), which otherwise leaks into later imports."""
    import os

    saved = {k: v for k, v in os.environ.items() if k.startswith("PHOSKINTIME_")}
    yield
    for k in [k for k in os.environ if k.startswith("PHOSKINTIME_")]:
        if k not in saved:
            del os.environ[k]
    os.environ.update(saved)
