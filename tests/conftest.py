"""Shared test policy for the official LIBERO backend.

The package normally initializes ``~/.libero/config.yaml`` interactively.  Tests
must never prompt or modify a developer's configuration, so point every test
session at an isolated configuration before importing :mod:`libero.libero`.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPOSITORY_ROOT / "libero" / "libero"
_CONFIG_DIRECTORY = Path(tempfile.mkdtemp(prefix="libero-pytest-"))


def _write_isolated_libero_config() -> None:
    config = "\n".join(
        (
            f"benchmark_root: {BENCHMARK_ROOT}",
            f"bddl_files: {BENCHMARK_ROOT / 'bddl_files'}",
            f"init_states: {BENCHMARK_ROOT / 'init_files'}",
            f"datasets: {_CONFIG_DIRECTORY / 'datasets'}",
            f"assets: {BENCHMARK_ROOT / 'assets'}",
            "",
        )
    )
    (_CONFIG_DIRECTORY / "config.yaml").write_text(config, encoding="utf-8")


_write_isolated_libero_config()
os.environ["LIBERO_CONFIG_PATH"] = str(_CONFIG_DIRECTORY)
atexit.register(shutil.rmtree, _CONFIG_DIRECTORY, ignore_errors=True)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "static: fast source-tree/package-data check")
    config.addinivalue_line(
        "markers",
        "official_integration: opt-in local robosuite / MuJoCo rendering test",
    )
    config.addinivalue_line(
        "markers", "nightly: requires LIBERO_RUN_NIGHTLY=1 and optional local test data"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Make expensive coverage deliberate instead of producing a false-green CI run."""

    run_integration = os.environ.get("LIBERO_RUN_OFFICIAL_INTEGRATION") == "1"
    run_nightly = os.environ.get("LIBERO_RUN_NIGHTLY") == "1"
    integration_skip = pytest.mark.skip(
        reason=(
            "official MuJoCo integration is opt-in; set "
            "LIBERO_RUN_OFFICIAL_INTEGRATION=1 on a local EGL/OSMesa-capable host"
        )
    )
    nightly_skip = pytest.mark.skip(
        reason="nightly coverage is opt-in; set LIBERO_RUN_NIGHTLY=1",
    )

    for item in items:
        if "nightly" in item.keywords and not run_nightly:
            item.add_marker(nightly_skip)
        elif "official_integration" in item.keywords and not run_integration:
            item.add_marker(integration_skip)
