"""Core LIBERO imports must not require the optional tensor / GPU stack."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import REPOSITORY_ROOT

pytestmark = pytest.mark.static


def _run_without_torch(source: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = str(tmp_path / "pycache")
    bootstrap = "\n".join(
        (
            "import importlib.abc",
            "import sys",
            "class DenyTorch(importlib.abc.MetaPathFinder):",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        if fullname == 'torch' or fullname.startswith('torch.'):",
            "            raise ImportError('Torch intentionally unavailable')",
            "        return None",
            "sys.meta_path.insert(0, DenyTorch())",
            source,
        )
    )
    return subprocess.run(
        [sys.executable, "-c", bootstrap],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )


def test_core_benchmark_and_utility_imports_do_not_require_torch(
    tmp_path: Path,
) -> None:
    result = _run_without_torch(
        "\n".join(
            (
                "from libero.libero import benchmark, get_libero_path",
                "from libero.libero.utils import utils",
                "assert callable(get_libero_path)",
                "assert benchmark.get_benchmark_dict()",
                "assert callable(utils.postprocess_model_xml)",
                "assert 'torch' not in sys.modules",
            )
        ),
        tmp_path,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_init_state_loader_reports_the_tensor_extra_when_torch_is_absent(
    tmp_path: Path,
) -> None:
    result = _run_without_torch(
        "\n".join(
            (
                "from libero.libero import benchmark",
                "suite = benchmark.get_benchmark('libero_spatial')()",
                "try:",
                "    suite.get_task_init_states(0)",
                "except RuntimeError as error:",
                "    assert 'libero-warp[tensor]' in str(error)",
                "else:",
                "    raise AssertionError('missing optional Torch error')",
            )
        ),
        tmp_path,
    )

    assert result.returncode == 0, result.stdout + result.stderr
