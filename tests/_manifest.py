"""Read the benchmark task manifest without importing simulator dependencies."""

from __future__ import annotations

import runpy

from .conftest import BENCHMARK_ROOT

_MANIFEST = runpy.run_path(
    str(BENCHMARK_ROOT / "benchmark" / "libero_suite_task_map.py")
)
LIBERO_TASK_MAP: dict[str, list[str]] = _MANIFEST["libero_task_map"]
