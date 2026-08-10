"""Reproducible M1 MJWarp spike benchmark harness.

The benchmark deliberately fails before the Warp backend is available instead
of silently measuring the official CPU adapter.  A future Warp runner supplies
the small protocol consumed here, keeping timing methodology independent of
backend implementation details.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

WORLD_COUNTS = (1, 16, 64, 128, 256)
MODES = ("physics-only", "render-only", "step+render", "partial-reset")
_CONTROL_MODES = frozenset({"physics-only", "step+render"})
WARMUP_STEPS = 100
MEASURE_STEPS = 1000
REPORT_SCHEMA_VERSION = 1
PROFILE_CAMERAS = (
    {
        "name": "agentview",
        "height": 128,
        "width": 128,
        "depth": False,
        "segmentation": None,
    },
    {
        "name": "robot0_eye_in_hand",
        "height": 128,
        "width": 128,
        "depth": False,
        "segmentation": None,
    },
    {
        "name": "sideview",
        "height": 128,
        "width": 128,
        "depth": False,
        "segmentation": None,
    },
)
M1_WARP_PROFILE = {
    "control_input": "recorded_actuator_ctrl",
    "control_substeps": 25,
    "rgb_format": "top-left NHWC uint8",
    "cameras": list(PROFILE_CAMERAS),
    "partial_reset_semantics": "raw_fullphysics_selected_worlds",
    "speedup_gate_mode": "step+render",
    "partial_reset_participates_in_speedup_gate": False,
}


class WarpSpikeRunner(Protocol):
    """Minimal device-resident operations required by the spike benchmark."""

    def physics_step(self) -> None: ...

    def render(self) -> None: ...

    def step_and_render(self) -> None: ...

    def partial_reset(self) -> None: ...

    def reset_case(self) -> None:
        """Restore trace initial state and cursor before one benchmark mode."""

    @property
    def partial_reset_worlds(self) -> int:
        """Number of worlds changed by one partial-reset operation."""

    def synchronize(self) -> None:
        """Synchronize the CUDA stream before or after a timed region."""

    def assert_healthy(self) -> None:
        """Check non-finite state / overflow after a synchronized measurement."""

    @property
    def trace_report(self) -> Mapping[str, Any]:
        """JSON-safe trace metadata and local model fingerprint for the report.

        The mapping must contain exactly ``metadata`` (the full, validated M1
        trace metadata) and ``fingerprint``.  ``fingerprint`` must expose the
        locally compiled ``mjb_sha256``, ``mjb_size``, ``mjb_matches_trace``,
        and portable model fingerprint.  This is intentionally the only
        reporting-specific runner property: the benchmark owns its fixed M1
        three-camera profile.
        """

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    worlds: int
    mode: str
    warmup_steps: int = WARMUP_STEPS
    measure_steps: int = MEASURE_STEPS


def _operation(runner: WarpSpikeRunner, mode: str) -> Callable[[], None]:
    operations = {
        "physics-only": runner.physics_step,
        "render-only": runner.render,
        "step+render": runner.step_and_render,
        "partial-reset": runner.partial_reset,
    }
    try:
        return operations[mode]
    except KeyError as error:
        raise ValueError(
            f"Unknown benchmark mode {mode!r}; expected one of {MODES}."
        ) from error


def _synchronize_boundary(runner: WarpSpikeRunner) -> None:
    """Synchronize runner work and the CUDA device at a timing boundary."""

    import torch

    runner.synchronize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _worlds_affected(runner: WarpSpikeRunner, case: BenchmarkCase) -> int:
    if case.mode != "partial-reset":
        return case.worlds
    value = getattr(runner, "partial_reset_worlds", None)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= case.worlds
    ):
        raise ValueError(
            "partial-reset benchmark runner must expose partial_reset_worlds "
            f"in [1, {case.worlds}], got {value!r}."
        )
    return value


def _mode_semantics(mode: str) -> dict[str, int | bool]:
    """Return fixed M1 work counts and speedup-gate role for one mode."""

    if mode not in MODES:
        raise ValueError(f"Unknown benchmark mode {mode!r}; expected one of {MODES}.")
    return {
        "physics_substeps_per_operation": 25 if mode in _CONTROL_MODES else 0,
        "rendered_cameras_per_operation": (
            len(PROFILE_CAMERAS) if mode in {"render-only", "step+render"} else 0
        ),
        "participates_in_speedup_gate": mode == "step+render",
    }


def _trace_report_from_runner(runner: WarpSpikeRunner) -> dict[str, Any]:
    """Validate and detach the trace identity supplied by a device runner."""

    from benchmarks.ctrl_trace import validate_m1_trace_metadata

    try:
        supplied = runner.trace_report
    except AttributeError as error:
        raise ValueError(
            "Warp benchmark runner must expose the trace_report reporting contract."
        ) from error
    if not isinstance(supplied, Mapping) or set(supplied) != {
        "metadata",
        "fingerprint",
    }:
        raise ValueError(
            "Warp benchmark trace_report must contain exactly metadata and fingerprint."
        )
    metadata = supplied["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("Warp benchmark trace_report.metadata must be a mapping.")
    validate_m1_trace_metadata(metadata)
    fingerprint = supplied["fingerprint"]
    if not isinstance(fingerprint, Mapping):
        raise TypeError("Warp benchmark trace_report.fingerprint must be a mapping.")
    required_fingerprint = {"mjb_sha256", "mjb_size", "mjb_matches_trace", "portable"}
    if set(fingerprint) != required_fingerprint:
        raise ValueError(
            "Warp benchmark trace_report.fingerprint must contain exactly "
            f"{sorted(required_fingerprint)}."
        )
    portable = fingerprint["portable"]
    if not isinstance(portable, Mapping):
        raise TypeError(
            "Warp benchmark trace_report.fingerprint.portable must be a mapping."
        )
    portable_fields = {"sha256", "field_count", "total_bytes", "physics_options"}
    if set(portable) != portable_fields:
        raise ValueError(
            "Warp benchmark portable fingerprint must contain exactly "
            f"{sorted(portable_fields)}."
        )
    if portable["sha256"] != metadata["model_arrays_sha256"]:
        raise ValueError(
            "Warp benchmark portable SHA256 does not match trace metadata."
        )
    if portable["field_count"] != metadata["model_array_field_count"]:
        raise ValueError(
            "Warp benchmark portable field count does not match trace metadata."
        )
    if portable["total_bytes"] != metadata["model_array_total_bytes"]:
        raise ValueError(
            "Warp benchmark portable byte count does not match trace metadata."
        )
    if portable["physics_options"] != metadata["model_physics_options"]:
        raise ValueError(
            "Warp benchmark portable physics options do not match trace metadata."
        )
    if not isinstance(fingerprint["mjb_matches_trace"], bool):
        raise TypeError("Warp benchmark fingerprint.mjb_matches_trace must be a bool.")
    try:
        serialized = json.dumps(supplied, sort_keys=True)
    except TypeError as error:
        raise TypeError(
            "Warp benchmark trace_report must be JSON serializable."
        ) from error
    return json.loads(serialized)


def run_case(runner: WarpSpikeRunner, case: BenchmarkCase) -> dict[str, Any]:
    """Warm up then time a case with explicit device synchronization boundaries."""

    operation = _operation(runner, case.mode)
    for _ in range(case.warmup_steps):
        operation()
    _synchronize_boundary(runner)
    start = time.perf_counter()
    for _ in range(case.measure_steps):
        operation()
    _synchronize_boundary(runner)
    elapsed_seconds = time.perf_counter() - start
    # Deliberately exclude health reductions and diagnostic copies from the
    # performance interval; the preceding synchronization makes this check
    # observe every measured device operation.
    runner.assert_healthy()
    worlds_affected = _worlds_affected(runner, case)
    # Physics / render operations advance every world, whereas one partial
    # reset affects only the runner-declared subset.
    batch_hz = case.measure_steps / elapsed_seconds
    semantics = _mode_semantics(case.mode)
    return {
        **asdict(case),
        "elapsed_seconds": elapsed_seconds,
        "worlds_affected": worlds_affected,
        "batch_hz": batch_hz,
        "per_world_hz": batch_hz,
        "per_world_control_hz": batch_hz if case.mode in _CONTROL_MODES else None,
        "aggregate_world_hz": batch_hz * worlds_affected,
        **semantics,
    }


def runtime_metadata() -> dict[str, Any]:
    """Capture hardware and package versions alongside every JSON result."""

    import torch

    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }
    if torch.cuda.is_available():
        metadata.update(
            cuda=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(),
            gpu_count=torch.cuda.device_count(),
        )
    try:
        import mujoco

        metadata["mujoco"] = mujoco.__version__
    except ImportError:
        metadata["mujoco"] = None
    try:
        import warp as wp

        metadata["warp"] = wp.__version__
    except ImportError:
        metadata["warp"] = None
    try:
        from importlib.metadata import version

        metadata["robosuite"] = version("robosuite")
    except ImportError:
        metadata["robosuite"] = None
    return metadata


def run_benchmark(
    make_runner: Callable[[int], WarpSpikeRunner],
    *,
    worlds: Sequence[int] = WORLD_COUNTS,
    modes: Sequence[str] = MODES,
) -> dict[str, Any]:
    """Run every requested world-count/mode case and return JSON-safe results."""

    results = []
    trace_report: dict[str, Any] | None = None
    for world_count in worlds:
        if world_count not in WORLD_COUNTS:
            raise ValueError(
                f"Unsupported world count {world_count}; "
                f"expected one of {WORLD_COUNTS}."
            )
        runner = make_runner(world_count)
        try:
            candidate_trace_report = _trace_report_from_runner(runner)
            if trace_report is None:
                trace_report = candidate_trace_report
            elif candidate_trace_report != trace_report:
                raise ValueError(
                    "Warp benchmark runners disagreed on trace identity across "
                    "world counts."
                )
            for mode in modes:
                runner.reset_case()
                results.append(run_case(runner, BenchmarkCase(world_count, mode)))
        finally:
            runner.close()
    if trace_report is None:
        raise ValueError("Warp benchmark requires at least one world count.")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "backend": "warp",
        "metadata": runtime_metadata(),
        "trace": trace_report,
        "profile": M1_WARP_PROFILE,
        "warmup_steps": WARMUP_STEPS,
        "measure_steps": MEASURE_STEPS,
        "results": results,
    }


def _load_warp_runner_factory() -> Callable[[int], WarpSpikeRunner]:
    """Load the future M1 Warp runner without falling back to official CPU code."""

    try:
        from libero.libero.runtime.warp import make_spike_benchmark_runner
    except ImportError as error:
        raise RuntimeError(
            "MJWarp spike backend is unavailable. Install/build the M1 Warp backend "
            "before running this benchmark; the official backend is intentionally "
            "not a fallback."
        ) from error
    return make_spike_benchmark_runner


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worlds", nargs="+", type=int, choices=WORLD_COUNTS, default=WORLD_COUNTS
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON to this path instead of stdout.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        result = run_benchmark(
            _load_warp_runner_factory(),
            worlds=tuple(args.worlds),
            modes=tuple(args.modes),
        )
    except RuntimeError as error:
        raise SystemExit(f"benchmark unavailable: {error}") from error
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(serialized)
    else:
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
