"""Measure the single-world official baseline for the M1 MJWarp spike.

This command deliberately replays the actuator ``ctrl`` trace at MuJoCo's raw
physics boundary.  It is therefore a control-replay baseline for M1, not a
benchmark of the 7-D policy-action / OSC path planned for M2.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from benchmarks.ctrl_trace import (
    CtrlTraceArtifact,
    load_trace,
    model_arrays_fingerprint,
    model_mjb_fingerprint,
    validate_m1_trace_metadata,
)
from libero.libero.runtime import CameraConfig, EnvConfig, TaskCompiler

MODES = ("physics-only", "render-only", "step+render", "partial-reset")
_CONTROL_MODES = frozenset({"physics-only", "step+render"})
WARMUP_STEPS = 100
MEASURE_STEPS = 1000
PROFILE_CAMERAS = (
    CameraConfig("agentview", 128, 128),
    CameraConfig("robot0_eye_in_hand", 128, 128),
    CameraConfig("sideview", 128, 128),
)
_FULLPHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One deterministic single-world baseline measurement."""

    worlds: int
    mode: str
    warmup_steps: int = WARMUP_STEPS
    measure_steps: int = MEASURE_STEPS


class OfficialTraceRunner:
    """Replay one validated M1 trace through its exact official MuJoCo model."""

    def __init__(self, artifact: CtrlTraceArtifact) -> None:
        validate_m1_trace_metadata(artifact.metadata)
        self.artifact = artifact
        self.config = _env_config_from_trace(artifact.metadata)
        self.compiled = TaskCompiler().compile(self.config)
        try:
            self._validate_trace_compatibility()
            self._control_env = self.compiled.official_env._env
            self.model = self.compiled.model
            self.data = self._control_env.sim.data._data
            self._initial_fullphysics = _validated_fullphysics(
                artifact.initial_fullphysics, self.model
            )
            self._ctrl = _validated_ctrl_trace(artifact.substep_ctrl, self.model)
            self._cursor = 0
            self._reset_before_next_control = False
            self._last_images: tuple[np.ndarray, ...] = ()
            self._validate_profile_cameras()
            self.reset_case()
        except Exception:
            self.compiled.close()
            raise

    def reset_case(self) -> None:
        """Restore the same trace boundary and control cursor before every case."""
        self._restore_initial_state()
        self._cursor = 0
        self._reset_before_next_control = False

    def physics_step(self) -> None:
        """Replay one control boundary: exactly 25 recorded actuator substeps."""
        if self._reset_before_next_control:
            self._restore_initial_state()
            self._reset_before_next_control = False
        ctrl = self._ctrl[self._cursor]
        for substep in ctrl:
            self.data.ctrl[:] = substep
            mujoco.mj_step(self.model, self.data)
        self._cursor += 1
        if self._cursor == len(self._ctrl):
            self._cursor = 0
            self._reset_before_next_control = True

    def render(self) -> None:
        """Render three top-left, contiguous NHWC uint8 profile images on CPU."""
        images = []
        for camera in PROFILE_CAMERAS:
            image = self._control_env.sim.render(
                camera_name=camera.name,
                width=camera.width,
                height=camera.height,
                depth=False,
            )
            value = np.asarray(image)
            if value.shape != (camera.height, camera.width, 3):
                raise RuntimeError(
                    f"Official renderer returned {value.shape} for {camera.name!r}; "
                    f"expected {(camera.height, camera.width, 3)}."
                )
            # MjSim.render is raw OpenGL bottom-left output. Unlike robosuite's
            # camera observable, this direct render call does not apply its
            # IMAGE_CONVENTION mapping, so always normalize it here.
            images.append(np.ascontiguousarray(value[::-1], dtype=np.uint8))
        self._last_images = tuple(images)

    def step_and_render(self) -> None:
        """Replay one control boundary then materialize the three RGB outputs."""
        self.physics_step()
        self.render()

    def partial_reset(self) -> None:
        """Reset the sole world through raw FULLPHYSICS state restoration.

        This is intentionally a physics state-scatter microbenchmark, matching
        M1's controller-free Warp partial reset rather than high-level episode
        reset semantics.
        """
        self._restore_initial_state()

    def close(self) -> None:
        """Release the official simulation and offscreen renderer."""
        self.compiled.close()

    def assert_healthy(self, phase: str) -> None:
        """Fail explicitly when a measured raw-MuJoCo state is non-finite."""
        values = {
            "time": np.asarray(self.data.time),
            "qpos": np.asarray(self.data.qpos),
            "qvel": np.asarray(self.data.qvel),
            "act": np.asarray(self.data.act),
            "ctrl": np.asarray(self.data.ctrl),
            "body_xpos": np.asarray(self.data.xpos),
            "body_xquat": np.asarray(self.data.xquat),
            "site_xpos": np.asarray(self.data.site_xpos),
            "site_xmat": np.asarray(self.data.site_xmat),
        }
        nonfinite = [
            name for name, value in values.items() if not np.isfinite(value).all()
        ]
        if nonfinite:
            raise RuntimeError(
                f"Official benchmark {phase} produced non-finite values in {nonfinite}."
            )

    @property
    def fingerprint_report(self) -> dict[str, Any]:
        """Return exact local MJB evidence and the portable hard-gate digest."""
        return self._fingerprint_report

    def _restore_initial_state(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(
            self.model, self.data, self._initial_fullphysics, _FULLPHYSICS
        )
        mujoco.mj_forward(self.model, self.data)

    def _validate_trace_compatibility(self) -> None:
        metadata = self.artifact.metadata
        if metadata.get("task_name") != self.compiled.metadata.task_name:
            raise ValueError(
                "Ctrl trace task_name does not match the exact compiled task: "
                f"{metadata.get('task_name')!r} != "
                f"{self.compiled.metadata.task_name!r}."
            )
        if (
            metadata.get("control_decimation")
            != self.compiled.metadata.control_substeps
        ):
            raise ValueError(
                "Ctrl trace control_decimation does not match the compiled model."
            )
        if metadata.get("timestep") != self.compiled.metadata.timestep:
            raise ValueError("Ctrl trace timestep does not match the compiled model.")
        for trace_name, compiled_name in (
            ("mujoco", "mujoco_version"),
            ("robosuite", "robosuite_version"),
        ):
            trace_version = metadata.get(trace_name)
            compiled_version = getattr(self.compiled.metadata, compiled_name)
            if trace_version != compiled_version:
                raise ValueError(
                    f"Ctrl trace {trace_name} version does not match the exact "
                    f"compiled task: {trace_version!r} != {compiled_version!r}."
                )

        arrays = model_arrays_fingerprint(self.compiled.model)
        for name, key in (
            ("sha256", "model_arrays_sha256"),
            ("field_count", "model_array_field_count"),
            ("total_bytes", "model_array_total_bytes"),
            ("physics_options", "model_physics_options"),
        ):
            if metadata.get(key) != arrays[name]:
                raise ValueError(
                    f"Ctrl trace portable model fingerprint mismatch: {key}."
                )

        mjb_hash, mjb_size = model_mjb_fingerprint(self.compiled.model)
        mjb_matches = (
            metadata.get("model_mjb_sha256") == mjb_hash
            and metadata.get("model_mjb_size") == mjb_size
        )
        if not mjb_matches:
            warnings.warn(
                "Ctrl trace MJB fingerprint differs across hosts; portable model "
                "arrays remain the hard compatibility gate. "
                f"trace_sha256={metadata.get('model_mjb_sha256')!r}, "
                f"compiled_sha256={mjb_hash!r}, "
                f"trace_size={metadata.get('model_mjb_size')!r}, "
                f"compiled_size={mjb_size!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._fingerprint_report = {
            "mjb_sha256": mjb_hash,
            "mjb_size": mjb_size,
            "mjb_matches_trace": mjb_matches,
            "portable": arrays,
        }

    def _validate_profile_cameras(self) -> None:
        missing = [
            camera.name
            for camera in PROFILE_CAMERAS
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera.name)
            < 0
        ]
        if missing:
            raise ValueError(
                "Exact trace model does not contain required M1 profile cameras: "
                f"{missing}."
            )


def _env_config_from_trace(metadata: Mapping[str, Any]) -> EnvConfig:
    raw = metadata["env_config"]
    if not isinstance(raw, Mapping):
        raise TypeError("Ctrl trace env_config must be a mapping.")
    cameras = tuple(CameraConfig(**camera) for camera in raw["cameras"])
    return EnvConfig(
        suite=raw["suite"],
        task_index=raw["task_index"],
        cameras=cameras,
        horizon=raw["horizon"],
        control_freq=raw["control_freq"],
        seed=raw["seed"],
    )


def _validated_fullphysics(value: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    state = np.ascontiguousarray(value, dtype=np.float64)
    expected = mujoco.mj_stateSize(model, _FULLPHYSICS)
    if state.ndim != 1 or state.shape[0] != expected:
        raise ValueError(
            "Trace initial_fullphysics must have shape "
            f"[{expected}], got {list(state.shape)}."
        )
    if not np.isfinite(state).all():
        raise ValueError("Trace initial_fullphysics contains non-finite values.")
    return state


def _validated_ctrl_trace(value: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    ctrl = np.ascontiguousarray(value, dtype=np.float64)
    if ctrl.ndim != 3 or ctrl.shape[0] == 0 or ctrl.shape[1:] != (25, model.nu):
        raise ValueError(
            "Trace substep_ctrl must have non-empty shape "
            f"[T, 25, {model.nu}], got {list(ctrl.shape)}."
        )
    if not np.isfinite(ctrl).all():
        raise ValueError("Trace substep_ctrl contains non-finite values.")
    return ctrl


def _operation(runner: OfficialTraceRunner, mode: str) -> Callable[[], None]:
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
            f"Unknown benchmark mode {mode!r}; expected {MODES}."
        ) from error


def _mode_semantics(mode: str) -> dict[str, int | bool]:
    """Return the fixed M1 work count and speedup-gate role for one mode."""
    if mode not in MODES:
        raise ValueError(f"Unknown benchmark mode {mode!r}; expected {MODES}.")
    return {
        "physics_substeps_per_operation": 25 if mode in _CONTROL_MODES else 0,
        "rendered_cameras_per_operation": (
            len(PROFILE_CAMERAS) if mode in {"render-only", "step+render"} else 0
        ),
        # M1's 10x/5x comparison is intentionally only the full control and
        # three-camera workload. Partial reset has distinct selected-world
        # semantics and is report-only.
        "participates_in_speedup_gate": mode == "step+render",
    }


def run_case(runner: OfficialTraceRunner, case: BenchmarkCase) -> dict[str, Any]:
    """Warm up and measure one case without CUDA synchronization or setup time."""
    if case.worlds != 1:
        raise ValueError("The official baseline supports exactly one world.")
    runner.reset_case()
    operation = _operation(runner, case.mode)
    for _ in range(case.warmup_steps):
        operation()
    runner.assert_healthy("warmup")
    # Official MuJoCo calls are synchronous CPU operations. Do not add CUDA
    # synchronization or include compile, trace loading, or renderer setup.
    start = time.perf_counter()
    for _ in range(case.measure_steps):
        operation()
    elapsed_seconds = time.perf_counter() - start
    runner.assert_healthy("measurement")
    batch_hz = case.measure_steps / elapsed_seconds
    semantics = _mode_semantics(case.mode)
    return {
        **asdict(case),
        "elapsed_seconds": elapsed_seconds,
        "batch_hz": batch_hz,
        "per_world_hz": batch_hz,
        "per_world_control_hz": batch_hz if case.mode in _CONTROL_MODES else None,
        "aggregate_world_hz": batch_hz,
        **semantics,
        "worlds_affected": 1,
        "worlds_affected_per_operation": 1,
    }


def runtime_metadata() -> dict[str, Any]:
    """Capture CPU, render, and package information needed to reproduce a run."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "mujoco": mujoco.__version__,
        "robosuite": version("robosuite"),
        "numpy": np.__version__,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }


def run_benchmark(
    artifact: CtrlTraceArtifact, *, modes: Sequence[str] = MODES
) -> dict[str, Any]:
    """Run every requested single-world M1 official baseline case."""
    runner = OfficialTraceRunner(artifact)
    try:
        results = [
            run_case(runner, BenchmarkCase(worlds=1, mode=mode)) for mode in modes
        ]
        return {
            "schema_version": 1,
            "backend": "official",
            "metadata": runtime_metadata(),
            "trace": {
                "metadata": artifact.metadata,
                "fingerprint": runner.fingerprint_report,
            },
            "profile": {
                "control_input": "recorded_actuator_ctrl",
                "control_substeps": 25,
                "rgb_format": "top-left NHWC uint8",
                "cameras": [asdict(camera) for camera in PROFILE_CAMERAS],
                "partial_reset_semantics": "raw_fullphysics_single_world",
                "speedup_gate_mode": "step+render",
                "partial_reset_participates_in_speedup_gate": False,
            },
            "warmup_steps": WARMUP_STEPS,
            "measure_steps": MEASURE_STEPS,
            "results": results,
        }
    finally:
        runner.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the trace input and fixed M1 benchmark selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace", type=Path, help="Versioned M1 official ctrl-trace NPZ."
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
    """Run the selected official baseline cases and emit reproducible JSON."""
    args = parse_args(argv)
    report = run_benchmark(load_trace(args.trace), modes=tuple(args.modes))
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(serialized)
    else:
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
