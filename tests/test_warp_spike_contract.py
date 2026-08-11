"""M1 spike trace and benchmark contracts.

The ctrl-trace replay is opt-in because it requires a downloaded successful
pilot demonstration.  Its recorder hooks ``mujoco.mj_step`` precisely where
the final actuator controls cross the physics boundary.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import benchmarks.benchmark_warp_spike as spike_benchmark
from benchmarks.benchmark_official_spike import OfficialTraceRunner
from benchmarks.benchmark_warp_spike import (
    MEASURE_STEPS,
    MODES,
    WARMUP_STEPS,
    WORLD_COUNTS,
    BenchmarkCase,
    run_benchmark,
    run_case,
)
from benchmarks.compare_m1_reports import compare_m1_reports
from benchmarks.ctrl_trace import (
    TRACE_FORMAT_VERSION,
    TRACE_MODEL_RESET_PROTOCOL,
    load_trace,
    make_artifact,
    model_arrays_fingerprint,
    model_mjb_fingerprint,
    save_trace,
    validate_m1_trace_metadata,
)
from libero.libero.runtime import warp as warp_runtime
from libero.libero.runtime.types import CameraConfig, EnvConfig

from ._manifest import LIBERO_TASK_MAP

PILOT_SUITE = "libero_spatial"
PILOT_TASK = LIBERO_TASK_MAP[PILOT_SUITE][0]
PILOT_SEED = 0
_SHA256 = "0" * 64


@pytest.mark.static
def test_graph_ctrl_staging_layout_preserves_expanded_world_rows() -> None:
    """Graph substep copies must address [substep, world, actuator]."""
    one_world_trace = torch.arange(25 * 3, dtype=torch.float32).reshape(1, 25, 3)
    controls = one_world_trace.expand(4, -1, -1)
    assert not controls.is_contiguous()

    staging = controls.permute(1, 0, 2).contiguous()
    for substep in (0, 7, 24):
        assert torch.equal(staging[substep], controls[:, substep])


def _valid_m1_metadata() -> dict[str, Any]:
    """Small JSON-safe M1 trace identity for static benchmark contracts."""

    return {
        "trace_format_version": TRACE_FORMAT_VERSION,
        "model_reset_protocol": TRACE_MODEL_RESET_PROTOCOL,
        "suite": PILOT_SUITE,
        "task_index": 0,
        "task_name": PILOT_TASK,
        "seed": PILOT_SEED,
        "mujoco": "3.11.0",
        "robosuite": "1.5.2",
        "timestep": 0.002,
        "control_freq": 20,
        "control_decimation": 25,
        "model_mjb_sha256": _SHA256,
        "model_mjb_size": 1,
        "model_arrays_sha256": _SHA256,
        "model_array_field_count": 1,
        "model_array_total_bytes": 1,
        "model_physics_options": {},
        "model": {"nq": 1, "nv": 1, "nu": 1, "nbody": 1, "nsite": 1},
        "env_config": {
            "suite": PILOT_SUITE,
            "task_index": 0,
            "cameras": [
                {
                    "name": "agentview",
                    "height": 64,
                    "width": 64,
                    "depth": False,
                    "segmentation": None,
                }
            ],
            "horizon": 1000,
            "control_freq": 20,
            "seed": PILOT_SEED,
        },
    }


def _trace_report() -> dict[str, Any]:
    metadata = _valid_m1_metadata()
    return {
        "metadata": metadata,
        "fingerprint": {
            "mjb_sha256": metadata["model_mjb_sha256"],
            "mjb_size": metadata["model_mjb_size"],
            "mjb_matches_trace": True,
            "portable": {
                "sha256": metadata["model_arrays_sha256"],
                "field_count": metadata["model_array_field_count"],
                "total_bytes": metadata["model_array_total_bytes"],
                "physics_options": metadata["model_physics_options"],
            },
        },
    }


class _FingerprintModel:
    """Small public-array stand-in for portable-fingerprint static tests."""

    def __init__(
        self,
        *,
        core_f64: np.ndarray | None = None,
        core_f32: np.ndarray | None = None,
        mesh_pathadr: np.ndarray | None = None,
        bvh_aabb: np.ndarray | None = None,
        mesh_normal: np.ndarray | None = None,
        mesh_polynormal: np.ndarray | None = None,
    ) -> None:
        self.core_f64 = (
            np.array([0.125, -0.0, 4.0e-8], dtype=np.float64)
            if core_f64 is None
            else core_f64
        )
        self.core_f32 = (
            np.array([0.25, -0.0, 4.0e-6], dtype=np.float32)
            if core_f32 is None
            else core_f32
        )
        self.core_ids = np.array([3, 5, 8], dtype=np.int32)
        self.mesh_pathadr = (
            np.array([0, 129, 258], dtype=np.int32)
            if mesh_pathadr is None
            else mesh_pathadr
        )
        self.bvh_aabb = (
            np.array([0.0, 1.0], dtype=np.float32) if bvh_aabb is None else bvh_aabb
        )
        self.mesh_normal = (
            np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if mesh_normal is None
            else mesh_normal
        )
        self.mesh_polynormal = (
            np.array([0.0, 1.0, 0.0], dtype=np.float32)
            if mesh_polynormal is None
            else mesh_polynormal
        )
        self.paths = b"/machine-specific/assets/mesh.stl\x00"
        self.opt = SimpleNamespace()


@pytest.mark.static
def test_portable_model_fingerprint_v2_excludes_path_and_platform_derived_fields() -> (
    None
):
    baseline = _FingerprintModel()
    host_variant = _FingerprintModel(
        mesh_pathadr=np.array([0, 247, 494], dtype=np.int32),
        bvh_aabb=np.array([99.0, -99.0], dtype=np.float32),
        mesh_normal=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        mesh_polynormal=np.array([0.0, -1.0, 0.0], dtype=np.float32),
    )

    assert model_arrays_fingerprint(baseline) == model_arrays_fingerprint(host_variant)


@pytest.mark.static
def test_portable_fingerprint_v2_canonicalizes_small_float_and_signed_zero_drift() -> (
    None
):
    baseline = _FingerprintModel()
    sub_tolerance_variant = _FingerprintModel(
        core_f64=np.array([0.12500004, 0.0, -4.0e-8], dtype=np.float64),
        core_f32=np.array([0.250004, 0.0, -4.0e-6], dtype=np.float32),
    )

    assert model_arrays_fingerprint(baseline) == model_arrays_fingerprint(
        sub_tolerance_variant
    )


@pytest.mark.static
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("core_f64", np.array([0.1250002, -0.0, 4.0e-8], dtype=np.float64)),
        ("core_f32", np.array([0.25002, -0.0, 4.0e-6], dtype=np.float32)),
    ),
)
def test_portable_model_fingerprint_v2_detects_core_changes_beyond_tolerance(
    field: str, value: np.ndarray
) -> None:
    baseline = _FingerprintModel()
    changed = _FingerprintModel(**{field: value})

    assert (
        model_arrays_fingerprint(baseline)["sha256"]
        != model_arrays_fingerprint(changed)["sha256"]
    )


@dataclass(frozen=True, slots=True)
class ControlBoundary:
    """Reference state at a control boundary, including render-relevant poses."""

    qpos: np.ndarray
    qvel: np.ndarray
    body_xpos: np.ndarray
    body_xquat: np.ndarray
    site_xpos: np.ndarray
    site_xmat: np.ndarray


@dataclass(frozen=True, slots=True)
class OfficialCtrlTrace:
    """Action-aligned controls captured immediately before every physics step."""

    initial_state: np.ndarray
    actions: np.ndarray
    substep_ctrl: tuple[np.ndarray, ...]
    boundaries: tuple[ControlBoundary, ...]
    metadata: dict[str, Any]


def _capture_boundary(data: Any) -> ControlBoundary:
    return ControlBoundary(
        qpos=np.asarray(data.qpos).copy(),
        qvel=np.asarray(data.qvel).copy(),
        body_xpos=np.asarray(data.xpos).copy(),
        body_xquat=np.asarray(data.xquat).copy(),
        site_xpos=np.asarray(data.site_xpos).copy(),
        site_xmat=np.asarray(data.site_xmat).copy(),
    )


def _trace_metadata(official_env: Any) -> dict[str, Any]:
    import mujoco

    model = official_env._env.sim.model._model
    timestep = float(model.opt.timestep)
    control_freq = int(official_env.config.control_freq)
    control_decimation = round(1.0 / (control_freq * timestep))
    model_mjb_sha256, model_mjb_size = model_mjb_fingerprint(model)
    model_arrays = model_arrays_fingerprint(model)
    config = official_env.config
    env_config = {
        "suite": config.suite,
        "task_index": config.task_index,
        "cameras": [
            {
                "name": camera.name,
                "height": camera.height,
                "width": camera.width,
                "depth": camera.depth,
                "segmentation": camera.segmentation,
            }
            for camera in config.cameras
        ],
        "horizon": config.horizon,
        "control_freq": config.control_freq,
        "seed": config.seed,
    }
    return {
        "model_reset_protocol": TRACE_MODEL_RESET_PROTOCOL,
        "env_config": env_config,
        "suite": official_env.config.suite,
        "task_index": official_env.config.task_index,
        "task_name": official_env.task.name,
        "seed": official_env.config.seed,
        "mujoco": mujoco.__version__,
        "robosuite": version("robosuite"),
        "timestep": timestep,
        "control_freq": control_freq,
        "control_decimation": control_decimation,
        "model_mjb_sha256": model_mjb_sha256,
        "model_mjb_size": model_mjb_size,
        "model_arrays_sha256": model_arrays["sha256"],
        "model_array_field_count": model_arrays["field_count"],
        "model_array_total_bytes": model_arrays["total_bytes"],
        "model_physics_options": model_arrays["physics_options"],
        "model": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
            "nsite": int(model.nsite),
        },
    }


def record_official_ctrl_trace(
    official_env: Any, actions: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> OfficialCtrlTrace:
    """Record final actuator ctrl at every raw MuJoCo substep of a demo replay.

    In robosuite 1.5.2, normal physics invokes ``MjSim.step`` while
    ``lite_physics`` invokes ``MjSim.step2``. Both are narrow boundaries
    immediately before their MuJoCo physics call. Patching the exact simulation
    instance avoids relying on a module-level pybind symbol that may have been
    captured by robosuite during import.
    """

    raw_env = official_env._env
    model = raw_env.sim.model._model
    data = raw_env.sim.data._data
    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.ndim != 2 or action_array.shape[1] != 7:
        raise ValueError(f"Expected demo actions [T, 7], got {action_array.shape}.")

    metadata = _trace_metadata(official_env)
    validate_m1_trace_metadata(metadata)
    if metadata["control_freq"] != 20 or metadata["timestep"] != pytest.approx(0.002):
        raise AssertionError(f"Unexpected M1 control metadata: {metadata}.")
    if metadata["control_decimation"] != 25:
        raise AssertionError(f"Expected 25 physics substeps, got {metadata}.")

    step_method_name = "step2" if raw_env.env.lite_physics else "step"
    original_sim_step = getattr(raw_env.sim, step_method_name)
    substep_ctrl: list[np.ndarray] = []
    initial_state = np.asarray(raw_env.get_sim_state()).reshape(-1).copy()
    boundaries = [_capture_boundary(data)]

    def record_final_ctrl(*args: Any, **kwargs: Any):
        substep_ctrl.append(np.asarray(data.ctrl).copy())
        return original_sim_step(*args, **kwargs)

    monkeypatch.setattr(raw_env.sim, step_method_name, record_final_ctrl)
    controls_by_action: list[np.ndarray] = []
    for action in action_array:
        start = len(substep_ctrl)
        official_env.step(torch.from_numpy(action).unsqueeze(0))
        controls = np.asarray(substep_ctrl[start:])
        if controls.shape != (25, int(model.nu)):
            raise AssertionError(
                "Control trace must capture the final ctrl before all 25 M1 "
                f"physics substeps; got {controls.shape}."
            )
        controls_by_action.append(controls)
        boundaries.append(_capture_boundary(data))

    return OfficialCtrlTrace(
        initial_state=initial_state,
        actions=action_array.copy(),
        substep_ctrl=tuple(controls_by_action),
        boundaries=tuple(boundaries),
        metadata=metadata,
    )


def assert_cpu_raw_ctrl_replay(
    official_env: Any, trace: OfficialCtrlTrace, *, atol: float = 1e-7
) -> float:
    """Replay captured ctrl through raw MuJoCo, independent of the controller.

    A separate ``MjData`` is created from the trace's *exact* compiled model.
    Constructing a second LIBERO environment would resample model-level fixture
    placements that are not encoded by a FULLPHYSICS state.
    """

    import mujoco

    raw_env = official_env._env
    model = raw_env.sim.model._model
    data = mujoco.MjData(model)
    mujoco.mj_setState(
        model,
        data,
        trace.initial_state,
        mujoco.mjtState.mjSTATE_FULLPHYSICS,
    )
    mujoco.mj_forward(model, data)
    assert len(trace.boundaries) == len(trace.actions) + 1
    assert len(trace.substep_ctrl) == len(trace.actions)

    max_abs_error = 0.0

    def assert_boundary(actual: ControlBoundary, expected: ControlBoundary) -> None:
        nonlocal max_abs_error
        for actual_value, expected_value in zip(
            (
                actual.qpos,
                actual.qvel,
                actual.body_xpos,
                actual.body_xquat,
                actual.site_xpos,
                actual.site_xmat,
            ),
            (
                expected.qpos,
                expected.qvel,
                expected.body_xpos,
                expected.body_xquat,
                expected.site_xpos,
                expected.site_xmat,
            ),
            strict=True,
        ):
            if actual_value.size:
                max_abs_error = max(
                    max_abs_error,
                    float(np.max(np.abs(actual_value - expected_value))),
                )
            np.testing.assert_allclose(
                actual_value, expected_value, rtol=0.0, atol=atol
            )

    assert_boundary(_capture_boundary(data), trace.boundaries[0])
    for controls, expected_boundary in zip(
        trace.substep_ctrl, trace.boundaries[1:], strict=True
    ):
        for ctrl in controls:
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
        assert_boundary(_capture_boundary(data), expected_boundary)
    return max_abs_error


class _FakeSpikeRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.synchronizations = 0
        self.case_resets = 0
        self.health_checks = 0
        self.partial_reset_worlds = 1
        self.closed = False

    def physics_step(self) -> None:
        self.calls.append("physics-only")

    def render(self) -> None:
        self.calls.append("render-only")

    def step_and_render(self) -> None:
        self.calls.append("step+render")

    def partial_reset(self) -> None:
        self.calls.append("partial-reset")

    def reset_case(self) -> None:
        self.case_resets += 1

    def synchronize(self) -> None:
        self.synchronizations += 1

    def assert_healthy(self) -> None:
        self.health_checks += 1

    @property
    def trace_report(self) -> dict[str, Any]:
        return _trace_report()

    def close(self) -> None:
        self.closed = True


@pytest.mark.static
def test_spike_benchmark_profile_is_complete_and_uses_sync_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert WORLD_COUNTS == (1, 16, 64, 128, 256)
    assert MODES == ("physics-only", "render-only", "step+render", "partial-reset")
    assert (WARMUP_STEPS, MEASURE_STEPS) == (100, 1000)

    runner = _FakeSpikeRunner()
    timestamps = iter((10.0, 12.0))
    monkeypatch.setattr(spike_benchmark.time, "perf_counter", lambda: next(timestamps))
    result = run_case(runner, BenchmarkCase(worlds=16, mode="physics-only"))
    assert runner.calls == ["physics-only"] * (WARMUP_STEPS + MEASURE_STEPS)
    assert runner.synchronizations == 2
    assert runner.health_checks == 1
    assert result["worlds"] == 16
    assert result["mode"] == "physics-only"
    assert result["batch_hz"] == 500.0
    assert result["per_world_hz"] == result["batch_hz"]
    assert result["per_world_control_hz"] == result["batch_hz"]
    assert result["aggregate_world_hz"] == result["batch_hz"] * result["worlds"]
    assert result["aggregate_world_hz"] == 8000.0
    assert result["physics_substeps_per_operation"] == 25
    assert result["rendered_cameras_per_operation"] == 0
    assert not result["participates_in_speedup_gate"]


@pytest.mark.static
def test_partial_reset_throughput_counts_only_the_reset_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _FakeSpikeRunner()
    runner.partial_reset_worlds = 2
    timestamps = iter((10.0, 12.0))
    monkeypatch.setattr(spike_benchmark.time, "perf_counter", lambda: next(timestamps))
    result = run_case(runner, BenchmarkCase(worlds=16, mode="partial-reset"))
    assert result["worlds_affected"] == 2
    assert result["batch_hz"] == result["per_world_hz"] == 500.0
    assert result["aggregate_world_hz"] == 1000.0
    assert result["per_world_control_hz"] is None
    assert result["physics_substeps_per_operation"] == 0
    assert result["rendered_cameras_per_operation"] == 0
    assert not result["participates_in_speedup_gate"]


@pytest.mark.static
def test_step_and_render_declares_the_exact_m1_speedup_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _FakeSpikeRunner()
    timestamps = iter((10.0, 12.0))
    monkeypatch.setattr(spike_benchmark.time, "perf_counter", lambda: next(timestamps))

    result = run_case(runner, BenchmarkCase(worlds=128, mode="step+render"))

    assert result["per_world_control_hz"] == result["per_world_hz"] == 500.0
    assert result["physics_substeps_per_operation"] == 25
    assert result["rendered_cameras_per_operation"] == 3
    assert result["participates_in_speedup_gate"]


@pytest.mark.static
def test_spike_benchmark_results_are_json_with_all_modes_and_world_counts() -> None:
    runners: list[_FakeSpikeRunner] = []

    def make_runner(_: int) -> _FakeSpikeRunner:
        runner = _FakeSpikeRunner()
        runners.append(runner)
        return runner

    report = run_benchmark(make_runner)
    assert len(report["results"]) == len(WORLD_COUNTS) * len(MODES)
    assert all(runner.closed for runner in runners)
    assert all(runner.synchronizations == 2 * len(MODES) for runner in runners)
    assert all(runner.case_resets == len(MODES) for runner in runners)
    assert all(runner.health_checks == len(MODES) for runner in runners)
    serialized = json.loads(json.dumps(report))
    assert serialized["schema_version"] == 1
    assert serialized["backend"] == "warp"
    assert serialized["trace"] == _trace_report()
    assert serialized["profile"] == spike_benchmark.M1_WARP_PROFILE
    assert serialized["warmup_steps"] == 100


@pytest.mark.static
def test_ctrl_trace_npz_round_trip_is_versioned_and_pickle_free(tmp_path: Path) -> None:
    steps = 2
    artifact = make_artifact(
        initial_fullphysics=np.arange(3, dtype=np.float64),
        actions=np.zeros((steps, 7), dtype=np.float32),
        substep_ctrl=np.zeros((steps, 25, 1), dtype=np.float32),
        qpos=np.zeros((steps + 1, 1), dtype=np.float64),
        qvel=np.zeros((steps + 1, 1), dtype=np.float64),
        body_xpos=np.zeros((steps + 1, 1, 3), dtype=np.float64),
        body_xquat=np.zeros((steps + 1, 1, 4), dtype=np.float64),
        site_xpos=np.zeros((steps + 1, 1, 3), dtype=np.float64),
        site_xmat=np.zeros((steps + 1, 1, 9), dtype=np.float64),
        metadata=_valid_m1_metadata(),
    )
    path = tmp_path / "ctrl-trace.npz"
    save_trace(path, artifact)
    loaded = load_trace(path)

    assert loaded.metadata["trace_format_version"] == TRACE_FORMAT_VERSION
    np.testing.assert_array_equal(
        loaded.initial_fullphysics, artifact.initial_fullphysics
    )
    np.testing.assert_array_equal(loaded.actions, artifact.actions)
    np.testing.assert_array_equal(loaded.substep_ctrl, artifact.substep_ctrl)
    with np.load(path, allow_pickle=False) as archive:
        assert "metadata_json" in archive.files


@pytest.mark.static
def test_ctrl_trace_rejects_an_empty_action_sequence() -> None:
    with pytest.raises(ValueError, match="at least one action"):
        make_artifact(
            initial_fullphysics=np.arange(3, dtype=np.float64),
            actions=np.zeros((0, 7), dtype=np.float32),
            substep_ctrl=np.zeros((0, 25, 1), dtype=np.float32),
            qpos=np.zeros((1, 1), dtype=np.float64),
            qvel=np.zeros((1, 1), dtype=np.float64),
            body_xpos=np.zeros((1, 1, 3), dtype=np.float64),
            body_xquat=np.zeros((1, 1, 4), dtype=np.float64),
            site_xpos=np.zeros((1, 1, 3), dtype=np.float64),
            site_xmat=np.zeros((1, 1, 9), dtype=np.float64),
            metadata=_valid_m1_metadata(),
        )


def _small_artifact_arrays() -> dict[str, np.ndarray]:
    return {
        "initial_fullphysics": np.zeros(3, dtype=np.float64),
        "actions": np.zeros((1, 7), dtype=np.float32),
        "substep_ctrl": np.zeros((1, 25, 1), dtype=np.float32),
        "qpos": np.zeros((2, 1), dtype=np.float64),
        "qvel": np.zeros((2, 1), dtype=np.float64),
        "body_xpos": np.zeros((2, 1, 3), dtype=np.float64),
        "body_xquat": np.zeros((2, 1, 4), dtype=np.float64),
        "site_xpos": np.zeros((2, 1, 3), dtype=np.float64),
        "site_xmat": np.zeros((2, 1, 9), dtype=np.float64),
    }


@pytest.mark.static
@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    (
        ("actions", np.full((1, 7), "bad"), "real numeric dtype"),
        ("qpos", np.zeros((2, 1), dtype=np.complex64), "real numeric dtype"),
        ("qvel", np.full((2, 1), np.nan), "only finite"),
        ("substep_ctrl", np.zeros((1, 25, 2)), "model dimensions"),
        ("body_xpos", np.zeros((2, 2, 3)), "model dimensions"),
        ("initial_fullphysics", np.zeros(4), "model dimensions"),
    ),
)
def test_ctrl_trace_rejects_nonreal_nonfinite_or_wrong_model_shapes(
    field: str, bad_value: np.ndarray, message: str
) -> None:
    arrays = _small_artifact_arrays()
    arrays[field] = bad_value
    with pytest.raises((TypeError, ValueError), match=message):
        make_artifact(**arrays, metadata=_valid_m1_metadata())


@pytest.mark.static
@pytest.mark.parametrize(
    ("trace_name", "compiled_name"),
    (("mujoco", "mujoco_version"), ("robosuite", "robosuite_version")),
)
def test_official_runner_rejects_trace_compiler_version_mismatch(
    trace_name: str, compiled_name: str
) -> None:
    metadata = _valid_m1_metadata()
    compiled_metadata = SimpleNamespace(
        task_name=metadata["task_name"],
        control_substeps=metadata["control_decimation"],
        timestep=metadata["timestep"],
        mujoco_version=metadata["mujoco"],
        robosuite_version=metadata["robosuite"],
    )
    setattr(compiled_metadata, compiled_name, "unexpected-version")
    runner = object.__new__(OfficialTraceRunner)
    runner.artifact = SimpleNamespace(metadata=metadata)
    runner.compiled = SimpleNamespace(metadata=compiled_metadata, model=object())

    with pytest.raises(ValueError, match=f"{trace_name} version"):
        runner._validate_trace_compatibility()


@pytest.mark.static
@pytest.mark.parametrize("trace_name", ("mujoco", "robosuite"))
def test_warp_runner_rejects_trace_compiler_version_mismatch(trace_name: str) -> None:
    metadata = _valid_m1_metadata()
    metadata[trace_name] = "unexpected-version"
    compiled_metadata = SimpleNamespace(
        task_name=metadata["task_name"],
        control_substeps=metadata["control_decimation"],
        timestep=metadata["timestep"],
        mujoco_version="3.11.0",
        robosuite_version="1.5.2",
        nu=1,
    )
    compiled = SimpleNamespace(metadata=compiled_metadata, model=object())
    config = EnvConfig(
        suite=PILOT_SUITE,
        task_index=0,
        cameras=(CameraConfig("agentview", 64, 64),),
        seed=PILOT_SEED,
    )

    with pytest.raises(ValueError, match=f"mismatch.*{trace_name}"):
        warp_runtime._validate_trace_metadata(compiled, metadata, config, (1, 25, 1))


@pytest.mark.static
def test_m1_trace_metadata_schema_freezes_reset_lifecycle_and_ordered_cameras() -> None:
    metadata = _valid_m1_metadata()
    validate_m1_trace_metadata(metadata)

    invalid = {**metadata, "env_config": {**metadata["env_config"], "seed": 1}}
    with pytest.raises(ValueError, match="env_config.seed"):
        validate_m1_trace_metadata(invalid)

    missing = {name: value for name, value in metadata.items() if name != "task_name"}
    with pytest.raises(ValueError, match="task_name"):
        validate_m1_trace_metadata(missing)

    invalid_decimation = {**metadata, "control_decimation": 24}
    with pytest.raises(ValueError, match="exactly 25"):
        validate_m1_trace_metadata(invalid_decimation)

    missing_version = {
        name: value
        for name, value in metadata.items()
        if name != "trace_format_version"
    }
    with pytest.raises(ValueError, match="trace_format_version"):
        validate_m1_trace_metadata(missing_version)

    unsupported_version = {**metadata, "trace_format_version": 3}
    with pytest.raises(ValueError, match="must be 2"):
        validate_m1_trace_metadata(unsupported_version)


def _benchmark_report(
    backend: str,
    *,
    worlds: int,
    batch_hz: float,
    worlds_affected: int,
    include_256: bool = False,
) -> dict[str, Any]:
    profile = deepcopy(spike_benchmark.M1_WARP_PROFILE)
    if backend == "official":
        profile["partial_reset_semantics"] = "raw_fullphysics_single_world"
    result = {
        "worlds": worlds,
        "mode": "step+render",
        "elapsed_seconds": MEASURE_STEPS / batch_hz,
        "batch_hz": batch_hz,
        "per_world_hz": batch_hz,
        "aggregate_world_hz": batch_hz * worlds_affected,
        "worlds_affected": worlds_affected,
        "warmup_steps": WARMUP_STEPS,
        "measure_steps": MEASURE_STEPS,
        "per_world_control_hz": batch_hz,
        "physics_substeps_per_operation": 25,
        "rendered_cameras_per_operation": 3,
        "participates_in_speedup_gate": True,
    }
    results = [result]
    if include_256:
        results.append(
            {
                "worlds": 256,
                "mode": "step+render",
                "elapsed_seconds": MEASURE_STEPS / 21.0,
                "batch_hz": 21.0,
                "per_world_hz": 21.0,
                "aggregate_world_hz": 256.0 * 21.0,
                "worlds_affected": 256,
                "warmup_steps": WARMUP_STEPS,
                "measure_steps": MEASURE_STEPS,
                "per_world_control_hz": 21.0,
                "physics_substeps_per_operation": 25,
                "rendered_cameras_per_operation": 3,
                "participates_in_speedup_gate": True,
            }
        )
    return {
        "schema_version": 1,
        "backend": backend,
        "metadata": {
            "hostname": "same-host",
            "platform": "same-platform",
            "mujoco": "3.11.0",
            "robosuite": "1.5.2",
        },
        "trace": _trace_report(),
        "profile": profile,
        "warmup_steps": WARMUP_STEPS,
        "measure_steps": MEASURE_STEPS,
        "results": results,
    }


@pytest.mark.static
def test_m1_report_gate_checks_trace_profile_128_per_world_and_aggregate_speedup() -> (
    None
):
    official = _benchmark_report("official", worlds=1, batch_hz=10.0, worlds_affected=1)
    warp = _benchmark_report(
        "warp", worlds=128, batch_hz=20.0, worlds_affected=128, include_256=True
    )

    verdict = compare_m1_reports(official, warp)

    assert verdict["hard_gate_passed"]
    assert verdict["go_no_go_5x"] == "go"
    assert verdict["aggregate_speedup"] == 256.0
    assert verdict["record_only_256"]["worlds"] == 256
    assert not verdict["partial_reset_participates_in_speedup_gate"]


@pytest.mark.static
def test_m1_report_gate_rejects_bad_denominator_and_reports_no_go() -> None:
    official = _benchmark_report("official", worlds=1, batch_hz=10.0, worlds_affected=1)
    slow_warp = _benchmark_report(
        "warp", worlds=128, batch_hz=0.03, worlds_affected=128
    )
    verdict = compare_m1_reports(official, slow_warp)
    assert not verdict["hard_gate_passed"]
    assert verdict["go_no_go_5x"] == "no-go"

    wrong_denominator = _benchmark_report(
        "warp", worlds=128, batch_hz=20.0, worlds_affected=128
    )
    wrong_denominator["results"][0]["aggregate_world_hz"] = 20.0
    with pytest.raises(ValueError, match="aggregate_world_hz"):
        compare_m1_reports(official, wrong_denominator)

    forged_batch_rate = _benchmark_report(
        "warp", worlds=128, batch_hz=20.0, worlds_affected=128
    )
    forged_batch_rate["results"][0]["elapsed_seconds"] = 100.0
    with pytest.raises(ValueError, match="measure_steps / elapsed_seconds"):
        compare_m1_reports(official, forged_batch_rate)

    fake_physics = _benchmark_report(
        "warp", worlds=128, batch_hz=20.0, worlds_affected=128
    )
    fake_physics["results"][0]["rendered_cameras_per_operation"] = 0
    with pytest.raises(ValueError, match="rendered_cameras_per_operation"):
        compare_m1_reports(official, fake_physics)


@pytest.mark.static
def test_m1_report_gate_rejects_cross_host_and_mutated_equal_profiles() -> None:
    official = _benchmark_report("official", worlds=1, batch_hz=10.0, worlds_affected=1)
    warp = _benchmark_report("warp", worlds=128, batch_hz=20.0, worlds_affected=128)
    warp["metadata"]["hostname"] = "different-host"
    with pytest.raises(ValueError, match="hostname differs"):
        compare_m1_reports(official, warp)

    warp["metadata"]["hostname"] = official["metadata"]["hostname"]
    for report in (official, warp):
        report["profile"]["cameras"][0]["width"] = 127
    with pytest.raises(ValueError, match="fixed M1 profile"):
        compare_m1_reports(official, warp)


@pytest.mark.static
def test_m1_report_gate_rejects_trace_and_runtime_version_tampering() -> None:
    official = _benchmark_report("official", worlds=1, batch_hz=10.0, worlds_affected=1)
    warp = _benchmark_report("warp", worlds=128, batch_hz=20.0, worlds_affected=128)

    warp["metadata"]["mujoco"] = "unexpected-version"
    with pytest.raises(ValueError, match="runtime mujoco version"):
        compare_m1_reports(official, warp)

    warp["trace"]["metadata"]["mujoco"] = "unexpected-version"
    with pytest.raises(ValueError, match="different trace mujoco"):
        compare_m1_reports(official, warp)

    warp["metadata"]["robosuite"] = "unexpected-version"
    with pytest.raises(ValueError, match="runtime robosuite version"):
        compare_m1_reports(official, warp)


def _pilot_demo_path() -> Path | None:
    explicit_path = os.environ.get("LIBERO_PILOT_DEMO")
    if explicit_path:
        return Path(explicit_path)
    demo_root = os.environ.get("LIBERO_DEMO_ROOT")
    if not demo_root:
        return None
    expected_name = f"{PILOT_TASK}_demo.hdf5"
    matches = sorted(Path(demo_root).rglob(expected_name))
    return matches[0] if matches else None


def _optional_trace_output_path() -> Path | None:
    configured = os.environ.get("LIBERO_CTRL_TRACE_OUTPUT")
    if not configured:
        return None
    path = Path(configured).expanduser().resolve()
    allowed_roots = (Path("/tmp").resolve(), (Path.home() / ".Trash").resolve())
    if not any(path.is_relative_to(root) for root in allowed_roots):
        raise ValueError(
            f"LIBERO_CTRL_TRACE_OUTPUT must be under /tmp or ~/.Trash; got {path}."
        )
    return path


def _trace_artifact(trace: OfficialCtrlTrace, *, cpu_replay_max_abs_error: float):
    boundaries = trace.boundaries
    return make_artifact(
        initial_fullphysics=trace.initial_state,
        actions=trace.actions,
        substep_ctrl=np.stack(trace.substep_ctrl),
        qpos=np.stack([boundary.qpos for boundary in boundaries]),
        qvel=np.stack([boundary.qvel for boundary in boundaries]),
        body_xpos=np.stack([boundary.body_xpos for boundary in boundaries]),
        body_xquat=np.stack([boundary.body_xquat for boundary in boundaries]),
        site_xpos=np.stack([boundary.site_xpos for boundary in boundaries]),
        site_xmat=np.stack([boundary.site_xmat for boundary in boundaries]),
        metadata={
            **trace.metadata,
            "cpu_raw_replay_max_abs_error": cpu_replay_max_abs_error,
        },
    )


@pytest.mark.nightly
@pytest.mark.official_integration
def test_public_successful_pilot_demo_ctrl_trace_and_cpu_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record all aligned demo actions and reproduce them through raw CPU MuJoCo."""

    demo_path = _pilot_demo_path()
    if demo_path is None:
        pytest.skip(
            "set LIBERO_PILOT_DEMO (or LIBERO_DEMO_ROOT containing the public "
            f"{PILOT_TASK}_demo.hdf5) to run the successful-demo ctrl trace"
        )
    if not demo_path.is_file():
        pytest.skip(f"pilot demo does not exist: {demo_path}")

    import h5py

    from libero.libero.runtime import CameraConfig, EnvConfig, make_env

    with h5py.File(demo_path, "r") as dataset:
        data = dataset["data"]
        demo_key = sorted(data.keys())[0]
        demo = data[demo_key]
        states = demo["states"][:]
        actions = demo["actions"][:]
    assert len(states) == len(actions)
    assert len(actions) > 0

    config = EnvConfig(
        suite=PILOT_SUITE,
        task_index=0,
        cameras=[CameraConfig("agentview", 64, 64)],
        horizon=max(1000, len(actions) + 1),
        seed=PILOT_SEED,
    )
    env = make_env(config)
    try:
        # Collector alignment: states[0] is the reset boundary for actions[0].
        env._env.reset()
        env._env.set_state(states[0])
        env._env.sim.forward()
        trace = record_official_ctrl_trace(env, actions, monkeypatch)
        assert trace.actions.shape == actions.shape
        assert trace.initial_state.ndim == 1
        assert len(trace.boundaries) == len(actions) + 1
        assert trace.metadata["control_decimation"] == 25
        assert trace.metadata["model_reset_protocol"] == TRACE_MODEL_RESET_PROTOCOL
        assert trace.metadata["env_config"] == {
            "suite": PILOT_SUITE,
            "task_index": 0,
            "cameras": [
                {
                    "name": "agentview",
                    "height": 64,
                    "width": 64,
                    "depth": False,
                    "segmentation": None,
                }
            ],
            "horizon": max(1000, len(actions) + 1),
            "control_freq": 20,
            "seed": PILOT_SEED,
        }
        assert trace.metadata["suite"] == PILOT_SUITE
        assert trace.metadata["task_index"] == 0
        assert trace.metadata["task_name"] == PILOT_TASK
        assert trace.metadata["seed"] == PILOT_SEED
        assert trace.metadata["model_mjb_size"] > 0
        assert trace.metadata["model_array_field_count"] > 0
        assert trace.metadata["model_array_total_bytes"] > 0
        assert env._env.check_success()

        from libero.libero.runtime import TaskCompiler

        compiled = TaskCompiler().compile(config)
        try:
            compiled_hash, compiled_size = model_mjb_fingerprint(compiled.model)
            compiled_arrays = model_arrays_fingerprint(compiled.model)
            assert compiled_hash == trace.metadata["model_mjb_sha256"]
            assert compiled_size == trace.metadata["model_mjb_size"]
            assert compiled_arrays["sha256"] == trace.metadata["model_arrays_sha256"]
            assert (
                compiled_arrays["field_count"]
                == trace.metadata["model_array_field_count"]
            )
            assert (
                compiled_arrays["total_bytes"]
                == trace.metadata["model_array_total_bytes"]
            )
        finally:
            compiled.close()

        cpu_replay_max_abs_error = assert_cpu_raw_ctrl_replay(env, trace)
        assert cpu_replay_max_abs_error <= 1e-7
        output_path = _optional_trace_output_path()
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_trace(
                output_path,
                _trace_artifact(
                    trace, cpu_replay_max_abs_error=cpu_replay_max_abs_error
                ),
            )
            saved_trace = load_trace(output_path)
            np.testing.assert_array_equal(saved_trace.actions, trace.actions)
            assert saved_trace.substep_ctrl.shape[1] == 25
    finally:
        env.close()
