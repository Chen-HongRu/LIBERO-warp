"""M1 MJWarp spike runner; policy-action OSC support is intentionally deferred."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any
from warnings import warn

import torch

from libero.libero.runtime.compiler import TaskCompiler
from libero.libero.runtime.types import CameraConfig, EnvConfig
from libero.libero.warp import MJWarpSpike


class _SpikeBenchmarkRunner:
    """Adapter consumed by :mod:`benchmarks.benchmark_warp_spike`."""

    def __init__(
        self,
        spike: MJWarpSpike,
        ctrl_trace: torch.Tensor,
        initial_fullphysics: torch.Tensor,
        trace_report: Mapping[str, Any],
    ) -> None:
        self.spike = spike
        self.trace = ctrl_trace
        self.cursor = 0
        self._reset_before_next_control = False
        self.initial_fullphysics = initial_fullphysics
        # Serialize through JSON so the cached artifact metadata and fingerprint
        # remain JSON-safe and detached from the trace archive's mutable dict.
        self._trace_report = json.loads(json.dumps(trace_report, sort_keys=True))
        self.reset_world_ids = torch.arange(
            max(1, spike.num_worlds // 8), device=spike.device, dtype=torch.int64
        )
        self.reset_state_ids = torch.zeros_like(self.reset_world_ids)

    def _next_ctrl(self) -> torch.Tensor:
        ctrl = self.trace[:, self.cursor]
        self.cursor += 1
        return ctrl

    def physics_step(self) -> None:
        # The trace is cyclic only at a control boundary.  Deferring this reset
        # lets step+render still render the final control state of a trace.
        if self._reset_before_next_control:
            self.spike.reset_fullphysics_prevalidated(self.initial_fullphysics)
            self._reset_before_next_control = False
        self.spike.replay_ctrl(self._next_ctrl(), check_health=False, validate=False)
        if self.cursor == self.trace.shape[1]:
            self.cursor = 0
            self._reset_before_next_control = True

    def render(self) -> None:
        self.spike.render(check_health=False)

    def step_and_render(self) -> None:
        self.physics_step()
        self.render()

    def partial_reset(self) -> None:
        self.spike.reset_partial_prevalidated(
            self.reset_world_ids, self.reset_state_ids
        )

    @property
    def partial_reset_worlds(self) -> int:
        """Number of prevalidated worlds reset by :meth:`partial_reset`."""
        return int(self.reset_world_ids.numel())

    @property
    def trace_report(self) -> Mapping[str, Any]:
        """Return a detached JSON-safe report captured before runner setup closes."""
        return json.loads(json.dumps(self._trace_report, sort_keys=True))

    def reset_case(self) -> None:
        """Restore the exact trace boundary before a benchmark case."""
        self.spike.reset_fullphysics_prevalidated(self.initial_fullphysics)
        self.cursor = 0
        self._reset_before_next_control = False

    def assert_healthy(self) -> None:
        """Run the explicit, synchronizing health diagnostic outside timing."""
        self.spike.assert_healthy("benchmark case")

    def synchronize(self) -> None:
        self.spike.synchronize()

    def close(self) -> None:
        self.spike.close()


def make_spike_benchmark_runner(num_worlds: int) -> _SpikeBenchmarkRunner:
    """Compile the fixed M1 pilot and create a no-CPU-fallback benchmark runner.

    The official trace is loaded by the caller through ``LIBERO_M1_CTRL_TRACE``
    to keep benchmark data out of the Python package.  It must contain a
    ``substep_ctrl`` array with shape ``[K, 25, nu]``.
    """
    import os
    from pathlib import Path

    from benchmarks.ctrl_trace import load_trace

    trace_path = os.environ.get("LIBERO_M1_CTRL_TRACE")
    if not trace_path:
        raise RuntimeError("Set LIBERO_M1_CTRL_TRACE to an official ctrl-trace NPZ.")
    artifact = load_trace(Path(trace_path))
    metadata = artifact.metadata
    if (
        artifact.substep_ctrl.ndim != 3
        or artifact.substep_ctrl.shape[0] == 0
        or artifact.substep_ctrl.shape[1] != 25
    ):
        raise ValueError("substep_ctrl must be non-empty with shape [K, 25, nu].")
    trace_config = _parse_trace_env_config(metadata)
    # Compile using the exact trace construction.  Rendering is configured
    # independently below, so the fixed benchmark profile cannot change its
    # model fingerprint.
    trace_config = replace(trace_config, backend="warp", num_worlds=num_worlds)
    profile_cameras = (
        CameraConfig("agentview", 128, 128),
        CameraConfig("robot0_eye_in_hand", 128, 128),
        CameraConfig("sideview", 128, 128),
    )
    compiled = TaskCompiler().compile(trace_config)
    spike: MJWarpSpike | None = None
    try:
        fingerprint = _validate_trace_metadata(
            compiled, metadata, trace_config, artifact.substep_ctrl.shape
        )
        spike = MJWarpSpike(compiled, num_worlds=num_worlds)
        ctrl = torch.as_tensor(
            artifact.substep_ctrl, device=spike.device, dtype=torch.float32
        )
        if not torch.isfinite(ctrl).all():
            raise ValueError("Ctrl trace substep_ctrl contains non-finite values.")
        initial = (
            torch.as_tensor(
                artifact.initial_fullphysics, device=spike.device, dtype=torch.float32
            )
            .unsqueeze(0)
            .expand(num_worlds, -1)
            .contiguous()
        )
        spike.reset_fullphysics(initial)
        spike.configure_renderer(profile_cameras)
    except Exception:
        if spike is None:
            compiled.close()
        else:
            spike.close()
        raise
    # [K, 25, nu] -> [N, K, 25, nu]; benchmark hands one [N,25,nu] control step.
    trace = ctrl.unsqueeze(0).expand(num_worlds, -1, -1, -1)
    return _SpikeBenchmarkRunner(
        spike,
        trace,
        initial,
        trace_report={"metadata": metadata, "fingerprint": fingerprint},
    )


def _parse_trace_env_config(metadata: dict[str, object]) -> EnvConfig:
    """Decode the versioned exact construction config embedded in a trace."""
    if metadata.get("model_reset_protocol") != "official-env-explicit-reset-once-v1":
        raise ValueError(
            "Ctrl trace must declare model_reset_protocol="
            "'official-env-explicit-reset-once-v1'."
        )
    raw = metadata.get("env_config")
    if not isinstance(raw, dict):
        raise TypeError("Ctrl trace metadata.env_config must be a JSON object.")
    expected = {
        "suite",
        "task_index",
        "cameras",
        "horizon",
        "control_freq",
        "seed",
    }
    actual = set(raw)
    unknown = actual - expected
    missing = expected - actual
    if missing or unknown:
        raise ValueError(
            "Ctrl trace env_config fields mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    suite = raw["suite"]
    if not isinstance(suite, str) or not suite.strip():
        raise TypeError("env_config.suite must be a non-empty string.")
    task_index = _strict_int(raw["task_index"], "env_config.task_index")
    horizon = _strict_int(raw["horizon"], "env_config.horizon")
    control_freq = _strict_int(raw["control_freq"], "env_config.control_freq")
    seed_value = raw["seed"]
    if seed_value is not None:
        seed_value = _strict_int(seed_value, "env_config.seed")
    raw_cameras = raw["cameras"]
    if not isinstance(raw_cameras, list) or not raw_cameras:
        raise TypeError(
            "env_config.cameras must be a non-empty JSON list; camera order is "
            "part of the trace fingerprint."
        )
    cameras = tuple(
        _parse_trace_camera(value, index) for index, value in enumerate(raw_cameras)
    )
    names = [camera.name for camera in cameras]
    if len(set(names)) != len(names):
        raise ValueError(
            f"env_config.cameras has duplicate names in its ordered list: {names}."
        )
    return EnvConfig(
        suite=suite,
        task_index=task_index,
        cameras=cameras,
        horizon=horizon,
        control_freq=control_freq,
        seed=seed_value,
    )


def _parse_trace_camera(value: object, index: int) -> CameraConfig:
    if not isinstance(value, dict):
        raise TypeError(f"env_config.cameras[{index}] must be a JSON object.")
    expected = {"name", "height", "width", "depth", "segmentation"}
    actual = set(value)
    unknown = actual - expected
    missing = expected - actual
    if missing or unknown:
        raise ValueError(
            f"env_config.cameras[{index}] fields mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}."
        )
    name = value["name"]
    if not isinstance(name, str) or not name.strip():
        raise TypeError(f"env_config.cameras[{index}].name must be a non-empty string.")
    depth = value["depth"]
    if not isinstance(depth, bool):
        raise TypeError(f"env_config.cameras[{index}].depth must be a bool.")
    segmentation = value["segmentation"]
    if segmentation not in {None, "instance", "class", "element"}:
        raise ValueError(
            f"env_config.cameras[{index}].segmentation is invalid: {segmentation!r}."
        )
    return CameraConfig(
        name=name,
        height=_strict_int(value["height"], f"env_config.cameras[{index}].height"),
        width=_strict_int(value["width"], f"env_config.cameras[{index}].width"),
        depth=depth,
        segmentation=segmentation,
    )


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    return value


def _validate_trace_metadata(
    compiled,
    metadata: dict[str, object],
    trace_config: EnvConfig,
    ctrl_shape: tuple[int, ...],
) -> dict[str, Any]:
    """Reject a trace from a different task/model before launching MJWarp."""
    from benchmarks.ctrl_trace import (
        TRACE_FORMAT_VERSION,
        model_arrays_fingerprint,
        model_mjb_fingerprint,
    )

    required = {
        "trace_format_version": TRACE_FORMAT_VERSION,
        "suite": trace_config.suite,
        "task_index": trace_config.task_index,
        "task_name": compiled.metadata.task_name,
        "seed": trace_config.seed,
        "control_freq": trace_config.control_freq,
        "control_decimation": compiled.metadata.control_substeps,
        "timestep": compiled.metadata.timestep,
        "mujoco": compiled.metadata.mujoco_version,
        "robosuite": compiled.metadata.robosuite_version,
    }
    for name, expected in required.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"Ctrl trace metadata mismatch for {name}: "
                f"expected {expected!r}, got {metadata.get(name)!r}."
            )
    if ctrl_shape[2] != compiled.metadata.nu:
        raise ValueError(
            "Ctrl trace nu="
            f"{ctrl_shape[2]} does not match model nu={compiled.metadata.nu}."
        )
    mjb_hash, mjb_size = model_mjb_fingerprint(compiled.model)
    if (
        metadata.get("model_mjb_sha256") != mjb_hash
        or metadata.get("model_mjb_size") != mjb_size
    ):
        warn(
            "Ctrl trace MJB fingerprint differs across hosts; portable model arrays "
            "remain the hard compatibility gate. "
            f"trace_sha256={metadata.get('model_mjb_sha256')!r}, "
            f"compiled_sha256={mjb_hash!r}, "
            f"trace_size={metadata.get('model_mjb_size')!r}, "
            f"compiled_size={mjb_size!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    arrays = model_arrays_fingerprint(compiled.model)
    for name, key in (
        ("sha256", "model_arrays_sha256"),
        ("field_count", "model_array_field_count"),
        ("total_bytes", "model_array_total_bytes"),
        ("physics_options", "model_physics_options"),
    ):
        expected = arrays[name]
        if metadata.get(key) != expected:
            raise ValueError(f"Ctrl trace portable model fingerprint mismatch: {key}.")
    return {
        "mjb_sha256": mjb_hash,
        "mjb_size": mjb_size,
        "mjb_matches_trace": (
            metadata.get("model_mjb_sha256") == mjb_hash
            and metadata.get("model_mjb_size") == mjb_size
        ),
        "portable": {
            "sha256": arrays["sha256"],
            "field_count": arrays["field_count"],
            "total_bytes": arrays["total_bytes"],
            "physics_options": arrays["physics_options"],
        },
    }
