"""Minimal MJWarp runtime plus the retained M1 benchmark runner.

The public runtime provides reset / state / render on CUDA and a transitional
N=1 policy-action seam. Robosuite currently advances the authoritative OSC
controller state on CPU while MJWarp replays the resulting actuator controls.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Integral
from typing import Any
from warnings import warn

import numpy as np
import torch

from libero.libero.runtime.compiler import TaskCompiler
from libero.libero.runtime.types import (
    CameraConfig,
    EnvConfig,
    ObservationBatch,
    RenderExactState,
    StepBatch,
)
from libero.libero.warp import MJWarpSpike


class WarpBatchEnv:
    """Single-world CUDA runtime with a transitional CPU OSC controller shadow.

    The default reset uses the compiler's first trusted LIBERO init state;
    callers may also provide any finite flattened FULLPHYSICS state with the
    exact model width. Visuals and MuJoCo state come from MJWarp.
    Proprioception is sampled by the official source environment at reset time
    and uploaded once; there is no CPU rendering fallback in the returned
    observation. Policy actions use the exact robosuite 1.5.2 controller state
    machine on CPU while Warp remains authoritative for physics and rendering.
    """

    def __init__(self, config: EnvConfig) -> None:
        if config.backend != "warp":
            raise ValueError("WarpBatchEnv requires EnvConfig.backend='warp'.")
        if config.num_worlds != 1:
            raise ValueError("The G3 Warp runtime currently supports num_worlds=1.")
        self.config = config
        self._closed = False
        self._last_proprio: torch.Tensor | None = None
        self._last_observation: ObservationBatch | None = None
        self._last_controller_reference: dict[str, np.ndarray] | None = None
        compiled = TaskCompiler().compile(config)
        spike: MJWarpSpike | None = None
        try:
            spike = MJWarpSpike(compiled, num_worlds=1)
            spike.configure_renderer(config.cameras)
        except Exception:
            if spike is None:
                compiled.close()
            else:
                spike.close()
            raise
        self._spike = spike
        self.task = compiled.official_env.task

    @property
    def device(self) -> torch.device:
        """CUDA device holding every returned observation tensor."""
        self._ensure_open()
        return self._spike.device

    def reset(
        self,
        *,
        init_state: torch.Tensor | np.ndarray | None = None,
        world_ids: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    ) -> ObservationBatch:
        """Reset world zero to the first trusted or a supplied flattened state."""
        self._ensure_open()
        self._validate_world_ids(world_ids)
        if init_state is None:
            state_index = 0
            legacy_state = self._spike.compiled.init_state_bank.legacy_flattened[
                state_index
            ]
            self._spike.reset_all(
                torch.tensor([state_index], device=self.device, dtype=torch.int64)
            )
        else:
            legacy_state = self._normalize_init_state(init_state)
            self._spike.reset_fullphysics(
                legacy_state.to(device=self.device, dtype=torch.float32).unsqueeze(0)
            )

        # Controller-derived proprio is not reconstructible from qpos/qvel alone.
        # The compiler-owned official environment is authoritative at this reset
        # boundary; upload its exact result once while Warp owns returned visuals,
        # physics state, and simulator time.
        official_observation = self._spike.compiled.official_env.reset(
            init_state=legacy_state
        )
        self._last_proprio = official_observation.proprio.to(
            device=self.device, dtype=torch.float32
        )
        self._last_controller_reference = None
        self._last_observation = self._make_observation()
        return self._last_observation

    def step(self, actions: torch.Tensor | np.ndarray) -> StepBatch:
        """Apply one real 7-D OSC action using the G4 CPU controller shadow.

        Robosuite 1.5.2 remains authoritative for controller goal updates and
        emits the final actuator control at each of the 25 MuJoCo substeps. Warp
        replays that control sequence from the same public state, then the CPU
        task is synchronized to the Warp result for observables and predicates.
        Raw actuator controls never enter or leave this public method.
        """
        self._ensure_open()
        if self._last_observation is None:
            raise RuntimeError("Call reset() before applying a Warp policy action.")
        action = self._normalize_action(actions)
        controls, official_done, official_info = self._capture_controller_controls(
            action
        )
        control_tensor = (
            torch.from_numpy(controls)
            .to(device=self.device, dtype=torch.float32)
            .unsqueeze(0)
        )
        self._spike.replay_ctrl(control_tensor)

        official_env = self._spike.compiled.official_env
        warp_state = self.get_state()[0].detach().to(device="cpu", dtype=torch.float64)
        raw_observation = official_env._env.regenerate_obs_from_state(
            warp_state.numpy()
        )
        official_env._last_raw_observation = raw_observation
        self._last_proprio = official_env._proprio_tensor(raw_observation).to(
            device=self.device, dtype=torch.float32
        )
        self._last_observation = self._make_observation()
        success = bool(official_env._env.check_success())
        reward = float(official_env._env.env.reward(action[0]))
        truncated = bool(official_env._env.env.timestep >= self.config.horizon)
        return StepBatch(
            observation=self._last_observation,
            reward=torch.tensor([reward], device=self.device, dtype=torch.float32),
            terminated=torch.tensor([success], device=self.device, dtype=torch.bool),
            truncated=torch.tensor([truncated], device=self.device, dtype=torch.bool),
            info={
                **dict(official_info),
                "controller_backend": "robosuite-cpu-shadow",
                "official_done": bool(official_done),
            },
        )

    def get_state(self) -> torch.Tensor:
        """Return current MuJoCo FULLPHYSICS state as a CUDA ``[1, D]`` tensor."""
        self._ensure_open()
        return self._spike.fullphysics_state()

    def get_proprio(self) -> torch.Tensor:
        """Return reset-boundary official proprioception on CUDA."""
        self._ensure_open()
        if self._last_proprio is None:
            raise RuntimeError("Call reset() before requesting proprioception.")
        return self._last_proprio

    def get_sim_time(self) -> torch.Tensor:
        """Return current MJWarp simulator time as a CUDA ``[1]`` tensor."""
        self._ensure_open()
        return self._spike.physics_readout()["time"].reshape(-1)

    def get_render_exact_state(self) -> RenderExactState:
        """Return a detached render snapshot without controller continuation state."""
        self._ensure_open()
        return RenderExactState(
            state=self.get_state().clone(), sim_time=self.get_sim_time().clone()
        )

    def legacy_observation(self) -> dict[str, np.ndarray]:
        """Synchronize the latest reset observation to the legacy NumPy dict.

        Warp is authoritative for every returned visual. The compiler-owned
        official shadow contributes controller/task observables that the Warp
        path does not yet derive on CUDA. This explicit compatibility conversion
        is outside the native tensor path.
        """
        self._ensure_open()
        if self._last_observation is None:
            raise RuntimeError("Call reset() before requesting a legacy observation.")
        official_env = self._spike.compiled.official_env
        raw_observation = official_env._last_raw_observation
        if raw_observation is None:
            raise RuntimeError("The official reset shadow has no observation.")
        legacy = {
            key: np.array(value, copy=True) for key, value in raw_observation.items()
        }
        for camera in self.config.cameras:
            legacy[f"{camera.name}_image"] = self._legacy_visual(
                self._last_observation.rgb[camera.name]
            )
            if camera.depth:
                metric_depth = self._legacy_visual(
                    self._last_observation.depth[camera.name]
                )
                legacy[f"{camera.name}_depth"] = self._normalized_depth(metric_depth)
            if camera.segmentation is not None:
                key = f"{camera.name}_segmentation_{camera.segmentation}"
                legacy[key] = self._legacy_visual(
                    self._last_observation.segmentation[camera.name]
                )
        return legacy

    def check_success(self) -> bool:
        """Evaluate the original predicate on the reset-synchronized CPU shadow."""
        self._ensure_open()
        if self._last_observation is None:
            raise RuntimeError("Call reset() before requesting task success.")
        return bool(self._spike.compiled.official_env._env.check_success())

    @property
    def shadow_task(self) -> Any:
        """Return the reset-synchronized official task used for legacy reads."""
        self._ensure_open()
        return self._spike.compiled.official_env._env.env

    def close(self) -> None:
        """Idempotently release CUDA, renderer, and compiler-owned resources."""
        if self._closed:
            return
        self._closed = True
        self._last_proprio = None
        self._last_observation = None
        self._last_controller_reference = None
        self._spike.close()

    def _make_observation(self) -> ObservationBatch:
        rendered = self._spike.render()
        return ObservationBatch(
            # Renderer and MJWarp readouts are reusable device views. Public
            # observations are snapshots: a later step must not mutate an older
            # frame merely because it reuses the same CUDA buffers.
            rgb={name: value.clone() for name, value in rendered.rgb.items()},
            depth={name: value.clone() for name, value in rendered.depth.items()},
            segmentation={
                name: value.clone() for name, value in rendered.segmentation.items()
            },
            proprio=self.get_proprio().clone(),
            state=self.get_state(),
            sim_time=self.get_sim_time().clone(),
        )

    def _normalize_init_state(
        self, init_state: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        if isinstance(init_state, torch.Tensor):
            value = init_state.detach().to(device="cpu")
        elif isinstance(init_state, np.ndarray):
            value = torch.from_numpy(init_state)
        else:
            raise TypeError("init_state must be a torch.Tensor or numpy.ndarray.")
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        elif value.ndim != 1:
            raise ValueError(
                "Warp init_state must have shape [D] or [1, D]; "
                f"got {list(value.shape)}."
            )
        if not torch.is_floating_point(value) and value.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("init_state must have a real numeric dtype.")
        value = value.to(dtype=torch.float64)
        if not torch.isfinite(value).all():
            raise ValueError("init_state must contain only finite values.")
        expected_width = self._spike.compiled.metadata.fullphysics_state_size
        if value.shape != (expected_width,):
            raise ValueError(
                f"init_state must contain {expected_width} values; got {value.numel()}."
            )
        return value.contiguous()

    def _normalize_action(self, actions: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(actions, torch.Tensor):
            value = actions.detach().to(device="cpu").numpy()
        elif isinstance(actions, np.ndarray):
            value = actions
        else:
            raise TypeError("actions must be a torch.Tensor or numpy.ndarray.")
        if value.shape != (1, 7):
            raise ValueError(
                f"Warp backend actions must have shape [1, 7]; got {list(value.shape)}."
            )
        if not (
            np.issubdtype(value.dtype, np.integer)
            or np.issubdtype(value.dtype, np.floating)
        ):
            raise TypeError("actions must have a real numeric dtype.")
        if not np.isfinite(value).all():
            raise ValueError("actions must contain only finite values.")
        return np.clip(value, -1.0, 1.0).astype(np.float32, copy=False)

    def _capture_controller_controls(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, bool, Mapping[str, Any]]:
        official_env = self._spike.compiled.official_env
        raw_env = official_env._env
        current_state = (
            self.get_state()[0].detach().to(device="cpu", dtype=torch.float64)
        )
        raw_env.regenerate_obs_from_state(current_state.numpy())
        sim = raw_env.sim
        step_method_name = "step2" if raw_env.env.lite_physics else "step"
        original_sim_step = getattr(sim, step_method_name)
        captured: list[np.ndarray] = []

        def record_final_ctrl(*args: Any, **kwargs: Any):
            captured.append(np.asarray(sim.data.ctrl).copy())
            return original_sim_step(*args, **kwargs)

        setattr(sim, step_method_name, record_final_ctrl)
        try:
            _, _, official_done, official_info = raw_env.step(action[0])
        finally:
            setattr(sim, step_method_name, original_sim_step)
        controls = np.asarray(captured)
        expected_shape = (
            self._spike.compiled.metadata.control_substeps,
            self._spike.compiled.metadata.nu,
        )
        if controls.shape != expected_shape:
            raise RuntimeError(
                "OSC controller did not emit one actuator control per physics "
                f"substep: got {controls.shape}, expected {expected_shape}."
            )
        data = raw_env.sim.data
        self._last_controller_reference = {
            "state": np.asarray(raw_env.get_sim_state()).reshape(-1).copy(),
            "qpos": np.asarray(data.qpos).copy(),
            "qvel": np.asarray(data.qvel).copy(),
            "ctrl": controls.copy(),
        }
        return controls, bool(official_done), dict(official_info)

    def _legacy_visual(self, tensor: torch.Tensor) -> np.ndarray:
        value = tensor[0].detach().to(device="cpu").numpy()
        source_convention = self._spike.compiled.official_env.source_image_convention
        if source_convention == "opengl":
            value = value[::-1]
        return np.ascontiguousarray(value)

    def _normalized_depth(self, metric_depth: np.ndarray) -> np.ndarray:
        model = self._spike.compiled.model
        extent = float(model.stat.extent)
        near = float(model.vis.map.znear) * extent
        far = float(model.vis.map.zfar) * extent
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = (1.0 - near / metric_depth) / (1.0 - near / far)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _validate_world_ids(
        world_ids: Sequence[int] | np.ndarray | torch.Tensor | None,
    ) -> None:
        if world_ids is None:
            return
        if isinstance(world_ids, torch.Tensor):
            values = tuple(world_ids.detach().to(device="cpu").reshape(-1).tolist())
        elif isinstance(world_ids, np.ndarray):
            values = tuple(world_ids.reshape(-1).tolist())
        elif isinstance(world_ids, Sequence) and not isinstance(
            world_ids, (str, bytes)
        ):
            values = tuple(world_ids)
        else:
            raise TypeError(
                "world_ids must be a sequence, numpy array, or torch tensor."
            )
        if len(values) != 1 or not isinstance(values[0], Integral) or values[0] != 0:
            raise ValueError("The G3 Warp runtime only accepts world_ids=[0].")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("This environment is closed.")


class _SpikeBenchmarkRunner:
    """Adapter consumed by :mod:`benchmarks.benchmark_warp_spike`."""

    def __init__(
        self,
        spike: MJWarpSpike,
        ctrl_trace: torch.Tensor,
        initial_fullphysics: torch.Tensor,
        trace_report: Mapping[str, Any],
        ctrl_graph_mode: str | None = None,
    ) -> None:
        self.spike = spike
        self.trace = ctrl_trace
        self.cursor = 0
        self._reset_before_next_control = False
        self.initial_fullphysics = initial_fullphysics
        self.ctrl_graph_mode = ctrl_graph_mode
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
        ctrl = self._next_ctrl()
        if self.ctrl_graph_mode is None:
            self.spike.replay_ctrl(ctrl, check_health=False, validate=False)
        else:
            self.spike.replay_captured_ctrl(ctrl, check_health=False, validate=False)
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
    graph_mode = os.environ.get("LIBERO_M1_CTRL_GRAPH")
    if graph_mode not in {None, "graph-1", "graph-25"}:
        raise ValueError(
            "LIBERO_M1_CTRL_GRAPH must be unset, 'graph-1', or 'graph-25'."
        )
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
    if graph_mode is not None:
        spike.capture_ctrl_replay_graph(trace[:, 0], mode=graph_mode)
    return _SpikeBenchmarkRunner(
        spike,
        trace,
        initial,
        trace_report={"metadata": metadata, "fingerprint": fingerprint},
        ctrl_graph_mode=graph_mode,
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
