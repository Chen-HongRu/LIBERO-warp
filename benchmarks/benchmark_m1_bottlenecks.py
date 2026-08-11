"""Measure and falsify the M1 MJWarp performance hypotheses on one CUDA host.

This is deliberately an *investigation* harness, not a replacement for the
fixed M1 gate.  It records device events, host enqueue/wait time, allocator
capacity, active contact/constraint occupancy, and CUDA-Graph feasibility.
All exact-parity cases keep the recorded [25, nu] actuator-control sequence;
the copy-once ablation is labelled invalid when that sequence varies.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from benchmarks.ctrl_trace import load_trace
from libero.libero.runtime.compiler import TaskCompiler
from libero.libero.runtime.types import CameraConfig
from libero.libero.runtime.warp import (
    _parse_trace_env_config,
    _SpikeBenchmarkRunner,
    _validate_trace_metadata,
)
from libero.libero.warp import MJWarpSpike

_M1_CAMERAS = (
    CameraConfig("agentview", 128, 128),
    CameraConfig("robot0_eye_in_hand", 128, 128),
    CameraConfig("sideview", 128, 128),
)


@dataclass(frozen=True, slots=True)
class Capacity:
    nconmax: int = 512
    nccdmax: int = 256
    njmax: int = 4096
    njmax_nnz: int = 65536
    naconmax: int | None = None
    nvmax: int | None = None


def _build_runner(
    worlds: int,
    capacity: Capacity,
    option_overrides: Mapping[str, int] | None = None,
    enable_flags: int = 0,
    disable_flags: int = 0,
    with_render: bool = False,
) -> _SpikeBenchmarkRunner:
    """Create an isolated experimental runner without changing gate defaults."""
    trace_path = Path(_required_env("LIBERO_M1_CTRL_TRACE"))
    artifact = load_trace(trace_path)
    config = dataclass_replace(
        _parse_trace_env_config(artifact.metadata), backend="warp", num_worlds=worlds
    )
    compiled = TaskCompiler().compile(config)
    spike: MJWarpSpike | None = None
    try:
        fingerprint = _validate_trace_metadata(
            compiled, artifact.metadata, config, artifact.substep_ctrl.shape
        )
        for field, value in (option_overrides or {}).items():
            setattr(compiled.model.opt, field, value)
        compiled.model.opt.enableflags |= enable_flags
        compiled.model.opt.disableflags |= disable_flags
        kwargs = {
            "nconmax": capacity.nconmax,
            "nccdmax": capacity.nccdmax,
            "njmax": capacity.njmax,
            "njmax_nnz": capacity.njmax_nnz,
        }
        # mujoco-warp 3.11 exposes naconmax/nvmax, while its legacy wrapper
        # accepts nconmax.  Pass the newer knobs only when requested.
        if capacity.naconmax is not None:
            kwargs["naconmax"] = capacity.naconmax
        if capacity.nvmax is not None:
            kwargs["nvmax"] = capacity.nvmax
        spike = MJWarpSpike(compiled, num_worlds=worlds, **kwargs)
        ctrl = torch.as_tensor(
            artifact.substep_ctrl, device=spike.device, dtype=torch.float32
        )
        initial = (
            torch.as_tensor(
                artifact.initial_fullphysics, device=spike.device, dtype=torch.float32
            )
            .unsqueeze(0)
            .expand(worlds, -1)
            .contiguous()
        )
        spike.reset_fullphysics(initial)
        if with_render:
            spike.configure_renderer(_M1_CAMERAS)
    except Exception:
        if spike is None:
            compiled.close()
        else:
            spike.close()
        raise
    trace = ctrl.unsqueeze(0).expand(worlds, -1, -1, -1)
    return _SpikeBenchmarkRunner(
        spike,
        trace,
        initial,
        trace_report={"metadata": artifact.metadata, "fingerprint": fingerprint},
    )


def _required_env(name: str) -> str:
    import os

    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} to the validated M1 trace path.")
    return value


def _gpu_value(value: Any) -> Any:
    """Copy diagnostics only after the timed region has synchronized."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return wp.to_torch(value).detach().cpu().tolist()


def _capture_state(runner: _SpikeBenchmarkRunner) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in runner.spike.physics_readout().items()
        if name in {"qpos", "qvel", "act", "time"}
    }


def _state_delta(
    expected: Mapping[str, torch.Tensor], actual: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    return {
        name: (
            float((expected[name] - actual[name]).abs().max().detach().cpu())
            if expected[name].numel()
            else 0.0
        )
        for name in expected
    }


def _current_operation(runner: _SpikeBenchmarkRunner, ctrl: torch.Tensor) -> None:
    runner.spike.replay_ctrl(ctrl, check_health=False, validate=False)


def _copy_once_operation(runner: _SpikeBenchmarkRunner, ctrl: torch.Tensor) -> None:
    """A deliberately non-exact ablation unless all 25 actuator rows agree."""
    target = wp.to_torch(runner.spike.data.ctrl)
    target.copy_(ctrl[:, 0].to(dtype=target.dtype))
    with wp.ScopedDevice(runner.spike._warp_device):
        for _ in range(ctrl.shape[1]):
            mjw.step(runner.spike.model, runner.spike.data)


def _with_render(
    runner: _SpikeBenchmarkRunner, operation: Callable[[], None], with_render: bool
) -> Callable[[], None]:
    if not with_render:
        return operation

    def step_and_render() -> None:
        operation()
        runner.spike.render(check_health=False)

    return step_and_render


def _measure_operation(
    runner: _SpikeBenchmarkRunner,
    operation: Callable[[], None],
    *,
    warmup: int,
    operations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    runner.synchronize()
    # Warp owns the stream used by mujoco-warp.  Torch events on its default
    # stream would not necessarily bracket these launches, so use Warp's CUDA
    # events for the device-time denominator.
    start_event = wp.Event(enable_timing=True)
    end_event = wp.Event(enable_timing=True)
    wp.record_event(start_event)
    wall_start = time.perf_counter()
    for _ in range(operations):
        operation()
    enqueue_seconds = time.perf_counter() - wall_start
    wp.record_event(end_event)
    wait_start = time.perf_counter()
    runner.synchronize()
    wait_seconds = time.perf_counter() - wait_start
    wall_seconds = time.perf_counter() - wall_start
    wp.synchronize_event(end_event)
    device_seconds = wp.get_event_elapsed_time(start_event, end_event) / 1000.0
    return {
        "operations": operations,
        "host_enqueue_seconds": enqueue_seconds,
        "host_wait_seconds": wait_seconds,
        "host_wall_seconds": wall_seconds,
        "device_event_seconds": device_seconds,
        "host_wall_hz": operations / wall_seconds,
        "device_event_hz": operations / device_seconds,
        "host_enqueue_us_per_op": enqueue_seconds * 1_000_000 / operations,
        "device_us_per_op": device_seconds * 1_000_000 / operations,
        "wait_us_per_op": wait_seconds * 1_000_000 / operations,
    }


def _occupancy(runner: _SpikeBenchmarkRunner, controls: torch.Tensor) -> dict[str, Any]:
    """Observe real trace maxima; all copies occur outside performance timing."""
    data = runner.spike.data
    maxima: dict[str, int] = {"nacon": 0, "nefc": 0, "solver_niter": 0, "nv_awake": 0}
    runner.reset_case()
    for index in range(controls.shape[1]):
        _current_operation(runner, controls[:, index])
        runner.synchronize()
        samples = {
            "nacon": _gpu_value(data.nacon),
            "nefc": _gpu_value(data.nefc),
            "solver_niter": _gpu_value(data.solver_niter),
            "nv_awake": _gpu_value(data.nv_awake),
        }
        for name, values in samples.items():
            maxima[name] = max(maxima[name], max(_flatten_ints(values), default=0))
    capacities = {
        name: getattr(data, name, None)
        for name in ("naconmax", "njmax", "njmax_nnz", "nvmax", "nvmax_pad")
    }
    return {"maxima": maxima, "capacities": capacities}


def _flatten_ints(value: Any) -> list[int]:
    if isinstance(value, list):
        return [item for part in value for item in _flatten_ints(part)]
    return [int(value)]


def _cuda_graph_result(
    worlds: int,
    capacity: Capacity,
    kind: str,
    option_overrides: Mapping[str, int],
    enable_flags: int,
    disable_flags: int,
    with_render: bool,
) -> dict[str, Any]:
    """Capture with Warp's supported API and prove exact mutable inputs."""
    runner = _build_runner(
        worlds,
        capacity,
        option_overrides,
        enable_flags=enable_flags,
        disable_flags=disable_flags,
        with_render=with_render,
    )
    try:
        ctrl_a = runner.trace[:, 0].clone()
        ctrl_b = runner.trace[:, 1].clone()
        target = wp.from_torch(wp.to_torch(runner.spike.data.ctrl).flatten())
        staging = ctrl_a.permute(1, 0, 2).contiguous()
        source = wp.from_torch(staging.flatten())
        row_size = ctrl_a.shape[0] * ctrl_a.shape[2]
        runner.reset_case()
        runner.synchronize()
        try:
            with wp.ScopedCapture(
                device=runner.spike._warp_device, force_module_load=False
            ) as capture:
                if kind == "graph-1":
                    mjw.step(runner.spike.model, runner.spike.data)
                else:
                    for index in range(ctrl_a.shape[1]):
                        wp.copy(
                            target,
                            source,
                            src_offset=index * row_size,
                            count=row_size,
                        )
                        mjw.step(runner.spike.model, runner.spike.data)
            runner.synchronize()
        except Exception as error:
            return {
                "kind": kind,
                "capture": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(limit=8),
            }

        # If capture ever becomes possible, prove the graph does not freeze the
        # input or state before reporting a speed number.
        def replay(ctrl: torch.Tensor) -> None:
            staging.copy_(ctrl.permute(1, 0, 2))
            torch.cuda.current_stream(runner.spike.device).synchronize()
            if kind == "graph-1":
                for index in range(ctrl.shape[1]):
                    wp.copy(
                        target,
                        source,
                        src_offset=index * row_size,
                        count=row_size,
                    )
                    wp.capture_launch(capture.graph)
            else:
                wp.capture_launch(capture.graph)

        runner.reset_case()
        _with_render(runner, lambda: replay(ctrl_a), with_render)()
        runner.synchronize()
        first = _capture_state(runner)
        runner.reset_case()
        _with_render(runner, lambda: replay(ctrl_b), with_render)()
        runner.synchronize()
        second = _capture_state(runner)
        return {
            "kind": kind,
            "capture": "ok",
            "first_time_range": _tensor_range(first["time"]),
            "second_time_range": _tensor_range(second["time"]),
            "state_changed_after_input_update": any(
                value > 0.0 for value in _state_delta(first, second).values()
            ),
            "parity_vs_eager": _graph_parity(
                runner, replay, ctrl_a, ctrl_b, with_render
            ),
        }
    finally:
        runner.close()


def _graph_parity(
    runner: _SpikeBenchmarkRunner,
    replay: Callable[[torch.Tensor], None],
    ctrl_a: torch.Tensor,
    ctrl_b: torch.Tensor,
    with_render: bool,
) -> dict[str, dict[str, float]]:
    """Compare each captured control sequence with the eager raw-ctrl path."""
    result: dict[str, dict[str, float]] = {}
    for name, ctrl in (("a", ctrl_a), ("b", ctrl_b)):
        runner.reset_case()
        _with_render(runner, lambda: replay(ctrl), with_render)()
        runner.synchronize()
        graph_state = _capture_state(runner)
        runner.reset_case()
        _with_render(runner, lambda: _current_operation(runner, ctrl), with_render)()
        runner.synchronize()
        result[name] = _state_delta(graph_state, _capture_state(runner))
    return result


def _tensor_range(value: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(value.min().detach().cpu()),
        "max": float(value.max().detach().cpu()),
    }


def _graph_operation(
    runner: _SpikeBenchmarkRunner, ctrl: torch.Tensor, kind: str
) -> Callable[[], None]:
    """Return one exact graph replay operation with a mutable staging input."""
    target = wp.from_torch(wp.to_torch(runner.spike.data.ctrl).flatten())
    staging = ctrl.permute(1, 0, 2).contiguous()
    source = wp.from_torch(staging.flatten())
    row_size = ctrl.shape[0] * ctrl.shape[2]
    with wp.ScopedCapture(
        device=runner.spike._warp_device, force_module_load=False
    ) as capture:
        if kind == "graph-1":
            mjw.step(runner.spike.model, runner.spike.data)
        else:
            for index in range(ctrl.shape[1]):
                wp.copy(target, source, src_offset=index * row_size, count=row_size)
                mjw.step(runner.spike.model, runner.spike.data)

    def operation() -> None:
        staging.copy_(ctrl.permute(1, 0, 2))
        torch.cuda.current_stream(runner.spike.device).synchronize()
        if kind == "graph-1":
            for index in range(ctrl.shape[1]):
                wp.copy(
                    target,
                    source,
                    src_offset=index * row_size,
                    count=row_size,
                )
                wp.capture_launch(capture.graph)
        else:
            wp.capture_launch(capture.graph)

    return operation


def _git_evidence() -> dict[str, str]:
    return {
        "head": _command_text(["git", "rev-parse", "HEAD"]),
        "status_short": _command_text(["git", "status", "--short", "--branch"]),
        "diff_stat": _command_text(["git", "diff", "--stat"]),
    }


def _command_text(command: Sequence[str]) -> str:
    return subprocess.run(command, check=False, text=True, capture_output=True).stdout


def _runtime_metadata() -> dict[str, Any]:
    import mujoco

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "mujoco": mujoco.__version__,
        "mujoco_warp": getattr(mjw, "__version__", None),
        "warp": wp.__version__,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    capacity = Capacity(
        nconmax=args.nconmax,
        nccdmax=args.nccdmax,
        njmax=args.njmax,
        njmax_nnz=args.njmax_nnz,
        naconmax=args.naconmax,
        nvmax=args.nvmax,
    )
    option_overrides = {
        name: value
        for name, value in {
            "iterations": args.iterations,
            "ls_iterations": args.ls_iterations,
        }.items()
        if value is not None
    }
    disable_flags = 0
    if args.disable_nativeccd:
        disable_flags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    if args.disable_multiccd:
        disable_flags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    enable_flags = int(mujoco.mjtEnableBit.mjENBL_SLEEP) if args.enable_sleep else 0
    runner = _build_runner(
        args.worlds,
        capacity,
        option_overrides,
        enable_flags=enable_flags,
        disable_flags=disable_flags,
        with_render=args.with_render,
    )
    try:
        ctrl = runner.trace[:, 0].clone()
        variation = (ctrl - ctrl[:, :1]).abs()
        control_constant = bool(torch.all(variation == 0).detach().cpu())
        result: dict[str, Any] = {
            "schema_version": 1,
            "purpose": "M1 bottleneck investigation; not a formal gate report",
            "runtime": _runtime_metadata(),
            "git": _git_evidence(),
            "arguments": {
                name: str(value) if isinstance(value, Path) else value
                for name, value in vars(args).items()
            },
            "capacity_request": asdict(capacity),
            "control_semantics": {
                "substeps": int(ctrl.shape[1]),
                "ctrl_constant_within_control": control_constant,
                "max_abs_delta_from_first_substep": float(variation.max().cpu()),
                "copy_once_is_exact": control_constant,
            },
            "occupancy": _occupancy(runner, runner.trace),
        }
        runner.reset_case()
        eager_state: dict[str, torch.Tensor]
        _current_operation(runner, ctrl)
        runner.synchronize()
        eager_state = _capture_state(runner)
        runner.reset_case()
        _copy_once_operation(runner, ctrl)
        runner.synchronize()
        result["copy_once_ablation"] = {
            "exact": control_constant,
            "state_delta_vs_recorded_ctrl": _state_delta(
                eager_state, _capture_state(runner)
            ),
        }
        runner.reset_case()
        result["timing"] = {
            "current_25_copy_plus_step": _measure_operation(
                runner,
                _with_render(
                    runner,
                    lambda: _current_operation(runner, ctrl),
                    args.with_render,
                ),
                warmup=args.warmup,
                operations=args.operations,
            ),
        }
        runner.reset_case()
        result["timing"]["copy_once_plus_25_step_invalid_if_ctrl_varies"] = (
            _measure_operation(
                runner,
                _with_render(
                    runner,
                    lambda: _copy_once_operation(runner, ctrl),
                    args.with_render,
                ),
                warmup=args.warmup,
                operations=args.operations,
            )
        )
        for kind in ("graph-1", "graph-25"):
            runner.reset_case()
            build_start = time.perf_counter()
            graph_operation = _graph_operation(runner, ctrl, kind)
            runner.synchronize()
            build_seconds = time.perf_counter() - build_start
            result["timing"][kind] = _measure_operation(
                runner,
                _with_render(runner, graph_operation, args.with_render),
                warmup=args.warmup,
                operations=args.operations,
            )
            result["timing"][kind]["capture_build_seconds"] = build_seconds
        runner.spike.assert_healthy("bottleneck benchmark")
    finally:
        runner.close()
    result["cuda_graph"] = [
        _cuda_graph_result(
            args.worlds,
            capacity,
            kind,
            option_overrides,
            enable_flags,
            disable_flags,
            args.with_render,
        )
        for kind in ("graph-1", "graph-25")
    ]
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--operations", type=int, default=50)
    parser.add_argument("--nconmax", type=int, default=512)
    parser.add_argument("--nccdmax", type=int, default=256)
    parser.add_argument("--njmax", type=int, default=4096)
    parser.add_argument("--njmax-nnz", type=int, default=65536)
    parser.add_argument("--naconmax", type=int, default=None)
    parser.add_argument("--nvmax", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--ls-iterations", type=int, default=None)
    parser.add_argument("--disable-nativeccd", action="store_true")
    parser.add_argument("--disable-multiccd", action="store_true")
    parser.add_argument("--enable-sleep", action="store_true")
    parser.add_argument(
        "--with-render",
        action="store_true",
        help="Include exact three-camera 128x128 RGB rendering after every control op.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
