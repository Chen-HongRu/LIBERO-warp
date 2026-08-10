"""Opt-in GPU parity gates for the fixed M1 MJWarp spike.

These tests intentionally never substitute the official CPU backend.  Run on
an AutoDL CUDA host with ``LIBERO_RUN_WARP_GPU=1`` and an exported control
trace, for example::

    LIBERO_RUN_WARP_GPU=1 \\
    LIBERO_M1_CTRL_TRACE=/tmp/libero-m1-pilot-ctrl-trace.npz \\
    MUJOCO_GL=egl uv run pytest tests/test_mjwarp_gpu_parity.py -q

Trace parity reconstructs the exact exported model configuration, then creates
the three-camera 128px renderer independently on that immutable model. It
never substitutes init-bank row zero for the recorded FULLPHYSICS state.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from benchmarks.ctrl_trace import (
    TRACE_MODEL_RESET_PROTOCOL,
    CtrlTraceArtifact,
    load_trace,
    model_arrays_fingerprint,
    model_mjb_fingerprint,
    validate_m1_trace_metadata,
)

from ._manifest import LIBERO_TASK_MAP

PILOT_SUITE = "libero_spatial"
PILOT_TASK_INDEX = 0
PILOT_TASK = LIBERO_TASK_MAP[PILOT_SUITE][PILOT_TASK_INDEX]
PILOT_SEED = 0
PILOT_TARGET_GEOM = "akita_black_bowl_1_g0"
MAX_TARGET_DEPTH_ABS_ERROR_METERS = 0.005
_OFFICIAL_VISIBLE_GEOM_GROUPS = np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8)
RAW_RESET_CONTRACT = (
    "MJWarpSpike needs reset_fullphysics(states: torch.Tensor, *, "
    "check_health: bool = True) -> None, accepting CUDA [N, D] "
    "mjSTATE_FULLPHYSICS rows without an init-state-bank indirection."
)
PARITY_READOUT_CONTRACT = (
    "MJWarpSpike.physics_readout() must expose CUDA body_xquat and site_xmat "
    "alongside qpos/qvel/act/body_xpos/site_xpos for orientation parity."
)
CLOSE_LIFECYCLE_CONTRACT = (
    "MJWarpSpike.close() is idempotent and then reset_all, reset_fullphysics, "
    "replay_ctrl, configure_renderer, render, physics_readout, and synchronize "
    "must all raise RuntimeError instead of retaining CUDA allocations."
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.fail("LIBERO_RUN_WARP_GPU=1 was set, but CUDA is unavailable")


def _trace_or_fail() -> CtrlTraceArtifact:
    value = os.environ.get("LIBERO_M1_CTRL_TRACE")
    if not value:
        pytest.fail("LIBERO_RUN_WARP_GPU=1 requires LIBERO_M1_CTRL_TRACE for M1 parity")
    path = Path(value).expanduser()
    if not path.is_file():
        pytest.fail(f"M1 ctrl trace does not exist: {path}")
    try:
        trace = load_trace(path)
        validate_m1_trace_metadata(trace.metadata)
    except (OSError, ValueError) as error:
        pytest.fail(f"invalid M1 ctrl trace {path}: {error}")
    if trace.metadata["control_decimation"] != 25:
        pytest.fail("M1 GPU parity requires a 25-substep control trace")
    return trace


def _cameras(*, render_aux: bool) -> tuple[Any, ...]:
    from libero.libero.runtime import CameraConfig

    camera_names = ("agentview", "robot0_eye_in_hand", "sideview")
    return tuple(
        CameraConfig(
            name,
            128,
            128,
            depth=render_aux,
            segmentation="element" if render_aux else None,
        )
        for name in camera_names
    )


def _trace_compile_cameras(trace: CtrlTraceArtifact) -> tuple[Any, ...]:
    from libero.libero.runtime import CameraConfig

    return tuple(
        CameraConfig(**camera) for camera in trace.metadata["env_config"]["cameras"]
    )


def _make_spike(*, worlds: int, trace: CtrlTraceArtifact):
    """Compile exactly the model configuration stored in the trace artifact."""

    from libero.libero.runtime import EnvConfig, TaskCompiler
    from libero.libero.warp import MJWarpSpike

    stored = trace.metadata["env_config"]
    config = EnvConfig(
        suite=stored["suite"],
        task_index=stored["task_index"],
        cameras=_trace_compile_cameras(trace),
        backend="warp",
        num_worlds=worlds,
        horizon=stored["horizon"],
        control_freq=stored["control_freq"],
        seed=stored["seed"],
    )
    compiled = TaskCompiler().compile(config)
    try:
        spike = MJWarpSpike(compiled, num_worlds=worlds)
    except BaseException:
        compiled.close()
        raise
    return compiled, spike


def _close_spike(compiled: Any, spike: Any) -> None:
    try:
        spike.close()
    finally:
        compiled.close()


def _assert_trace_model_fingerprint(compiled: Any, trace: CtrlTraceArtifact) -> bool:
    mjb_hash, mjb_size = model_mjb_fingerprint(compiled.model)
    portable = model_arrays_fingerprint(compiled.model)
    assert trace.metadata["suite"] == PILOT_SUITE
    assert trace.metadata["task_index"] == PILOT_TASK_INDEX
    assert trace.metadata["task_name"] == PILOT_TASK
    assert trace.metadata["seed"] == PILOT_SEED
    assert portable["sha256"] == trace.metadata["model_arrays_sha256"]
    assert portable["field_count"] == trace.metadata["model_array_field_count"]
    assert portable["total_bytes"] == trace.metadata["model_array_total_bytes"]
    assert portable["physics_options"] == trace.metadata["model_physics_options"]
    mjb_matches = (mjb_hash, mjb_size) == (
        trace.metadata["model_mjb_sha256"],
        trace.metadata["model_mjb_size"],
    )
    if not mjb_matches:
        warnings.warn(
            "M1 trace MJB fingerprint differs across hosts; portable model "
            "fingerprint matched and remains the cross-host hard gate.",
            RuntimeWarning,
            stacklevel=2,
        )
    return mjb_matches


def _reset_trace_state(spike: Any, trace: CtrlTraceArtifact) -> None:
    states = torch.as_tensor(
        trace.initial_fullphysics, dtype=torch.float32, device=spike.device
    ).unsqueeze(0)
    states = states.expand(spike.num_worlds, -1).contiguous()
    spike.reset_fullphysics(states, check_health=True)


def _reference_initial_state(
    compiled: Any, trace: CtrlTraceArtifact
) -> dict[str, np.ndarray]:
    """Read the trace state through its exact official model without resetting it."""

    raw_env = compiled.official_env._env
    raw_env.set_state(trace.initial_fullphysics)
    raw_env.sim.forward()
    data = raw_env.sim.data._data
    return {
        "qpos": np.asarray(data.qpos).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "act": np.asarray(data.act).copy(),
        "body_xpos": np.asarray(data.xpos).copy(),
        "body_xquat": np.asarray(data.xquat).copy(),
        "site_xpos": np.asarray(data.site_xpos).copy(),
        "site_xmat": np.asarray(data.site_xmat).copy(),
    }


def _official_camera_reference(
    compiled: Any, trace: CtrlTraceArtifact, cameras: tuple[Any, ...]
) -> dict[str, dict[str, torch.Tensor]]:
    """Render 128px references from the trace's exact MjModel.

    Robosuite's camera observables are deliberately not used here: enabling a
    different observable set changes the compiled model fingerprint. MjSim can
    render any named model camera at an arbitrary resolution without doing so.
    Its direct ``mjr_readPixels`` output is always OpenGL bottom-left, so this
    helper flips every modality unconditionally into public top-left order.
    """

    from robosuite.utils.camera_utils import get_real_depth_map

    raw_env = compiled.official_env._env
    raw_env.set_state(trace.initial_fullphysics)
    raw_env.sim.forward()
    reference: dict[str, dict[str, torch.Tensor]] = {}
    for camera in cameras:
        rgb, normalized_depth = raw_env.sim.render(
            width=camera.width,
            height=camera.height,
            camera_name=camera.name,
            depth=True,
        )
        segmentation = raw_env.sim.render(
            width=camera.width,
            height=camera.height,
            camera_name=camera.name,
            segmentation=True,
        )
        reference[camera.name] = {
            "rgb": torch.from_numpy(
                np.ascontiguousarray(np.asarray(rgb)[::-1])
            ).unsqueeze(0),
            "depth": torch.from_numpy(
                np.ascontiguousarray(
                    get_real_depth_map(raw_env.sim, normalized_depth)[::-1, ..., None]
                ).astype(np.float32, copy=False)
            ).unsqueeze(0),
            "normalized_depth": torch.from_numpy(
                np.ascontiguousarray(np.asarray(normalized_depth)[::-1, ..., None])
            ).unsqueeze(0),
            "segmentation": torch.from_numpy(
                np.ascontiguousarray(np.asarray(segmentation)[::-1, ..., 1:2])
            ).unsqueeze(0),
        }
    return reference


def _camera_ray_direction(
    *, row: int, column: int, width: int, height: int, fovy_degrees: float
) -> np.ndarray:
    """Return one normalized top-left pixel ray in MuJoCo camera coordinates."""

    if not (0 <= row < height and 0 <= column < width):
        raise ValueError("Camera pixel is outside the configured image bounds.")
    focal = 0.5 * height / np.tan(np.deg2rad(fovy_degrees) / 2.0)
    direction = np.array(
        (
            (column + 0.5 - width / 2.0) / focal,
            (height / 2.0 - (row + 0.5)) / focal,
            -1.0,
        ),
        dtype=np.float64,
    )
    return direction / np.linalg.norm(direction)


def _camera_z_from_ray_distance(distance: float, camera_direction: np.ndarray) -> float:
    """Convert an ``mj_ray`` hit distance to metric planar camera-Z depth."""

    if distance < 0 or camera_direction.shape != (3,) or camera_direction[2] >= 0:
        raise ValueError("Ray depth requires a forward-facing camera ray hit.")
    return float(-distance * camera_direction[2])


def _require_official_visible_geom_groups(geom_groups: np.ndarray) -> None:
    """Fail closed if the CPU ray oracle diverges from offscreen visibility."""

    actual = np.asarray(geom_groups, dtype=np.uint8)
    if actual.shape != _OFFICIAL_VISIBLE_GEOM_GROUPS.shape or not np.array_equal(
        actual, _OFFICIAL_VISIBLE_GEOM_GROUPS
    ):
        raise AssertionError(
            "Official offscreen vopt.geomgroup differs from the M1 ray oracle: "
            f"expected={_OFFICIAL_VISIBLE_GEOM_GROUPS.tolist()}, "
            f"actual={actual.tolist()}."
        )


def _require_full_target_ray_coverage(
    target_mask: np.ndarray, hit_geom_ids: np.ndarray, target_geom_id: int
) -> None:
    """Reject an oracle that did not ray-hit every official target pixel."""

    if target_mask.dtype != np.bool_ or target_mask.shape != hit_geom_ids.shape:
        raise ValueError(
            "Target mask and ray-hit geometry IDs must share one image shape."
        )
    missing = target_mask & (hit_geom_ids != target_geom_id)
    if np.any(missing):
        coordinates = np.argwhere(missing)
        raise AssertionError(
            "MuJoCo ray oracle did not cover every official target pixel; "
            f"missing={len(coordinates)}, sample={coordinates[:4].tolist()}."
        )


def _official_target_ray_depth(
    compiled: Any,
    trace: CtrlTraceArtifact,
    camera: Any,
    target_geom_id: int,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Return MuJoCo ``mj_ray`` camera-Z for every official target pixel.

    OpenGL depth can differ at transparent / self-occluding raster edges. This
    CPU oracle traces the same camera-center rays through the official visible
    geom groups and rejects incomplete target coverage instead of selecting a
    convenient low-error subset.
    """

    import mujoco

    if target_mask.dtype != np.bool_ or target_mask.shape != (
        camera.height,
        camera.width,
    ):
        raise ValueError(
            "Target ray mask must be a top-left [height, width] boolean image."
        )
    raw_env = compiled.official_env._env
    raw_env.set_state(trace.initial_fullphysics)
    raw_env.sim.forward()
    model = compiled.model
    data = raw_env.sim.data._data
    camera_id = compiled.metadata.camera_ids[camera.name]
    origin = np.asarray(data.cam_xpos[camera_id], dtype=np.float64).copy()
    camera_rotation = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(
        3, 3
    )
    fovy = float(model.cam_fovy[camera_id])
    ray_depth = np.full(target_mask.shape, np.nan, dtype=np.float32)
    hit_geom_ids = np.full(target_mask.shape, -1, dtype=np.int32)
    for row, column in np.argwhere(target_mask):
        camera_direction = _camera_ray_direction(
            row=int(row),
            column=int(column),
            width=camera.width,
            height=camera.height,
            fovy_degrees=fovy,
        )
        world_direction = camera_rotation @ camera_direction
        geom_id = np.empty(1, dtype=np.int32)
        distance = mujoco.mj_ray(
            model,
            data,
            origin,
            world_direction,
            _OFFICIAL_VISIBLE_GEOM_GROUPS,
            True,
            -1,
            geom_id,
        )
        hit_geom_ids[row, column] = geom_id[0]
        if distance >= 0:
            ray_depth[row, column] = _camera_z_from_ray_distance(
                distance, camera_direction
            )
    _require_full_target_ray_coverage(target_mask, hit_geom_ids, target_geom_id)
    return ray_depth


def _max_abs(actual: torch.Tensor, expected: np.ndarray) -> float:
    if actual.numel() == 0:
        return 0.0
    target = torch.as_tensor(expected, device=actual.device, dtype=actual.dtype)
    return float((actual - target).abs().max().item())


def _require_orientation_readout(readout: dict[str, torch.Tensor]) -> None:
    missing = {"body_xquat", "site_xmat"} - set(readout)
    if missing:
        pytest.fail(f"{PARITY_READOUT_CONTRACT} Missing: {sorted(missing)}")


def _orientation_error_degrees(
    actual_quat: torch.Tensor, expected_quat: np.ndarray
) -> float:
    if actual_quat.numel() == 0:
        return 0.0
    expected = torch.as_tensor(
        expected_quat, device=actual_quat.device, dtype=actual_quat.dtype
    )
    dot = (actual_quat * expected).sum(dim=-1).abs().clamp(max=1.0)
    return float(torch.rad2deg(2.0 * torch.acos(dot)).max().item())


def _target_keypoint_error(
    official: torch.Tensor, warp: torch.Tensor, target_geom_id: int
) -> float:
    """Compare one fixed task-object element centroid as an image keypoint."""

    official_labels = official.squeeze(-1).to(device=warp.device)
    warp_labels = warp.squeeze(-1)
    official_xy = torch.nonzero(official_labels == target_geom_id, as_tuple=False).to(
        torch.float32
    )
    warp_xy = torch.nonzero(warp_labels == target_geom_id, as_tuple=False).to(
        torch.float32
    )
    if min(len(official_xy), len(warp_xy)) < 4:
        raise AssertionError(
            "Target element must have at least four visible pixels in both "
            f"renders; geom_id={target_geom_id}, official={len(official_xy)}, "
            f"warp={len(warp_xy)}."
        )
    return float(torch.linalg.vector_norm(official_xy.mean(0) - warp_xy.mean(0)).item())


def _target_silhouette_iou(
    official: torch.Tensor, warp: torch.Tensor, target_geom_id: int
) -> float:
    official_mask = official.squeeze(-1).to(device=warp.device) == target_geom_id
    warp_mask = warp.squeeze(-1) == target_geom_id
    union = torch.logical_or(official_mask, warp_mask).sum()
    if int(union.item()) == 0:
        raise AssertionError("Element segmentation has no visible silhouette.")
    intersection = torch.logical_and(official_mask, warp_mask).sum()
    return float((intersection / union).item())


def _rgb_orientation_errors(
    official: torch.Tensor, warp: torch.Tensor
) -> tuple[float, float]:
    """Return direct and vertically reversed RGB content errors."""

    expected = official.to(device=warp.device, dtype=torch.float32)
    actual = warp.to(torch.float32)
    direct = float((actual - expected).abs().mean().item())
    reversed_rows = float((actual - expected.flip(dims=(1,))).abs().mean().item())
    return direct, reversed_rows


def _target_metric_depth_max_abs_error(
    official_depth: torch.Tensor, warp_depth: torch.Tensor, target_mask: torch.Tensor
) -> float:
    """Return the strict target-object camera-Z error in metres."""

    if not bool(target_mask.any().item()):
        raise AssertionError("Target depth comparison requires visible target pixels.")
    expected = official_depth.to(device=warp_depth.device, dtype=warp_depth.dtype)
    return float((warp_depth - expected).abs()[target_mask].max().item())


@pytest.mark.static
def test_warp_gpu_marker_and_raw_trace_requirements_are_explicit(
    pytestconfig: pytest.Config,
) -> None:
    markers = pytestconfig.getini("markers")
    assert any(marker.startswith("warp_gpu:") for marker in markers)
    assert "reset_fullphysics" in RAW_RESET_CONTRACT
    assert "FULLPHYSICS" in RAW_RESET_CONTRACT
    assert "body_xquat" in PARITY_READOUT_CONTRACT
    assert "idempotent" in CLOSE_LIFECYCLE_CONTRACT
    assert "physics_readout" in CLOSE_LIFECYCLE_CONTRACT
    assert TRACE_MODEL_RESET_PROTOCOL == "official-env-explicit-reset-once-v1"


@pytest.mark.static
def test_target_geometry_helpers_do_not_treat_full_scene_coverage_as_object_iou() -> (
    None
):
    target = 17
    official = torch.tensor(
        [[[[target], [target], [0]], [[target], [target], [0]], [[0], [0], [0]]]],
        dtype=torch.int32,
    )
    matching = official.clone()
    misplaced = torch.tensor(
        [[[[0], [0], [0]], [[0], [target], [target]], [[0], [target], [target]]]],
        dtype=torch.int32,
    )
    assert _target_keypoint_error(official, matching, target) == 0.0
    assert _target_silhouette_iou(official, matching, target) == 1.0
    assert _target_silhouette_iou(official, misplaced, target) < 0.2


@pytest.mark.static
def test_target_metric_depth_gate_is_an_absolute_camera_z_measurement() -> None:
    official = torch.tensor([[[[1.0], [1.5]]]], dtype=torch.float32)
    within_tolerance = official + 0.004
    outside_tolerance = official + 0.006
    target_mask = torch.tensor([[[[True], [True]]]])

    assert (
        _target_metric_depth_max_abs_error(official, within_tolerance, target_mask)
        <= MAX_TARGET_DEPTH_ABS_ERROR_METERS
    )
    assert (
        _target_metric_depth_max_abs_error(official, outside_tolerance, target_mask)
        > MAX_TARGET_DEPTH_ABS_ERROR_METERS
    )


@pytest.mark.static
def test_mujoco_camera_ray_uses_top_left_pixels_and_planar_camera_z() -> None:
    center = _camera_ray_direction(
        row=1, column=1, width=3, height=3, fovy_degrees=90.0
    )
    top_center = _camera_ray_direction(
        row=0, column=1, width=3, height=3, fovy_degrees=90.0
    )

    np.testing.assert_allclose(center, [0.0, 0.0, -1.0], atol=1e-12)
    assert top_center[1] > 0.0
    assert _camera_z_from_ray_distance(2.0, center) == pytest.approx(2.0)
    assert _camera_z_from_ray_distance(
        10.0, np.array([0.6, 0.0, -0.8])
    ) == pytest.approx(8.0)


@pytest.mark.static
def test_mujoco_ray_oracle_rejects_offscreen_geom_group_drift() -> None:
    _require_official_visible_geom_groups(np.array([0, 1, 1, 0, 0, 0], dtype=np.uint8))
    with pytest.raises(AssertionError, match="vopt.geomgroup"):
        _require_official_visible_geom_groups(
            np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
        )


@pytest.mark.static
def test_target_ray_oracle_requires_full_official_mask_coverage() -> None:
    target_geom = 17
    target_mask = np.array([[True, False], [True, True]], dtype=np.bool_)
    complete = np.array([[target_geom, -1], [target_geom, target_geom]], dtype=np.int32)
    incomplete = complete.copy()
    incomplete[1, 0] = 3

    _require_full_target_ray_coverage(target_mask, complete, target_geom)
    with pytest.raises(AssertionError, match="every official target pixel"):
        _require_full_target_ray_coverage(target_mask, incomplete, target_geom)


@pytest.mark.warp_gpu
def test_mjwarp_trace_initial_state_reset_matches_official_at_one_e6() -> None:
    _require_cuda()
    trace = _trace_or_fail()
    compiled, spike = _make_spike(worlds=1, trace=trace)
    try:
        _assert_trace_model_fingerprint(compiled, trace)
        expected = _reference_initial_state(compiled, trace)
        _reset_trace_state(spike, trace)
        readout = spike.physics_readout()
        for name in ("qpos", "qvel", "act"):
            assert _max_abs(readout[name][0], expected[name]) <= 1e-6
        spike.assert_healthy("trace initial-state reset")
    finally:
        _close_spike(compiled, spike)


@pytest.mark.warp_gpu
def test_mjwarp_first_control_and_contact_drift_match_the_official_trace(
    record_property: pytest.RecordProperty,
) -> None:
    _require_cuda()
    trace = _trace_or_fail()
    compiled, spike = _make_spike(worlds=1, trace=trace)
    try:
        _assert_trace_model_fingerprint(compiled, trace)
        _reset_trace_state(spike, trace)
        first_ctrl = torch.as_tensor(
            trace.substep_ctrl[0], device=spike.device, dtype=torch.float32
        ).unsqueeze(0)
        spike.replay_ctrl(first_ctrl)
        readout = spike.physics_readout()
        _require_orientation_readout(readout)
        reference = trace
        qpos_error = _max_abs(readout["qpos"][0, :7], reference.qpos[1, :7])
        position_error = max(
            _max_abs(readout["body_xpos"][0], reference.body_xpos[1]),
            _max_abs(readout["site_xpos"][0], reference.site_xpos[1]),
        )
        orientation_error = _orientation_error_degrees(
            readout["body_xquat"][0], reference.body_xquat[1]
        )
        assert qpos_error <= 1e-3
        assert position_error <= 2e-3
        assert orientation_error <= 0.5

        # Contact trajectories are informative but deliberately report-only at
        # 10 / 50 control boundaries, matching the M1 acceptance boundary.
        for steps in (10, 50):
            _reset_trace_state(spike, trace)
            for control in trace.substep_ctrl[:steps]:
                spike.replay_ctrl(
                    torch.as_tensor(
                        control, device=spike.device, dtype=torch.float32
                    ).unsqueeze(0),
                    check_health=False,
                )
            drift = _max_abs(spike.physics_readout()["qpos"][0], trace.qpos[steps])
            record_property(f"qpos_drift_{steps}_controls", drift)
        spike.assert_healthy("trace contact-drift replay")
    finally:
        _close_spike(compiled, spike)


@pytest.mark.warp_gpu
def test_mjwarp_readout_and_render_stay_on_cuda_without_host_outputs() -> None:
    _require_cuda()
    trace = _trace_or_fail()
    compiled, spike = _make_spike(worlds=1, trace=trace)
    try:
        import warp as wp

        spike.reset_all(torch.zeros(1, device=spike.device, dtype=torch.int64))
        readout = spike.physics_readout()
        assert all(
            tensor.is_cuda and tensor.device == spike.device
            for tensor in readout.values()
        )
        assert readout["qpos"].data_ptr() == wp.to_torch(spike.data.qpos).data_ptr()
        assert readout["qvel"].data_ptr() == wp.to_torch(spike.data.qvel).data_ptr()

        spike.configure_renderer(_cameras(render_aux=True))
        render = spike.render()
        for images in (render.rgb, render.depth, render.segmentation):
            assert all(
                image.is_cuda and image.device == spike.device
                for image in images.values()
            )
        for image in render.rgb.values():
            assert (
                image.dtype == torch.uint8 and image.ndim == 4 and image.shape[-1] == 3
            )
        for image in render.depth.values():
            assert image.dtype == torch.float32 and torch.isfinite(image).all()
            assert torch.all(image >= 0)
        for image in render.segmentation.values():
            assert image.dtype == torch.int32 and image.shape[-1] == 1
    finally:
        _close_spike(compiled, spike)


@pytest.mark.warp_gpu
def test_mjwarp_three_camera_geometry_and_element_silhouettes_match_official(
    record_property: pytest.RecordProperty,
) -> None:
    _require_cuda()
    trace = _trace_or_fail()
    cameras = _cameras(render_aux=True)
    compiled, spike = _make_spike(worlds=1, trace=trace)
    try:
        _assert_trace_model_fingerprint(compiled, trace)
        try:
            target_geom_id = compiled.metadata.geom_ids[PILOT_TARGET_GEOM]
        except KeyError as error:
            pytest.fail(
                f"M1 pilot model is missing target geom {PILOT_TARGET_GEOM!r}: {error}"
            )
        official = _official_camera_reference(compiled, trace, cameras)
        _require_official_visible_geom_groups(
            compiled.official_env._env.sim._render_context_offscreen.vopt.geomgroup
        )
        _reset_trace_state(spike, trace)
        spike.configure_renderer(cameras)
        warp = spike.render()
        assert tuple(warp.rgb) == tuple(camera.name for camera in cameras)
        for camera in cameras:
            name = camera.name
            official_rgb = official[name]["rgb"]
            warp_rgb = warp.rgb[name]
            assert warp_rgb.shape == official_rgb.shape == (1, 128, 128, 3)
            assert warp_rgb.dtype == torch.uint8
            assert (
                warp.depth[name].shape
                == official[name]["depth"].shape
                == (
                    1,
                    128,
                    128,
                    1,
                )
            )
            assert warp.depth[name].dtype == torch.float32
            assert torch.isfinite(warp.depth[name]).all() and torch.all(
                warp.depth[name] >= 0
            )
            assert (
                warp.segmentation[name].shape
                == official[name]["segmentation"].shape
                == (1, 128, 128, 1)
            )
            assert warp.segmentation[name].dtype == torch.int32
            direct_rgb_error, reversed_rgb_error = _rgb_orientation_errors(
                official_rgb, warp_rgb
            )
            assert direct_rgb_error < reversed_rgb_error
            official_target_mask = (
                official[name]["segmentation"][0, ..., 0].numpy() == target_geom_id
            )
            assert int(official_target_mask.sum()) >= 4
            target_visible = (warp.segmentation[name] == target_geom_id) & (
                official[name]["segmentation"].to(device=spike.device) == target_geom_id
            )
            assert int(target_visible.sum().item()) >= 4
            keypoint_error = _target_keypoint_error(
                official[name]["segmentation"][0],
                warp.segmentation[name][0],
                target_geom_id,
            )
            silhouette_iou = _target_silhouette_iou(
                official[name]["segmentation"][0],
                warp.segmentation[name][0],
                target_geom_id,
            )
            assert keypoint_error <= 1.0
            assert silhouette_iou >= 0.95

            ray_depth = _official_target_ray_depth(
                compiled,
                trace,
                camera,
                target_geom_id,
                official_target_mask,
            )
            ray_depth_tensor = torch.from_numpy(ray_depth).unsqueeze(0).unsqueeze(-1)
            target_mask = torch.from_numpy(official_target_mask).to(spike.device)
            max_ray_depth_error = _target_metric_depth_max_abs_error(
                ray_depth_tensor,
                warp.depth[name],
                target_mask.unsqueeze(0).unsqueeze(-1),
            )
            assert max_ray_depth_error <= MAX_TARGET_DEPTH_ABS_ERROR_METERS
            gl_vs_ray_max_error = _target_metric_depth_max_abs_error(
                ray_depth_tensor,
                official[name]["depth"],
                torch.from_numpy(official_target_mask).unsqueeze(0).unsqueeze(-1),
            )
            record_property(
                f"{name}_target_gl_vs_ray_depth_max_abs_error_m", gl_vs_ray_max_error
            )
    finally:
        _close_spike(compiled, spike)


@pytest.mark.warp_gpu
def test_mjwarp_rejects_unknown_camera_nonfinite_ctrl_and_overflow() -> None:
    _require_cuda()
    trace = _trace_or_fail()
    compiled, spike = _make_spike(worlds=1, trace=trace)
    try:
        import warp as wp

        from libero.libero.runtime import CameraConfig

        with pytest.raises(ValueError, match="Unknown cameras"):
            spike.configure_renderer((CameraConfig("unknown_m1_camera", 8, 8),))

        spike.reset_all(torch.zeros(1, device=spike.device, dtype=torch.int64))
        nonfinite = torch.zeros((1, 25, spike.nu), device=spike.device)
        nonfinite[0, 0, 0] = torch.nan
        with pytest.raises(ValueError, match="non-finite"):
            spike.replay_ctrl(nonfinite)

        overflow = wp.to_torch(spike.data.overflow)
        overflow.zero_()
        overflow[0] = 1
        with pytest.raises(RuntimeError, match=r"overflow_worlds=\[0\]"):
            spike.assert_healthy("forced overflow")
        overflow.zero_()
    finally:
        _close_spike(compiled, spike)


@pytest.mark.warp_gpu
@pytest.mark.parametrize("worlds", (1, 16))
def test_mjwarp_batch_reset_and_one_control_step_smoke(worlds: int) -> None:
    _require_cuda()
    trace = _trace_or_fail()
    compiled, spike = _make_spike(worlds=worlds, trace=trace)
    try:
        state_indices = torch.zeros(worlds, device=spike.device, dtype=torch.int64)
        spike.reset_all(state_indices)
        ctrl = torch.zeros((worlds, 25, spike.nu), device=spike.device)
        spike.replay_ctrl(ctrl)
        readout = spike.physics_readout()
        assert readout["qpos"].shape[0] == worlds
        assert all(torch.isfinite(value).all() for value in readout.values())
        spike.assert_healthy("N-world reset plus one control step")
    finally:
        _close_spike(compiled, spike)


@pytest.mark.warp_gpu
def test_mjwarp_close_is_idempotent_and_invalidates_public_cuda_operations() -> None:
    _require_cuda()
    trace = _trace_or_fail()
    compiled, spike = _make_spike(worlds=1, trace=trace)
    try:
        fullphysics = torch.as_tensor(
            trace.initial_fullphysics, dtype=torch.float32, device=spike.device
        ).unsqueeze(0)
        state_indices = torch.zeros(1, dtype=torch.int64, device=spike.device)
        ctrl = torch.zeros((1, 25, spike.nu), dtype=torch.float32, device=spike.device)
        spike.close()
        spike.close()

        closed_operations = (
            lambda: spike.reset_all(state_indices),
            lambda: spike.reset_fullphysics(fullphysics),
            lambda: spike.replay_ctrl(ctrl),
            lambda: spike.configure_renderer(_cameras(render_aux=False)),
            spike.render,
            spike.physics_readout,
            spike.synchronize,
        )
        for operation in closed_operations:
            with pytest.raises(RuntimeError, match="closed"):
                operation()
    finally:
        compiled.close()
