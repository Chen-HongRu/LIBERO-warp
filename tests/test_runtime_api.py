"""Contracts for the frozen M1 public runtime API.

The cheap tests exercise data-model and validation behavior.  The simulator
test is deliberately opt-in because it constructs an official robosuite task.
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import fields
from types import ModuleType
from typing import get_type_hints

import numpy as np
import pytest
import torch

from ._manifest import LIBERO_TASK_MAP
from .conftest import BENCHMARK_ROOT


def _runtime_api():
    from libero.libero.runtime import (
        CameraConfig,
        EnvConfig,
        ObservationBatch,
        StepBatch,
        make_env,
    )

    return CameraConfig, EnvConfig, ObservationBatch, StepBatch, make_env


def _pilot_config(*, horizon: int = 1000):
    CameraConfig, EnvConfig, *_ = _runtime_api()
    cameras = [
        CameraConfig("agentview", height=48, width=80),
        CameraConfig("robot0_eye_in_hand", height=64, width=40, depth=True),
        CameraConfig("sideview", height=32, width=96, segmentation="instance"),
    ]
    return EnvConfig(
        suite="libero_spatial",
        task_index=0,
        cameras=cameras,
        backend="official",
        num_worlds=1,
        horizon=horizon,
    )


def _valid_observation_batch_kwargs() -> dict[str, object]:
    return {
        "rgb": {"camera": torch.zeros((1, 5, 7, 3), dtype=torch.uint8)},
        "depth": {},
        "segmentation": {},
        "proprio": torch.zeros((1, 4), dtype=torch.float32),
        "state": torch.zeros((1, 8), dtype=torch.float32),
        "sim_time": torch.zeros(1, dtype=torch.float64),
    }


def _non_cpu_test_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "meta")


@pytest.mark.static
def test_runtime_public_types_have_the_frozen_batch_fields() -> None:
    CameraConfig, EnvConfig, ObservationBatch, StepBatch, make_env = _runtime_api()

    assert [field.name for field in fields(CameraConfig)] == [
        "name",
        "height",
        "width",
        "depth",
        "segmentation",
    ]
    assert [field.name for field in fields(EnvConfig)] == [
        "suite",
        "task_index",
        "cameras",
        "backend",
        "num_worlds",
        "horizon",
        "control_freq",
        "seed",
    ]
    assert [field.name for field in fields(ObservationBatch)] == [
        "rgb",
        "depth",
        "segmentation",
        "proprio",
        "state",
        "sim_time",
    ]
    assert [field.name for field in fields(StepBatch)] == [
        "observation",
        "reward",
        "terminated",
        "truncated",
        "info",
    ]
    assert callable(make_env)


@pytest.mark.static
def test_camera_configs_preserve_an_arbitrary_ordered_unequal_resolution_list() -> None:
    CameraConfig, EnvConfig, *_ = _runtime_api()
    cameras = [
        CameraConfig("custom_camera", 17, 31),
        CameraConfig("another_camera", 29, 13, depth=True, segmentation="element"),
    ]
    config = EnvConfig(suite="libero_spatial", task_index=0, cameras=cameras)

    assert tuple(config.cameras) == tuple(cameras)
    assert [(camera.height, camera.width) for camera in config.cameras] == [
        (17, 31),
        (29, 13),
    ]
    assert config.cameras[0].depth is False
    assert config.cameras[0].segmentation is None
    assert config.cameras[1].depth is True
    assert config.cameras[1].segmentation == "element"


@pytest.mark.static
def test_runtime_configuration_does_not_encode_fixed_camera_roles() -> None:
    _, EnvConfig, *_ = _runtime_api()
    field_names = {field.name for field in fields(EnvConfig)}
    assert "cameras" in field_names
    assert (
        not {
            "agentview",
            "eye_in_hand",
            "sideview",
            "birdview",
            "deployable_cameras",
            "privileged_cameras",
        }
        & field_names
    )


@pytest.mark.static
def test_make_env_signature_accepts_one_env_config() -> None:
    *_, make_env = _runtime_api()
    signature = inspect.signature(make_env)
    assert list(signature.parameters) == ["config"]


@pytest.mark.static
def test_make_env_return_type_is_a_backend_neutral_protocol() -> None:
    from libero.libero.runtime import OfficialBatchEnv, make_env

    return_type = get_type_hints(make_env)["return"]
    assert return_type is not OfficialBatchEnv
    assert getattr(return_type, "_is_protocol", False)


@pytest.mark.static
def test_make_env_dispatches_warp_without_importing_it_for_official(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    CameraConfig, EnvConfig, *_, make_env = _runtime_api()
    sentinel = object()
    module = ModuleType("libero.libero.runtime.warp")
    module.WarpBatchEnv = lambda config: (sentinel, config)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libero.libero.runtime.warp", module)
    config = EnvConfig(
        suite="libero_spatial",
        task_index=0,
        cameras=[CameraConfig("agentview", 16, 16)],
        backend="warp",
    )

    result = make_env(config)

    assert result == (sentinel, config)


@pytest.mark.static
@pytest.mark.parametrize("modality", ("depth", "segmentation"))
@pytest.mark.parametrize("mismatched_shape", ((1, 6, 7, 1), (1, 5, 8, 1)))
def test_observation_batch_rejects_auxiliary_resolution_mismatch(
    modality: str, mismatched_shape: tuple[int, int, int, int]
) -> None:
    *_, ObservationBatch, _, _ = _runtime_api()
    values = _valid_observation_batch_kwargs()
    dtype = torch.float32 if modality == "depth" else torch.int32
    values[modality] = {"camera": torch.zeros(mismatched_shape, dtype=dtype)}

    with pytest.raises(ValueError, match="H/W|resolution|height|width|shape"):
        ObservationBatch(**values)


@pytest.mark.static
def test_observation_batch_rejects_cross_device_tensors() -> None:
    *_, ObservationBatch, _, _ = _runtime_api()
    values = _valid_observation_batch_kwargs()
    values["proprio"] = torch.zeros((1, 4), device=_non_cpu_test_device())

    with pytest.raises(ValueError, match="device"):
        ObservationBatch(**values)


@pytest.mark.static
def test_step_batch_requires_float32_reward_and_one_device() -> None:
    *_, ObservationBatch, StepBatch, _ = _runtime_api()
    observation = ObservationBatch(**_valid_observation_batch_kwargs())

    with pytest.raises(TypeError, match="reward.*float32"):
        StepBatch(
            observation=observation,
            reward=torch.zeros(1, dtype=torch.float64),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.zeros(1, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="device"):
        StepBatch(
            observation=observation,
            reward=torch.zeros(1, dtype=torch.float32, device=_non_cpu_test_device()),
            terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.zeros(1, dtype=torch.bool),
        )


@pytest.mark.static
def test_render_exact_state_rejects_cross_device_tensors() -> None:
    from libero.libero.runtime import RenderExactState

    with pytest.raises(ValueError, match="device"):
        RenderExactState(
            state=torch.zeros((1, 8), dtype=torch.float32),
            sim_time=torch.zeros(1, dtype=torch.float64, device=_non_cpu_test_device()),
        )


@pytest.mark.static
def test_env_config_rejects_the_removed_libero_100_aggregate_suite() -> None:
    CameraConfig, EnvConfig, *_, make_env = _runtime_api()
    config = EnvConfig(
        suite="libero_100",
        task_index=0,
        cameras=[CameraConfig("agentview", 32, 32)],
    )
    with pytest.raises(ValueError, match="libero_100"):
        make_env(config)


@pytest.mark.official_integration
def test_official_runtime_pilot_observation_and_boundary_contract() -> None:
    import torch

    CameraConfig, EnvConfig, ObservationBatch, StepBatch, make_env = _runtime_api()
    del CameraConfig, EnvConfig
    env = make_env(_pilot_config(horizon=2))
    try:
        observation = env.reset()
        assert isinstance(observation, ObservationBatch)
        assert tuple(observation.rgb) == (
            "agentview",
            "robot0_eye_in_hand",
            "sideview",
        )
        assert env.task.name == LIBERO_TASK_MAP["libero_spatial"][0]
        expected_shapes = {
            "agentview": (1, 48, 80, 3),
            "robot0_eye_in_hand": (1, 64, 40, 3),
            "sideview": (1, 32, 96, 3),
        }
        for camera_name, expected_shape in expected_shapes.items():
            image = observation.rgb[camera_name]
            assert image.shape == expected_shape
            assert image.dtype == torch.uint8
            assert image.device.type == "cpu"
        raw_agentview = env._last_raw_observation["agentview_image"]
        expected_agentview = (
            raw_agentview[::-1]
            if env.source_image_convention == "opengl"
            else raw_agentview
        )
        np.testing.assert_array_equal(
            observation.rgb["agentview"][0].numpy(), expected_agentview
        )
        assert tuple(observation.depth) == ("robot0_eye_in_hand",)
        assert tuple(observation.segmentation) == ("sideview",)
        for modalities, expected_dtype in (
            (observation.depth, torch.float32),
            (observation.segmentation, torch.int32),
        ):
            for camera_name, modality in modalities.items():
                rgb = observation.rgb[camera_name]
                assert modality.shape == (*rgb.shape[:3], 1)
                assert modality.dtype == expected_dtype
                assert modality.device.type == "cpu"
        from robosuite.utils.camera_utils import get_real_depth_map

        raw_depth = env._last_raw_observation["robot0_eye_in_hand_depth"]
        metric_depth = get_real_depth_map(env._env.sim, raw_depth)
        expected_depth = (
            metric_depth[::-1]
            if env.source_image_convention == "opengl"
            else metric_depth
        )
        expected_depth = np.asarray(expected_depth, dtype=np.float32)
        if expected_depth.ndim == 2:
            expected_depth = expected_depth[..., None]
        public_depth = observation.depth["robot0_eye_in_hand"][0].numpy()
        np.testing.assert_allclose(public_depth, expected_depth, rtol=1e-6, atol=1e-6)
        assert np.isfinite(public_depth).all()
        assert (public_depth >= 0.0).all()
        assert observation.proprio.device.type == "cpu"
        assert observation.state.device.type == "cpu"
        assert observation.sim_time.device.type == "cpu"
        assert torch.isfinite(observation.proprio).all()
        assert torch.isfinite(observation.state).all()

        pilot_init_path = (
            BENCHMARK_ROOT
            / "init_files"
            / "libero_spatial"
            / (
                "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
                "and_place_it_on_the_plate.pruned_init"
            )
        )
        init_states = torch.load(
            pilot_init_path,
            map_location="cpu",
            weights_only=False,
        )
        init_state = torch.as_tensor(init_states[0])
        initialized_observation = env.reset(init_state=init_state)
        assert initialized_observation.state.shape == observation.state.shape
        batched_initialized_observation = env.reset(init_state=init_state.unsqueeze(0))
        assert batched_initialized_observation.state.shape == observation.state.shape
        with pytest.raises(ValueError, match="init_state.*shape"):
            env.reset(init_state=torch.zeros((2, init_state.numel())))
        nonfinite_init_state = init_state.clone()
        nonfinite_init_state[0] = float("nan")
        with pytest.raises(ValueError, match="init_state.*finite"):
            env.reset(init_state=nonfinite_init_state)

        with pytest.raises(ValueError):
            env.step(torch.zeros(7))
        with pytest.raises(ValueError):
            env.step(torch.full((1, 7), float("nan")))
        with pytest.raises(ValueError):
            env.step(np.full((1, 7), np.inf, dtype=np.float32))

        captured_actions = []
        official_step = env._env.step

        def capture_clipped_action(action):
            captured_actions.append(np.asarray(action).copy())
            return official_step(action)

        env._env.step = capture_clipped_action
        step = env.step(torch.full((1, 7), 2.0))
        np.testing.assert_array_equal(captured_actions, [np.ones(7, dtype=np.float32)])
        assert isinstance(step, StepBatch)
        assert step.reward.shape == (1,)
        assert step.terminated.shape == (1,)
        assert step.truncated.shape == (1,)
        assert step.reward.device.type == "cpu"
        assert step.terminated.device.type == "cpu"
        assert step.truncated.device.type == "cpu"
        assert step.truncated.item() is False
        assert step.terminated.item() is bool(env._env.check_success())
        assert step.terminated.dtype == torch.bool
        assert step.truncated.dtype == torch.bool
        assert torch.equal(env.get_state(), step.observation.state)
        assert torch.equal(env.get_proprio(), step.observation.proprio)
        assert torch.equal(env.get_sim_time(), step.observation.sim_time)
        assert step.observation.sim_time.item() > observation.sim_time.item()

        horizon_step = env.step(torch.zeros((1, 7)))
        assert horizon_step.truncated.item() is True
        assert env._env.env.timestep == 2

        reset_observation = env.reset(init_state=init_state)
        assert env._env.env.timestep == 0
        post_reset_step = env.step(torch.zeros((1, 7)))
        assert post_reset_step.truncated.item() is False
        assert (
            post_reset_step.observation.sim_time.item()
            > reset_observation.sim_time.item()
        )

        render_exact = env.get_render_exact_state()
        assert render_exact is not None
        assert not hasattr(env, "restore")
        assert not hasattr(env, "set_state")

        env.reset(world_ids=[0])
        env.reset(world_ids=np.array([0], dtype=np.int64))
        env.reset(world_ids=torch.tensor([0], dtype=torch.int64))
        with pytest.raises(ValueError):
            env.reset(world_ids=[1])
        with pytest.raises(ValueError):
            env.reset(world_ids=[0, 0])
        with pytest.raises(ValueError):
            env.reset(world_ids=np.array([1], dtype=np.int64))
        with pytest.raises(ValueError):
            env.reset(world_ids=torch.tensor([1], dtype=torch.int64))
    finally:
        env.close()


@pytest.mark.official_integration
def test_official_runtime_rejects_multiple_worlds_before_constructing() -> None:
    CameraConfig, EnvConfig, *_, make_env = _runtime_api()
    config = EnvConfig(
        suite="libero_spatial",
        task_index=0,
        cameras=[CameraConfig("agentview", 32, 32)],
        num_worlds=2,
    )
    with pytest.raises(ValueError, match="num_worlds"):
        make_env(config)


@pytest.mark.warp_gpu
def test_warp_runtime_n1_reset_render_state_and_close_contract() -> None:
    """Exercise the smallest useful public Warp runtime on a real CUDA host."""
    if not torch.cuda.is_available():
        pytest.fail("LIBERO_RUN_WARP_GPU=1 was set, but CUDA is unavailable")
    CameraConfig, EnvConfig, ObservationBatch, StepBatch, make_env = _runtime_api()
    config = EnvConfig(
        suite="libero_spatial",
        task_index=0,
        cameras=(
            CameraConfig("agentview", 48, 64),
            CameraConfig("robot0_eye_in_hand", 32, 40, depth=True),
            CameraConfig("sideview", 24, 36, segmentation="instance"),
        ),
        backend="warp",
        num_worlds=1,
        seed=0,
    )
    env = make_env(config)
    try:
        observation = env.reset()
        assert isinstance(observation, ObservationBatch)
        assert env.device.type == "cuda"
        assert tuple(observation.rgb) == (
            "agentview",
            "robot0_eye_in_hand",
            "sideview",
        )
        assert tuple(observation.depth) == ("robot0_eye_in_hand",)
        assert tuple(observation.segmentation) == ("sideview",)
        assert observation.rgb["agentview"].shape == (1, 48, 64, 3)
        assert observation.rgb["robot0_eye_in_hand"].shape == (1, 32, 40, 3)
        assert observation.rgb["sideview"].shape == (1, 24, 36, 3)
        assert observation.depth["robot0_eye_in_hand"].shape == (1, 32, 40, 1)
        assert observation.segmentation["sideview"].shape == (1, 24, 36, 1)
        tensors = (
            *observation.rgb.values(),
            *observation.depth.values(),
            *observation.segmentation.values(),
            observation.proprio,
            observation.state,
            observation.sim_time,
        )
        assert all(value.device == env.device for value in tensors)
        assert all(torch.isfinite(value).all() for value in tensors)
        assert observation.sim_time.tolist() == [0.0]
        assert torch.equal(env.get_state(), observation.state)
        assert torch.equal(env.get_proprio(), observation.proprio)
        assert torch.equal(env.get_sim_time(), observation.sim_time)

        trusted_state = env._spike.compiled.init_state_bank.legacy_flattened[1]
        second = env.reset(init_state=trusted_state, world_ids=[0])
        expected = env._spike.compiled.init_state_bank.fullphysics[1].to(
            env.device, dtype=torch.float32
        )
        torch.testing.assert_close(second.state[0], expected, rtol=0.0, atol=0.0)
        custom_state = trusted_state.clone()
        custom_state[0] = 0.125
        custom = env.reset(init_state=custom_state)
        torch.testing.assert_close(
            custom.state[0], custom_state.to(env.device, dtype=torch.float32)
        )
        with pytest.raises(ValueError, match="finite"):
            env.reset(init_state=torch.full_like(trusted_state, float("nan")))
        with pytest.raises(ValueError, match=r"world_ids=\[0\]"):
            env.reset(world_ids=[1])

        snapshot = env.get_render_exact_state()
        assert snapshot.state.device == env.device
        assert snapshot.sim_time.device == env.device
        step = env.step(torch.zeros((1, 7), device=env.device))
        assert isinstance(step, StepBatch)
        assert step.observation.state.device == env.device
        assert step.reward.device == env.device
        assert step.info["controller_backend"] == "robosuite-cpu-shadow"
    finally:
        env.close()
        env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.get_state()


@pytest.mark.warp_gpu
def test_legacy_offscreen_env_warp_reset_numpy_dict_and_capabilities() -> None:
    """Keep the original single-env reset/state API while Warp owns visuals."""
    if not torch.cuda.is_available():
        pytest.fail("LIBERO_RUN_WARP_GPU=1 was set, but CUDA is unavailable")
    from libero.libero.envs import OffScreenRenderEnv, UnsupportedBackendOperation

    task_name = LIBERO_TASK_MAP["libero_spatial"][0]
    bddl_path = BENCHMARK_ROOT / "bddl_files" / "libero_spatial" / f"{task_name}.bddl"
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        backend="warp",
        camera_names=["agentview", "robot0_eye_in_hand", "sideview"],
        camera_heights=[48, 32, 24],
        camera_widths=[64, 40, 36],
        camera_depths=[False, True, False],
        camera_segmentations=[None, None, "instance"],
        seed=0,
    )
    try:
        assert env.backend_info.actual_backend == "warp"
        assert env.backend_info.selection_source == "constructor"
        assert env.backend_info.device == {
            "compute": "cuda:0",
            "physics": "mujoco-warp",
            "render": "cuda:0",
        }
        assert "step_osc_pose_7d" in env.backend_info.capabilities
        observation = env.reset()
        assert isinstance(observation, dict)
        expected_visuals = {
            "agentview_image": ((48, 64, 3), np.uint8),
            "robot0_eye_in_hand_image": ((32, 40, 3), np.uint8),
            "robot0_eye_in_hand_depth": ((32, 40, 1), np.float32),
            "sideview_image": ((24, 36, 3), np.uint8),
            "sideview_segmentation_instance": ((24, 36, 1), np.int32),
        }
        for key, (shape, dtype) in expected_visuals.items():
            assert observation[key].shape == shape
            assert observation[key].dtype == dtype
            assert np.isfinite(observation[key]).all()
        assert "robot0_proprio-state" in observation
        assert np.isfinite(observation["robot0_proprio-state"]).all()
        state = env.get_sim_state()
        assert state.ndim == 1
        assert state.dtype == np.float64
        assert np.isfinite(state).all()
        restored = env.set_init_state(state)
        assert isinstance(restored, dict)
        np.testing.assert_allclose(env.get_sim_state(), state, rtol=0.0, atol=0.0)
        assert env.check_success() is bool(env.env._check_success())
        step_observation, reward, done, info = env.step(np.zeros(7, dtype=np.float32))
        assert isinstance(step_observation, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert info["controller_backend"] == "robosuite-cpu-shadow"
        with pytest.raises(UnsupportedBackendOperation, match="model_xml_reset"):
            env.reset_from_xml_string(env.sim.model.get_xml())
    finally:
        env.close()


@pytest.mark.warp_gpu
def test_warp_hybrid_osc_one_step_and_short_rollout_match_controller_reference() -> (
    None
):
    """Freeze the G4 public action and 25-substep CPU-controller contract."""
    if not torch.cuda.is_available():
        pytest.fail("LIBERO_RUN_WARP_GPU=1 was set, but CUDA is unavailable")
    CameraConfig, EnvConfig, _, StepBatch, make_env = _runtime_api()
    env = make_env(
        EnvConfig(
            suite="libero_spatial",
            task_index=0,
            cameras=(CameraConfig("agentview", 28, 36),),
            backend="warp",
            num_worlds=1,
            horizon=3,
            control_freq=20,
            seed=0,
        )
    )
    actions = (
        np.zeros((1, 7), dtype=np.float32),
        np.array(
            [[0.05, -0.03, 0.02, 0.01, -0.02, 0.03, -1.0]],
            dtype=np.float32,
        ),
        np.array(
            [[-0.04, 0.02, -0.01, -0.02, 0.01, -0.03, 1.0]],
            dtype=np.float32,
        ),
    )
    try:
        env.reset()
        env.step(np.full((1, 7), 2.0, dtype=np.float32))
        saturated_controls = env._last_controller_reference["ctrl"].copy()
        env.reset()
        env.step(np.ones((1, 7), dtype=np.float32))
        np.testing.assert_allclose(
            env._last_controller_reference["ctrl"],
            saturated_controls,
            rtol=0.0,
            atol=0.0,
        )

        reset = env.reset()
        reset_rgb = reset.rgb["agentview"].clone()
        with pytest.raises(ValueError, match=r"shape \[1, 7\]"):
            env.step(np.zeros(7, dtype=np.float32))
        with pytest.raises(ValueError, match="finite"):
            env.step(np.full((1, 7), np.nan, dtype=np.float32))

        qpos_errors = []
        qvel_errors = []
        for index, action in enumerate(actions):
            step = env.step(action)
            assert isinstance(step, StepBatch)
            assert step.info["controller_backend"] == "robosuite-cpu-shadow"
            assert step.observation.state.is_cuda
            assert step.reward.is_cuda
            assert step.terminated.is_cuda
            assert step.truncated.is_cuda
            assert step.observation.sim_time.item() == pytest.approx(
                (index + 1) / 20.0, abs=1e-6
            )
            reference = env._last_controller_reference
            assert reference is not None
            assert reference["ctrl"].shape == (
                env._spike.compiled.metadata.control_substeps,
                env._spike.compiled.metadata.nu,
            )
            assert np.isfinite(reference["ctrl"]).all()
            readout = env._spike.physics_readout()
            qpos_errors.append(
                float(
                    np.max(np.abs(readout["qpos"][0].cpu().numpy() - reference["qpos"]))
                )
            )
            qvel_errors.append(
                float(
                    np.max(np.abs(readout["qvel"][0].cpu().numpy() - reference["qvel"]))
                )
            )
            assert step.terminated.item() is env.check_success()
            assert step.reward.item() == float(env.check_success())
            assert step.truncated.item() is (index == len(actions) - 1)

        assert qpos_errors[0] <= 1e-5
        assert qvel_errors[0] <= 1e-4
        assert max(qpos_errors) <= 1e-3
        assert max(qvel_errors) <= 5e-3
        assert reset.sim_time.item() == 0.0
        assert torch.equal(reset.rgb["agentview"], reset_rgb)
    finally:
        env.close()


@pytest.mark.warp_gpu
@pytest.mark.parametrize(
    ("suite", "camera"),
    (
        ("libero_object", ("agentview", 21, 35, False, "class")),
        ("libero_goal", ("sideview", 27, 31, True, "element")),
        ("libero_10", ("robot0_eye_in_hand", 25, 29, True, None)),
    ),
)
def test_warp_reset_oracle_spans_suites_scenes_and_camera_modalities(
    suite: str,
    camera: tuple[str, int, int, bool, str | None],
) -> None:
    """Compile representative non-pilot tasks without fixed camera assumptions."""
    if not torch.cuda.is_available():
        pytest.fail("LIBERO_RUN_WARP_GPU=1 was set, but CUDA is unavailable")
    CameraConfig, EnvConfig, *_, make_env = _runtime_api()
    name, height, width, depth, segmentation = camera
    env = make_env(
        EnvConfig(
            suite=suite,
            task_index=0,
            cameras=(
                CameraConfig(
                    name,
                    height,
                    width,
                    depth=depth,
                    segmentation=segmentation,
                ),
            ),
            backend="warp",
            num_worlds=1,
            seed=11,
        )
    )
    try:
        observation = env.reset()
        assert observation.rgb[name].shape == (1, height, width, 3)
        assert observation.rgb[name].dtype == torch.uint8
        if depth:
            assert observation.depth[name].shape == (1, height, width, 1)
            assert observation.depth[name].dtype == torch.float32
        else:
            assert name not in observation.depth
        if segmentation is not None:
            assert observation.segmentation[name].shape == (1, height, width, 1)
            assert observation.segmentation[name].dtype == torch.int32
        else:
            assert name not in observation.segmentation

        official = env._spike.compiled.official_env
        official_state = official.get_state().to(env.device, dtype=torch.float32)
        torch.testing.assert_close(
            observation.state, official_state, rtol=0.0, atol=0.0
        )
        assert env.check_success() is bool(official._env.check_success())
        legacy = env.legacy_observation()
        assert legacy[f"{name}_image"].shape == (height, width, 3)
        assert legacy[f"{name}_image"].dtype == np.uint8
        if depth:
            normalized_depth = legacy[f"{name}_depth"]
            assert normalized_depth.shape == (height, width, 1)
            assert normalized_depth.dtype == np.float32
            assert np.all((0.0 <= normalized_depth) & (normalized_depth <= 1.0))
        if segmentation is not None:
            legacy_segmentation = legacy[f"{name}_segmentation_{segmentation}"]
            assert legacy_segmentation.shape == (height, width, 1)
            assert legacy_segmentation.dtype == np.int32

        step = env.step(np.zeros((1, 7), dtype=np.float32))
        reference = env._last_controller_reference
        readout = env._spike.physics_readout()
        qpos_error = float(
            np.max(np.abs(readout["qpos"][0].cpu().numpy() - reference["qpos"]))
        )
        qvel_delta = np.abs(readout["qvel"][0].cpu().numpy() - reference["qvel"])
        max_qvel_index = int(np.argmax(qvel_delta))
        model = env._spike.compiled.model
        max_qvel_joint = model.joint(int(model.dof_jntid[max_qvel_index])).name
        robot = official._env.env.robots[0]
        controlled_qvel_indices = np.asarray(
            [
                *robot._ref_joint_vel_indexes,
                *(
                    index
                    for indices in robot._ref_gripper_joint_vel_indexes.values()
                    for index in indices
                ),
            ],
            dtype=np.int64,
        )
        passive_qvel_indices = np.setdiff1d(
            np.arange(model.nv), controlled_qvel_indices
        )
        controlled_qvel_error = float(np.max(qvel_delta[controlled_qvel_indices]))
        passive_qvel_error = float(np.max(qvel_delta[passive_qvel_indices]))
        assert qpos_error <= 1e-3
        assert controlled_qvel_error <= 5e-3
        # MJWarp and MuJoCo may resolve a resting free-object contact to
        # different instantaneous velocities while its pose remains aligned.
        # Keep that known physics boundary separate from the strict robot OSC
        # check instead of hiding it in one global tolerance.
        assert passive_qvel_error <= 1e-1, (
            f"max passive qvel error {passive_qvel_error} at dof {max_qvel_index} "
            f"({max_qvel_joint})"
        )
        assert step.terminated.item() is env.check_success()
        assert step.reward.item() == float(env.check_success())
    finally:
        env.close()


@pytest.mark.warp_gpu
def test_warp_repeated_construct_close_has_bounded_cuda_memory_growth() -> None:
    """Three identical lifecycles must not retain one model per construction."""
    if not torch.cuda.is_available():
        pytest.fail("LIBERO_RUN_WARP_GPU=1 was set, but CUDA is unavailable")
    import gc

    CameraConfig, EnvConfig, *_, make_env = _runtime_api()
    config = EnvConfig(
        suite="libero_spatial",
        task_index=0,
        cameras=(CameraConfig("agentview", 20, 28),),
        backend="warp",
        num_worlds=1,
        seed=0,
    )
    free_after_close = []
    for _ in range(3):
        env = make_env(config)
        try:
            observation = env.reset()
            assert observation.rgb["agentview"].is_cuda
        finally:
            env.close()
        del env
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_after_close.append(torch.cuda.mem_get_info()[0])

    # Warp and CUDA may retain reusable module/mempool pages after the first
    # lifecycle. Repeating the identical lifecycle must not retain another full
    # model/renderer allocation each time.
    assert free_after_close[-1] >= free_after_close[0] - 64 * 1024 * 1024


@pytest.mark.warp_gpu
def test_warp_runtime_camera_geometry_depth_and_segmentation_oracle() -> None:
    """Compare the public G3 renderer with official MuJoCo and CPU ray oracles."""
    if not torch.cuda.is_available():
        pytest.fail("LIBERO_RUN_WARP_GPU=1 was set, but CUDA is unavailable")
    from types import SimpleNamespace

    from .test_mjwarp_gpu_parity import (
        MAX_TARGET_DEPTH_ABS_ERROR_METERS,
        PILOT_TARGET_GEOM,
        _official_camera_reference,
        _official_target_ray_depth,
        _require_official_visible_geom_groups,
        _rgb_orientation_errors,
        _target_keypoint_error,
        _target_metric_depth_max_abs_error,
        _target_silhouette_iou,
    )

    CameraConfig, EnvConfig, *_, make_env = _runtime_api()
    cameras = (CameraConfig("agentview", 64, 64, depth=True, segmentation="element"),)
    env = make_env(
        EnvConfig(
            suite="libero_spatial",
            task_index=0,
            cameras=cameras,
            backend="warp",
            num_worlds=1,
            seed=0,
        )
    )
    try:
        observation = env.reset()
        compiled = env._spike.compiled
        state = observation.state[0].detach().to(device="cpu").numpy()
        state_reference = SimpleNamespace(initial_fullphysics=state)
        official = _official_camera_reference(compiled, state_reference, cameras)
        _require_official_visible_geom_groups(
            compiled.official_env._env.sim._render_context_offscreen.vopt.geomgroup
        )
        target_geom_id = compiled.metadata.geom_ids[PILOT_TARGET_GEOM]
        name = cameras[0].name
        warp_rgb = observation.rgb[name]
        warp_depth = observation.depth[name]
        warp_segmentation = observation.segmentation[name]
        direct_rgb_error, reversed_rgb_error = _rgb_orientation_errors(
            official[name]["rgb"], warp_rgb
        )
        assert direct_rgb_error < reversed_rgb_error

        official_target_mask = (
            official[name]["segmentation"][0, ..., 0].numpy() == target_geom_id
        )
        assert int(official_target_mask.sum()) >= 4
        keypoint_error = _target_keypoint_error(
            official[name]["segmentation"][0],
            warp_segmentation[0],
            target_geom_id,
        )
        silhouette_iou = _target_silhouette_iou(
            official[name]["segmentation"][0],
            warp_segmentation[0],
            target_geom_id,
        )
        assert keypoint_error <= 1.0
        assert silhouette_iou >= 0.95

        ray_depth = _official_target_ray_depth(
            compiled,
            state_reference,
            cameras[0],
            target_geom_id,
            official_target_mask,
        )
        ray_depth_tensor = torch.from_numpy(ray_depth).unsqueeze(0).unsqueeze(-1)
        target_mask = (
            torch.from_numpy(official_target_mask)
            .to(env.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        assert (
            _target_metric_depth_max_abs_error(
                ray_depth_tensor, warp_depth, target_mask
            )
            <= MAX_TARGET_DEPTH_ABS_ERROR_METERS
        )
    finally:
        env.close()
