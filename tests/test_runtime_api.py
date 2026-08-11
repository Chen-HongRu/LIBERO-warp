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
    import numpy as np
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
    CameraConfig, EnvConfig, ObservationBatch, _, make_env = _runtime_api()
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
        with pytest.raises(NotImplementedError, match="OSC_POSE"):
            env.step(torch.zeros((1, 7), device=env.device))
    finally:
        env.close()
        env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.get_state()
