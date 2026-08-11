"""G1 contracts for the legacy official wrapper and vector environments."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .conftest import BENCHMARK_ROOT


class _FakeState:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def flatten(self) -> np.ndarray:
        return self._values.copy()


class _FakeSim:
    def __init__(self) -> None:
        self.values = np.array([1.0, 2.0, 3.0])
        self.forward_calls = 0

    def get_state(self) -> _FakeState:
        return _FakeState(self.values)

    def set_state_from_flattened(self, values: np.ndarray) -> None:
        self.values = np.asarray(values).copy()

    def forward(self) -> None:
        self.forward_calls += 1


class _FakeOfficialTask:
    constructed: list[_FakeOfficialTask] = []

    def __init__(self, bddl_file_name: str, **kwargs: object) -> None:
        self.bddl_file_name = bddl_file_name
        self.kwargs = kwargs
        self.seed = kwargs.get("seed")
        self.rng = np.random.default_rng(self.seed)
        self.obj_of_interest = ["target"]
        self.robots = [object()]
        self.sim = _FakeSim()
        self._visualizations = {"sites": False}
        self.closed = False
        self.xml = None
        self.post_process_calls = 0
        self.observable_updates: list[bool] = []
        type(self).constructed.append(self)

    def reset(self) -> dict[str, np.ndarray]:
        return {"sample": np.array([np.random.randint(0, 2**16)])}

    def step(self, action: np.ndarray):
        return {"action": action}, 1.25, False, {"source": "official"}

    def _check_success(self) -> bool:
        return True

    def _post_process(self) -> None:
        self.post_process_calls += 1

    def _update_observables(self, *, force: bool = False) -> None:
        self.observable_updates.append(force)

    def _get_observations(self) -> dict[str, np.ndarray]:
        return {"state": self.sim.values.copy()}

    def reset_from_xml_string(self, xml_string: str) -> None:
        self.xml = xml_string

    def close(self) -> None:
        self.closed = True


class _VectorGoldenEnv:
    """Picklable legacy four-value environment used by both vector workers."""

    def __init__(self, env_id: int) -> None:
        self.env_id = env_id
        self.counter = 0
        self.seed_value = None
        self.state = np.array([env_id, 0], dtype=np.int64)
        self.closed = False
        self.unwrapped = self

    def reset(self, **kwargs: object) -> np.ndarray:
        if "seed" in kwargs:
            self.seed(kwargs["seed"])
        self.counter = 0
        self.state = np.array([self.env_id, self.counter], dtype=np.int64)
        return self.state.copy()

    def step(self, action: np.ndarray):
        self.counter += 1
        self.state = np.array([self.env_id, self.counter], dtype=np.int64)
        return self.state.copy(), float(action[0]), self.counter >= 2, {}

    def seed(self, seed: int | None = None) -> list[int | None]:
        self.seed_value = seed
        return [seed]

    def check_success(self) -> bool:
        return self.counter >= 1

    def get_sim_state(self) -> np.ndarray:
        return self.state.copy()

    def set_init_state(self, state: np.ndarray) -> np.ndarray:
        self.state = np.asarray(state).copy()
        return self.state.copy()

    def get_segmentation_of_interest(self, image: np.ndarray) -> np.ndarray:
        return image

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_official_wrapper(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from libero.libero.envs import env_wrapper

    bddl_path = tmp_path / "task.bddl"
    bddl_path.write_text("(define (problem fake))\n", encoding="utf-8")
    _FakeOfficialTask.constructed.clear()
    monkeypatch.setattr(
        env_wrapper.BDDLUtils,
        "get_problem_info",
        lambda _: {
            "problem_name": "fake",
            "domain_name": "robosuite",
            "language_instruction": ["do the fake task"],
        },
    )
    monkeypatch.setitem(env_wrapper.TASK_MAPPING, "fake", _FakeOfficialTask)
    monkeypatch.setattr(
        env_wrapper.suite,
        "load_part_controller_config",
        lambda *, default_controller: {"controller": default_controller},
    )
    monkeypatch.setattr(
        env_wrapper,
        "refactor_composite_controller_config",
        lambda config, robot, arms: {
            **config,
            "robot": robot,
            "arms": tuple(arms),
        },
    )
    return env_wrapper, bddl_path


@pytest.mark.static
def test_default_official_backend_preserves_legacy_constructor_and_returns(
    fake_official_wrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_wrapper, bddl_path = fake_official_wrapper
    monkeypatch.delenv("LIBERO_SIM_BACKEND", raising=False)
    wrapper = env_wrapper.ControlEnv(str(bddl_path), has_offscreen_renderer=False)

    task = _FakeOfficialTask.constructed[-1]
    assert "backend" not in task.kwargs
    assert wrapper.backend_info.requested_backend == "official"
    assert wrapper.backend_info.actual_backend == "official"
    assert wrapper.backend_info.selection_source == "default"
    assert wrapper.backend_info.device == {
        "compute": "cpu",
        "physics": "mujoco",
        "render": "disabled",
    }
    capability_vocabulary = {
        "action_exact_continuation",
        "camera_calibration",
        "cuda_observations",
        "metric_depth",
        "model_xml_read",
        "model_xml_reset",
        "native_batch",
        "partial_reset",
        "predicate_success",
        "private_sim_read_proxy",
        "render_exact_restore",
        "reset",
        "rgb",
        "segmentation",
        "single_env_numpy_api",
        "state_flattened_read",
        "state_flattened_write",
        "step_osc_pose_7d",
    }
    assert wrapper.backend_info.capabilities <= capability_vocabulary
    assert wrapper.backend_info.capabilities.isdisjoint(
        {
            "action_exact_continuation",
            "cuda_observations",
            "native_batch",
            "partial_reset",
            "private_sim_read_proxy",
            "render_exact_restore",
        }
    )
    assert wrapper.obj_of_interest is task.obj_of_interest
    assert wrapper.robots is task.robots
    assert wrapper.sim is task.sim
    action = np.array([0.5])
    observation, reward, done, info = wrapper.step(action)
    assert observation["action"] is action
    assert (reward, done, info) == (1.25, False, {"source": "official"})

    backend_dict = wrapper.backend_info.to_dict()
    assert backend_dict["capabilities"] == sorted(wrapper.backend_info.capabilities)
    assert json.loads(json.dumps(backend_dict, sort_keys=True)) == backend_dict
    with pytest.raises(FrozenInstanceError):
        wrapper.backend_info.actual_backend = "warp"
    with pytest.raises(TypeError):
        wrapper.backend_info.dependency_versions["MuJoCo"] = "changed"

    wrapper.close()
    assert task.closed
    assert not hasattr(wrapper, "env")


@pytest.mark.static
def test_backend_selection_precedence_and_fail_closed_behavior(
    fake_official_wrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_wrapper, bddl_path = fake_official_wrapper
    monkeypatch.setenv("LIBERO_SIM_BACKEND", "official")
    from_environment = env_wrapper.ControlEnv(str(bddl_path))
    assert from_environment.backend_info.selection_source == "environment"
    from_environment.close()

    monkeypatch.setenv("LIBERO_SIM_BACKEND", "warp")
    explicit_official = env_wrapper.ControlEnv(str(bddl_path), backend="official")
    assert explicit_official.backend_info.selection_source == "constructor"
    explicit_official.close()

    constructed_before = len(_FakeOfficialTask.constructed)
    with pytest.raises(
        env_wrapper.UnsupportedBackendOperation,
        match="backend 'warp'.*single_env_numpy_api",
    ):
        env_wrapper.ControlEnv(str(bddl_path))
    assert len(_FakeOfficialTask.constructed) == constructed_before

    monkeypatch.setenv("LIBERO_SIM_BACKEND", "official")
    with pytest.raises(env_wrapper.UnsupportedBackendOperation):
        env_wrapper.ControlEnv(str(bddl_path), backend="warp")
    for invalid_backend in ("", "Official", "unknown"):
        with pytest.raises(ValueError, match="exactly 'official' or 'warp'"):
            env_wrapper.ControlEnv(str(bddl_path), backend=invalid_backend)
    with pytest.raises(TypeError, match="backend"):
        env_wrapper.ControlEnv(str(bddl_path), backend=1)


@pytest.mark.static
def test_backend_info_resolves_authoritative_renderer_device(
    fake_official_wrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_wrapper, bddl_path = fake_official_wrapper
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    wrapper = env_wrapper.ControlEnv(str(bddl_path), render_gpu_device_id=-1)
    try:
        assert wrapper.backend_info.device["render"] == "egl:3"
    finally:
        wrapper.close()

    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "2")
    explicit_device = env_wrapper.ControlEnv(str(bddl_path), render_gpu_device_id=7)
    try:
        assert explicit_device.backend_info.device["render"] == "egl:2"
    finally:
        explicit_device.close()

    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    constructed_before = len(_FakeOfficialTask.constructed)
    with pytest.raises(AssertionError, match="must be one of the devices visible"):
        env_wrapper.ControlEnv(str(bddl_path), render_gpu_device_id=2)
    assert len(_FakeOfficialTask.constructed) == constructed_before


@pytest.mark.static
def test_seed_state_xml_and_observation_regeneration_match_legacy_contract(
    fake_official_wrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_wrapper, bddl_path = fake_official_wrapper
    monkeypatch.delenv("LIBERO_SIM_BACKEND", raising=False)
    first = env_wrapper.ControlEnv(str(bddl_path))
    second = env_wrapper.ControlEnv(str(bddl_path))
    try:
        first.seed(37)
        second.seed(37)
        first_reset = first.reset()
        second_reset = second.reset()
        np.testing.assert_array_equal(first_reset["sample"], second_reset["sample"])

        np.testing.assert_array_equal(first.get_sim_state(), [1.0, 2.0, 3.0])
        restored_state = np.array([4.0, 5.0, 6.0])
        observation = first.set_init_state(restored_state)
        np.testing.assert_array_equal(observation["state"], restored_state)
        np.testing.assert_array_equal(first.get_sim_state(), restored_state)
        assert first.env.sim.forward_calls == 1
        assert first.env.post_process_calls == 1
        assert first.env.observable_updates == [True]
        assert first.check_success() is True

        first.reset_from_xml_string("<mujoco model='golden'/>")
        assert first.env.xml == "<mujoco model='golden'/>"
    finally:
        first.close()
        second.close()


@pytest.mark.static
def test_segmentation_helpers_fail_closed_for_unknown_or_missing_instances() -> None:
    from libero.libero.envs.env_wrapper import SegmentationRenderEnv

    wrapper = object.__new__(SegmentationRenderEnv)
    wrapper.env = SimpleNamespace(obj_of_interest=["target"])
    wrapper.instance_to_id = {"other": 2}
    wrapper.robot_segmentation_ids = frozenset({1})
    wrapper.segmentation_id_mapping = {1: "other"}

    with pytest.raises(KeyError, match="target"):
        wrapper.get_segmentation_of_interest(np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match="absent from the current model mapping"):
        wrapper.get_segmentation_instances(np.array([[0, 1, 3]]))


@pytest.mark.static
def test_robot_compatibility_names_are_real_robosuite_15_aliases() -> None:
    from robosuite.robots.fixed_base_robot import FixedBaseRobot

    from libero.libero.envs import ROBOT_CLASS_MAPPING, SingleArm
    from libero.libero.envs.robots import MountedPanda, OnTheGroundPanda

    assert SingleArm is FixedBaseRobot
    assert ROBOT_CLASS_MAPPING["MountedPanda"] is FixedBaseRobot
    assert ROBOT_CLASS_MAPPING["OnTheGroundPanda"] is FixedBaseRobot

    mounted = object.__new__(MountedPanda)
    on_the_ground = object.__new__(OnTheGroundPanda)
    assert mounted.default_mount == mounted.default_base == "RethinkMount"
    assert on_the_ground.default_base == "NullMount"
    assert on_the_ground.default_mount is None
    with pytest.raises(AttributeError):
        mounted.default_mount = "changed"


@pytest.mark.static
@pytest.mark.parametrize("vector_class_name", ("DummyVectorEnv", "SubprocVectorEnv"))
def test_legacy_vector_reset_step_state_and_close_golden(
    vector_class_name: str,
) -> None:
    from libero.libero import envs

    vector_class = getattr(envs, vector_class_name)
    vector = vector_class([lambda: _VectorGoldenEnv(0), lambda: _VectorGoldenEnv(1)])
    try:
        np.testing.assert_array_equal(vector.reset(), [[0, 0], [1, 0]])
        assert vector.seed(10) == [[10], [11]]
        observations, rewards, dones, infos = vector.step(np.array([[0.25], [0.75]]))
        np.testing.assert_array_equal(observations, [[0, 1], [1, 1]])
        np.testing.assert_array_equal(rewards, [0.25, 0.75])
        np.testing.assert_array_equal(dones, [False, False])
        assert [info["env_id"] for info in infos] == [0, 1]
        assert vector.check_success() == [True, True]
        np.testing.assert_array_equal(vector.get_sim_state(), [[0, 1], [1, 1]])
        np.testing.assert_array_equal(
            vector.set_init_state(np.array([[7, 8], [9, 10]])),
            [[7, 8], [9, 10]],
        )
    finally:
        vector.close()


@pytest.mark.official_integration
def test_real_default_backend_headless_state_and_step_golden() -> None:
    """Exercise the real official backend without requiring a GL context."""
    from libero.libero.envs.env_wrapper import ControlEnv

    fixture_root = Path(__file__).with_name("fixtures")
    metadata = json.loads(
        (fixture_root / "g1_official_headless_v1.json").read_text(encoding="utf-8")
    )

    task_name = (
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
        "and_place_it_on_the_plate"
    )
    bddl_path = BENCHMARK_ROOT / "bddl_files" / "libero_spatial" / f"{task_name}.bddl"
    env = ControlEnv(
        bddl_file_name=str(bddl_path),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        seed=17,
    )
    try:
        with np.load(fixture_root / "g1_official_headless_v1.npz") as golden:
            assert metadata["backend_argument"] is None
            assert metadata["actual_backend"] == "official"
            assert (
                metadata["compatibility_target"]
                == env.backend_info.libero_compat_target
            )
            assert env.backend_info.actual_backend == "official"
            assert env.backend_info.device["render"] == "disabled"

            observation = env.reset()
            assert set(observation) == set(metadata["reset_observation_arrays"])
            for key, array_name in metadata["reset_observation_arrays"].items():
                np.testing.assert_allclose(
                    observation[key], golden[array_name], rtol=0.0, atol=1e-9
                )

            state = env.get_sim_state()
            np.testing.assert_allclose(
                state, golden["reset_state"], rtol=0.0, atol=1e-9
            )
            restored = env.set_init_state(state)
            assert isinstance(restored, dict)
            np.testing.assert_allclose(env.get_sim_state(), state, rtol=0.0, atol=0.0)

            step_observation, reward, done, info = env.step(golden["action"])
            assert set(step_observation) == set(metadata["step_observation_arrays"])
            for key, array_name in metadata["step_observation_arrays"].items():
                np.testing.assert_allclose(
                    step_observation[key], golden[array_name], rtol=0.0, atol=1e-9
                )
            np.testing.assert_allclose(
                env.get_sim_state(), golden["step_state"], rtol=0.0, atol=1e-9
            )
            assert reward == metadata["reward"]
            assert done is metadata["done"]
            assert info == metadata["info"]
    finally:
        env.close()


@pytest.mark.official_integration
def test_real_default_backend_state_xml_and_metadata_golden() -> None:
    """One renderer-backed official oracle; the 130-task sweep remains separate."""
    from libero.libero.envs import OffScreenRenderEnv

    task_name = (
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
        "and_place_it_on_the_plate"
    )
    bddl_path = BENCHMARK_ROOT / "bddl_files" / "libero_spatial" / f"{task_name}.bddl"
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_names=["agentview"],
        camera_heights=32,
        camera_widths=48,
    )
    try:
        assert env.backend_info.selection_source == "default"
        assert env.backend_info.actual_backend == "official"
        env.seed(0)
        observation = env.reset()
        assert observation["agentview_image"].shape == (32, 48, 3)
        assert observation["agentview_image"].dtype == np.uint8
        state = env.get_sim_state()
        model_xml = env.sim.model.get_xml()
        env.reset_from_xml_string(model_xml)
        restored = env.set_init_state(state)
        np.testing.assert_array_equal(
            restored["agentview_image"], observation["agentview_image"]
        )
        np.testing.assert_allclose(env.get_sim_state(), state, rtol=0.0, atol=0.0)
    finally:
        env.close()


@pytest.mark.official_integration
@pytest.mark.parametrize("segmentation_mode", ("instance", "class", "element"))
def test_real_segmentation_env_construct_reset_helpers_and_close(
    segmentation_mode: str,
) -> None:
    """Qualify the legacy segmentation wrapper against the real official stack."""
    from libero.libero.envs import SegmentationRenderEnv

    task_name = (
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_"
        "and_place_it_on_the_plate"
    )
    bddl_path = BENCHMARK_ROOT / "bddl_files" / "libero_spatial" / f"{task_name}.bddl"
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_names=["agentview"],
        camera_heights=32,
        camera_widths=48,
        camera_segmentations=segmentation_mode,
        seed=17,
    )
    try:
        observation = env.reset()
        segmentation = observation[f"agentview_segmentation_{segmentation_mode}"]
        assert segmentation.shape == (32, 48, 1)
        if segmentation_mode == "instance":
            assert env.segmentation_id_mapping
            assert env.instance_to_id
            assert env.robot_segmentation_ids

            instances = env.get_segmentation_instances(segmentation)
            assert set(instances) == {
                "robot",
                *env.segmentation_id_mapping.values(),
            }
            assert all(mask.shape == segmentation.shape for mask in instances.values())
            interest = env.get_segmentation_of_interest(segmentation)
            assert interest.shape == segmentation.shape
            assert set(np.unique(interest)) <= {-1, 0, 1}
        else:
            assert not env.segmentation_id_mapping
            with pytest.raises(ValueError, match="require.*instance"):
                env.get_segmentation_instances(segmentation)
            with pytest.raises(ValueError, match="require.*instance"):
                env.get_segmentation_of_interest(segmentation)
    finally:
        env.close()
    assert not hasattr(env, "env")
