"""Opt-in integration coverage for robosuite 1.5 / MuJoCo 3 official evaluation.

Run on a rendering-capable workstation, for example::

    MUJOCO_GL=egl LIBERO_RUN_OFFICIAL_INTEGRATION=1 \\
        uv run pytest -m official_integration
"""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

import pytest

from ._manifest import LIBERO_TASK_MAP
from .conftest import BENCHMARK_ROOT

CAMERAS = ("agentview", "robot0_eye_in_hand", "sideview", "birdview")


def all_tasks() -> list[tuple[str, str]]:
    return [
        (suite_name, task_name)
        for suite_name, task_names in LIBERO_TASK_MAP.items()
        for task_name in task_names
    ]


def actions_from_hdf_state_action_alignment(states, actions):
    """Return all actions that start from the first stored collector state.

    robosuite's ``DataCollectionWrapper`` writes the reset state as
    ``states[0]`` before recording the post-action states.  LIBERO's gathering
    step drops only the final surplus state, leaving one initial boundary for
    ``actions[0]``.  Consequently every stored action must be replayed.
    """

    assert len(states) == len(actions)
    return actions


@pytest.mark.static
def test_hdf_state_action_alignment_replays_first_action_from_initial_state() -> None:
    states = ("reset", "after_action_0")
    actions = ("action_0", "action_1")

    assert actions_from_hdf_state_action_alignment(states, actions) == actions


def _make_env(bddl_path: Path, camera_names: tuple[str, ...] = ("agentview",)):
    pytest.importorskip(
        "robosuite", reason="robosuite must be installed for official backend tests"
    )
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_names=list(camera_names),
        camera_heights=64,
        camera_widths=64,
        render_gpu_device_id=-1,
    )


@pytest.mark.official_integration
def test_robosuite_152_mujoco_311_official_pilot_smoke() -> None:
    """Fast compatibility gate for the first spatial task and default official stack."""

    from importlib.metadata import version

    import mujoco
    import numpy as np
    import torch

    assert version("robosuite") == "1.5.2"
    assert version("mujoco") == "3.11.0"
    task_name = LIBERO_TASK_MAP["libero_spatial"][0]
    bddl_path = BENCHMARK_ROOT / "bddl_files" / "libero_spatial" / f"{task_name}.bddl"
    init_path = (
        BENCHMARK_ROOT / "init_files" / "libero_spatial" / f"{task_name}.pruned_init"
    )
    env = _make_env(bddl_path)
    try:
        assert env.reset()
        init_states = torch.load(init_path, map_location="cpu", weights_only=False)
        assert len(init_states) > 0
        assert env.set_init_state(init_states[0])
        action = np.zeros(env.robots[0].action_dim, dtype=np.float32)
        start_time = env.sim.data.time
        _, _, _, info = env.step(action)
        assert isinstance(info, dict)
        assert np.isclose(env.sim.data.time - start_time, 1.0 / 20.0)

        controller = env.robots[0].part_controllers["right"]
        controller.update(force=True)
        mass_matrix = controller.mass_matrix
        assert mass_matrix.shape == (7, 7)
        assert np.isfinite(mass_matrix).all()
        np.testing.assert_allclose(mass_matrix, mass_matrix.T, rtol=1e-10, atol=1e-10)

        joint_vector = np.linspace(-0.5, 0.5, num=7, dtype=np.float64)
        full_vector = np.zeros(env.sim.model.nv, dtype=np.float64)
        full_vector[controller.qvel_index] = joint_vector
        raw_product = np.zeros(env.sim.model.nv, dtype=np.float64)
        mujoco.mj_mulM(
            env.sim.model._model,
            env.sim.data._data,
            raw_product,
            full_vector,
        )
        np.testing.assert_allclose(
            mass_matrix @ joint_vector,
            raw_product[controller.qvel_index],
            rtol=1e-10,
            atol=1e-10,
        )
    finally:
        env.close()


@pytest.mark.official_integration
@pytest.mark.parametrize(("suite_name", "task_name"), all_tasks())
def test_all_tasks_construct_reset_load_init_state_and_step(
    suite_name: str, task_name: str
) -> None:
    """The 130-task official-backend gate; deliberately excluded from normal CI."""

    import numpy as np
    import torch

    bddl_path = BENCHMARK_ROOT / "bddl_files" / suite_name / f"{task_name}.bddl"
    init_path = BENCHMARK_ROOT / "init_files" / suite_name / f"{task_name}.pruned_init"
    env = _make_env(bddl_path)
    try:
        reset_observation = env.reset()
        assert reset_observation
        init_states = torch.load(init_path, map_location="cpu", weights_only=False)
        assert len(init_states) > 0
        init_observation = env.set_init_state(init_states[0])
        assert init_observation
        action = np.zeros(env.robots[0].action_dim, dtype=np.float32)
        _, _, _, info = env.step(action)
        assert isinstance(info, dict)
    finally:
        env.close()


@pytest.mark.official_integration
@pytest.mark.parametrize("robot_name", ("MountedPanda", "OnTheGroundPanda"))
def test_custom_panda_variants_expose_robosuite_15_interfaces(robot_name: str) -> None:
    pytest.importorskip(
        "robosuite", reason="robosuite must be installed for robot tests"
    )
    from libero.libero.envs.robots import MountedPanda, OnTheGroundPanda

    robot_class = {"MountedPanda": MountedPanda, "OnTheGroundPanda": OnTheGroundPanda}[
        robot_name
    ]
    robot = robot_class()
    assert robot.default_gripper == {"right": "PandaGripper"}
    assert robot.default_controller_config == "default_panda"
    assert robot.init_qpos.shape == (7,)
    assert robot.default_base in {"RethinkMount", "NullMount"}


@pytest.mark.official_integration
@pytest.mark.parametrize(
    ("suite_name", "task_name"),
    (
        ("libero_spatial", LIBERO_TASK_MAP["libero_spatial"][0]),
        ("libero_10", LIBERO_TASK_MAP["libero_10"][0]),
    ),
)
def test_required_camera_observations_are_available(
    suite_name: str, task_name: str
) -> None:
    bddl_path = BENCHMARK_ROOT / "bddl_files" / suite_name / f"{task_name}.bddl"
    env = _make_env(bddl_path, CAMERAS)
    try:
        observation = env.reset()
        for camera_name in CAMERAS:
            image_key = f"{camera_name}_image"
            assert image_key in observation, (
                f"missing {image_key}; available={sorted(observation)}"
            )
            image = observation[image_key]
            assert image.shape[:2] == (64, 64)
            assert image.shape[-1] == 3
    finally:
        env.close()


@pytest.mark.nightly
def test_local_demo_replay_hook() -> None:
    """Replay a successful local demonstration without downloading datasets in tests.

    ``LIBERO_DEMO_ROOT`` must point to the directory that contains suite demo
    HDF5 files.  The hook intentionally skips when that private/local dataset is
    unavailable rather than performing a network download.
    """

    demo_root = os.environ.get("LIBERO_DEMO_ROOT")
    if not demo_root:
        pytest.skip(
            "set LIBERO_DEMO_ROOT to a locally downloaded LIBERO demo directory"
        )
    h5py = pytest.importorskip("h5py", reason="h5py is required to inspect local demos")
    pytest.importorskip(
        "robosuite", reason="robosuite is required to replay local demos"
    )

    demo_files = sorted(Path(demo_root).rglob("*_demo.hdf5"))
    if not demo_files:
        pytest.skip(f"no *_demo.hdf5 files found under LIBERO_DEMO_ROOT={demo_root}")

    import numpy as np

    task_by_demo_name = {
        f"{task_name}_demo.hdf5": (suite_name, task_name)
        for suite_name, task_names in LIBERO_TASK_MAP.items()
        for task_name in task_names
    }
    attempted_demos = []
    for demo_path in demo_files:
        task_entry = task_by_demo_name.get(demo_path.name)
        if task_entry is None:
            continue
        with h5py.File(demo_path, "r") as dataset:
            data = dataset["data"]
            for demo_key in sorted(data.keys()):
                demo = data[demo_key]
                assert "actions" in demo and "states" in demo
                assert len(demo["actions"]) == len(demo["states"])
                assert len(demo["actions"]) > 0
                actions = demo["actions"][:]
                states = demo["states"][:]

                suite_name, task_name = task_entry
                bddl_path = (
                    BENCHMARK_ROOT / "bddl_files" / suite_name / f"{task_name}.bddl"
                )
                env = _make_env(bddl_path)
                try:
                    env.reset()
                    env.set_state(states[0])
                    env.sim.forward()
                    for action in actions_from_hdf_state_action_alignment(
                        states, actions
                    ):
                        _, _, _, info = env.step(action)
                        assert isinstance(info, dict)
                        assert np.isfinite(env.get_sim_state()).all()
                    attempted_demos.append(f"{demo_path.name}:{demo_key}")
                    if env.check_success():
                        return
                finally:
                    env.close()

    if not attempted_demos:
        pytest.skip("no LIBERO task-named demonstrations found in LIBERO_DEMO_ROOT")
    pytest.fail(
        "no complete local demonstration replay satisfied the LIBERO success "
        f"predicate; attempted {attempted_demos}"
    )


@pytest.mark.nightly
@pytest.mark.official_integration
@pytest.mark.xfail(
    strict=True,
    reason=(
        "published demo uses a robosuite 1.4 controller boundary that the 1.5 "
        "exact-model compatibility path does not yet reproduce"
    ),
)
def test_task_compiler_exact_model_replays_fixed_positive_source_demo() -> None:
    """Use a demo's recorded XML, not a seed-rebuilt BDDL placement, for replay.

    This is the positive regression for the adapter path used before handing an
    HDF5 demo model to MJWarp.  It deliberately accepts any locally available
    successful LIBERO task-named demo and skips when private demo data is not
    configured.
    """
    demo_root = os.environ.get("LIBERO_DEMO_ROOT")
    if not demo_root:
        pytest.skip(
            "set LIBERO_DEMO_ROOT to a locally downloaded LIBERO demo directory"
        )
    h5py = pytest.importorskip("h5py", reason="h5py is required to inspect local demos")
    pytest.importorskip(
        "robosuite", reason="robosuite is required to replay local demos"
    )
    from libero.libero.runtime import CameraConfig, EnvConfig, TaskCompiler

    demo_name = "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
    matches = sorted(Path(demo_root).rglob(demo_name))
    if not matches:
        pytest.skip(f"exact-model positive regression demo is missing: {demo_name}")
    demo_path = matches[0]
    with h5py.File(demo_path, "r") as dataset:
        demo = dataset["data/demo_0"]
        model_xml = demo.attrs["model_file"]
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8")
        actions = demo["actions"][:]
        states = demo["states"][:]

    config = EnvConfig(
        suite="libero_10",
        task_index=2,
        cameras=[CameraConfig("agentview", 64, 64)],
        horizon=max(1000, len(actions) + 1),
        seed=0,
    )
    source_sha256 = sha256(model_xml.encode("utf-8")).hexdigest()
    compiled = TaskCompiler().compile(
        config,
        exact_model_xml=model_xml,
        expected_model_xml_sha256=source_sha256,
    )
    try:
        import numpy as np

        assert compiled.metadata.source_model_xml_sha256 == source_sha256
        assert compiled.metadata.remapped_model_xml_sha256 is not None
        body_pos_before = compiled.model.body_pos.copy()
        with pytest.raises(RuntimeError, match="would resample model-level"):
            compiled.official_env.reset(init_state=states[0])
        compiled.official_env.reset_exact_state(init_state=states[0])
        success_steps = []
        for action_index, action in enumerate(
            actions_from_hdf_state_action_alignment(states, actions)
        ):
            transition = compiled.official_env.step(action[None])
            assert transition.info["official_done"] in {True, False}
            if bool(transition.terminated[0]):
                success_steps.append(action_index)
        np.testing.assert_array_equal(compiled.model.body_pos, body_pos_before)
        assert success_steps
        assert bool(compiled.official_env._env.check_success())
    finally:
        compiled.close()
