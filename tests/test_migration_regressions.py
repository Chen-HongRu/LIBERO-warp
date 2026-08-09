"""Regression contracts discovered while reviewing the robosuite 1.5 migration."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from .conftest import REPOSITORY_ROOT

DEMO_SCRIPTS = (
    REPOSITORY_ROOT / "scripts" / "collect_demonstration.py",
    REPOSITORY_ROOT / "scripts" / "libero_100_collect_demonstrations.py",
)
BDDL_PROBLEM_PATTERN = re.compile(r"\(define \(problem ([^)]+)\)")


def _arm_argument_choices(script_path: Path) -> list[str] | None:
    """Read argparse's ``--arm`` choices without importing hardware dependencies."""

    module = ast.parse(script_path.read_text(encoding="utf-8"), filename=script_path)
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if not isinstance(node.args[0], ast.Constant) or node.args[0].value != "--arm":
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices" and isinstance(keyword.value, ast.List):
                return [
                    item.value
                    for item in keyword.value.elts
                    if isinstance(item, ast.Constant)
                ]
        return None
    raise AssertionError(f"{script_path} has no --arm argparse argument")


@pytest.mark.static
@pytest.mark.parametrize("script_path", DEMO_SCRIPTS, ids=lambda path: path.stem)
def test_demo_collectors_reject_unsupported_left_arm(script_path: Path) -> None:
    assert _arm_argument_choices(script_path) == ["right"]


def _robot_argument_parser(script_path: Path) -> argparse.ArgumentParser:
    """Reconstruct the ``--robots`` contract without importing input backends."""

    module = ast.parse(script_path.read_text(encoding="utf-8"), filename=script_path)
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if (
            not isinstance(node.args[0], ast.Constant)
            or node.args[0].value != "--robots"
        ):
            continue
        keyword_values = {keyword.arg: keyword.value for keyword in node.keywords}
        nargs = keyword_values.get("nargs")
        robot_type = keyword_values.get("type")
        assert isinstance(nargs, ast.Constant) and nargs.value == "+"
        assert isinstance(robot_type, ast.Name) and robot_type.id == "str", (
            f"{script_path.name} must parse each explicit robot name as a string"
        )
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--robots", nargs="+", type=str)
        return parser
    raise AssertionError(f"{script_path} has no --robots argparse argument")


@pytest.mark.static
@pytest.mark.parametrize("script_path", DEMO_SCRIPTS, ids=lambda path: path.stem)
def test_demo_collectors_preserve_explicit_robot_names(script_path: Path) -> None:
    robots = (
        _robot_argument_parser(script_path).parse_args(["--robots", "Panda"]).robots
    )
    assert robots == ["Panda"]


@pytest.mark.official_integration
def test_control_env_reset_retries_only_randomization_errors() -> None:
    from robosuite.utils.errors import RandomizationError

    from libero.libero.envs.env_wrapper import ControlEnv

    wrapper = object.__new__(ControlEnv)
    calls = 0

    def reset() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("unexpected simulation failure")

    wrapper.env = SimpleNamespace(reset=reset)
    with pytest.raises(ValueError, match="unexpected simulation failure"):
        wrapper.reset()
    assert calls == 1

    results = iter((RandomizationError("placement"), {"observation": 1}))

    def reset_after_randomization_error():
        result = next(results)
        if isinstance(result, BaseException):
            raise result
        return result

    wrapper.env = SimpleNamespace(reset=reset_after_randomization_error)
    assert wrapper.reset() == {"observation": 1}


def _load_demo_conversion_module(monkeypatch: pytest.MonkeyPatch, script_path: Path):
    """Load conversion code without importing a renderer or input device backend."""

    def refactor_composite_controller_config(*args, **kwargs):
        return args, kwargs

    fake_robosuite = ModuleType("robosuite")
    fake_robosuite.__path__ = []
    fake_robosuite.__version__ = "1.5.2"
    fake_robosuite.load_part_controller_config = lambda **_: {}
    fake_controllers = ModuleType("robosuite.controllers")
    fake_controllers.__path__ = []
    fake_composite = ModuleType("robosuite.controllers.composite")
    fake_composite.__path__ = []
    fake_factory = ModuleType(
        "robosuite.controllers.composite.composite_controller_factory"
    )
    fake_factory.refactor_composite_controller_config = (
        refactor_composite_controller_config
    )
    fake_wrappers = ModuleType("robosuite.wrappers")
    fake_wrappers.DataCollectionWrapper = object
    fake_wrappers.VisualizationWrapper = object
    fake_utils = ModuleType("robosuite.utils")
    fake_utils.__path__ = []
    fake_errors = ModuleType("robosuite.utils.errors")
    fake_errors.RandomizationError = RuntimeError

    fake_libero = ModuleType("libero")
    fake_libero.__path__ = []
    fake_libero_package = ModuleType("libero.libero")
    fake_libero_package.__path__ = []
    fake_envs = ModuleType("libero.libero.envs")
    fake_envs.__path__ = []
    fake_envs.TASK_MAPPING = {}
    fake_bddl_utils = ModuleType("libero.libero.envs.bddl_utils")
    fake_bddl_base_domain = ModuleType("libero.libero.envs.bddl_base_domain")
    fake_bddl_base_domain.TASK_MAPPING = {}
    fake_init_path = ModuleType("init_path")
    fake_cv2 = ModuleType("cv2")
    fake_termcolor = ModuleType("termcolor")
    fake_termcolor.colored = lambda value, *_, **__: value
    fake_libero.libero = fake_libero_package
    fake_libero_package.envs = fake_envs
    fake_envs.bddl_utils = fake_bddl_utils
    fake_envs.bddl_base_domain = fake_bddl_base_domain

    fake_modules = {
        "robosuite": fake_robosuite,
        "robosuite.controllers": fake_controllers,
        "robosuite.controllers.composite": fake_composite,
        "robosuite.controllers.composite.composite_controller_factory": fake_factory,
        "robosuite.wrappers": fake_wrappers,
        "robosuite.utils": fake_utils,
        "robosuite.utils.errors": fake_errors,
        "libero": fake_libero,
        "libero.libero": fake_libero_package,
        "libero.libero.envs": fake_envs,
        "libero.libero.envs.bddl_utils": fake_bddl_utils,
        "libero.libero.envs.bddl_base_domain": fake_bddl_base_domain,
        "init_path": fake_init_path,
        "cv2": fake_cv2,
        "termcolor": fake_termcolor,
    }
    for module_name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)

    module_name = f"_demo_conversion_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.official_integration
@pytest.mark.parametrize("script_path", DEMO_SCRIPTS, ids=lambda path: path.stem)
def test_demo_hdf5_conversion_embeds_bddl_source_text(
    monkeypatch: pytest.MonkeyPatch, script_path: Path, tmp_path: Path
) -> None:
    import h5py
    import numpy as np

    module = _load_demo_conversion_module(monkeypatch, script_path)
    raw_directory = tmp_path / "raw"
    episode_directory = raw_directory / "episode_0"
    episode_directory.mkdir(parents=True)
    (episode_directory / "model.xml").write_text("<mujoco/>", encoding="utf-8")
    np.savez(
        episode_directory / "state_000.npz",
        env=np.array("LIBERO_Test"),
        states=np.array([[0.0, 1.0], [2.0, 3.0]]),
        action_infos=np.array([{"actions": np.array([0.0, 0.0])}], dtype=object),
    )
    bddl_text = "(define (problem LIBERO_Test) (:domain robosuite))\n"
    bddl_path = tmp_path / "task.bddl"
    bddl_path.write_text(bddl_text, encoding="utf-8")
    module.problem_info = {"problem_name": "libero_test"}
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    args = SimpleNamespace(bddl_file=str(bddl_path))

    module.gather_demonstrations_as_hdf5(
        str(raw_directory), str(output_directory), "{}", args
    )

    with h5py.File(output_directory / "demo.hdf5", "r") as dataset:
        assert dataset["data"].attrs["bddl_file_content"] == bddl_text


def _segmentation_wrapper_with_instances(instance_names: list[str]):
    import numpy as np

    from libero.libero.envs.env_wrapper import SegmentationRenderEnv

    class MountedPanda:
        pass

    class RethinkMount:
        pass

    class PandaGripper:
        pass

    robot = SimpleNamespace(
        robot_model=MountedPanda(),
        gripper={"right": PandaGripper()},
    )
    robot.robot_model.base = RethinkMount()
    wrapper = object.__new__(SegmentationRenderEnv)
    wrapper.env = SimpleNamespace(
        reset=lambda: {"agentview_image": np.zeros((1, 1, 3), dtype=np.uint8)},
        robots=[robot],
        model=SimpleNamespace(
            instances_to_ids={name: index for index, name in enumerate(instance_names)}
        ),
    )
    return wrapper


@pytest.mark.official_integration
def test_segmentation_reset_reports_missing_robot_instance() -> None:
    wrapper = _segmentation_wrapper_with_instances(["table0", "cup0"])
    with pytest.raises(RuntimeError, match="robot.*instance"):
        wrapper.reset()


@pytest.mark.official_integration
def test_segmentation_keeps_non_robot_instances_out_of_robot_threshold() -> None:
    import numpy as np

    wrapper = _segmentation_wrapper_with_instances(["table0", "MountedPanda0", "cup0"])
    wrapper.reset()
    instances = wrapper.get_segmentation_instances(np.array([[3]], dtype=np.int32))
    assert instances["cup0"][0, 0] == 3
    assert instances["robot"][0, 0] == 0


@pytest.mark.official_integration
def test_segmentation_of_interest_uses_explicit_robot_ids_not_background() -> None:
    import numpy as np

    from libero.libero.envs.env_wrapper import SegmentationRenderEnv

    wrapper = object.__new__(SegmentationRenderEnv)
    wrapper.env = SimpleNamespace(obj_of_interest=["target"])
    wrapper.instance_to_id = {"target": 3, "other": 4}
    wrapper.robot_segmentation_ids = frozenset({1, 2})
    segmentation = np.array([[0, 1, 2, 3, 4]], dtype=np.int32)

    interest = wrapper.get_segmentation_of_interest(segmentation)

    np.testing.assert_array_equal(interest, np.array([[0, -1, -1, 1, 0]]))


@pytest.mark.official_integration
def test_segmentation_random_color_helper_is_deterministic_uint8_rgb() -> None:
    import numpy as np

    from libero.libero.envs.env_wrapper import SegmentationRenderEnv

    wrapper = object.__new__(SegmentationRenderEnv)
    segmentation = np.array([[0, 1, 2], [255, 256, 257]], dtype=np.int32)
    first = wrapper.segmentation_to_rgb(segmentation, random_colors=True)
    second = wrapper.segmentation_to_rgb(segmentation, random_colors=True)
    assert first.shape == segmentation.shape + (3,)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)


@pytest.mark.official_integration
def test_envs_public_import_registers_every_packaged_problem() -> None:
    from libero.libero.envs import (
        TASK_MAPPING,
        OffScreenRenderEnv,
        SegmentationRenderEnv,
    )

    problem_names = set()
    for bddl_file in (REPOSITORY_ROOT / "libero" / "libero" / "bddl_files").rglob(
        "*.bddl"
    ):
        match = BDDL_PROBLEM_PATTERN.search(bddl_file.read_text(encoding="utf-8"))
        assert match is not None, bddl_file
        problem_names.add(match.group(1).lower())

    assert problem_names <= TASK_MAPPING.keys()
    assert OffScreenRenderEnv.__name__ == "OffScreenRenderEnv"
    assert SegmentationRenderEnv.__name__ == "SegmentationRenderEnv"
