"""Isolated contracts for the narrow robosuite 1.5.2 / MuJoCo 3.11 patch."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
import sys
from types import ModuleType

import pytest

from .conftest import REPOSITORY_ROOT

COMPATIBILITY_SOURCE = REPOSITORY_ROOT / "libero/libero/envs/mujoco3_compat.py"


def _load_compatibility_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    robosuite_version: str,
    mujoco_version: str,
    exposes_legacy_qm: bool,
):
    """Load the compatibility module against fake dependencies, never real globals."""

    class FakeMjData:
        pass

    if exposes_legacy_qm:
        FakeMjData.qM = object()

    class FakeController:
        def update(self, force: bool = False) -> None:
            del force

    FakeController.update.__module__ = "robosuite.controllers.parts.controller"
    original_update = FakeController.update
    fake_mujoco = ModuleType("mujoco")
    fake_mujoco.MjData = FakeMjData
    fake_mujoco.__version__ = mujoco_version
    fake_numpy = ModuleType("numpy")
    fake_robosuite = ModuleType("robosuite")
    fake_robosuite.__path__ = []
    fake_robosuite.__version__ = robosuite_version
    fake_controllers = ModuleType("robosuite.controllers")
    fake_controllers.__path__ = []
    fake_parts = ModuleType("robosuite.controllers.parts")
    fake_parts.__path__ = []
    fake_controller_module = ModuleType("robosuite.controllers.parts.controller")
    fake_controller_module.Controller = FakeController

    fake_robosuite.controllers = fake_controllers
    fake_controllers.parts = fake_parts
    fake_parts.controller = fake_controller_module
    for module_name, module in {
        "mujoco": fake_mujoco,
        "numpy": fake_numpy,
        "robosuite": fake_robosuite,
        "robosuite.controllers": fake_controllers,
        "robosuite.controllers.parts": fake_parts,
        "robosuite.controllers.parts.controller": fake_controller_module,
    }.items():
        monkeypatch.setitem(sys.modules, module_name, module)

    def fake_version(distribution_name: str) -> str:
        return {
            "robosuite": robosuite_version,
            "mujoco": mujoco_version,
        }[distribution_name]

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    monkeypatch.setattr(inspect, "getsource", lambda _: "self.sim.data.qM")
    module_name = "_isolated_mujoco3_compatibility"
    spec = importlib.util.spec_from_file_location(module_name, COMPATIBILITY_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, FakeController, original_update


@pytest.mark.static
def test_compatibility_patch_activates_only_for_supported_versions_without_legacy_qm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, controller, original_update = _load_compatibility_module(
        monkeypatch,
        robosuite_version="1.5.2",
        mujoco_version="3.11.0",
        exposes_legacy_qm=False,
    )
    assert controller.update is module._update_with_mujoco_3_mass_matrix
    assert controller.update is not original_update
    assert controller._libero_mujoco3_patch is True


@pytest.mark.static
def test_compatibility_patch_does_not_monkeypatch_unknown_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="only.*supports"):
        _load_compatibility_module(
            monkeypatch,
            robosuite_version="1.5.3",
            mujoco_version="3.11.0",
            exposes_legacy_qm=False,
        )


@pytest.mark.static
def test_compatibility_patch_skips_bindings_that_expose_legacy_qm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, controller, original_update = _load_compatibility_module(
        monkeypatch,
        robosuite_version="1.5.2",
        mujoco_version="3.11.0",
        exposes_legacy_qm=True,
    )
    assert controller.update is original_update
    assert not hasattr(controller, "_libero_mujoco3_patch")
