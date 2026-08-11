import configparser
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.generate_libero_api_manifest import (
    _coverage,
    _parse_module,
    _resolve_exports,
    build_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
MANIFEST_PATH = REPO_ROOT / "tests" / "api_manifest" / f"upstream-{TARGET_COMMIT}.json"
MATRIX_PATH = REPO_ROOT / "docs" / "LIBERO_API_COMPATIBILITY_MATRIX.md"
pytestmark = pytest.mark.static


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_frozen_target_is_exact_local_upstream_commit_and_ancestor():
    manifest = _manifest()

    assert manifest["compatibility_target"] == {
        "commit": TARGET_COMMIT,
        "committed_at": "2025-03-15T20:13:56+08:00",
        "ref": TARGET_COMMIT,
        "remote_url": "https://github.com/Lifelong-Robot-Learning/LIBERO.git",
        "tree": "99f4ada3f1d62e026fc9ff2390eb4ff8a1760e60",
    }
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", TARGET_COMMIT, "HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )


def test_manifest_is_complete_and_reproducible_from_git_objects():
    recorded = _manifest()
    regenerated = build_manifest(REPO_ROOT, recorded["compatibility_target"]["ref"])

    recorded_head = recorded["current"]["head_commit"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", recorded_head, "HEAD"],
        cwd=REPO_ROOT,
        check=True,
    )
    regenerated["current"]["head_commit"] = recorded_head
    assert regenerated == recorded
    assert recorded["schema_version"] == 2
    assert recorded["generator_contract"] == {
        "ast_renderer": "stdlib",
        "python": ">=3.11,<3.13",
    }
    assert len(recorded["target"]["modules"]) == 97
    assert len(recorded["target"]["source_data_files"]) == 1002
    assert all(
        module in recorded["comparison"]["module_coverage"]
        for module in recorded["target"]["modules"]
    )


def test_manifest_paths_independently_cover_the_complete_target_git_tree():
    recorded = _manifest()
    raw_paths = _git(
        "ls-tree",
        "-r",
        "--name-only",
        TARGET_COMMIT,
        "--",
        *recorded["scope"]["source_roots"],
    ).splitlines()
    expected_python_paths = {
        path
        for path in raw_paths
        if path.endswith(".py")
        and any(
            path.startswith(f"{root}/") for root in recorded["scope"]["module_roots"]
        )
    }
    expected_data_paths = {
        path
        for path in raw_paths
        if not path.endswith(".py")
        or path.startswith("templates/")
        or "/templates/" in path
    }
    manifest_python_paths = {
        module["path"] for module in recorded["target"]["modules"].values()
    }

    assert manifest_python_paths == expected_python_paths
    assert set(recorded["target"]["source_data_files"]) == expected_data_paths
    assert "templates.problem_class_template" not in recorded["target"]["modules"]
    assert (
        "templates/problem_class_template.py" in recorded["target"]["source_data_files"]
    )
    assert not any(
        "parse_error" in module for module in recorded["target"]["modules"].values()
    )


def test_each_target_module_digest_matches_its_git_blob():
    modules = _manifest()["target"]["modules"]

    for module in modules.values():
        source = _git("show", f"{TARGET_COMMIT}:{module['path']}")
        assert hashlib.sha256(source.encode()).hexdigest() == module["sha256"]


def test_each_target_source_data_record_matches_git_tree_oid_and_size():
    data_files = _manifest()["target"]["source_data_files"]
    raw = _git(
        "ls-tree",
        "-r",
        "-l",
        TARGET_COMMIT,
        "--",
        *_manifest()["scope"]["source_roots"],
    )
    git_entries = {}
    for line in raw.splitlines():
        metadata, path = line.split("\t", 1)
        _mode, object_type, oid, size = metadata.split()
        if object_type == "blob":
            git_entries[path] = {"git_blob_oid": oid, "size": int(size)}

    assert data_files == {path: git_entries[path] for path in data_files}


def test_static_export_parser_handles_constant_all_composition():
    record = _parse_module(
        "example.py",
        '__all__ = ["first"] + ["second"]\n'
        "first = 1\nsecond = 2\nthird = 3\n"
        '__all__.extend(["third"])\n',
    )
    modules = {"example": record}

    _resolve_exports(modules)

    assert set(modules["example"]["exports"]) == {"first", "second", "third"}
    assert not modules["example"]["explicit_all_unresolved"]


def test_static_export_parser_captures_conditional_top_level_bindings():
    record = _parse_module(
        "example.py",
        "try:\n    AVAILABLE = True\nexcept ImportError:\n    AVAILABLE = False\n",
    )
    binding = record["declared_symbols"]["AVAILABLE"]

    assert binding["kind"] == "conditional_binding"
    assert {item["value"] for item in binding["alternatives"]} == {False, True}


def test_static_coverage_detects_changed_symbol_origin():
    target = {"pkg": _parse_module("pkg/__init__.py", "from origin_a import Thing\n")}
    current = {"pkg": _parse_module("pkg/__init__.py", "from origin_b import Thing\n")}
    _resolve_exports(target)
    _resolve_exports(current)

    coverage = _coverage(target, current)["pkg"]

    assert coverage["status"] == "static_drift"
    assert coverage["symbol_changes"][0]["symbol"] == "Thing"


def test_manifest_contains_constructor_and_method_signatures():
    modules = _manifest()["target"]["modules"]
    wrapper = modules["libero.libero.envs.env_wrapper"]["exports"]

    assert wrapper["ControlEnv"]["signature"].startswith("(self, bddl_file_name")
    assert wrapper["ControlEnv"]["members"]["step"]["signature"] == "(self, action)"
    assert wrapper["OffScreenRenderEnv"]["signature"] == "(self, **kwargs)"
    assert (
        wrapper["DemoRenderEnv"]["members"]["_get_observations"]["signature"]
        == "(self)"
    )


def test_frozen_public_import_contract_executes_in_isolated_process():
    source = "\n".join(
        (
            "from libero.libero import benchmark, get_libero_path",
            "from libero.libero.envs import (",
            "    DummyVectorEnv, OffScreenRenderEnv, SegmentationRenderEnv,",
            "    SubprocVectorEnv,",
            ")",
            "from libero.libero.envs.env_wrapper import ControlEnv, DemoRenderEnv",
            "assert benchmark is not None and callable(get_libero_path)",
            "assert all(item is not None for item in (",
            "    DummyVectorEnv, OffScreenRenderEnv, SegmentationRenderEnv,",
            "    SubprocVectorEnv, ControlEnv, DemoRenderEnv,",
            "))",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_manifest_captures_real_conditional_binding_and_base_class_drift():
    manifest = _manifest()
    availability = manifest["target"]["modules"]["libero.libero.utils.download_utils"][
        "declared_symbols"
    ]["HUGGINGFACE_AVAILABLE"]
    domain_changes = manifest["comparison"]["module_coverage"][
        "libero.libero.envs.bddl_base_domain"
    ]["symbol_changes"]
    domain = next(
        change for change in domain_changes if change["symbol"] == "BDDLBaseDomain"
    )

    assert availability["kind"] == "conditional_binding"
    assert {item["value"] for item in availability["alternatives"]} == {False, True}
    assert domain["target"]["bases"] == ["SingleArmEnv"]
    assert domain["current"]["bases"] == ["ManipulationEnv"]


def test_distribution_and_cli_metadata_are_explicit():
    manifest = _manifest()

    assert manifest["target"]["distribution"]["name"] == "libero"
    assert manifest["current"]["distribution"]["name"] == "libero-warp"
    assert manifest["target"]["console_scripts"] == {
        "libero.config_copy": "scripts.config_copy:main",
        "libero.create_template": "scripts.create_template:main",
        "lifelong.eval": "libero.lifelong.evaluate:main",
        "lifelong.main": "libero.lifelong.main:main",
    }
    assert manifest["comparison"]["changed_console_scripts"] == {}


def test_built_wheel_gap_matches_the_frozen_source_inventory(tmp_path):
    manifest = _manifest()
    wheel_directory = tmp_path / "wheel"
    build = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_directory)],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheels = tuple(wheel_directory.glob("libero_warp-*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_paths = set(archive.namelist())
        entrypoint_path = next(
            path for path in wheel_paths if path.endswith(".dist-info/entry_points.txt")
        )
        parser = configparser.ConfigParser()
        parser.read_string(archive.read(entrypoint_path).decode())

    # ``setuptools.data-files`` stores compatibility files under the wheel's
    # installation data scheme. Normalize those archive locations back to the
    # frozen source paths before comparing the complete target inventory.
    wheel_source_paths = set(wheel_paths)
    data_marker = ".data/data/"
    wheel_source_paths.update(
        path.split(data_marker, maxsplit=1)[1]
        for path in wheel_paths
        if data_marker in path
    )

    target_module_paths = {
        module["path"] for module in manifest["target"]["modules"].values()
    }
    expected_missing_module_paths = {
        manifest["target"]["modules"][module_name]["path"]
        for module_name, coverage in manifest["comparison"]["module_coverage"].items()
        if coverage["status"] == "missing_module"
    }
    target_data_paths = set(manifest["target"]["source_data_files"])
    wheel_console_scripts = dict(parser["console_scripts"])

    assert target_module_paths - wheel_source_paths == expected_missing_module_paths
    assert target_data_paths - wheel_source_paths == set(
        manifest["comparison"]["missing_source_data_files"]
    )
    assert set(manifest["target"]["console_scripts"]) - set(
        wheel_console_scripts
    ) == set(manifest["comparison"]["missing_console_scripts"])
    assert all(
        wheel_console_scripts[name] == target
        for name, target in manifest["target"]["console_scripts"].items()
        if name in wheel_console_scripts
    )


def test_every_missing_target_module_and_cli_has_an_explicit_g0_decision():
    manifest = _manifest()
    matrix = MATRIX_PATH.read_text()
    missing_modules = [
        module
        for module, coverage in manifest["comparison"]["module_coverage"].items()
        if coverage["status"] == "missing_module"
    ]

    for module in missing_modules:
        assert f"`{module}`" in matrix
    for command in manifest["comparison"]["missing_console_scripts"]:
        assert f"`{command}`" in matrix


def test_downstream_private_usage_and_backend_schema_are_frozen():
    matrix = MATRIX_PATH.read_text()
    required_accesses = (
        "`action_spec()`",
        "`action_space.low/high`",
        "`sim.reset(); sim.forward()`",
        "`get_real_depth_map(sim, depth)`",
        "`deterministic_reset`",
        "`hard_reset`",
        "`robot.gripper.current_action`",
        "`_render_context_offscreen.gl_ctx.make_current()`",
        "`_update_observables(force=True)`",
        "`_get_observations()`",
    )
    capabilities = (
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
    )

    assert all(access in matrix for access in required_accesses)
    assert all(f"\n{capability}\n" in matrix for capability in capabilities)
    assert "`selection_source`" in matrix


def test_environment_contract_decouples_python_and_gpu_profiles():
    matrix = MATRIX_PATH.read_text()

    assert "Python 3.12" in matrix
    assert "`>=3.12,<3.13`" in matrix
    assert "`>=2.4,<3`" in matrix
    assert "Never pin or install `nvidia-*` packages directly" in matrix
    assert "must not run a full `uv sync`" in matrix
    assert all(
        f"\n{profile} " in matrix
        for profile in ("core", "official", "tensor", "warp", "legacy")
    )


def test_g1_inventory_restores_every_target_module():
    manifest = _manifest()
    coverage = manifest["comparison"]["module_coverage"].values()
    counts = {
        status: sum(item["status"] == status for item in coverage)
        for status in {item["status"] for item in coverage}
    }

    assert counts == {
        "compatible_static_surface": 73,
        "static_drift": 24,
    }
    assert len(manifest["current"]["modules"]) == 111
    assert manifest["comparison"]["missing_source_data_files"] == []
    assert len(manifest["comparison"]["extra_source_data_files"]) == 2
    assert manifest["comparison"]["missing_console_scripts"] == []
