"""Source-level packaging contracts that need no simulator or hardware imports."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import zipfile
from importlib.metadata import distribution
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 test matrix
    import tomli as tomllib

from .conftest import REPOSITORY_ROOT

CONSOLE_ENTRYPOINTS = {
    "libero.config_copy": "scripts.config_copy:main",
    "libero.create_template": "scripts.create_template:main",
    "libero.download_datasets": "benchmark_scripts.download_libero_datasets:main",
    "lifelong.eval": "libero.lifelong.evaluate:main",
    "lifelong.main": "libero.lifelong.main:main",
}


@pytest.mark.static
def test_package_data_declares_runtime_assets_and_templates() -> None:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    package_data = project["tool"]["setuptools"]["package-data"]
    assert set(package_data["libero.libero"]) >= {
        "assets/**/*",
        "bddl_files/**/*",
        "init_files/**/*",
    }
    assert "templates/*" in package_data["libero"]
    assert "configs/**/*.yaml" in package_data["libero"]
    for relative_path in (
        "libero/libero/assets",
        "libero/libero/bddl_files",
        "libero/libero/init_files",
        "libero/templates/scene_template.xml",
        "libero/templates/problem_class_template.py",
        "libero/configs/config.yaml",
        "templates/scene_template.xml",
        "templates/problem_class_template.py",
    ):
        assert (REPOSITORY_ROOT / relative_path).exists(), relative_path


@pytest.mark.static
def test_python_and_dependency_profiles_keep_accelerators_optional() -> None:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)["project"]

    assert project["requires-python"] == ">=3.12,<3.13"
    core_names = {
        requirement.split("==", maxsplit=1)[0].split(">=", maxsplit=1)[0]
        for requirement in project["dependencies"]
    }
    assert {"mujoco", "robosuite"} <= core_names
    assert {"torch", "mujoco-warp", "warp-lang"}.isdisjoint(core_names)

    profiles = project["optional-dependencies"]
    assert set(profiles) >= {"official", "tensor", "warp", "legacy"}
    assert profiles["official"] == []
    assert any(requirement.startswith("torch>=") for requirement in profiles["tensor"])
    assert any(
        requirement.startswith("mujoco-warp==") for requirement in profiles["warp"]
    )
    assert any(
        requirement.startswith("warp-lang==") for requirement in profiles["warp"]
    )


@pytest.mark.static
def test_console_entrypoints_are_declared_in_the_packaged_source_tree() -> None:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "benchmark_scripts*",
        "benchmarks*",
        "libero*",
        "scripts*",
    ]
    assert CONSOLE_ENTRYPOINTS.items() <= project["project"]["scripts"].items()

    for entrypoint_name, target in CONSOLE_ENTRYPOINTS.items():
        assert project["project"]["scripts"][entrypoint_name] == target
        module_name, callable_name = target.split(":", maxsplit=1)
        module_path = REPOSITORY_ROOT / Path(*module_name.split(".")).with_suffix(".py")
        module = ast.parse(
            module_path.read_text(encoding="utf-8"), filename=module_path
        )
        assert any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == callable_name
            for node in module.body
        ), f"{target} must target a module-level callable"


@pytest.mark.static
def test_wheel_contains_and_imports_runtime_benchmark_modules(tmp_path: Path) -> None:
    """Runtime's ``benchmarks.ctrl_trace`` import must work outside the source tree."""

    wheel_directory = tmp_path / "wheel"
    build = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_directory)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheels = tuple(wheel_directory.glob("libero_warp-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    expected_paths = {
        "benchmarks/__init__.py",
        "benchmarks/ctrl_trace.py",
        "benchmarks/compare_m1_reports.py",
        "benchmarks/benchmark_warp_spike.py",
        "benchmarks/benchmark_official_spike.py",
        "scripts/config_copy.py",
        "libero/configs/config.yaml",
        "libero/configs/policy/bc_rnn_policy.yaml",
        "libero/lifelong/__init__.py",
        "libero/lifelong/main.py",
        "libero/lifelong/evaluate.py",
    }
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert expected_paths <= names
        assert any(name.endswith("/templates/scene_template.xml") for name in names)
        assert any(
            name.endswith("/templates/problem_class_template.py") for name in names
        )

    environment = _isolated_cli_environment(tmp_path)
    environment["PYTHONPATH"] = str(wheel)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import benchmarks.benchmark_warp_spike as warp; "
                "import benchmarks.compare_m1_reports as compare; "
                "import benchmarks.ctrl_trace as trace; "
                "import libero.libero.envs as envs; "
                "from libero.libero import benchmark; "
                "from libero.libero.envs.env_wrapper import ControlEnv, DemoRenderEnv; "
                "assert '.whl/' in warp.__file__; "
                "assert '.whl/' in compare.__file__; "
                "assert '.whl/' in trace.__file__; "
                "assert '.whl/' in envs.__file__; "
                "assert benchmark is not None; "
                "assert ControlEnv is not None and DemoRenderEnv is not None"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    config_destination = tmp_path / "wheel-configs"
    config_copy = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.config_copy",
            "--destination",
            str(config_destination),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert config_copy.returncode == 0, config_copy.stdout + config_copy.stderr
    assert (config_destination / "config.yaml").is_file()

    legacy_import = subprocess.run(
        [sys.executable, "-c", "import libero.lifelong.algos"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert legacy_import.returncode != 0
    assert "pip install libero-warp[legacy]" in legacy_import.stderr

    installed_environment = tmp_path / "installed-wheel"
    create_environment = subprocess.run(
        [
            "uv",
            "venv",
            "--python",
            sys.executable,
            str(installed_environment),
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert create_environment.returncode == 0, (
        create_environment.stdout + create_environment.stderr
    )
    executable_directory = installed_environment / (
        "Scripts" if sys.platform == "win32" else "bin"
    )
    installed_python = executable_directory / (
        "python.exe" if sys.platform == "win32" else "python"
    )
    install = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(installed_python),
            "--no-deps",
            str(wheel),
        ],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert install.returncode == 0, install.stdout + install.stderr

    installed_config_destination = tmp_path / "installed-configs"
    installed_cli = executable_directory / (
        "libero.config_copy.exe" if sys.platform == "win32" else "libero.config_copy"
    )
    installed_smoke = subprocess.run(
        [
            str(installed_cli),
            "--destination",
            str(installed_config_destination),
        ],
        cwd=tmp_path,
        env=_isolated_cli_environment(tmp_path),
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert installed_smoke.returncode == 0, (
        installed_smoke.stdout + installed_smoke.stderr
    )
    assert (installed_config_destination / "config.yaml").is_file()

    installed_import = subprocess.run(
        [
            str(installed_python),
            "-c",
            (
                "from importlib.metadata import distribution; "
                "import libero.configs, libero.lifelong; "
                "entries=distribution('libero-warp').entry_points; "
                "names={entry.name for entry in entries}; "
                f"assert {set(CONSOLE_ENTRYPOINTS)!r} <= names"
            ),
        ],
        cwd=tmp_path,
        env=_isolated_cli_environment(tmp_path),
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert installed_import.returncode == 0, (
        installed_import.stdout + installed_import.stderr
    )

    for legacy_command in ("lifelong.eval", "lifelong.main"):
        legacy_cli = executable_directory / (
            f"{legacy_command}.exe" if sys.platform == "win32" else legacy_command
        )
        missing_extra = subprocess.run(
            [str(legacy_cli), "--help"],
            cwd=tmp_path,
            env=_isolated_cli_environment(tmp_path),
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        assert missing_extra.returncode != 0
        assert "pip install libero-warp[legacy]" in missing_extra.stderr


@pytest.mark.static
def test_installed_package_metadata_exposes_required_console_entrypoints() -> None:
    """Check the entry-point metadata generated by the package installation."""

    entrypoints = {
        entrypoint.name: entrypoint.value
        for entrypoint in distribution("libero-warp").entry_points
        if entrypoint.group == "console_scripts"
    }
    assert CONSOLE_ENTRYPOINTS.items() <= entrypoints.items()


@pytest.mark.static
def test_dataset_conversion_module_has_no_legacy_top_level_imports() -> None:
    """``python -m scripts.create_dataset`` must work from an installed wheel."""

    module_path = REPOSITORY_ROOT / "scripts" / "create_dataset.py"
    module = ast.parse(module_path.read_text(encoding="utf-8"), filename=module_path)
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            assert all(alias.name != "init_path" for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module == "libero.libero.envs":
            assert all(alias.name != "*" for alias in node.names)


@pytest.mark.static
def test_retained_packaged_sources_do_not_import_legacy_init_path() -> None:
    """Removed bootstrap code must not remain a runtime dependency in a wheel."""

    retained_roots = (
        REPOSITORY_ROOT / "libero",
        REPOSITORY_ROOT / "scripts",
        REPOSITORY_ROOT / "benchmark_scripts",
    )
    for root in retained_roots:
        for module_path in root.rglob("*.py"):
            module = ast.parse(
                module_path.read_text(encoding="utf-8"), filename=module_path
            )
            for node in module.body:
                if isinstance(node, ast.Import):
                    assert all(alias.name != "init_path" for alias in node.names), (
                        f"{module_path.relative_to(REPOSITORY_ROOT)} imports "
                        "removed init_path"
                    )
                if isinstance(node, ast.ImportFrom):
                    assert node.module != "init_path", (
                        f"{module_path.relative_to(REPOSITORY_ROOT)} imports "
                        "removed init_path"
                    )


@pytest.mark.static
def test_config_copy_tool_uses_installed_package_resources(tmp_path: Path) -> None:
    destination = tmp_path / "copied-configs"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.config_copy",
            "--destination",
            str(destination),
        ],
        cwd=tmp_path,
        env=_isolated_cli_environment(tmp_path),
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (destination / "config.yaml").is_file()
    assert (destination / "policy" / "bc_rnn_policy.yaml").is_file()


def _isolated_cli_environment(tmp_path: Path) -> dict[str, str]:
    config_dir = tmp_path / "libero-config"
    config_dir.mkdir(exist_ok=True)
    benchmark_root = REPOSITORY_ROOT / "libero" / "libero"
    (config_dir / "config.yaml").write_text(
        "\n".join(
            (
                f"benchmark_root: {benchmark_root}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"init_states: {benchmark_root / 'init_files'}",
                f"datasets: {tmp_path / 'datasets'}",
                f"assets: {benchmark_root / 'assets'}",
                "",
            )
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(tmp_path / "home"),
            "LIBERO_CONFIG_PATH": str(config_dir),
            "MUJOCO_GL": "disable",
        }
    )
    return environment


@pytest.mark.official_integration
@pytest.mark.parametrize(
    "module_name",
    ("benchmark_scripts.download_libero_datasets", "scripts.create_dataset"),
)
def test_retained_dataset_entrypoints_offer_offline_help(
    module_name: str, tmp_path: Path
) -> None:
    """CLI help must not download data, prompt, or use a user's LIBERO config."""

    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=REPOSITORY_ROOT,
        env=_isolated_cli_environment(tmp_path),
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.static
def test_get_affordance_info_is_an_installed_module_smoke(tmp_path: Path) -> None:
    """Retained utility modules must execute without relying on shell path hacks."""

    result = subprocess.run(
        [sys.executable, "-m", "scripts.get_affordance_info"],
        cwd=REPOSITORY_ROOT,
        env=_isolated_cli_environment(tmp_path),
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
