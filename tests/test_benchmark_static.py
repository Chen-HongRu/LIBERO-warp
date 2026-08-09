"""Fast, deterministic checks for the LIBERO benchmark manifest and package data."""

from __future__ import annotations

import xml.etree.ElementTree as element_tree

import pytest

from ._manifest import LIBERO_TASK_MAP
from .conftest import BENCHMARK_ROOT

EXPECTED_SUITE_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}


def benchmark_entries() -> list[tuple[str, str]]:
    return [
        (suite_name, task_name)
        for suite_name, task_names in LIBERO_TASK_MAP.items()
        for task_name in task_names
    ]


@pytest.mark.static
def test_benchmark_task_catalog_is_complete_and_unique() -> None:
    assert {
        suite: len(tasks) for suite, tasks in LIBERO_TASK_MAP.items()
    } == EXPECTED_SUITE_COUNTS
    entries = benchmark_entries()
    assert len(entries) == 130
    assert len(entries) == len(set(entries))
    for suite_name, task_names in LIBERO_TASK_MAP.items():
        assert len(task_names) == len(set(task_names)), suite_name


@pytest.mark.static
@pytest.mark.parametrize(("suite_name", "task_name"), benchmark_entries())
def test_every_task_has_bddl_and_init_state(suite_name: str, task_name: str) -> None:
    bddl_path = BENCHMARK_ROOT / "bddl_files" / suite_name / f"{task_name}.bddl"
    init_state_path = (
        BENCHMARK_ROOT / "init_files" / suite_name / f"{task_name}.pruned_init"
    )
    assert bddl_path.is_file(), bddl_path
    assert bddl_path.stat().st_size > 0, bddl_path
    assert init_state_path.is_file(), init_state_path
    assert init_state_path.stat().st_size > 0, init_state_path


@pytest.mark.static
def test_all_packaged_mjcf_files_are_well_formed_xml() -> None:
    asset_root = BENCHMARK_ROOT / "assets"
    xml_files = sorted(asset_root.rglob("*.xml"))
    assert xml_files, "LIBERO assets must include MuJoCo XML files"
    malformed = []
    for xml_file in xml_files:
        try:
            element_tree.parse(xml_file)
        except element_tree.ParseError as error:
            malformed.append(f"{xml_file}: {error}")
    assert not malformed, "\n".join(malformed)


@pytest.mark.static
def test_benchmark_package_directories_exist() -> None:
    for directory_name in ("bddl_files", "init_files", "assets", "benchmark"):
        assert (BENCHMARK_ROOT / directory_name).is_dir()
