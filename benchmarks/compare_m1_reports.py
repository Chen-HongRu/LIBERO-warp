"""Compare reproducible official and MJWarp M1 benchmark reports.

The comparison is intentionally separate from either benchmark runner: it
cannot accidentally time report parsing, and it makes the 128-world M1 hard
gate executable on the machine that collected the Warp report.  MJB bytes are
kept as local evidence but are never a cross-host gate; portable model arrays
and the exact trace construction config are.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.ctrl_trace import validate_m1_trace_metadata

REPORT_SCHEMA_VERSION = 1
SPEEDUP_MODE = "step+render"
HARD_GATE_WORLDS = 128
RECORD_ONLY_WORLDS = 256
MIN_PER_WORLD_CONTROL_HZ = 20.0
GO_NO_GO_SPEEDUP = 5.0
HARD_GATE_SPEEDUP = 10.0
WARMUP_STEPS = 100
MEASURE_STEPS = 1000
_PROFILE_PARITY_FIELDS = (
    "control_input",
    "control_substeps",
    "rgb_format",
    "cameras",
    "speedup_gate_mode",
    "partial_reset_participates_in_speedup_gate",
)
_PORTABLE_FIELDS = ("sha256", "field_count", "total_bytes", "physics_options")
_FIXED_PROFILE = {
    "control_input": "recorded_actuator_ctrl",
    "control_substeps": 25,
    "rgb_format": "top-left NHWC uint8",
    "cameras": [
        {
            "name": "agentview",
            "height": 128,
            "width": 128,
            "depth": False,
            "segmentation": None,
        },
        {
            "name": "robot0_eye_in_hand",
            "height": 128,
            "width": 128,
            "depth": False,
            "segmentation": None,
        },
        {
            "name": "sideview",
            "height": 128,
            "width": 128,
            "depth": False,
            "segmentation": None,
        },
    ],
    "speedup_gate_mode": SPEEDUP_MODE,
    "partial_reset_participates_in_speedup_gate": False,
}


def compare_m1_reports(
    official_report: Mapping[str, Any], warp_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate comparable reports and return the M1 performance-gate verdict.

    A false result is a real M1 rejection, not an exception: callers can still
    persist the JSON verdict for a go/no-go decision. Malformed or incomparable
    inputs raise ``ValueError`` / ``TypeError`` with an actionable diagnosis.
    """

    official = _validate_report(official_report, backend="official")
    warp = _validate_report(warp_report, backend="warp")
    _validate_same_trace_and_profile(official, warp)
    _validate_same_host(official, warp)

    official_case = _result_for(official, worlds=1, mode=SPEEDUP_MODE)
    warp_case = _result_for(warp, worlds=HARD_GATE_WORLDS, mode=SPEEDUP_MODE)
    _validate_result_throughput(
        official_case, expected_worlds=1, require_speedup_workload=True
    )
    _validate_result_throughput(
        warp_case, expected_worlds=HARD_GATE_WORLDS, require_speedup_workload=True
    )

    official_hz = _positive_float(
        official_case["aggregate_world_hz"], "official aggregate_world_hz"
    )
    warp_per_world_hz = _positive_float(warp_case["per_world_hz"], "warp per_world_hz")
    warp_aggregate_hz = _positive_float(
        warp_case["aggregate_world_hz"], "warp aggregate_world_hz"
    )
    aggregate_speedup = warp_aggregate_hz / official_hz
    per_world_passed = warp_per_world_hz >= MIN_PER_WORLD_CONTROL_HZ
    speedup_passed = aggregate_speedup >= HARD_GATE_SPEEDUP

    record_only_256 = _optional_result_for(
        warp, worlds=RECORD_ONLY_WORLDS, mode=SPEEDUP_MODE
    )
    if record_only_256 is not None:
        _validate_result_throughput(
            record_only_256,
            expected_worlds=RECORD_ONLY_WORLDS,
            require_speedup_workload=True,
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "speedup_mode": SPEEDUP_MODE,
        "trace": {
            "env_config": official["trace"]["metadata"]["env_config"],
            "portable_model_sha256": official["trace"]["metadata"][
                "model_arrays_sha256"
            ],
            "official_mjb_matches_trace": official["trace"]["fingerprint"][
                "mjb_matches_trace"
            ],
            "warp_mjb_matches_trace": warp["trace"]["fingerprint"]["mjb_matches_trace"],
        },
        "official_single_world_hz": official_hz,
        "warp_128_per_world_hz": warp_per_world_hz,
        "warp_128_aggregate_world_hz": warp_aggregate_hz,
        "aggregate_speedup": aggregate_speedup,
        "per_world_20hz_passed": per_world_passed,
        "aggregate_10x_passed": speedup_passed,
        "hard_gate_passed": per_world_passed and speedup_passed,
        "go_no_go_5x": "go" if aggregate_speedup >= GO_NO_GO_SPEEDUP else "no-go",
        "partial_reset_participates_in_speedup_gate": False,
        "record_only_256": record_only_256,
    }


def _validate_report(report: Mapping[str, Any], *, backend: str) -> Mapping[str, Any]:
    if not isinstance(report, Mapping):
        raise TypeError(f"{backend} report must be a JSON object.")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"{backend} report schema_version must be {REPORT_SCHEMA_VERSION!r}."
        )
    if report.get("backend") != backend:
        raise ValueError(
            f"Expected backend={backend!r}, got {report.get('backend')!r}."
        )
    if report.get("warmup_steps") != WARMUP_STEPS:
        raise ValueError(
            f"{backend} report warmup_steps must be {WARMUP_STEPS}, "
            f"got {report.get('warmup_steps')!r}."
        )
    if report.get("measure_steps") != MEASURE_STEPS:
        raise ValueError(
            f"{backend} report measure_steps must be {MEASURE_STEPS}, "
            f"got {report.get('measure_steps')!r}."
        )
    trace = report.get("trace")
    if not isinstance(trace, Mapping) or set(trace) != {"metadata", "fingerprint"}:
        raise ValueError(
            f"{backend} report.trace must contain exactly metadata and fingerprint."
        )
    metadata = trace["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{backend} report.trace.metadata must be a mapping.")
    validate_m1_trace_metadata(metadata)
    _validate_fingerprint(trace["fingerprint"], metadata, backend=backend)

    profile = report.get("profile")
    if not isinstance(profile, Mapping):
        raise TypeError(f"{backend} report.profile must be a mapping.")
    missing_profile = set(_PROFILE_PARITY_FIELDS) - set(profile)
    if missing_profile:
        raise ValueError(
            f"{backend} report.profile is missing fields: {sorted(missing_profile)}."
        )
    if profile["speedup_gate_mode"] != SPEEDUP_MODE:
        raise ValueError(
            f"{backend} report speedup_gate_mode must be {SPEEDUP_MODE!r}."
        )
    if profile["partial_reset_participates_in_speedup_gate"] is not False:
        raise ValueError(
            f"{backend} report must exclude partial reset from the speedup gate."
        )
    for name, expected in _FIXED_PROFILE.items():
        if profile[name] != expected:
            raise ValueError(
                f"{backend} report.profile.{name} must equal the fixed M1 profile."
            )
    runtime = report.get("metadata")
    if not isinstance(runtime, Mapping):
        raise TypeError(f"{backend} report.metadata must be a mapping.")
    for name in ("hostname", "platform", "mujoco", "robosuite"):
        value = runtime.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{backend} report.metadata.{name} must be a non-empty string."
            )
    for name in ("mujoco", "robosuite"):
        if runtime[name] != metadata[name]:
            raise ValueError(
                f"{backend} runtime {name} version does not match its trace version."
            )
    if not isinstance(report.get("results"), list):
        raise TypeError(f"{backend} report.results must be a list.")
    return report


def _validate_fingerprint(
    fingerprint: Any, metadata: Mapping[str, Any], *, backend: str
) -> None:
    expected = {"mjb_sha256", "mjb_size", "mjb_matches_trace", "portable"}
    if not isinstance(fingerprint, Mapping) or set(fingerprint) != expected:
        raise ValueError(
            f"{backend} report.trace.fingerprint must contain exactly "
            f"{sorted(expected)}."
        )
    if not isinstance(fingerprint["mjb_matches_trace"], bool):
        raise TypeError(f"{backend} report MJB match flag must be a bool.")
    portable = fingerprint["portable"]
    if not isinstance(portable, Mapping) or set(portable) != set(_PORTABLE_FIELDS):
        raise ValueError(
            f"{backend} report portable fingerprint must contain exactly "
            f"{list(_PORTABLE_FIELDS)}."
        )
    metadata_fields = (
        ("sha256", "model_arrays_sha256"),
        ("field_count", "model_array_field_count"),
        ("total_bytes", "model_array_total_bytes"),
        ("physics_options", "model_physics_options"),
    )
    for fingerprint_key, metadata_key in metadata_fields:
        if portable[fingerprint_key] != metadata[metadata_key]:
            raise ValueError(
                f"{backend} report portable fingerprint {fingerprint_key} does not "
                "match its trace metadata."
            )


def _validate_same_trace_and_profile(
    official: Mapping[str, Any], warp: Mapping[str, Any]
) -> None:
    official_metadata = official["trace"]["metadata"]
    warp_metadata = warp["trace"]["metadata"]
    trace_fields = (
        "trace_format_version",
        "model_reset_protocol",
        "suite",
        "task_index",
        "task_name",
        "seed",
        "mujoco",
        "robosuite",
        "control_freq",
        "control_decimation",
        "env_config",
        "model_arrays_sha256",
        "model_array_field_count",
        "model_array_total_bytes",
        "model_physics_options",
    )
    for name in trace_fields:
        if official_metadata[name] != warp_metadata[name]:
            raise ValueError(f"Official and Warp reports use different trace {name}.")
    for name in _PROFILE_PARITY_FIELDS:
        if official["profile"][name] != warp["profile"][name]:
            raise ValueError(f"Official and Warp reports use different profile {name}.")


def _validate_same_host(official: Mapping[str, Any], warp: Mapping[str, Any]) -> None:
    """Reject cross-host CPU/GPU ratios; M1 speedup is a same-host comparison."""

    for name in ("hostname", "platform", "mujoco", "robosuite"):
        if official["metadata"][name] != warp["metadata"][name]:
            raise ValueError(
                "Official and Warp performance reports must come from the same "
                f"host and platform; metadata.{name} differs."
            )


def _result_for(
    report: Mapping[str, Any], *, worlds: int, mode: str
) -> Mapping[str, Any]:
    matches = [
        result
        for result in report["results"]
        if isinstance(result, Mapping)
        and result.get("worlds") == worlds
        and result.get("mode") == mode
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one worlds={worlds}, mode={mode!r} result; "
            f"found {len(matches)}."
        )
    return matches[0]


def _optional_result_for(
    report: Mapping[str, Any], *, worlds: int, mode: str
) -> Mapping[str, Any] | None:
    matches = [
        result
        for result in report["results"]
        if isinstance(result, Mapping)
        and result.get("worlds") == worlds
        and result.get("mode") == mode
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Found duplicate record-only worlds={worlds}, mode={mode!r} results."
        )
    return matches[0] if matches else None


def _validate_result_throughput(
    result: Mapping[str, Any], *, expected_worlds: int, require_speedup_workload: bool
) -> None:
    if result.get("worlds") != expected_worlds:
        raise ValueError(f"Benchmark result worlds must be {expected_worlds}.")
    if result.get("warmup_steps") != WARMUP_STEPS:
        raise ValueError(
            f"Benchmark result warmup_steps must be {WARMUP_STEPS}, "
            f"got {result.get('warmup_steps')!r}."
        )
    if result.get("measure_steps") != MEASURE_STEPS:
        raise ValueError(
            f"Benchmark result measure_steps must be {MEASURE_STEPS}, "
            f"got {result.get('measure_steps')!r}."
        )
    worlds_affected = result.get("worlds_affected")
    if isinstance(worlds_affected, bool) or worlds_affected != expected_worlds:
        raise ValueError(
            "Speedup-gate result must affect every world; "
            f"expected {expected_worlds}, got {worlds_affected!r}."
        )
    elapsed_seconds = _positive_float(result.get("elapsed_seconds"), "elapsed_seconds")
    batch_hz = _positive_float(result.get("batch_hz"), "batch_hz")
    per_world_hz = _positive_float(result.get("per_world_hz"), "per_world_hz")
    aggregate_world_hz = _positive_float(
        result.get("aggregate_world_hz"), "aggregate_world_hz"
    )
    expected_batch_hz = MEASURE_STEPS / elapsed_seconds
    # ``batch_hz`` is a measured quantity, not a caller-provided claim.  A
    # JSON report may round its elapsed duration slightly, hence a small
    # relative tolerance, but it may not substitute a faster rate.
    if not math.isclose(batch_hz, expected_batch_hz, rel_tol=1e-6, abs_tol=0.0):
        raise ValueError(
            "Benchmark batch_hz must equal measure_steps / elapsed_seconds."
        )
    if not math.isclose(batch_hz, per_world_hz, rel_tol=1e-9, abs_tol=0.0):
        raise ValueError("Benchmark per_world_hz must equal batch_hz.")
    if not math.isclose(
        aggregate_world_hz,
        batch_hz * expected_worlds,
        rel_tol=1e-9,
        abs_tol=0.0,
    ):
        raise ValueError(
            "Benchmark aggregate_world_hz must equal batch_hz * worlds_affected."
        )
    if require_speedup_workload:
        expected = {
            "physics_substeps_per_operation": 25,
            "rendered_cameras_per_operation": 3,
            "participates_in_speedup_gate": True,
        }
        for name, value in expected.items():
            if result.get(name) != value:
                raise ValueError(
                    f"Speedup-gate result {name} must be {value!r}, "
                    f"got {result.get(name)!r}."
                )
        if result.get("per_world_control_hz") != per_world_hz:
            raise ValueError(
                "Speedup-gate result per_world_control_hz must equal per_world_hz."
            )


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("official_report", type=Path)
    parser.add_argument("warp_report", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    official = json.loads(args.official_report.read_text(encoding="utf-8"))
    warp = json.loads(args.warp_report.read_text(encoding="utf-8"))
    verdict = compare_m1_reports(official, warp)
    serialized = json.dumps(verdict, indent=2, sort_keys=True)
    if args.output is None:
        print(serialized)
    else:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if not verdict["hard_gate_passed"]:
        raise SystemExit("M1 128-world performance hard gate failed.")


if __name__ == "__main__":
    main()
