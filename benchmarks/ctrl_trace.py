"""Versioned, pickle-free exchange format for official control traces.

Use this format to move an official CPU trace to a separate GPU machine for
MJWarp parity checks.  It deliberately stores only numeric NPZ arrays and a
JSON metadata string; no Python objects are serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TRACE_FORMAT_VERSION = 2
TRACE_MODEL_RESET_PROTOCOL = "official-env-explicit-reset-once-v1"
_PATH_OR_PLATFORM_DERIVED_FIELDS = frozenset({"mesh_normal", "mesh_polynormal"})
_FLOAT64_DECIMALS = 7
_FLOAT32_DECIMALS = 5
_M1_ENV_CONFIG_KEYS = frozenset(
    {"suite", "task_index", "cameras", "horizon", "control_freq", "seed"}
)
_M1_REQUIRED_METADATA_FIELDS = frozenset(
    {
        "trace_format_version",
        "model_reset_protocol",
        "env_config",
        "suite",
        "task_index",
        "task_name",
        "seed",
        "mujoco",
        "robosuite",
        "timestep",
        "control_freq",
        "control_decimation",
        "model_mjb_sha256",
        "model_mjb_size",
        "model_arrays_sha256",
        "model_array_field_count",
        "model_array_total_bytes",
        "model_physics_options",
        "model",
    }
)
_M1_MODEL_KEYS = frozenset({"nq", "nv", "nu", "nbody", "nsite"})
_ARRAY_FIELDS = (
    "initial_fullphysics",
    "actions",
    "substep_ctrl",
    "qpos",
    "qvel",
    "body_xpos",
    "body_xquat",
    "site_xpos",
    "site_xmat",
)


def validate_m1_trace_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate the self-describing metadata required for M1 GPU replay.

    The MJB is local exact evidence; portable model arrays are the stable
    cross-host gate.  Storing the construction config makes it possible to
    rebuild the exact model that produced an exported control trace.
    """

    missing = _M1_REQUIRED_METADATA_FIELDS - set(metadata)
    if missing:
        raise ValueError(f"M1 trace metadata is missing fields: {sorted(missing)}.")
    if metadata["trace_format_version"] != TRACE_FORMAT_VERSION:
        raise ValueError(
            "M1 trace metadata.trace_format_version must be "
            f"{TRACE_FORMAT_VERSION}, got {metadata['trace_format_version']!r}."
        )
    if metadata["model_reset_protocol"] != TRACE_MODEL_RESET_PROTOCOL:
        raise ValueError(
            f"Unsupported M1 model reset protocol {metadata['model_reset_protocol']!r}."
        )
    _require_nonempty_string(metadata["suite"], "M1 trace metadata.suite")
    _require_nonempty_string(metadata["task_name"], "M1 trace metadata.task_name")
    _require_nonnegative_int(metadata["task_index"], "M1 trace metadata.task_index")
    _require_seed(metadata["seed"], "M1 trace metadata.seed")
    _require_nonempty_string(metadata["mujoco"], "M1 trace metadata.mujoco")
    _require_nonempty_string(metadata["robosuite"], "M1 trace metadata.robosuite")
    timestep = _require_positive_float(
        metadata["timestep"], "M1 trace metadata.timestep"
    )
    control_freq = _require_positive_int(
        metadata["control_freq"], "M1 trace metadata.control_freq"
    )
    control_decimation = _require_positive_int(
        metadata["control_decimation"], "M1 trace metadata.control_decimation"
    )
    if control_decimation != 25:
        raise ValueError(
            f"M1 trace control_decimation must be exactly 25, got {control_decimation}."
        )
    if not math.isclose(
        timestep * control_freq * control_decimation, 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            "M1 trace timestep, control_freq, and control_decimation must describe "
            "one control second: timestep * control_freq * control_decimation == 1."
        )
    _require_sha256(metadata["model_mjb_sha256"], "M1 trace metadata.model_mjb_sha256")
    _require_positive_int(
        metadata["model_mjb_size"], "M1 trace metadata.model_mjb_size"
    )
    _require_sha256(
        metadata["model_arrays_sha256"], "M1 trace metadata.model_arrays_sha256"
    )
    _require_nonnegative_int(
        metadata["model_array_field_count"],
        "M1 trace metadata.model_array_field_count",
    )
    _require_nonnegative_int(
        metadata["model_array_total_bytes"],
        "M1 trace metadata.model_array_total_bytes",
    )
    if not isinstance(metadata["model_physics_options"], Mapping):
        raise TypeError("M1 trace metadata.model_physics_options must be a mapping.")
    model = metadata["model"]
    if not isinstance(model, Mapping) or set(model) != _M1_MODEL_KEYS:
        raise ValueError(
            f"M1 trace metadata.model must contain exactly {sorted(_M1_MODEL_KEYS)}."
        )
    for name, value in model.items():
        _require_nonnegative_int(value, f"M1 trace metadata.model.{name}")

    config = metadata["env_config"]
    if not isinstance(config, Mapping) or set(config) != _M1_ENV_CONFIG_KEYS:
        raise ValueError(
            f"M1 trace env_config must contain exactly {sorted(_M1_ENV_CONFIG_KEYS)}."
        )
    if config["suite"] != metadata["suite"]:
        raise ValueError("M1 trace env_config.suite does not match metadata.suite.")
    if config["task_index"] != metadata["task_index"]:
        raise ValueError(
            "M1 trace env_config.task_index does not match metadata.task_index."
        )
    if config["seed"] != metadata["seed"]:
        raise ValueError("M1 trace env_config.seed does not match metadata.seed.")
    if config["control_freq"] != metadata["control_freq"]:
        raise ValueError(
            "M1 trace env_config.control_freq does not match metadata.control_freq."
        )
    _require_nonempty_string(config["suite"], "M1 trace env_config.suite")
    _require_nonnegative_int(config["task_index"], "M1 trace env_config.task_index")
    _require_positive_int(config["horizon"], "M1 trace env_config.horizon")
    _require_positive_int(config["control_freq"], "M1 trace env_config.control_freq")
    _require_seed(config["seed"], "M1 trace env_config.seed")
    if not isinstance(config["cameras"], list) or not config["cameras"]:
        raise ValueError(
            "M1 trace env_config.cameras must be a non-empty ordered list."
        )
    expected_camera_keys = {"name", "height", "width", "depth", "segmentation"}
    camera_names: set[str] = set()
    for index, camera in enumerate(config["cameras"]):
        if not isinstance(camera, Mapping) or set(camera) != expected_camera_keys:
            raise ValueError(
                "Each M1 trace camera must contain exactly "
                f"{sorted(expected_camera_keys)}."
            )
        name = camera["name"]
        _require_nonempty_string(name, f"M1 trace env_config.cameras[{index}].name")
        if name in camera_names:
            raise ValueError(f"M1 trace env_config has duplicate camera {name!r}.")
        camera_names.add(name)
        _require_positive_int(
            camera["height"], f"M1 trace env_config.cameras[{index}].height"
        )
        _require_positive_int(
            camera["width"], f"M1 trace env_config.cameras[{index}].width"
        )
        if not isinstance(camera["depth"], bool):
            raise TypeError(
                f"M1 trace env_config.cameras[{index}].depth must be a bool."
            )
        if camera["segmentation"] not in {None, "instance", "class", "element"}:
            raise ValueError(
                "M1 trace env_config.cameras["
                f"{index}].segmentation is invalid: {camera['segmentation']!r}."
            )


def _require_nonempty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string.")


def _require_nonnegative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer.")


def _require_positive_int(value: Any, name: str) -> int:
    _require_nonnegative_int(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _require_seed(value: Any, name: str) -> None:
    if value is not None:
        _require_nonnegative_int(value, name)


def _require_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return result


def _require_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise TypeError(f"{name} must be a lowercase 64-character SHA256 hex string.")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA256 hex string.")


def model_mjb_fingerprint(model: Any) -> tuple[str, int]:
    """Return the SHA256 and byte size of a MuJoCo MJB saved only in memory."""

    import mujoco

    size = mujoco.mj_sizeModel(model)
    buffer = np.empty(size, dtype=np.uint8)
    mujoco.mj_saveModel(model, filename=None, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest(), size


def _json_value(value: Any) -> Any:
    """Convert scalar MuJoCo option values into canonical JSON primitives."""

    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def model_arrays_fingerprint(model: Any) -> dict[str, Any]:
    """Hash portable model fields and canonical physics options.

    MJB bytes are ideal local evidence but can contain host-specific details.
    This v2 digest is intended for cross-host parity triage. It excludes
    filesystem-path offsets and platform-derived BVH / mesh-normal buffers,
    then streams the remaining public numeric / byte arrays in sorted field-name
    order. Floats are rounded at a type-specific stable precision and encoded
    as little-endian float64; integers and byte arrays are little-endian,
    contiguous bytes. Solver-relevant ``model.opt`` values are added as
    canonical JSON.
    """

    hasher = hashlib.sha256()
    field_count = 0
    total_bytes = 0
    for name in sorted(name for name in dir(model) if not name.startswith("_")):
        if _exclude_portability_derived_field(name):
            continue
        try:
            value = getattr(model, name)
        except (AttributeError, TypeError, ValueError):
            continue
        if callable(value):
            continue
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if array.ndim == 0 or array.dtype.hasobject:
            continue
        if array.dtype.kind not in "biufcSUV":
            continue
        contiguous = _canonical_portable_array(array)
        descriptor = json.dumps(
            {"dtype": contiguous.dtype.str, "name": name, "shape": contiguous.shape},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        hasher.update(len(descriptor).to_bytes(8, "little"))
        hasher.update(descriptor)
        if contiguous.nbytes:
            hasher.update(memoryview(contiguous).cast("B"))
        field_count += 1
        total_bytes += contiguous.nbytes

    option_names = (
        "timestep",
        "integrator",
        "solver",
        "cone",
        "jacobian",
        "iterations",
        "tolerance",
        "ls_iterations",
        "ls_tolerance",
        "noslip_iterations",
        "noslip_tolerance",
        "mpr_iterations",
        "mpr_tolerance",
        "gravity",
        "wind",
        "magnetic",
        "density",
        "viscosity",
        "impratio",
    )
    physics_options = {
        name: _json_value(getattr(model.opt, name))
        for name in option_names
        if hasattr(model.opt, name)
    }
    canonical_options = json.dumps(
        physics_options, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    hasher.update(len(canonical_options).to_bytes(8, "little"))
    hasher.update(canonical_options)
    return {
        "sha256": hasher.hexdigest(),
        "field_count": field_count,
        "total_bytes": total_bytes,
        "physics_options": physics_options,
    }


def _exclude_portability_derived_field(name: str) -> bool:
    """Return whether a public model field is host/path-derived metadata."""

    return (
        name.startswith("bvh_")
        or name.endswith("_pathadr")
        or name in _PATH_OR_PLATFORM_DERIVED_FIELDS
    )


def _canonical_portable_array(array: np.ndarray) -> np.ndarray:
    """Return host-independent bytes for one model-array fingerprint field."""

    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.kind == "f":
        if contiguous.dtype.itemsize == np.dtype(np.float64).itemsize:
            decimals = _FLOAT64_DECIMALS
        elif contiguous.dtype.itemsize == np.dtype(np.float32).itemsize:
            decimals = _FLOAT32_DECIMALS
        else:
            raise TypeError(
                "Portable model fingerprints support only float32 and float64 arrays; "
                f"got {contiguous.dtype}."
            )
        quantum = 10.0**-decimals
        normalized = np.round(contiguous.astype(np.float64), decimals=decimals)
        # Assignment canonicalizes both negative zero and values below one-half
        # of the source precision quantum to positive zero.
        normalized[np.abs(normalized) < quantum / 2.0] = 0.0
        return np.ascontiguousarray(normalized.astype("<f8", copy=False))
    if contiguous.dtype.kind in "biu":
        return np.ascontiguousarray(
            contiguous.astype(contiguous.dtype.newbyteorder("<"))
        )
    if contiguous.dtype.kind in "SUV":
        return np.ascontiguousarray(
            contiguous.astype(contiguous.dtype.newbyteorder("<"))
        )
    # The public-model filter admits complex arrays for backwards compatibility,
    # even though MuJoCo currently exposes none. Keep their byte order stable.
    if contiguous.dtype.kind == "c":
        return np.ascontiguousarray(
            contiguous.astype(contiguous.dtype.newbyteorder("<"))
        )
    raise TypeError(f"Unsupported portable model array dtype {contiguous.dtype}.")


@dataclass(frozen=True, slots=True)
class CtrlTraceArtifact:
    """Numeric official trace arrays plus JSON-safe versioned metadata."""

    initial_fullphysics: np.ndarray
    actions: np.ndarray
    substep_ctrl: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    body_xpos: np.ndarray
    body_xquat: np.ndarray
    site_xpos: np.ndarray
    site_xmat: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")
        validate_m1_trace_metadata(self.metadata)
        json.dumps(self.metadata, sort_keys=True)

        values = {name: getattr(self, name) for name in _ARRAY_FIELDS}
        for name, value in values.items():
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray.")
            if value.dtype.kind not in "fiu":
                raise TypeError(f"{name} must use a real numeric dtype.")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values.")

        model = self.metadata["model"]
        steps = len(self.actions)
        if self.initial_fullphysics.ndim != 1:
            raise ValueError("initial_fullphysics must be flattened [D].")
        # The fixed M1 LIBERO pilot has no activation state (na == 0), so its
        # FULLPHYSICS layout is time + qpos + qvel.  Preserve trace format v1
        # while binding the state vector to the recorded model dimensions.
        expected_fullphysics = 1 + model["nq"] + model["nv"]
        if self.initial_fullphysics.shape != (expected_fullphysics,):
            raise ValueError(
                "initial_fullphysics must match the M1 model dimensions: "
                f"expected [{expected_fullphysics}], got "
                f"{list(self.initial_fullphysics.shape)}."
            )
        if self.actions.ndim != 2 or self.actions.shape[1] != 7:
            raise ValueError("actions must have shape [T, 7].")
        if steps == 0:
            raise ValueError("M1 ctrl traces must contain at least one action.")
        expected_shapes = {
            "substep_ctrl": (steps, 25, model["nu"]),
            "qpos": (steps + 1, model["nq"]),
            "qvel": (steps + 1, model["nv"]),
            "body_xpos": (steps + 1, model["nbody"], 3),
            "body_xquat": (steps + 1, model["nbody"], 4),
            "site_xpos": (steps + 1, model["nsite"], 3),
            "site_xmat": (steps + 1, model["nsite"], 9),
        }
        for name, expected_shape in expected_shapes.items():
            if values[name].shape != expected_shape:
                raise ValueError(
                    f"{name} must match the M1 model dimensions: expected "
                    f"{list(expected_shape)}, got {list(values[name].shape)}."
                )


def make_artifact(
    *,
    initial_fullphysics: np.ndarray,
    actions: np.ndarray,
    substep_ctrl: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    body_xpos: np.ndarray,
    body_xquat: np.ndarray,
    site_xpos: np.ndarray,
    site_xmat: np.ndarray,
    metadata: Mapping[str, Any],
) -> CtrlTraceArtifact:
    """Construct a validated artifact and insert the format version."""

    versioned_metadata = {"trace_format_version": TRACE_FORMAT_VERSION, **metadata}
    return CtrlTraceArtifact(
        initial_fullphysics=np.asarray(initial_fullphysics),
        actions=np.asarray(actions),
        substep_ctrl=np.asarray(substep_ctrl),
        qpos=np.asarray(qpos),
        qvel=np.asarray(qvel),
        body_xpos=np.asarray(body_xpos),
        body_xquat=np.asarray(body_xquat),
        site_xpos=np.asarray(site_xpos),
        site_xmat=np.asarray(site_xmat),
        metadata=versioned_metadata,
    )


def save_trace(path: Path, artifact: CtrlTraceArtifact) -> None:
    """Write a compressed numeric NPZ trace without pickled Python objects."""

    path = Path(path)
    np.savez_compressed(
        path,
        **{name: getattr(artifact, name) for name in _ARRAY_FIELDS},
        metadata_json=np.asarray(json.dumps(artifact.metadata, sort_keys=True)),
    )


def load_trace(path: Path) -> CtrlTraceArtifact:
    """Load and validate a trace while explicitly disallowing pickle payloads."""

    with np.load(Path(path), allow_pickle=False) as archive:
        missing = (set(_ARRAY_FIELDS) | {"metadata_json"}) - set(archive.files)
        if missing:
            raise ValueError(f"Trace is missing fields: {sorted(missing)}.")
        metadata = json.loads(str(archive["metadata_json"].item()))
        return CtrlTraceArtifact(
            **{name: archive[name].copy() for name in _ARRAY_FIELDS}, metadata=metadata
        )


def _load_numeric_array(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{path} must contain one .npy numeric array.")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser(
        "pack", help="Pack .npy arrays and metadata JSON into a trace NPZ."
    )
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--metadata", type=Path, required=True)
    for name in _ARRAY_FIELDS:
        pack.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    inspect = commands.add_parser(
        "inspect", help="Load a trace and print JSON-safe summary metadata."
    )
    inspect.add_argument("trace", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "pack":
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        artifact = make_artifact(
            **{
                name: _load_numeric_array(getattr(args, name)) for name in _ARRAY_FIELDS
            },
            metadata=metadata,
        )
        save_trace(args.output, artifact)
        return
    artifact = load_trace(args.trace)
    print(
        json.dumps(
            {
                "metadata": artifact.metadata,
                "steps": int(artifact.actions.shape[0]),
                "control_substeps": int(artifact.substep_ctrl.shape[1]),
                "state_size": int(artifact.initial_fullphysics.size),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
