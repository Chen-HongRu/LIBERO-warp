"""Official-task compilation and init-state utilities for batched runtimes.

This module deliberately stops at model compilation and state-bank preparation.
It does not provide a renderer, physics stepping, or a controller for Warp.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

import mujoco
import numpy as np
import robosuite
import torch

from .official import OfficialBatchEnv
from .types import EnvConfig

# LIBERO demonstration HDF5 files preserve the model XML verbatim.  Historical
# dataset builders embedded these two absolute source roots in ``file``
# attributes.  Remap only those exact roots: replacing arbitrary absolute paths
# would hide an incomplete artifact instead of making it reproducible.
_HISTORICAL_LIBERO_ASSET_ROOT = "/home/yifengz/workspace/libero-dev/chiliocosm/assets"
_HISTORICAL_ROBOSUITE_ASSET_ROOT = (
    "/home/yifengz/workspace/robosuite-master/robosuite/models/assets"
)


def _referenced_asset_suffixes(model_xml: str, historical_root: str) -> set[str]:
    pattern = re.compile(re.escape(historical_root) + r'/([^"\']+)')
    return set(pattern.findall(model_xml))


def _select_complete_asset_root(
    model_xml: str,
    historical_root: str,
    candidates: list[Path],
    *,
    label: str,
) -> Path | None:
    suffixes = _referenced_asset_suffixes(model_xml, historical_root)
    if not suffixes:
        return None
    unique_candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    for candidate in unique_candidates:
        if all((candidate / suffix).is_file() for suffix in suffixes):
            return candidate
    missing = {
        str(candidate): sorted(
            suffix for suffix in suffixes if not (candidate / suffix).is_file()
        )[:3]
        for candidate in unique_candidates
    }
    raise FileNotFoundError(
        f"No complete {label} asset root could load the recorded model XML; "
        f"checked {missing}."
    )


def remap_demo_model_xml_assets(
    model_xml: str,
    *,
    libero_asset_root: str | Path | None = None,
    robosuite_asset_root: str | Path | None = None,
) -> str:
    """Rewrite known historical demo asset roots to this installation's roots.

    The XML itself remains the source of static scene placement.  This helper
    changes just the two old asset-directory prefixes needed to load that XML
    on a different host; unknown absolute paths are intentionally left intact
    so MuJoCo reports a useful missing-asset error.
    """
    if not isinstance(model_xml, str) or not model_xml.strip():
        raise ValueError("model_xml must be a non-empty string.")
    libero_candidates = [
        Path(libero_asset_root)
        if libero_asset_root is not None
        else Path(__file__).resolve().parents[1] / "assets"
    ]
    if robosuite_asset_root is not None:
        robosuite_candidates = [Path(robosuite_asset_root)]
    else:
        robosuite_candidates = [
            Path(robosuite.__file__).resolve().parent / "models" / "assets",
            *(
                Path(entry) / "robosuite" / "models" / "assets"
                for entry in sys.path
                if entry
            ),
        ]
    libero_assets = _select_complete_asset_root(
        model_xml,
        _HISTORICAL_LIBERO_ASSET_ROOT,
        libero_candidates,
        label="LIBERO",
    )
    robosuite_assets = _select_complete_asset_root(
        model_xml,
        _HISTORICAL_ROBOSUITE_ASSET_ROOT,
        robosuite_candidates,
        label="robosuite",
    )
    remapped = model_xml
    if libero_assets is not None:
        remapped = remapped.replace(_HISTORICAL_LIBERO_ASSET_ROOT, str(libero_assets))
    if robosuite_assets is not None:
        remapped = remapped.replace(
            _HISTORICAL_ROBOSUITE_ASSET_ROOT, str(robosuite_assets)
        )
    return remapped


def _port_legacy_single_panda_xml(model_xml: str) -> str:
    """Port robosuite 1.4 single-arm names to the 1.5 Panda contract.

    The published demos predate composite-controller arm qualifiers. Dynamic
    state sizes are unchanged; this migration changes identifiers only and
    adds the reference-only center site expected by robosuite 1.5.
    """
    if 'name="gripper0_eef"' not in model_xml:
        return model_xml
    root = ET.fromstring(model_xml)
    for element in root.iter():
        for key, value in tuple(element.attrib.items()):
            value = value.replace("mount0_", "fixed_mount0_")
            value = value.replace("gripper0_", "gripper0_right_")
            element.set(key, value)
    if root.find(".//site[@name='robot0_right_center']") is None:
        link0 = root.find(".//body[@name='robot0_link0']")
        if link0 is None:
            raise ValueError("Legacy Panda exact model is missing body 'robot0_link0'.")
        ET.SubElement(
            link0,
            "site",
            {
                "name": "robot0_right_center",
                "pos": "0 0 0",
                "size": "0.01",
                "group": "2",
                "rgba": "1 0.3 0.3 -1",
            },
        )
    return ET.tostring(root, encoding="unicode")


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _validate_sha256(value: str, *, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{name} must be a 64-character SHA-256 hex digest."
        ) from error


def _freeze_id_mapping(values: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType(dict(values))


def _validate_tensor(
    name: str,
    value: torch.Tensor,
    *,
    ndim: int,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.shape}.")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}.")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}.")


def _validate_finite(name: str, value: torch.Tensor) -> None:
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have a floating-point dtype.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")


def _validate_indices(
    name: str,
    indices: torch.Tensor,
    *,
    upper_bound: int,
    device: torch.device | None = None,
) -> None:
    _validate_tensor(name, indices, ndim=1, dtype=torch.int64, device=device)
    if indices.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if torch.any(indices < 0) or torch.any(indices >= upper_bound):
        raise IndexError(
            f"{name} must be in [0, {upper_bound}), got {indices.tolist()}."
        )


@dataclass(frozen=True, slots=True)
class TaskRuntimeMetadata:
    """Immutable task/model metadata derived from the official BDDL environment.

    ``model`` is the exact ``mujoco.MjModel`` used by ``official_env``. It is a
    mutable MuJoCo object by nature, but the metadata binding and every mapping
    stored here are immutable.
    """

    suite: str
    task_index: int
    task_name: str
    bddl_file: str
    target_backend: str
    target_num_worlds: int
    model: mujoco.MjModel
    mujoco_version: str
    robosuite_version: str
    timestep: float
    control_frequency: int
    control_substeps: int
    nq: int
    nv: int
    nu: int
    na: int
    fullphysics_state_size: int
    source_model_xml_sha256: str | None
    remapped_model_xml_sha256: str | None
    body_ids: Mapping[str, int]
    site_ids: Mapping[str, int]
    geom_ids: Mapping[str, int]
    joint_ids: Mapping[str, int]
    camera_ids: Mapping[str, int]
    actuator_ids: Mapping[str, int]
    requested_camera_ids: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.target_num_worlds <= 0:
            raise ValueError("target_num_worlds must be positive.")
        if self.timestep <= 0 or self.control_frequency <= 0:
            raise ValueError("timestep and control_frequency must be positive.")
        if self.control_substeps <= 0:
            raise ValueError("control_substeps must be positive.")
        for name in ("source_model_xml_sha256", "remapped_model_xml_sha256"):
            value = getattr(self, name)
            if value is not None:
                _validate_sha256(value, name=name)
        for name in (
            "body_ids",
            "site_ids",
            "geom_ids",
            "joint_ids",
            "camera_ids",
            "actuator_ids",
            "requested_camera_ids",
        ):
            object.__setattr__(self, name, _freeze_id_mapping(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class PhysicsStateBatch:
    """Explicit MuJoCo full-physics state fields for one or more worlds."""

    fullphysics: torch.Tensor
    time: torch.Tensor
    qpos: torch.Tensor
    qvel: torch.Tensor
    act: torch.Tensor

    def __post_init__(self) -> None:
        _validate_tensor("fullphysics", self.fullphysics, ndim=2)
        _validate_tensor("time", self.time, ndim=1, device=self.fullphysics.device)
        for name in ("qpos", "qvel", "act"):
            _validate_tensor(
                name, getattr(self, name), ndim=2, device=self.fullphysics.device
            )
        tensors = (self.fullphysics, self.time, self.qpos, self.qvel, self.act)
        if any(not torch.is_floating_point(value) for value in tensors):
            raise TypeError("PhysicsStateBatch tensors must use floating-point dtypes.")
        batch_size = self.fullphysics.shape[0]
        if any(value.shape[0] != batch_size for value in tensors[1:]):
            raise ValueError("PhysicsStateBatch tensors must share a batch dimension.")
        for name, value in zip(("fullphysics", "time", "qpos", "qvel", "act"), tensors):
            _validate_finite(name, value)


@dataclass(frozen=True, slots=True)
class InitStateBank:
    """Trusted local init states decoded with official MuJoCo state APIs."""

    legacy_flattened: torch.Tensor
    fullphysics: torch.Tensor
    time: torch.Tensor
    qpos: torch.Tensor
    qvel: torch.Tensor
    act: torch.Tensor

    def __post_init__(self) -> None:
        fields = (
            ("legacy_flattened", self.legacy_flattened, 2),
            ("fullphysics", self.fullphysics, 2),
            ("time", self.time, 1),
            ("qpos", self.qpos, 2),
            ("qvel", self.qvel, 2),
            ("act", self.act, 2),
        )
        batch_size: int | None = None
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        for name, value, ndim in fields:
            _validate_tensor(name, value, ndim=ndim)
            _validate_finite(name, value)
            device = value.device if device is None else device
            dtype = value.dtype if dtype is None else dtype
            if value.device != device or value.dtype != dtype:
                raise ValueError(
                    "InitStateBank tensors must share one device and dtype."
                )
            batch_size = value.shape[0] if batch_size is None else batch_size
            if value.shape[0] != batch_size:
                raise ValueError("InitStateBank tensors must share a batch dimension.")
        if self.fullphysics.shape != self.legacy_flattened.shape:
            raise ValueError(
                "Legacy and MuJoCo FULLPHYSICS init-state shapes must match exactly."
            )

    @property
    def size(self) -> int:
        """Number of trusted initialization states."""
        return self.fullphysics.shape[0]

    @property
    def device(self) -> torch.device:
        """Device holding this bank; compiler output is canonical CPU float64."""
        return self.fullphysics.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype shared by every state tensor."""
        return self.fullphysics.dtype

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> InitStateBank:
        """Explicitly upload/cast this bank once for later device-local gathers."""
        target_device = torch.device(device)
        target_dtype = self.dtype if dtype is None else dtype
        if not torch.empty((), dtype=target_dtype).is_floating_point():
            raise TypeError("InitStateBank dtype must be floating point.")

        def transfer(value: torch.Tensor) -> torch.Tensor:
            return value.to(device=target_device, dtype=target_dtype)

        return InitStateBank(
            legacy_flattened=transfer(self.legacy_flattened),
            fullphysics=transfer(self.fullphysics),
            time=transfer(self.time),
            qpos=transfer(self.qpos),
            qvel=transfer(self.qvel),
            act=transfer(self.act),
        )


def gather_init_states(
    bank: InitStateBank,
    state_indices: torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> PhysicsStateBatch:
    """Gather selected states from a bank already uploaded to the target device."""
    if not isinstance(bank, InitStateBank):
        raise TypeError("bank must be an InitStateBank.")
    target_device = torch.device(device) if device is not None else state_indices.device
    if bank.device != target_device:
        raise ValueError(
            f"InitStateBank is on {bank.device}, but gather target is {target_device}. "
            "Upload once with bank.to(target_device, dtype=...) before gathering."
        )
    _validate_indices(
        "state_indices",
        state_indices,
        upper_bound=bank.size,
        device=target_device,
    )

    def gather(value: torch.Tensor) -> torch.Tensor:
        return value.index_select(0, state_indices)

    return PhysicsStateBatch(
        fullphysics=gather(bank.fullphysics),
        time=gather(bank.time),
        qpos=gather(bank.qpos),
        qvel=gather(bank.qvel),
        act=gather(bank.act),
    )


def scatter_init_states(
    fullphysics_target: torch.Tensor,
    bank: InitStateBank,
    world_ids: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    allow_duplicate_world_ids: bool = False,
) -> torch.Tensor:
    """Scatter trusted states into selected worlds in-place and return the view.

    Duplicate state indices are intentionally allowed: multiple worlds may reset
    to the same initial state. Duplicate world IDs are rejected by default so a
    partial reset has a deterministic one-state-per-world meaning.
    """
    _validate_tensor("fullphysics_target", fullphysics_target, ndim=2)
    if not torch.is_floating_point(fullphysics_target):
        raise TypeError("fullphysics_target must use a floating-point dtype.")
    if fullphysics_target.shape[1] != bank.fullphysics.shape[1]:
        raise ValueError(
            "fullphysics_target has incompatible state width "
            f"{fullphysics_target.shape[1]}; expected {bank.fullphysics.shape[1]}."
        )
    _validate_indices(
        "world_ids",
        world_ids,
        upper_bound=fullphysics_target.shape[0],
        device=fullphysics_target.device,
    )
    _validate_indices(
        "state_indices",
        state_indices,
        upper_bound=bank.size,
        device=fullphysics_target.device,
    )
    if world_ids.numel() != state_indices.numel():
        raise ValueError("world_ids and state_indices must have equal lengths.")
    if (
        not allow_duplicate_world_ids
        and torch.unique(world_ids).numel() != world_ids.numel()
    ):
        raise ValueError("Duplicate world_ids are not allowed for deterministic reset.")
    if bank.device != fullphysics_target.device:
        raise ValueError(
            f"InitStateBank is on {bank.device}, but target is on "
            f"{fullphysics_target.device}. Upload once with bank.to(target.device, "
            "dtype=...) before scattering."
        )
    if bank.dtype != fullphysics_target.dtype:
        raise ValueError(
            f"InitStateBank uses {bank.dtype}, but target uses "
            f"{fullphysics_target.dtype}. Cast once with bank.to(target.device, "
            "dtype=target.dtype) before scattering."
        )
    gathered = gather_init_states(bank, state_indices, device=fullphysics_target.device)
    fullphysics_target.index_copy_(0, world_ids, gathered.fullphysics)
    return fullphysics_target


def reset_all_worlds(
    fullphysics_target: torch.Tensor,
    bank: InitStateBank,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """Reset every world in a target view from one state index per world."""
    _validate_tensor("fullphysics_target", fullphysics_target, ndim=2)
    _validate_indices(
        "state_indices",
        state_indices,
        upper_bound=bank.size,
        device=fullphysics_target.device,
    )
    if state_indices.numel() != fullphysics_target.shape[0]:
        raise ValueError("Full reset requires one state index for every world.")
    world_ids = torch.arange(
        fullphysics_target.shape[0], dtype=torch.int64, device=fullphysics_target.device
    )
    return scatter_init_states(fullphysics_target, bank, world_ids, state_indices)


@dataclass(frozen=True, slots=True)
class CompiledTask:
    """Official task construction, immutable metadata, and its trusted init bank."""

    official_env: OfficialBatchEnv
    metadata: TaskRuntimeMetadata
    init_state_bank: InitStateBank

    @property
    def model(self) -> mujoco.MjModel:
        """The exact MuJoCo model owned by the compiled official environment."""
        return self.metadata.model

    def get_state(self) -> torch.Tensor:
        """Read the current official state using the frozen runtime convention."""
        return self.official_env.get_state()

    def get_proprio(self) -> torch.Tensor:
        """Read current official proprioception using the frozen runtime convention."""
        return self.official_env.get_proprio()

    def get_sim_time(self) -> torch.Tensor:
        """Read current official simulator time using the frozen runtime convention."""
        return self.official_env.get_sim_time()

    def close(self) -> None:
        """Close the underlying official environment after compilation use."""
        self.official_env.close()


class TaskCompiler:
    """Compile one canonical LIBERO task through the official BDDL backend."""

    def compile(
        self,
        config: EnvConfig,
        *,
        exact_model_xml: str | None = None,
        expected_model_xml_sha256: str | None = None,
    ) -> CompiledTask:
        """Construct and decode one task without creating a Warp runtime.

        ``exact_model_xml`` is an opt-in path for replaying a LIBERO HDF5 demo
        against its recorded static MuJoCo scene rather than reconstructing a
        seed-dependent BDDL placement.  It never changes the default compiler
        path.  When supplied, ``expected_model_xml_sha256`` verifies the raw
        HDF5 payload *before* host-specific asset paths are remapped.
        """
        if not isinstance(config, EnvConfig):
            raise TypeError("TaskCompiler.compile requires an EnvConfig instance.")
        if exact_model_xml is None and expected_model_xml_sha256 is not None:
            raise ValueError(
                "expected_model_xml_sha256 requires exact_model_xml to be supplied."
            )
        if expected_model_xml_sha256 is not None:
            _validate_sha256(
                expected_model_xml_sha256, name="expected_model_xml_sha256"
            )
        source_xml_sha256: str | None = None
        remapped_xml_sha256: str | None = None
        remapped_model_xml: str | None = None
        if exact_model_xml is not None:
            if not isinstance(exact_model_xml, str) or not exact_model_xml.strip():
                raise ValueError("exact_model_xml must be a non-empty string.")
            source_xml_sha256 = _sha256_text(exact_model_xml)
            if (
                expected_model_xml_sha256 is not None
                and source_xml_sha256 != expected_model_xml_sha256.lower()
            ):
                raise ValueError(
                    "exact_model_xml SHA-256 did not match expected_model_xml_sha256."
                )
            remapped_model_xml = _port_legacy_single_panda_xml(
                remap_demo_model_xml_assets(exact_model_xml)
            )
            remapped_xml_sha256 = _sha256_text(remapped_model_xml)
        official_compile_config = replace(config, backend="official", num_worlds=1)
        official_env = OfficialBatchEnv(official_compile_config)
        try:
            # Let robosuite complete one official hard reset before binding the
            # model. Subsequent compiler and public resets stay soft so this
            # exact MjModel remains stable for MJWarp handoff.
            official_env.reset()
            if remapped_model_xml is not None:
                # robosuite owns controller construction.  Rebind its simulator
                # through its public XML reset API, then reset only MuJoCo data;
                # calling a subsequent hard env reset would regenerate the BDDL
                # placement and discard this recorded static scene.
                official_env.reset_from_xml_string(remapped_model_xml)
                official_env._env.sim.reset()
                official_env._env.sim.forward()
            official_env._env.env.hard_reset = False
            model = official_env._env.sim.model._model
            if not isinstance(model, mujoco.MjModel):
                raise TypeError(
                    "Official backend did not expose a mujoco.MjModel through "
                    "sim.model._model."
                )
            metadata = self._compile_metadata(
                official_env,
                model,
                target_config=config,
                source_model_xml_sha256=source_xml_sha256,
                remapped_model_xml_sha256=remapped_xml_sha256,
            )
            init_state_bank = self._load_init_state_bank(official_env, metadata)
            if model is not official_env._env.sim.model._model:
                raise RuntimeError(
                    "Official compilation changed MjModel identity while decoding "
                    "init states."
                )
            return CompiledTask(
                official_env=official_env,
                metadata=metadata,
                init_state_bank=init_state_bank,
            )
        except Exception:
            official_env.close()
            raise

    @staticmethod
    def _compile_metadata(
        official_env: OfficialBatchEnv,
        model: mujoco.MjModel,
        *,
        target_config: EnvConfig,
        source_model_xml_sha256: str | None,
        remapped_model_xml_sha256: str | None,
    ) -> TaskRuntimeMetadata:
        camera_ids = _name_to_id(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
        requested_camera_names = [camera.name for camera in official_env.config.cameras]
        if len(set(requested_camera_names)) != len(requested_camera_names):
            raise ValueError(
                "Requested camera names must be unique during compilation."
            )
        missing_cameras = [
            name for name in requested_camera_names if name not in camera_ids
        ]
        if missing_cameras:
            raise ValueError(
                "Requested cameras are absent from the compiled MuJoCo model: "
                f"{missing_cameras}. Available cameras: {sorted(camera_ids)}."
            )
        ratio = 1.0 / (official_env.config.control_freq * model.opt.timestep)
        control_substeps = round(ratio)
        if control_substeps <= 0 or not np.isclose(
            ratio, control_substeps, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                "Control frequency and MuJoCo timestep do not form an integral "
                f"decimation: 1 / ({official_env.config.control_freq} * "
                f"{model.opt.timestep}) = {ratio}."
            )
        fullphysics_state_size = mujoco.mj_stateSize(
            model, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
        return TaskRuntimeMetadata(
            suite=official_env.config.suite,
            task_index=official_env.config.task_index,
            task_name=official_env.task.name,
            bddl_file=official_env._suite.get_task_bddl_file_path(
                official_env.config.task_index
            ),
            target_backend=target_config.backend,
            target_num_worlds=target_config.num_worlds,
            model=model,
            mujoco_version=mujoco.__version__,
            robosuite_version=getattr(robosuite, "__version__", "unknown"),
            timestep=float(model.opt.timestep),
            control_frequency=official_env.config.control_freq,
            control_substeps=control_substeps,
            nq=model.nq,
            nv=model.nv,
            nu=model.nu,
            na=model.na,
            fullphysics_state_size=fullphysics_state_size,
            source_model_xml_sha256=source_model_xml_sha256,
            remapped_model_xml_sha256=remapped_model_xml_sha256,
            body_ids=_name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
            site_ids=_name_to_id(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite),
            geom_ids=_name_to_id(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
            joint_ids=_name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
            camera_ids=camera_ids,
            actuator_ids=_name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
            requested_camera_ids={
                name: camera_ids[name] for name in requested_camera_names
            },
        )

    @staticmethod
    def _load_init_state_bank(
        official_env: OfficialBatchEnv, metadata: TaskRuntimeMetadata
    ) -> InitStateBank:
        """Decode trusted legacy states using MuJoCo, never manual offsets."""
        raw_states = official_env._suite.get_task_init_states(
            official_env.config.task_index
        )
        legacy = np.asarray(raw_states, dtype=np.float64)
        if legacy.ndim != 2 or legacy.shape[0] == 0:
            raise ValueError(
                "Trusted init-state bank must be a non-empty two-dimensional array."
            )
        if legacy.shape[1] != metadata.fullphysics_state_size:
            raise ValueError(
                "Legacy init-state width does not match MuJoCo FULLPHYSICS size: "
                f"{legacy.shape[1]} != {metadata.fullphysics_state_size}."
            )
        if not np.isfinite(legacy).all():
            raise ValueError("Trusted init-state bank contains non-finite values.")

        state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        count = legacy.shape[0]
        fullphysics = np.empty_like(legacy)
        time = np.empty((count,), dtype=np.float64)
        qpos = np.empty((count, metadata.nq), dtype=np.float64)
        qvel = np.empty((count, metadata.nv), dtype=np.float64)
        act = np.empty((count, metadata.na), dtype=np.float64)
        control_env = official_env._env
        clean_state = np.ascontiguousarray(control_env.get_sim_state().copy())
        data = control_env.sim.data._data
        try:
            for index, state in enumerate(legacy):
                # Use LIBERO's official set_state_from_flattened path rather
                # than assuming the legacy flattened field ordering. Avoid a
                # hard reset here: robosuite would rebuild the model object.
                control_env.set_state(state)
                control_env.sim.forward()
                mujoco.mj_getState(metadata.model, data, fullphysics[index], state_spec)
                time[index] = data.time
                qpos[index] = data.qpos
                qvel[index] = data.qvel
                if metadata.na:
                    act[index] = data.act
        finally:
            # Restore the initial official state without rebuilding its model.
            raw_observation = control_env.regenerate_obs_from_state(clean_state)
            official_env._last_raw_observation = raw_observation
        return InitStateBank(
            legacy_flattened=torch.from_numpy(np.ascontiguousarray(legacy)),
            fullphysics=torch.from_numpy(np.ascontiguousarray(fullphysics)),
            time=torch.from_numpy(time),
            qpos=torch.from_numpy(qpos),
            qvel=torch.from_numpy(qvel),
            act=torch.from_numpy(act),
        )


def _name_to_id(
    model: mujoco.MjModel, object_type: Any, count: int
) -> Mapping[str, int]:
    """Return a total named-object map and reject ambiguous MuJoCo names."""
    mapping: dict[str, int] = {}
    for object_id in range(count):
        name = mujoco.mj_id2name(model, object_type, object_id)
        if name is None:
            continue
        if name in mapping:
            raise ValueError(f"Duplicate MuJoCo name {name!r} for {object_type}.")
        mapping[name] = object_id
    return mapping
