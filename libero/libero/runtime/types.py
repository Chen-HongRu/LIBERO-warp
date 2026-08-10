"""Frozen public data contracts for LIBERO runtime backends.

The data contracts intentionally do not prescribe a device. The official
adapter materializes CPU tensors, while a future Warp adapter may return CUDA
tensors without changing the public shapes or camera mapping semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np
import torch


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy a mapping while preserving its insertion (camera) order."""
    if not isinstance(values, Mapping):
        raise TypeError(f"Expected a mapping, got {type(values).__name__}.")
    return MappingProxyType(dict(values))


def _validate_tensor(name: str, value: torch.Tensor, ndim: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.shape}.")


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """One requested camera output, in the caller's declared order.

    RGB observations are returned as top-left-origin, NHWC ``uint8`` tensors.
    When ``depth`` is enabled, it is metric camera-Z depth in meters. Cameras
    do not have fixed roles or slots, and can use different resolutions.
    """

    name: str
    height: int
    width: int
    depth: bool = False
    segmentation: Literal["instance", "class", "element"] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("CameraConfig.name must be a non-empty string.")
        if isinstance(self.height, bool) or not isinstance(self.height, int):
            raise TypeError("CameraConfig.height must be an integer.")
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise TypeError("CameraConfig.width must be an integer.")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("CameraConfig height and width must be positive.")
        if not isinstance(self.depth, bool):
            raise TypeError("CameraConfig.depth must be a bool.")
        if self.segmentation not in {None, "instance", "class", "element"}:
            raise ValueError(
                "CameraConfig.segmentation must be one of 'instance', 'class', "
                "'element', or None."
            )


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Immutable configuration for exactly one LIBERO task/model.

    ``suite`` and ``task_index`` resolve one official BDDL task. Camera order is
    supplied exclusively by ``cameras``; no backend assigns view roles.
    """

    suite: str
    task_index: int
    cameras: tuple[CameraConfig, ...]
    backend: Literal["official", "warp"] = "official"
    num_worlds: int = 1
    horizon: int = 1000
    control_freq: int = 20
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.suite, str) or not self.suite.strip():
            raise ValueError("EnvConfig.suite must be a non-empty string.")
        object.__setattr__(self, "suite", self.suite.lower())

        if not isinstance(self.cameras, (tuple, list)):
            raise TypeError("EnvConfig.cameras must be an ordered tuple or list.")
        cameras = tuple(self.cameras)
        if not cameras:
            raise ValueError("EnvConfig.cameras must contain at least one camera.")
        if any(not isinstance(camera, CameraConfig) for camera in cameras):
            raise TypeError("EnvConfig.cameras must contain only CameraConfig values.")
        names = [camera.name for camera in cameras]
        if len(set(names)) != len(names):
            raise ValueError("EnvConfig camera names must be unique.")
        object.__setattr__(self, "cameras", cameras)

        if self.backend not in {"official", "warp"}:
            raise ValueError("EnvConfig.backend must be 'official' or 'warp'.")
        for name in ("task_index", "num_worlds", "horizon", "control_freq"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"EnvConfig.{name} must be an integer.")
        if self.task_index < 0:
            raise ValueError("EnvConfig.task_index must be non-negative.")
        if self.num_worlds <= 0:
            raise ValueError("EnvConfig.num_worlds must be positive.")
        if self.horizon <= 0 or self.control_freq <= 0:
            raise ValueError("EnvConfig.horizon and control_freq must be positive.")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise TypeError("EnvConfig.seed must be an integer or None.")
            # LIBERO's retained placement samplers still use numpy.random's
            # legacy RandomState alongside robosuite's Generator.  Reject
            # values that RandomState cannot seed before constructing a model.
            if not 0 <= self.seed <= np.iinfo(np.uint32).max:
                raise ValueError("EnvConfig.seed must be in [0, 2**32 - 1].")


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """Batch observations with ordered per-camera maps rather than image stacks.

    Every RGB tensor has shape ``[N, H, W, 3]`` and dtype ``torch.uint8``.
    ``depth`` and ``segmentation`` are keyed by the same caller-selected camera
    names when requested, allowing cameras to have unequal resolutions. Depth
    is metric camera-Z depth in meters with dtype ``torch.float32``;
    segmentation uses ``torch.int32``. Both have a singleton final channel.
    """

    rgb: Mapping[str, torch.Tensor]
    depth: Mapping[str, torch.Tensor]
    segmentation: Mapping[str, torch.Tensor]
    proprio: torch.Tensor
    state: torch.Tensor
    sim_time: torch.Tensor

    def __post_init__(self) -> None:
        rgb = _freeze_mapping(self.rgb)
        depth = _freeze_mapping(self.depth)
        segmentation = _freeze_mapping(self.segmentation)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "segmentation", segmentation)
        if not rgb:
            raise ValueError("ObservationBatch.rgb must contain at least one camera.")

        batch_size: int | None = None
        device: torch.device | None = None
        for camera_name, image in rgb.items():
            _validate_tensor(f"rgb[{camera_name!r}]", image, 4)
            if image.dtype != torch.uint8:
                raise TypeError(f"rgb[{camera_name!r}] must have dtype torch.uint8.")
            if image.shape[-1] != 3:
                raise ValueError(
                    f"rgb[{camera_name!r}] must have three color channels."
                )
            batch_size = image.shape[0] if batch_size is None else batch_size
            device = image.device if device is None else device
            if image.shape[0] != batch_size:
                raise ValueError("All camera tensors must share a batch dimension.")
            if image.device != device:
                raise ValueError("All observation tensors must share one device.")
        for modality, values in (("depth", depth), ("segmentation", segmentation)):
            for camera_name, image in values.items():
                _validate_tensor(f"{modality}[{camera_name!r}]", image, 4)
                if camera_name not in rgb:
                    raise ValueError(
                        f"{modality}[{camera_name!r}] has no matching RGB camera."
                    )
                if image.shape[0] != batch_size:
                    raise ValueError("All camera tensors must share a batch dimension.")
                if image.device != device:
                    raise ValueError("All observation tensors must share one device.")
                if image.shape[1:3] != rgb[camera_name].shape[1:3]:
                    raise ValueError(
                        f"{modality}[{camera_name!r}] must match its RGB H/W."
                    )
                if image.shape[-1] != 1:
                    raise ValueError(
                        f"{modality}[{camera_name!r}] must have one final channel."
                    )
                expected_dtype = torch.float32 if modality == "depth" else torch.int32
                if image.dtype != expected_dtype:
                    raise TypeError(
                        f"{modality}[{camera_name!r}] must have dtype {expected_dtype}."
                    )
        _validate_tensor("proprio", self.proprio, 2)
        _validate_tensor("state", self.state, 2)
        _validate_tensor("sim_time", self.sim_time, 1)
        if (
            self.proprio.shape[0] != batch_size
            or self.state.shape[0] != batch_size
            or self.sim_time.shape[0] != batch_size
        ):
            raise ValueError("Observation tensors must share a batch dimension.")
        if any(
            value.device != device
            for value in (self.proprio, self.state, self.sim_time)
        ):
            raise ValueError("All observation tensors must share one device.")


@dataclass(frozen=True, slots=True)
class StepBatch:
    """One explicit environment transition; it never performs an auto-reset."""

    observation: ObservationBatch
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_tensor("reward", self.reward, 1)
        _validate_tensor("terminated", self.terminated, 1)
        _validate_tensor("truncated", self.truncated, 1)
        if self.reward.dtype != torch.float32:
            raise TypeError("reward must have dtype torch.float32.")
        if self.terminated.dtype != torch.bool or self.truncated.dtype != torch.bool:
            raise TypeError("terminated and truncated must have dtype torch.bool.")
        batch_size = self.observation.sim_time.shape[0]
        if any(
            value.shape[0] != batch_size
            for value in (self.reward, self.terminated, self.truncated)
        ):
            raise ValueError("StepBatch tensors must share a batch dimension.")
        observation_device = self.observation.sim_time.device
        if any(
            value.device != observation_device
            for value in (self.reward, self.terminated, self.truncated)
        ):
            raise ValueError("StepBatch tensors must share the observation device.")
        object.__setattr__(self, "info", _freeze_mapping(self.info))


@dataclass(frozen=True, slots=True)
class RenderExactState:
    """A render-exact state snapshot, not a continuation checkpoint.

    This public snapshot is useful for inspection and rendering comparisons.
    It intentionally has no restore API because controller and solver transient
    state required for action-continuation parity is not yet represented.
    """

    state: torch.Tensor
    sim_time: torch.Tensor

    def __post_init__(self) -> None:
        _validate_tensor("state", self.state, 2)
        _validate_tensor("sim_time", self.sim_time, 1)
        if self.state.shape[0] != self.sim_time.shape[0]:
            raise ValueError("RenderExactState tensors must share a batch dimension.")
        if self.state.device != self.sim_time.device:
            raise ValueError("RenderExactState tensors must share one device.")


class RuntimeEnv(Protocol):
    """Backend-neutral runtime environment contract."""

    def reset(
        self,
        *,
        init_state: torch.Tensor | np.ndarray | None = None,
        world_ids: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    ) -> ObservationBatch: ...

    def step(self, actions: torch.Tensor | np.ndarray) -> StepBatch: ...

    def get_state(self) -> torch.Tensor: ...

    def get_proprio(self) -> torch.Tensor: ...

    def get_sim_time(self) -> torch.Tensor: ...

    def get_render_exact_state(self) -> RenderExactState: ...

    def close(self) -> None: ...
