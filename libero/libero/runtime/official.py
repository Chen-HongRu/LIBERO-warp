"""Single-world adapter for the official LIBERO MuJoCo backend."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np
import torch
from robosuite.utils.camera_utils import get_real_depth_map

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from .types import (
    CameraConfig,
    EnvConfig,
    ObservationBatch,
    RenderExactState,
    StepBatch,
)

_CANONICAL_SUITES = frozenset(
    {"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"}
)


class OfficialBatchEnv:
    """Adapt one official environment to the frozen batched runtime contract.

    The official backend only supports one world and always returns CPU tensors.
    Robosuite captures its image convention while camera sensors are created; we
    record that convention and normalize every visual modality to OpenCV's
    top-left image origin before exposing it through :class:`ObservationBatch`.
    """

    def __init__(self, config: EnvConfig) -> None:
        if config.backend != "official":
            raise ValueError("OfficialBatchEnv requires EnvConfig.backend='official'.")
        if config.num_worlds != 1:
            raise ValueError("The official backend only supports num_worlds=1.")
        self.config = config
        self._closed = False
        self._last_raw_observation: Mapping[str, np.ndarray] | None = None

        if config.suite not in _CANONICAL_SUITES:
            raise ValueError(
                f"Unsupported LIBERO suite {config.suite!r}; supported suites: "
                f"{sorted(_CANONICAL_SUITES)}."
            )
        suite_class = benchmark.get_benchmark_dict().get(config.suite)
        if suite_class is None:
            available = sorted(benchmark.get_benchmark_dict())
            raise ValueError(
                f"Unknown LIBERO suite {config.suite!r}; available suites: {available}."
            )
        self._suite = suite_class()
        if config.task_index >= self._suite.get_num_tasks():
            raise IndexError(
                f"task_index {config.task_index} is outside suite {config.suite!r} "
                f"with {self._suite.get_num_tasks()} tasks."
            )
        self.task = self._suite.get_task(config.task_index)

        # RobotEnv captures this convention in each camera sensor closure during
        # construction. Store it so conversion cannot silently follow later
        # global macro changes.
        import robosuite.macros as macros

        self._source_image_convention = macros.IMAGE_CONVENTION
        if self._source_image_convention not in {"opengl", "opencv"}:
            raise RuntimeError(
                "Unsupported robosuite IMAGE_CONVENTION "
                f"{self._source_image_convention!r}; expected 'opengl' or 'opencv'."
            )

        camera_segmentations = [
            None if camera.segmentation is None else [camera.segmentation]
            for camera in config.cameras
        ]
        self._env = OffScreenRenderEnv(
            bddl_file_name=self._suite.get_task_bddl_file_path(config.task_index),
            robots=["Panda"],
            controller="OSC_POSE",
            use_camera_obs=True,
            camera_names=[camera.name for camera in config.cameras],
            camera_heights=[camera.height for camera in config.cameras],
            camera_widths=[camera.width for camera in config.cameras],
            camera_depths=[camera.depth for camera in config.cameras],
            camera_segmentations=(
                None
                if all(value is None for value in camera_segmentations)
                else camera_segmentations
            ),
            control_freq=config.control_freq,
            horizon=config.horizon,
            ignore_done=False,
            # Robosuite 1.5.2 consumes this at construction to initialize its
            # per-environment Generator. Calling ``env.seed(...)`` afterward
            # is invalid: ``seed`` is an integer attribute, not a method.
            seed=config.seed,
        )

    @property
    def source_image_convention(self) -> str:
        """Robosuite convention captured when this environment was constructed."""
        return self._source_image_convention

    def reset(
        self,
        *,
        init_state: torch.Tensor | np.ndarray | None = None,
        world_ids: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    ) -> ObservationBatch:
        """Reset the only official world, optionally to a trusted init state.

        ``world_ids`` is accepted for API parity but must be exactly ``[0]``
        when supplied. Reset never performs any implicit action continuation.
        """
        self._ensure_open()
        self._validate_world_ids(world_ids)
        raw_observation = self._env.reset()
        if init_state is not None:
            raw_observation = self._env.set_init_state(
                self._normalize_init_state(init_state)
            )
        self._last_raw_observation = raw_observation
        return self._make_observation(raw_observation)

    def step(self, actions: torch.Tensor | np.ndarray) -> StepBatch:
        """Apply one ``[1, 7]`` action after finite-value validation and clipping."""
        self._ensure_open()
        action = self._validate_action(actions)
        raw_observation, reward, official_done, info = self._env.step(action[0])
        self._last_raw_observation = raw_observation
        observation = self._make_observation(raw_observation)
        terminated = bool(self._env.check_success())
        truncated = bool(self._env.env.timestep >= self.config.horizon)
        return StepBatch(
            observation=observation,
            reward=torch.tensor([reward], dtype=torch.float32, device="cpu"),
            terminated=torch.tensor([terminated], dtype=torch.bool, device="cpu"),
            truncated=torch.tensor([truncated], dtype=torch.bool, device="cpu"),
            info={
                "official_done": bool(official_done),
                "source_image_convention": self._source_image_convention,
                **dict(info),
            },
        )

    def get_state(self) -> torch.Tensor:
        """Return the current full MuJoCo state as a CPU ``[1, D]`` tensor."""
        self._ensure_open()
        return self._state_tensor()

    def get_proprio(self) -> torch.Tensor:
        """Return ``robot0_proprio-state`` as a CPU ``[1, D]`` tensor."""
        self._ensure_open()
        if self._last_raw_observation is None:
            raise RuntimeError("Call reset() before requesting proprioception.")
        return self._proprio_tensor(self._last_raw_observation)

    def get_sim_time(self) -> torch.Tensor:
        """Return the official simulator time as a CPU ``[1]`` tensor."""
        self._ensure_open()
        return torch.tensor(
            [self._env.sim.data.time], dtype=torch.float64, device="cpu"
        )

    def get_render_exact_state(self) -> RenderExactState:
        """Return a render-exact snapshot; continuation restore is unsupported."""
        self._ensure_open()
        return RenderExactState(
            state=self._state_tensor().clone(), sim_time=self.get_sim_time().clone()
        )

    def close(self) -> None:
        """Release official rendering and simulation resources."""
        if not self._closed:
            self._env.close()
            self._closed = True

    def _make_observation(self, raw_observation: Mapping[str, Any]) -> ObservationBatch:
        rgb: OrderedDict[str, torch.Tensor] = OrderedDict()
        depth: OrderedDict[str, torch.Tensor] = OrderedDict()
        segmentation: OrderedDict[str, torch.Tensor] = OrderedDict()
        for camera in self.config.cameras:
            rgb[camera.name] = self._visual_tensor(
                raw_observation, f"{camera.name}_image", camera, "rgb"
            )
            if camera.depth:
                depth[camera.name] = self._visual_tensor(
                    raw_observation, f"{camera.name}_depth", camera, "depth"
                )
            if camera.segmentation is not None:
                segmentation[camera.name] = self._visual_tensor(
                    raw_observation,
                    f"{camera.name}_segmentation_{camera.segmentation}",
                    camera,
                    "segmentation",
                )
        return ObservationBatch(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            proprio=self._proprio_tensor(raw_observation),
            state=self._state_tensor(),
            sim_time=self.get_sim_time(),
        )

    def _visual_tensor(
        self,
        raw_observation: Mapping[str, Any],
        key: str,
        camera: CameraConfig,
        modality: str,
    ) -> torch.Tensor:
        if key not in raw_observation:
            raise RuntimeError(
                f"Official backend did not return requested {modality} key {key!r}."
            )
        value = np.asarray(raw_observation[key])
        if modality == "depth":
            # Robosuite exposes normalized MuJoCo depth. Convert it to the
            # public metric camera-Z depth convention before orienting pixels.
            value = get_real_depth_map(self._env.sim, value)
        value = self._to_top_left(value)
        if value.shape[:2] != (camera.height, camera.width):
            raise RuntimeError(
                f"Camera {camera.name!r} returned {value.shape[:2]}, expected "
                f"{(camera.height, camera.width)}."
            )
        if modality == "rgb":
            if value.ndim != 3 or value.shape[-1] != 3:
                raise RuntimeError(
                    f"Camera {camera.name!r} RGB must be HWC with three channels; "
                    f"got {value.shape}."
                )
            value = value.astype(np.uint8, copy=False)
        elif value.ndim == 2:
            value = value[..., None]
        if value.ndim != 3:
            raise RuntimeError(
                f"Camera {camera.name!r} {modality} must be HW or HWC; "
                f"got {value.shape}."
            )
        if modality != "rgb":
            if value.shape[-1] != 1:
                raise RuntimeError(
                    f"Camera {camera.name!r} {modality} must have one channel; "
                    f"got {value.shape}."
                )
            dtype = np.float32 if modality == "depth" else np.int32
            value = value.astype(dtype, copy=False)
        return torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0)

    def _to_top_left(self, value: np.ndarray) -> np.ndarray:
        """Normalize the construction-time robosuite camera convention to top-left."""
        if self._source_image_convention == "opengl":
            return value[::-1]
        return value

    def _proprio_tensor(self, raw_observation: Mapping[str, Any]) -> torch.Tensor:
        key = "robot0_proprio-state"
        if key not in raw_observation:
            raise RuntimeError(
                f"Official backend did not return required proprio key {key!r}."
            )
        value = np.ascontiguousarray(np.asarray(raw_observation[key]))
        return torch.from_numpy(value).unsqueeze(0)

    def _state_tensor(self) -> torch.Tensor:
        value = np.ascontiguousarray(np.asarray(self._env.get_sim_state()))
        return torch.from_numpy(value).unsqueeze(0)

    def _validate_action(self, actions: torch.Tensor | np.ndarray) -> np.ndarray:
        action = self._as_numpy(actions, name="actions")
        if action.shape != (1, 7):
            raise ValueError(
                "Official backend actions must have shape [1, 7]; "
                f"got {list(action.shape)}."
            )
        if not self._is_real_numeric_dtype(action.dtype):
            raise TypeError("actions must have a real numeric dtype.")
        if not np.isfinite(action).all():
            raise ValueError("actions must contain only finite values.")
        return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _as_numpy(value: torch.Tensor | np.ndarray, *, name: str) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu").numpy()
        if isinstance(value, np.ndarray):
            return value
        raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray.")

    @staticmethod
    def _validate_world_ids(
        world_ids: Sequence[int] | np.ndarray | torch.Tensor | None,
    ) -> None:
        if world_ids is None:
            return
        if isinstance(world_ids, torch.Tensor):
            values = tuple(world_ids.detach().to(device="cpu").reshape(-1).tolist())
        elif isinstance(world_ids, np.ndarray):
            values = tuple(world_ids.reshape(-1).tolist())
        elif isinstance(world_ids, Sequence) and not isinstance(
            world_ids, (str, bytes)
        ):
            values = tuple(world_ids)
        else:
            raise TypeError(
                "world_ids must be a sequence, numpy array, or torch tensor."
            )
        if len(values) != 1 or not isinstance(values[0], Integral) or values[0] != 0:
            raise ValueError("The official backend only accepts world_ids=[0].")

    def _normalize_init_state(
        self, init_state: torch.Tensor | np.ndarray
    ) -> np.ndarray:
        value = self._as_numpy(init_state, name="init_state")
        if not self._is_real_numeric_dtype(value.dtype):
            raise TypeError("init_state must have a real numeric dtype.")
        if value.ndim == 1:
            flat_state = value
        elif value.ndim == 2 and value.shape[0] == 1:
            flat_state = value[0]
        else:
            raise ValueError(
                "Official init_state must have shape [D] or [1, D]; "
                f"got {list(value.shape)}."
            )
        if not np.isfinite(flat_state).all():
            raise ValueError("init_state must contain only finite values.")
        expected_size = self._env.get_sim_state().size
        if flat_state.size != expected_size:
            raise ValueError(
                f"init_state must contain {expected_size} values; "
                f"got {flat_state.size}."
            )
        return np.ascontiguousarray(flat_state)

    @staticmethod
    def _is_real_numeric_dtype(dtype: np.dtype[Any]) -> bool:
        """Accept real integer / floating arrays without lossy complex casts."""
        return np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("This environment is closed.")
