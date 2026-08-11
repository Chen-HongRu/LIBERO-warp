import colorsys
import importlib.metadata
import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import cv2
import matplotlib.cm as cm
import numpy as np
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)
from robosuite.utils.errors import RandomizationError

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs.bddl_base_domain import TASK_MAPPING

LIBERO_COMPAT_TARGET = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
_BACKEND_ENVIRONMENT_VARIABLE = "LIBERO_SIM_BACKEND"
_OFFICIAL_CAPABILITIES = frozenset(
    {
        "camera_calibration",
        "metric_depth",
        "model_xml_read",
        "model_xml_reset",
        "predicate_success",
        "reset",
        "rgb",
        "segmentation",
        "single_env_numpy_api",
        "state_flattened_read",
        "state_flattened_write",
        "step_osc_pose_7d",
    }
)


class UnsupportedBackendOperation(RuntimeError):
    """An explicitly selected backend cannot perform the requested operation."""

    def __init__(
        self,
        *,
        operation: str,
        backend: str,
        capability: str,
        replacement: str,
    ) -> None:
        self.operation = operation
        self.backend = backend
        self.capability = capability
        self.replacement = replacement
        super().__init__(
            f"Operation {operation!r} is unsupported by backend {backend!r}: "
            f"missing capability {capability!r}. Use {replacement}."
        )


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Immutable backend selection, capability, and build provenance."""

    schema_version: int
    requested_backend: str
    selection_source: str
    actual_backend: str
    capabilities: frozenset[str]
    libero_warp_version: str | None
    libero_compat_target: str
    dependency_versions: Mapping[str, str | None]
    device: Mapping[str, str | int | None]
    build: Mapping[str, str | None]

    def to_dict(self) -> dict[str, Any]:
        """Return the frozen schema in deterministic JSON-compatible form."""
        return {
            "schema_version": self.schema_version,
            "requested_backend": self.requested_backend,
            "selection_source": self.selection_source,
            "actual_backend": self.actual_backend,
            "capabilities": sorted(self.capabilities),
            "libero_warp_version": self.libero_warp_version,
            "libero_compat_target": self.libero_compat_target,
            "dependency_versions": dict(sorted(self.dependency_versions.items())),
            "device": dict(sorted(self.device.items())),
            "build": dict(sorted(self.build.items())),
        }


def _distribution_version(*distribution_names: str) -> str | None:
    for distribution_name in distribution_names:
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _dependency_versions() -> Mapping[str, str | None]:
    versions = {
        "MJWarp": _distribution_version("mujoco-warp", "mjwarp"),
        "MuJoCo": _distribution_version("mujoco"),
        "NumPy": _distribution_version("numpy"),
        "Torch": _distribution_version("torch"),
        "Warp": _distribution_version("warp-lang", "warp"),
        "robosuite": _distribution_version("robosuite"),
    }
    return MappingProxyType(dict(sorted(versions.items())))


def _resolve_backend(backend: str | None) -> tuple[str, str]:
    if backend is not None:
        requested_backend = backend
        selection_source = "constructor"
    elif _BACKEND_ENVIRONMENT_VARIABLE in os.environ:
        requested_backend = os.environ[_BACKEND_ENVIRONMENT_VARIABLE]
        selection_source = "environment"
    else:
        requested_backend = "official"
        selection_source = "default"

    if not isinstance(requested_backend, str):
        raise TypeError("backend must be exactly 'official' or 'warp'")
    if requested_backend not in {"official", "warp"}:
        raise ValueError(
            "backend must be exactly 'official' or 'warp'; "
            f"received {requested_backend!r} from {selection_source}"
        )
    if requested_backend == "warp":
        raise UnsupportedBackendOperation(
            operation="construct ControlEnv",
            backend="warp",
            capability="single_env_numpy_api",
            replacement='backend="official" until the G3 Warp compatibility slice',
        )
    return requested_backend, selection_source


def _official_backend_info(
    *,
    requested_backend: str,
    selection_source: str,
    render_device: str,
) -> BackendInfo:
    return BackendInfo(
        schema_version=1,
        requested_backend=requested_backend,
        selection_source=selection_source,
        actual_backend="official",
        capabilities=_OFFICIAL_CAPABILITIES,
        libero_warp_version=_distribution_version("libero-warp"),
        libero_compat_target=LIBERO_COMPAT_TARGET,
        dependency_versions=_dependency_versions(),
        device=MappingProxyType(
            {
                "compute": "cpu",
                "physics": "mujoco",
                "render": render_device,
            }
        ),
        build=MappingProxyType(
            {
                "cuda": None,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "python_runtime": sys.implementation.name,
            }
        ),
    )


def _official_render_device_identity(
    *, render_gpu_device_id: int, rendering_enabled: bool
) -> str:
    """Resolve the renderer backend and device without reporting ``auto``."""
    if not rendering_enabled:
        return "disabled"

    render_backend = os.environ.get("MUJOCO_GL", "").lower().strip()
    if not render_backend:
        render_backend = {
            "Darwin": "cgl",
            "Linux": "egl",
            "Windows": "wgl",
        }.get(platform.system(), "platform-default")
    if render_backend == "osmesa":
        return "osmesa:cpu"
    if render_backend != "egl":
        return render_backend

    selected_devices = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if selected_devices is None:
        selected_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selected_devices is None:
        selected_device = 0 if render_gpu_device_id == -1 else render_gpu_device_id
    elif selected_devices.isdigit():
        selected_device = int(selected_devices)
    else:
        visible_devices = [int(value) for value in selected_devices.split(",")]
        selected_device = (
            visible_devices[0] if render_gpu_device_id == -1 else render_gpu_device_id
        )
        if render_gpu_device_id != -1 and render_gpu_device_id not in visible_devices:
            raise AssertionError(
                "render_gpu_device_id must be one of the devices visible through "
                "MUJOCO_EGL_DEVICE_ID or CUDA_VISIBLE_DEVICES"
            )
    return f"egl:{selected_device}"


def randomize_colors(count, bright=True):
    """Return distinct RGB colors for segmentation visualization."""
    brightness = 1.0 if bright else 0.7
    return np.array(
        [colorsys.hsv_to_rgb(index / count, 1.0, brightness) for index in range(count)]
    )


class LiberoBackendSession(Protocol):
    """Internal backend lifecycle used by the legacy-compatible wrappers."""

    task: Any

    def reset(self): ...

    def step(self, action): ...

    def seed(self, seed): ...

    def check_success(self): ...

    def get_sim_state(self): ...

    def set_state(self, mujoco_state): ...

    def reset_from_xml_string(self, xml_string): ...

    def regenerate_obs_from_state(self, mujoco_state): ...

    def close(self): ...


class OfficialLiberoSession:
    """Official robosuite task ownership behind the internal backend seam.

    The robosuite task remains authoritative for camera observations, flattened
    MuJoCo state, predicates, and controller state. This session only delegates
    lifecycle operations and preserves LIBERO's legacy RNG behavior.
    """

    def __init__(
        self,
        *,
        task_factory,
        bddl_file_name,
        robots,
        controller,
        task_kwargs,
    ):
        seed = task_kwargs.get("seed")
        self._legacy_rng_state = None
        caller_rng_state = None
        if seed is not None:
            caller_rng_state = np.random.get_state()
            np.random.seed(seed)

        controller_configs = suite.load_part_controller_config(
            default_controller=controller
        )
        robot_type = robots[0] if isinstance(robots, list) else robots
        controller_configs = refactor_composite_controller_config(
            controller_configs, robot_type, ["right"]
        )

        try:
            self.task = task_factory(
                bddl_file_name,
                robots=robots,
                controller_configs=controller_configs,
                **task_kwargs,
            )
        finally:
            if caller_rng_state is not None:
                self._legacy_rng_state = np.random.get_state()
                np.random.set_state(caller_rng_state)

    def step(self, action):
        return self.task.step(action)

    def reset(self):
        while True:
            try:
                return self._reset_with_legacy_rng()
            except RandomizationError:
                continue

    def _reset_with_legacy_rng(self):
        legacy_rng_state = self._legacy_rng_state
        if legacy_rng_state is None:
            return self.task.reset()
        caller_rng_state = np.random.get_state()
        np.random.set_state(legacy_rng_state)
        try:
            return self.task.reset()
        finally:
            self._legacy_rng_state = np.random.get_state()
            np.random.set_state(caller_rng_state)

    def seed(self, seed):
        self.task.seed = seed
        self.task.rng = np.random.default_rng(seed)
        if seed is None:
            self._legacy_rng_state = None
            return
        caller_rng_state = np.random.get_state()
        try:
            np.random.seed(seed)
            self._legacy_rng_state = np.random.get_state()
        finally:
            np.random.set_state(caller_rng_state)

    def check_success(self):
        return self.task._check_success()

    def get_sim_state(self):
        return self.task.sim.get_state().flatten()

    def set_state(self, mujoco_state):
        self.task.sim.set_state_from_flattened(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self.task.reset_from_xml_string(xml_string)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.task.sim.forward()
        self.check_success()
        self.task._post_process()
        self.task._update_observables(force=True)
        return self.task._get_observations()

    def close(self):
        self.task.close()
        del self.task


class ControlEnv:
    def __init__(
        self,
        bddl_file_name,
        robots=["Panda"],
        controller="OSC_POSE",
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=[
            "agentview",
            "robot0_eye_in_hand",
        ],
        camera_heights=128,
        camera_widths=128,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        *,
        backend=None,
        **kwargs,
    ):
        requested_backend, selection_source = _resolve_backend(backend)
        render_device = _official_render_device_identity(
            render_gpu_device_id=render_gpu_device_id,
            rendering_enabled=has_renderer or has_offscreen_renderer,
        )
        assert os.path.exists(bddl_file_name), (
            f"[error] {bddl_file_name} does not exist!"
        )

        problem_info = BDDLUtils.get_problem_info(bddl_file_name)
        self.problem_name = problem_info["problem_name"]
        self.domain_name = problem_info["domain_name"]
        self.language_instruction = problem_info["language_instruction"]
        task_kwargs = {
            "gripper_types": gripper_types,
            "initialization_noise": initialization_noise,
            "use_camera_obs": use_camera_obs,
            "has_renderer": has_renderer,
            "has_offscreen_renderer": has_offscreen_renderer,
            "render_camera": render_camera,
            "render_collision_mesh": render_collision_mesh,
            "render_visual_mesh": render_visual_mesh,
            "render_gpu_device_id": render_gpu_device_id,
            "control_freq": control_freq,
            "horizon": horizon,
            "ignore_done": ignore_done,
            "hard_reset": hard_reset,
            "camera_names": camera_names,
            "camera_heights": camera_heights,
            "camera_widths": camera_widths,
            "camera_depths": camera_depths,
            "camera_segmentations": camera_segmentations,
            "renderer": renderer,
            "renderer_config": renderer_config,
            **kwargs,
        }
        self._session: LiberoBackendSession = OfficialLiberoSession(
            task_factory=TASK_MAPPING[self.problem_name],
            bddl_file_name=bddl_file_name,
            robots=robots,
            controller=controller,
            task_kwargs=task_kwargs,
        )
        # Preserve the legacy public escape hatch while session ownership stays
        # internal and replaceable.
        self.env = self._session.task
        self._backend_info = _official_backend_info(
            requested_backend=requested_backend,
            selection_source=selection_source,
            render_device=render_device,
        )

    @property
    def backend_info(self) -> BackendInfo:
        return self._backend_info

    @property
    def obj_of_interest(self):
        return self.env.obj_of_interest

    def step(self, action):
        return self._session.step(action)

    def reset(self):
        session = getattr(self, "_session", None)
        if session is not None:
            return session.reset()
        while True:
            try:
                return self.env.reset()
            except RandomizationError:
                continue

    def check_success(self):
        return self._session.check_success()

    @property
    def _visualizations(self):
        return self.env._visualizations

    @property
    def robots(self):
        return self.env.robots

    @property
    def sim(self):
        return self.env.sim

    def get_sim_state(self):
        return self._session.get_sim_state()

    def _post_process(self):
        return self.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        self._session.set_state(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self._session.reset_from_xml_string(xml_string)

    def seed(self, seed):
        """Reset the robosuite 1.5.2 environment RNG to ``seed``.

        In robosuite 1.5.2, ``MujocoEnv.seed`` is a numeric instance
        attribute set during construction rather than a callable API. Keep
        this legacy wrapper usable without treating that attribute as a
        method.
        """
        self._session.seed(seed)

    def _reset_with_legacy_rng(self):
        """Reset with this environment's saved BDDL placement RNG state."""
        return self._session._reset_with_legacy_rng()

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        return self._session.regenerate_obs_from_state(mujoco_state)

    def close(self):
        self._session.close()
        del self.env
        del self._session


class OffScreenRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True
        super().__init__(**kwargs)


class SegmentationRenderEnv(OffScreenRenderEnv):
    """
    This wrapper will additionally generate the segmentation mask of objects,
    which is useful for comparing attention.
    """

    def __init__(
        self,
        camera_segmentations="instance",
        camera_heights=128,
        camera_widths=128,
        **kwargs,
    ):
        assert camera_segmentations is not None
        kwargs["camera_segmentations"] = camera_segmentations
        kwargs["camera_heights"] = camera_heights
        kwargs["camera_widths"] = camera_widths
        self._segmentation_modes = frozenset(
            {camera_segmentations}
            if isinstance(camera_segmentations, str)
            else camera_segmentations
        )
        self.segmentation_id_mapping = {}
        self.instance_to_id = {}
        self.robot_segmentation_ids = frozenset()
        super().__init__(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = super().reset()
        self.segmentation_id_mapping = {}
        self.instance_to_id = {}
        self.robot_segmentation_ids = frozenset()

        segmentation_modes = getattr(
            self, "_segmentation_modes", frozenset({"instance"})
        )
        if "instance" not in segmentation_modes:
            return obs

        robot_instance_names = set()
        for idx, robot in enumerate(self.env.robots):
            robot_instance_names.add(f"{type(robot.robot_model).__name__}{idx}")
            robot_instance_names.add(f"{type(robot.robot_model.base).__name__}{idx}")
            for arm, gripper in robot.gripper.items():
                robot_instance_names.add(f"{type(gripper).__name__}{idx}_{arm}")

        instance_names = list(self.env.model.instances_to_ids.keys())
        robot_instances = {
            i + 1
            for i, instance_name in enumerate(instance_names)
            if instance_name in robot_instance_names
        }
        if not robot_instances:
            raise RuntimeError(
                "Could not identify robot instance IDs for segmentation. "
                f"Expected one of {sorted(robot_instance_names)}, but the "
                "model exposes "
                f"{instance_names}."
            )
        self.robot_segmentation_ids = frozenset(robot_instances)

        for i, instance_name in enumerate(instance_names):
            if instance_name not in robot_instance_names:
                self.segmentation_id_mapping[i] = instance_name

        self.instance_to_id = {
            v: k + 1 for k, v in self.segmentation_id_mapping.items()
        }
        return obs

    def get_segmentation_instances(self, segmentation_image):
        # get all instances' segmentation separately
        self._require_instance_segmentation_mode()
        seg_img_dict = {}
        robot_mask = np.isin(segmentation_image, tuple(self.robot_segmentation_ids))
        seg_img_dict["robot"] = segmentation_image * robot_mask

        self._validate_segmentation_ids(segmentation_image)

        for seg_id, instance_name in self.segmentation_id_mapping.items():
            instance_id = seg_id + 1
            seg_img_dict[instance_name] = segmentation_image * (
                segmentation_image == instance_id
            )
        return seg_img_dict

    def _validate_segmentation_ids(self, segmentation_image):
        known_ids = {0, *self.robot_segmentation_ids, *self.instance_to_id.values()}
        unknown_ids = np.setdiff1d(np.unique(segmentation_image), list(known_ids))
        if unknown_ids.size:
            raise ValueError(
                "Received segmentation IDs absent from the current model mapping: "
                f"{unknown_ids.tolist()}. Known robot IDs: "
                f"{sorted(self.robot_segmentation_ids)}."
            )

    def get_segmentation_of_interest(self, segmentation_image):
        # get the combined segmentation of obj of interest
        self._require_instance_segmentation_mode()
        # 1 for obj_of_interest
        # -1.0 for robot
        # 0 for other things
        missing_objects = [
            obj for obj in self.obj_of_interest if obj not in self.instance_to_id
        ]
        if missing_objects:
            raise KeyError(
                "Objects of interest are missing from the segmentation mapping: "
                f"{missing_objects}. Available instances: "
                f"{sorted(self.instance_to_id)}."
            )
        self._validate_segmentation_ids(segmentation_image)
        ret_seg = np.zeros_like(segmentation_image)
        for obj in self.obj_of_interest:
            ret_seg[segmentation_image == self.instance_to_id[obj]] = 1.0
        ret_seg[np.isin(segmentation_image, tuple(self.robot_segmentation_ids))] = -1.0
        return ret_seg

    def _require_instance_segmentation_mode(self):
        modes = getattr(self, "_segmentation_modes", frozenset({"instance"}))
        if "instance" not in modes:
            raise ValueError(
                "Segmentation instance helpers require "
                "camera_segmentations='instance'; received "
                f"{sorted(modes)}."
            )

    def segmentation_to_rgb(self, seg_im, random_colors=False):
        """
        Helper function to visualize segmentations as RGB frames.
        NOTE: assumes that geom IDs go up to 255 at most - if not,
        multiple geoms might be assigned to the same color.
        """
        # ensure all values lie within [0, 255]
        seg_im = np.mod(seg_im, 256)

        if random_colors:
            colors = randomize_colors(256, bright=True)
            return (255.0 * colors[seg_im]).astype(np.uint8)
        else:
            # Deterministically map each geom ID to a value in [0, 255].
            rstate = np.random.RandomState(seed=2)
            inds = np.arange(256)
            rstate.shuffle(inds)
            seg_img = (
                np.array(255.0 * cm.rainbow(inds[seg_im], 10))
                .astype(np.uint8)[..., :3]
                .astype(np.uint8)
                .squeeze(-2)
            )
            print(seg_img.shape)
            cv2.imshow("Seg Image", seg_img[::-1])
            cv2.waitKey(1)
            # use @inds to map each geom ID to a color
            return seg_img


class DemoRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True
        kwargs["render_camera"] = "frontview"

        super().__init__(**kwargs)

    def _get_observations(self):
        return self.env._get_observations()
