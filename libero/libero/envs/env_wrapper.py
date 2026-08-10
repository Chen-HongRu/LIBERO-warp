import colorsys
import os

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


def randomize_colors(count, bright=True):
    """Return distinct RGB colors for segmentation visualization."""
    brightness = 1.0 if bright else 0.7
    return np.array(
        [colorsys.hsv_to_rgb(index / count, 1.0, brightness) for index in range(count)]
    )


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
        **kwargs,
    ):
        assert os.path.exists(bddl_file_name), (
            f"[error] {bddl_file_name} does not exist!"
        )

        # LIBERO's legacy BDDL placement samplers use ``numpy.random``'s
        # process-global RandomState, while robosuite 1.5.2 owns a separate
        # Generator. Scope a saved global state to this wrapper so a seeded
        # environment is deterministic without leaking its seed to callers or
        # to other environments.
        seed = kwargs.get("seed")
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

        problem_info = BDDLUtils.get_problem_info(bddl_file_name)
        self.problem_name = problem_info["problem_name"]
        self.domain_name = problem_info["domain_name"]
        self.language_instruction = problem_info["language_instruction"]
        try:
            self.env = TASK_MAPPING[self.problem_name](
                bddl_file_name,
                robots=robots,
                controller_configs=controller_configs,
                gripper_types=gripper_types,
                initialization_noise=initialization_noise,
                use_camera_obs=use_camera_obs,
                has_renderer=has_renderer,
                has_offscreen_renderer=has_offscreen_renderer,
                render_camera=render_camera,
                render_collision_mesh=render_collision_mesh,
                render_visual_mesh=render_visual_mesh,
                render_gpu_device_id=render_gpu_device_id,
                control_freq=control_freq,
                horizon=horizon,
                ignore_done=ignore_done,
                hard_reset=hard_reset,
                camera_names=camera_names,
                camera_heights=camera_heights,
                camera_widths=camera_widths,
                camera_depths=camera_depths,
                camera_segmentations=camera_segmentations,
                renderer=renderer,
                renderer_config=renderer_config,
                **kwargs,
            )
        finally:
            if caller_rng_state is not None:
                self._legacy_rng_state = np.random.get_state()
                np.random.set_state(caller_rng_state)

    @property
    def obj_of_interest(self):
        return self.env.obj_of_interest

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        while True:
            try:
                return self._reset_with_legacy_rng()
            except RandomizationError:
                continue

    def check_success(self):
        return self.env._check_success()

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
        return self.env.sim.get_state().flatten()

    def _post_process(self):
        return self.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        self.env.sim.set_state_from_flattened(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self.env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        """Reset the robosuite 1.5.2 environment RNG to ``seed``.

        In robosuite 1.5.2, ``MujocoEnv.seed`` is a numeric instance
        attribute set during construction rather than a callable API. Keep
        this legacy wrapper usable without treating that attribute as a
        method.
        """
        self.env.seed = seed
        self.env.rng = np.random.default_rng(seed)
        if seed is None:
            self._legacy_rng_state = None
            return
        caller_rng_state = np.random.get_state()
        try:
            np.random.seed(seed)
            self._legacy_rng_state = np.random.get_state()
        finally:
            np.random.set_state(caller_rng_state)

    def _reset_with_legacy_rng(self):
        """Reset with this environment's saved BDDL placement RNG state."""
        legacy_rng_state = getattr(self, "_legacy_rng_state", None)
        if legacy_rng_state is None:
            return self.env.reset()
        caller_rng_state = np.random.get_state()
        np.random.set_state(legacy_rng_state)
        try:
            return self.env.reset()
        finally:
            self._legacy_rng_state = np.random.get_state()
            np.random.set_state(caller_rng_state)

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.sim.forward()
        self.check_success()
        self._post_process()
        self._update_observables(force=True)
        return self.env._get_observations()

    def close(self):
        self.env.close()
        del self.env


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
        self.segmentation_id_mapping = {}
        self.instance_to_id = {}
        self.robot_segmentation_ids = frozenset()
        super().__init__(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = super().reset()
        self.segmentation_id_mapping = {}
        self.robot_segmentation_ids = frozenset()

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
