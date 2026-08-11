from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.fixed_base_robot import FixedBaseRobot

from .mounted_panda import MountedPanda
from .on_the_ground_panda import OnTheGroundPanda

# robosuite 1.5 replaced the old SingleArm implementation with the real
# FixedBaseRobot implementation. Preserve LIBERO's import name without
# introducing a compatibility-only robot class.
SingleArm = FixedBaseRobot

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": SingleArm,
        "OnTheGroundPanda": SingleArm,
    }
)

__all__ = [
    "MountedPanda",
    "OnTheGroundPanda",
    "ROBOT_CLASS_MAPPING",
    "SingleArm",
]
