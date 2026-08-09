"""Compatibility fix for robosuite 1.5.2 with MuJoCo 3.11.

Remove this module once LIBERO requires a robosuite release whose controller
uses the MuJoCo 3 ``mj_fullM(model, data, destination)`` binding directly.
"""

import inspect

import mujoco
import numpy as np
import robosuite
from robosuite.controllers.parts.controller import Controller

_SUPPORTED_MUJOCO_VERSION = "3.11.0"
_SUPPORTED_ROBOSUITE_VERSION = "1.5.2"


def _update_with_mujoco_3_mass_matrix(self, force=False):
    """Update robosuite controller state using MuJoCo 3's ``mj_fullM`` API."""
    if not (self.new_update or force):
        return

    if not (self.lite_physics and not force):
        self.sim.forward()

    if self.ref_name is not None:
        self.update_reference_data()

    self.joint_pos = np.array(self.sim.data.qpos[self.qpos_index])
    self.joint_vel = np.array(self.sim.data.qvel[self.qvel_index])

    mass_matrix = np.empty((self.sim.model.nv, self.sim.model.nv), dtype=np.float64)
    # MuJoCo 3.11 accepts MjData directly. Older bindings exposed the sparse
    # qM array instead, which is what robosuite 1.5.2 still passes here.
    mujoco_data = getattr(self.sim.data, "_data", None)
    if mujoco_data is None:
        raise RuntimeError(
            "LIBERO's MuJoCo 3.11 compatibility patch requires robosuite's "
            "MjData wrapper to expose its underlying MjData as '_data'."
        )
    mujoco.mj_fullM(self.sim.model._model, mujoco_data, mass_matrix)
    self.mass_matrix = mass_matrix[np.ix_(self.qvel_index, self.qvel_index)]
    self.new_update = False


def install_robosuite_mujoco3_compatibility():
    """Install the narrow, version-gated patch required when ``MjData.qM`` is absent."""
    if getattr(Controller, "_libero_mujoco3_patch", False):
        return

    if hasattr(mujoco.MjData, "qM"):
        return
    if (
        mujoco.__version__ != _SUPPORTED_MUJOCO_VERSION
        or robosuite.__version__ != _SUPPORTED_ROBOSUITE_VERSION
    ):
        raise RuntimeError(
            "MjData.qM is unavailable, but LIBERO's compatibility patch only "
            f"supports mujoco=={_SUPPORTED_MUJOCO_VERSION} and "
            f"robosuite=={_SUPPORTED_ROBOSUITE_VERSION}; found "
            f"mujoco=={mujoco.__version__} and robosuite=={robosuite.__version__}."
        )
    if Controller.update.__module__ != "robosuite.controllers.parts.controller":
        raise RuntimeError("Refusing to patch an unknown robosuite Controller.update.")
    if "self.sim.data.qM" not in inspect.getsource(Controller.update):
        raise RuntimeError(
            "robosuite Controller.update no longer has the expected qM call; "
            "refusing to apply LIBERO's compatibility patch."
        )

    Controller.update = _update_with_mujoco_3_mass_matrix
    Controller._libero_mujoco3_patch = True


install_robosuite_mujoco3_compatibility()
