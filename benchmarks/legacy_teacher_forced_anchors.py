"""Build teacher-forced one-control anchors in the legacy LIBERO runtime.

The output is a Tier-1 local-dynamics artifact.  Every selected action starts
from its recorded HDF state, reinitializes the OSC controller at that state,
captures all 25 actuator-control vectors, and records the legacy MuJoCo result.
It does not claim action-exact long-horizon replay or task-level qualification.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def _load_legacy_helpers(path: Path):
    spec = importlib.util.spec_from_file_location("legacy_reference_replay", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_env(demo_hdf5: Path, demo_name: str, libero_root: Path, helpers):
    from libero.libero.envs import TASK_MAPPING

    with h5py.File(demo_hdf5, "r") as dataset:
        env_args = json.loads(dataset["data"].attrs["env_args"])
        demo = dataset[f"data/{demo_name}"]
        actions = np.asarray(demo["actions"][:], dtype=np.float64)
        states = np.asarray(demo["states"][:], dtype=np.float64)
        model_xml = demo.attrs["model_file"]
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8")

    env_kwargs = dict(env_args["env_kwargs"])
    env_kwargs["bddl_file_name"] = helpers._resolve_bddl(env_args, libero_root)
    env = TASK_MAPPING[env_args["problem_name"]](**env_kwargs)
    env.reset()
    remapped_xml = helpers._remap_model_xml(
        model_xml, libero_root / "libero" / "libero" / "assets"
    )
    env.reset_from_xml_string(remapped_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()
    return env, model_xml, remapped_xml, states, actions


def _state_playback_success(env, states: np.ndarray) -> np.ndarray:
    success = []
    for state in states:
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        success.append(bool(env._check_success()))
    return np.asarray(success, dtype=np.bool_)


def _anchor_indices(states: np.ndarray, env, count: int) -> list[int]:
    contact_counts = []
    for state in states:
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        contact_counts.append(int(env.sim.data.ncon))
    contact_counts = np.asarray(contact_counts)
    changes = np.flatnonzero(np.diff(contact_counts) != 0) + 1
    candidates = set(np.linspace(0, len(states) - 2, count, dtype=int).tolist())
    for index in changes[: max(0, count // 2)]:
        candidates.update({max(0, int(index) - 1), int(index)})
    candidates.update({max(0, len(states) - 3), len(states) - 2})
    selected = sorted(candidates)
    if len(selected) > count:
        positions = np.linspace(0, len(selected) - 1, count, dtype=int)
        selected = [selected[position] for position in positions]
    return selected


def _reload_controller_at_current_state(env) -> None:
    for robot in env.robots:
        robot._load_controller()
        robot.controller.update_base_pose(robot.base_pos, robot.base_ori)


def run(args: argparse.Namespace) -> dict:
    import mujoco
    import robosuite

    helpers = _load_legacy_helpers(args.helper)
    env, source_xml, remapped_xml, states, actions = _build_env(
        args.demo_hdf5, args.demo, args.libero_root, helpers
    )
    try:
        state_success = _state_playback_success(env, states)
        success_indices = np.flatnonzero(state_success)
        indices = _anchor_indices(states, env, args.anchor_count)

        controls = []
        output_states = []
        qpos = []
        qvel = []
        ncon_before = []
        ncon_after = []
        health = []

        original_step = env.sim.step
        captured: list[np.ndarray] = []

        def recording_step() -> None:
            captured.append(np.asarray(env.sim.data.ctrl).copy())
            original_step()

        env.sim.step = recording_step
        for index in indices:
            env.sim.set_state_from_flattened(states[index])
            env.sim.forward()
            _reload_controller_at_current_state(env)
            ncon_before.append(int(env.sim.data.ncon))
            start = len(captured)
            env.step(actions[index])
            anchor_ctrl = np.asarray(captured[start:])
            if anchor_ctrl.shape != (25, env.sim.model.nu):
                raise RuntimeError(
                    f"anchor {index} captured {anchor_ctrl.shape}, expected "
                    f"(25, {env.sim.model.nu})"
                )
            state = env.sim.get_state().flatten().copy()
            controls.append(anchor_ctrl)
            output_states.append(state)
            qpos.append(np.asarray(env.sim.data.qpos).copy())
            qvel.append(np.asarray(env.sim.data.qvel).copy())
            ncon_after.append(int(env.sim.data.ncon))
            health.append(bool(np.isfinite(state).all()))

        np.savez_compressed(
            args.output_npz,
            source_model_xml=np.asarray(source_xml),
            remapped_model_xml=np.asarray(remapped_xml),
            anchor_indices=np.asarray(indices, dtype=np.int64),
            input_fullphysics=states[np.asarray(indices)],
            actions=actions[np.asarray(indices)],
            substep_ctrl=np.asarray(controls),
            legacy_output_fullphysics=np.asarray(output_states),
            legacy_qpos=np.asarray(qpos),
            legacy_qvel=np.asarray(qvel),
            legacy_ncon_before=np.asarray(ncon_before),
            legacy_ncon_after=np.asarray(ncon_after),
            state_playback_success=state_success,
        )
        report = {
            "qualification_tier": "tier1_teacher_forced_local_dynamics_only",
            "runtime": {
                "robosuite": robosuite.__version__,
                "mujoco": mujoco.__version__,
            },
            "state_playback": {
                "count": len(states),
                "ever_success": bool(success_indices.size),
                "first_success": (
                    int(success_indices[0]) if success_indices.size else None
                ),
                "last_success": (
                    int(success_indices[-1]) if success_indices.size else None
                ),
                "final_success": bool(state_success[-1]),
                "success_count": int(success_indices.size),
            },
            "anchors": {
                "indices": indices,
                "count": len(indices),
                "ctrl_shape": list(np.asarray(controls).shape),
                "ncon_before": ncon_before,
                "ncon_after": ncon_after,
                "all_healthy": bool(all(health)),
                "controller_initialization": (
                    "reload OSC at HDF state, then one policy action and 25 substeps"
                ),
            },
            "artifact_npz": str(args.output_npz),
        }
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return report
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-hdf5", type=Path, required=True)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--libero-root", type=Path, default=Path("/root/LIBERO"))
    parser.add_argument(
        "--helper",
        type=Path,
        default=Path(__file__).with_name("legacy_reference_replay.py"),
    )
    parser.add_argument("--anchor-count", type=int, default=12)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
