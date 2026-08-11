"""Replay one LIBERO HDF5 demo in the pinned legacy runtime.

This diagnostic is intentionally self-contained: run it with the legacy
robosuite 1.4 / MuJoCo 2.3 uv environment, not the project environment.  It
loads the exact per-demo XML and state, records the actuator control at every
physics substep, and preserves numeric artifacts for cross-runtime replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _model_common_fingerprint(model: Any) -> dict[str, Any]:
    """Hash stable model arrays available in both legacy and current MuJoCo."""

    fields = (
        "body_pos",
        "body_quat",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "jnt_pos",
        "jnt_axis",
        "actuator_ctrlrange",
    )
    digest = hashlib.sha256()
    shapes: dict[str, list[int]] = {}
    for field in fields:
        array = np.asarray(getattr(model, field))
        canonical = np.ascontiguousarray(array.astype("<f8", copy=False))
        digest.update(field.encode("utf-8"))
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
        shapes[field] = list(canonical.shape)
    return {"sha256": digest.hexdigest(), "fields": shapes}


def _remap_model_xml(model_xml: str, asset_root: Path) -> str:
    from libero.libero.utils import utils as libero_utils

    remapped = libero_utils.postprocess_model_xml(model_xml, {})
    legacy_prefixes = (
        "/home/yifengz/workspace/libero-dev/chiliocosm/assets",
        "/home/yifengz/workspace/libero-dev/libero/libero/assets",
    )
    for prefix in legacy_prefixes:
        remapped = remapped.replace(prefix, str(asset_root))
    return remapped


def _resolve_bddl(env_args: dict[str, Any], libero_root: Path) -> str:
    source = Path(env_args["bddl_file"])
    candidate = (
        libero_root
        / "libero"
        / "libero"
        / "bddl_files"
        / source.parent.name
        / source.name
    )
    if not candidate.exists():
        matches = list(
            (libero_root / "libero" / "libero" / "bddl_files").rglob(source.name)
        )
        if len(matches) != 1:
            rendered_matches = [str(path) for path in matches]
            raise FileNotFoundError(
                f"could not uniquely resolve {source.name}: {rendered_matches}"
            )
        candidate = matches[0]
    return str(candidate)


def replay(args: argparse.Namespace) -> dict[str, Any]:
    import mujoco
    import robosuite

    from libero.libero.envs import TASK_MAPPING

    with h5py.File(args.demo_hdf5, "r") as dataset:
        env_args = json.loads(dataset["data"].attrs["env_args"])
        demo = dataset[f"data/{args.demo}"]
        actions = np.asarray(demo["actions"][:], dtype=np.float64)
        states = np.asarray(demo["states"][:], dtype=np.float64)
        model_xml = demo.attrs["model_file"]
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8")

    source_xml_sha256 = hashlib.sha256(model_xml.encode("utf-8")).hexdigest()
    env_kwargs = dict(env_args["env_kwargs"])
    env_kwargs["bddl_file_name"] = _resolve_bddl(env_args, args.libero_root)
    env = TASK_MAPPING[env_args["problem_name"]](**env_kwargs)
    try:
        # Match the official LIBERO replay sequence exactly.
        env.reset()
        remapped_xml = _remap_model_xml(
            model_xml, args.libero_root / "libero" / "libero" / "assets"
        )
        remapped_xml_sha256 = hashlib.sha256(remapped_xml.encode("utf-8")).hexdigest()
        env.reset_from_xml_string(remapped_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        initial_state = env.sim.get_state().flatten().copy()
        model_fingerprint = _model_common_fingerprint(env.sim.model)
        body_pos_before = np.asarray(env.sim.model.body_pos).copy()

        substep_ctrl: list[np.ndarray] = []
        action_ctrl: list[np.ndarray] = []
        action_states: list[np.ndarray] = []
        qpos: list[np.ndarray] = []
        qvel: list[np.ndarray] = []
        success: list[bool] = []
        state_l2_to_hdf_next: list[float] = []

        original_step = env.sim.step

        def recording_step() -> None:
            substep_ctrl.append(np.asarray(env.sim.data.ctrl).copy())
            original_step()

        env.sim.step = recording_step
        for action_index, action in enumerate(actions):
            start = len(substep_ctrl)
            env.step(action)
            captured = np.asarray(substep_ctrl[start:])
            if captured.shape != (25, env.sim.model.nu):
                raise RuntimeError(
                    f"action {action_index} captured {captured.shape}, expected "
                    f"(25, {env.sim.model.nu})"
                )
            action_ctrl.append(captured)
            state = env.sim.get_state().flatten().copy()
            action_states.append(state)
            qpos.append(np.asarray(env.sim.data.qpos).copy())
            qvel.append(np.asarray(env.sim.data.qvel).copy())
            success.append(bool(env._check_success()))
            if action_index + 1 < len(states):
                state_l2_to_hdf_next.append(
                    float(np.linalg.norm(state - states[action_index + 1]))
                )

        body_pos_after = np.asarray(env.sim.model.body_pos)
        success_indices = np.flatnonzero(success)
        np.savez_compressed(
            args.output_npz,
            source_model_xml=np.asarray(model_xml),
            remapped_model_xml=np.asarray(remapped_xml),
            initial_fullphysics=initial_state,
            actions=actions,
            hdf_states=states,
            substep_ctrl=np.asarray(action_ctrl),
            action_fullphysics=np.asarray(action_states),
            qpos=np.asarray(qpos),
            qvel=np.asarray(qvel),
            success=np.asarray(success),
        )
        return {
            "protocol": "official-create-dataset-exact-model-state0-v1",
            "demo_hdf5": str(args.demo_hdf5),
            "demo": args.demo,
            "runtime": {
                "robosuite": robosuite.__version__,
                "mujoco": getattr(mujoco, "__version__", "unknown"),
            },
            "source_model_xml_sha256": source_xml_sha256,
            "remapped_model_xml_sha256": remapped_xml_sha256,
            "model_common_fingerprint": model_fingerprint,
            "model": {
                "nq": int(env.sim.model.nq),
                "nv": int(env.sim.model.nv),
                "nu": int(env.sim.model.nu),
            },
            "actions": len(actions),
            "ctrl_shape": list(np.asarray(action_ctrl).shape),
            "operationally_valid": True,
            "behavior": {
                "ever_success": bool(success_indices.size),
                "first_success": (
                    int(success_indices[0]) if success_indices.size else None
                ),
                "final_success": bool(success[-1]),
                "success_count": int(np.count_nonzero(success)),
            },
            "body_pos_max_abs_mutation": float(
                np.max(np.abs(body_pos_after - body_pos_before))
            ),
            "hdf_next_state_l2": {
                "count": len(state_l2_to_hdf_next),
                "max": float(np.max(state_l2_to_hdf_next)),
                "median": float(np.median(state_l2_to_hdf_next)),
                "first_over_0_01": next(
                    (
                        index
                        for index, value in enumerate(state_l2_to_hdf_next)
                        if value > 0.01
                    ),
                    None,
                ),
            },
            "artifact_npz": str(args.output_npz),
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-hdf5", type=Path, required=True)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--libero-root", type=Path, default=Path("/root/LIBERO"))
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    report = replay(args)
    args.output_json.write_text(
        json.dumps(report, indent=2, default=_jsonable) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()
