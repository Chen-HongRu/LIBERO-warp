"""Regenerate the frozen G1 official-backend headless golden fixture."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from tests.conftest import BENCHMARK_ROOT

FIXTURE_ROOT = Path(__file__).resolve().parent
METADATA_PATH = FIXTURE_ROOT / "g1_official_headless_v1.json"
ARRAYS_PATH = FIXTURE_ROOT / "g1_official_headless_v1.npz"
TASK_NAME = (
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
)
SEED = 17


def _capture_observation(
    prefix: str, observation: dict[str, np.ndarray], arrays: dict[str, np.ndarray]
) -> dict[str, str]:
    mapping = {}
    for key, value in sorted(observation.items()):
        array_name = f"{prefix}__{key}"
        arrays[array_name] = np.asarray(value)
        mapping[key] = array_name
    return mapping


def main() -> None:
    from libero.libero.envs.env_wrapper import ControlEnv

    bddl_path = BENCHMARK_ROOT / "bddl_files" / "libero_spatial" / f"{TASK_NAME}.bddl"
    env = ControlEnv(
        bddl_file_name=str(bddl_path),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        seed=SEED,
    )
    try:
        arrays: dict[str, np.ndarray] = {}
        reset_observation = env.reset()
        arrays["reset_state"] = env.get_sim_state()
        reset_arrays = _capture_observation("reset", reset_observation, arrays)
        env.set_init_state(arrays["reset_state"])

        action = np.zeros(env.env.action_dim, dtype=np.float64)
        arrays["action"] = action
        step_observation, reward, done, info = env.step(action)
        arrays["step_state"] = env.get_sim_state()
        step_arrays = _capture_observation("step", step_observation, arrays)
        backend_info = env.backend_info.to_dict()
    finally:
        env.close()

    np.savez_compressed(ARRAYS_PATH, **arrays)
    metadata = {
        "schema_version": 1,
        "compatibility_target": backend_info["libero_compat_target"],
        "backend_argument": None,
        "actual_backend": backend_info["actual_backend"],
        "task_suite": "libero_spatial",
        "task_name": TASK_NAME,
        "seed": SEED,
        "use_camera_obs": False,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "dependency_versions": {
            "mujoco": version("mujoco"),
            "numpy": version("numpy"),
            "robosuite": version("robosuite"),
        },
        "reset_observation_arrays": reset_arrays,
        "step_observation_arrays": step_arrays,
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
