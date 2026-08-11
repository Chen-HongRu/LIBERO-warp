"""Compare current MuJoCo and MJWarp on legacy teacher-forced anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp


def _common_fingerprint(model: mujoco.MjModel) -> str:
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
    for field in fields:
        array = np.ascontiguousarray(
            np.asarray(getattr(model, field)).astype("<f8", copy=False)
        )
        digest.update(field.encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _run_current_cpu(
    model: mujoco.MjModel, states: np.ndarray, controls: np.ndarray
) -> dict[str, np.ndarray]:
    qpos = []
    qvel = []
    ncon_before = []
    ncon_after = []
    healthy = []
    for state, anchor_ctrl in zip(states, controls, strict=True):
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, data)
        ncon_before.append(data.ncon)
        for ctrl in anchor_ctrl:
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        ncon_after.append(data.ncon)
        healthy.append(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    return {
        "qpos": np.asarray(qpos),
        "qvel": np.asarray(qvel),
        "ncon_before": np.asarray(ncon_before),
        "ncon_after": np.asarray(ncon_after),
        "healthy": np.asarray(healthy),
    }


def _reset_warp(model, data, states: torch.Tensor, index: int) -> None:
    mjw.reset_data(model, data)
    mjw.set_state(
        model,
        data,
        wp.from_torch(states[index : index + 1]),
        int(mjw.State.FULLPHYSICS),
    )
    mjw.forward(model, data)


def _warp_readout(data) -> tuple[np.ndarray, np.ndarray, int, bool]:
    wp.synchronize()
    qpos = wp.to_torch(data.qpos)[0].cpu().numpy().copy()
    qvel = wp.to_torch(data.qvel)[0].cpu().numpy().copy()
    ncon = int(wp.to_torch(data.nacon)[0].cpu().item())
    healthy = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())
    return qpos, qvel, ncon, healthy


def _run_warp(
    mj_model: mujoco.MjModel,
    states_np: np.ndarray,
    controls_np: np.ndarray,
    *,
    graph: bool,
) -> dict[str, np.ndarray]:
    device = wp.get_device("cuda:0")
    states = torch.as_tensor(states_np, device="cuda", dtype=torch.float32).contiguous()
    controls = torch.as_tensor(
        controls_np, device="cuda", dtype=torch.float32
    ).contiguous()
    torch.cuda.synchronize()
    with wp.ScopedDevice(device):
        model = mjw.put_model(mj_model, batch_sizes={})
        data = mjw.make_data(mj_model, nworld=1, nconmax=512, naconmax=512, njmax=512)
        ctrl_target = wp.from_torch(wp.to_torch(data.ctrl).flatten())
        ctrl_source = wp.from_torch(controls.flatten())
        row_size = mj_model.nu

        # Materialize lazy kernels and allocations before optional capture.
        _reset_warp(model, data, states, 0)
        mjw.step(model, data)
        wp.synchronize()
        captured_graph = None
        if graph:
            _reset_warp(model, data, states, 0)
            with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                for substep in range(controls.shape[1]):
                    wp.copy(
                        ctrl_target,
                        ctrl_source,
                        src_offset=substep * row_size,
                        count=row_size,
                    )
                    mjw.step(model, data)
            captured_graph = capture.graph

        qpos = []
        qvel = []
        ncon_before = []
        ncon_after = []
        healthy = []
        for index in range(len(states_np)):
            _reset_warp(model, data, states, index)
            wp.synchronize()
            ncon_before.append(int(wp.to_torch(data.nacon)[0].cpu().item()))
            if captured_graph is None:
                for substep in range(controls.shape[1]):
                    wp.copy(
                        ctrl_target,
                        ctrl_source,
                        src_offset=(index * controls.shape[1] + substep) * row_size,
                        count=row_size,
                    )
                    mjw.step(model, data)
            else:
                # Graph copy nodes point at anchor zero. Move this anchor into that
                # stable staging region before launch; the cost is outside this
                # correctness-only diagnostic.
                controls[0].copy_(controls[index])
                torch.cuda.synchronize()
                wp.capture_launch(captured_graph)
            read_qpos, read_qvel, read_ncon, read_health = _warp_readout(data)
            qpos.append(read_qpos)
            qvel.append(read_qvel)
            ncon_after.append(read_ncon)
            healthy.append(read_health)
        return {
            "qpos": np.asarray(qpos),
            "qvel": np.asarray(qvel),
            "ncon_before": np.asarray(ncon_before),
            "ncon_after": np.asarray(ncon_after),
            "healthy": np.asarray(healthy),
        }


def _errors(reference: dict, candidate: dict) -> dict:
    qpos = np.max(np.abs(reference["qpos"] - candidate["qpos"]), axis=1)
    qvel = np.max(np.abs(reference["qvel"] - candidate["qvel"]), axis=1)
    return {
        "qpos_max_abs_per_anchor": qpos.tolist(),
        "qvel_max_abs_per_anchor": qvel.tolist(),
        "qpos_max_abs": float(qpos.max()),
        "qvel_max_abs": float(qvel.max()),
        "first_qpos_over_1e-3": next(
            (index for index, value in enumerate(qpos) if value > 1e-3), None
        ),
        "all_healthy": bool(np.all(candidate["healthy"])),
        "ncon_before": candidate["ncon_before"].tolist(),
        "ncon_after": candidate["ncon_after"].tolist(),
    }


def run(args: argparse.Namespace) -> dict:
    artifact = np.load(args.input_npz, allow_pickle=False)
    remapped_xml = str(artifact["remapped_model_xml"].item())
    states = artifact["input_fullphysics"]
    controls = artifact["substep_ctrl"]
    legacy = {
        "qpos": artifact["legacy_qpos"],
        "qvel": artifact["legacy_qvel"],
    }
    model = mujoco.MjModel.from_xml_string(remapped_xml)
    current_cpu = _run_current_cpu(model, states, controls)
    warp_eager = _run_warp(model, states, controls, graph=False)
    warp_graph25 = _run_warp(model, states, controls, graph=True)
    report = {
        "qualification_tier": "tier1_teacher_forced_local_dynamics_only",
        "runtime": {
            "mujoco": mujoco.__version__,
            "mujoco_warp": mjw.__version__,
            "warp": wp.__version__,
        },
        "model": {
            "common_fingerprint_sha256": _common_fingerprint(model),
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
        },
        "anchor_indices": artifact["anchor_indices"].tolist(),
        "current_cpu_vs_legacy": _errors(legacy, current_cpu),
        "warp_eager_vs_legacy": _errors(legacy, warp_eager),
        "warp_eager_vs_current_cpu": _errors(current_cpu, warp_eager),
        "warp_graph25_vs_eager": _errors(warp_eager, warp_graph25),
    }
    np.savez_compressed(
        args.output_npz,
        anchor_indices=artifact["anchor_indices"],
        current_cpu_qpos=current_cpu["qpos"],
        current_cpu_qvel=current_cpu["qvel"],
        warp_eager_qpos=warp_eager["qpos"],
        warp_eager_qvel=warp_eager["qvel"],
        warp_graph25_qpos=warp_graph25["qpos"],
        warp_graph25_qvel=warp_graph25["qvel"],
    )
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
