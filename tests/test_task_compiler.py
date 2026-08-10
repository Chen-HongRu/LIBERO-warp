"""Contracts for M1 task compilation and trusted init-state preparation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.ctrl_trace import model_arrays_fingerprint, model_mjb_fingerprint

from ._manifest import LIBERO_TASK_MAP


def _runtime_compiler_api():
    from libero.libero.runtime import (
        CameraConfig,
        EnvConfig,
        InitStateBank,
        TaskCompiler,
        gather_init_states,
        reset_all_worlds,
        scatter_init_states,
    )

    return (
        CameraConfig,
        EnvConfig,
        InitStateBank,
        TaskCompiler,
        gather_init_states,
        reset_all_worlds,
        scatter_init_states,
    )


def _init_state_bank():
    *_, InitStateBank, _, _, _, _ = _runtime_compiler_api()
    fullphysics = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [20.0, 21.0, 22.0, 23.0, 24.0],
        ],
        dtype=torch.float64,
    )
    return InitStateBank(
        legacy_flattened=fullphysics.clone(),
        fullphysics=fullphysics,
        time=fullphysics[:, 0],
        qpos=fullphysics[:, :2],
        qvel=fullphysics[:, :3],
        act=fullphysics[:, :1],
    )


def _assert_random_state_equal(
    actual: tuple[object, ...], expected: tuple[object, ...]
) -> None:
    """Assert exact preservation of the caller-owned legacy NumPy RNG state."""

    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


@pytest.mark.static
def test_public_compiler_api_exports_trusted_state_utilities() -> None:
    from libero.libero import runtime

    required_symbols = {
        "TaskCompiler",
        "CompiledTask",
        "TaskRuntimeMetadata",
        "InitStateBank",
        "PhysicsStateBatch",
        "gather_init_states",
        "scatter_init_states",
        "reset_all_worlds",
    }
    assert required_symbols <= set(runtime.__all__)
    assert all(callable(getattr(runtime, symbol)) for symbol in required_symbols)


@pytest.mark.static
def test_init_state_gather_and_partial_scatter_preserve_selected_world_mapping() -> (
    None
):
    *_, gather_init_states, _, scatter_init_states = _runtime_compiler_api()
    bank = _init_state_bank()

    gathered = gather_init_states(bank, torch.tensor([2, 0], dtype=torch.int64))
    torch.testing.assert_close(gathered.fullphysics, bank.fullphysics[[2, 0]])
    torch.testing.assert_close(gathered.qpos, bank.qpos[[2, 0]])
    assert gathered.fullphysics.device.type == "cpu"

    target = torch.full((4, 5), -1.0, dtype=torch.float64)
    scattered = scatter_init_states(
        target,
        bank,
        world_ids=torch.tensor([3, 1], dtype=torch.int64),
        state_indices=torch.tensor([2, 0], dtype=torch.int64),
    )
    assert scattered.data_ptr() == target.data_ptr()
    torch.testing.assert_close(target[3], bank.fullphysics[2])
    torch.testing.assert_close(target[1], bank.fullphysics[0])
    torch.testing.assert_close(
        target[[0, 2]], torch.full((2, 5), -1.0, dtype=torch.float64)
    )

    with pytest.raises(ValueError, match="Duplicate world_ids"):
        scatter_init_states(
            target,
            bank,
            world_ids=torch.tensor([1, 1], dtype=torch.int64),
            state_indices=torch.tensor([0, 1], dtype=torch.int64),
        )


@pytest.mark.static
def test_reset_all_worlds_requires_one_trusted_index_per_world() -> None:
    *_, _, reset_all_worlds, _ = _runtime_compiler_api()
    bank = _init_state_bank()
    target = torch.full((3, 5), -1.0, dtype=torch.float64)

    reset_all_worlds(target, bank, torch.tensor([1, 2, 0], dtype=torch.int64))
    torch.testing.assert_close(target, bank.fullphysics[[1, 2, 0]])
    with pytest.raises(ValueError, match="one state index"):
        reset_all_worlds(target, bank, torch.tensor([0, 1], dtype=torch.int64))


@pytest.mark.static
def test_init_state_bank_requires_explicit_one_time_device_and_dtype_transfer() -> None:
    *_, gather_init_states, _, scatter_init_states = _runtime_compiler_api()
    bank = _init_state_bank()
    float32_bank = bank.to("cpu", dtype=torch.float32)
    assert bank.device.type == "cpu"
    assert bank.dtype == torch.float64
    assert float32_bank.device.type == "cpu"
    assert float32_bank.dtype == torch.float32

    gathered = gather_init_states(float32_bank, torch.tensor([2, 0], dtype=torch.int64))
    assert gathered.fullphysics.dtype == torch.float32
    target = torch.full((3, 5), -1.0, dtype=torch.float32)
    scatter_init_states(
        target,
        float32_bank,
        world_ids=torch.tensor([2, 0], dtype=torch.int64),
        state_indices=torch.tensor([1, 2], dtype=torch.int64),
    )
    torch.testing.assert_close(target[2], float32_bank.fullphysics[1])
    torch.testing.assert_close(target[0], float32_bank.fullphysics[2])

    with pytest.raises(ValueError, match="Upload once"):
        gather_init_states(bank, torch.tensor([0], dtype=torch.int64), device="meta")
    with pytest.raises(ValueError, match="Cast once"):
        scatter_init_states(
            target,
            bank,
            world_ids=torch.tensor([1], dtype=torch.int64),
            state_indices=torch.tensor([0], dtype=torch.int64),
        )


@pytest.mark.static
def test_init_state_bank_cuda_transfer_is_device_local_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; CPU/meta mismatch is covered above")
    *_, gather_init_states, _, scatter_init_states = _runtime_compiler_api()
    device = torch.device("cuda")
    bank = _init_state_bank().to(device, dtype=torch.float32)
    indices = torch.tensor([2, 0], dtype=torch.int64, device=device)
    gathered = gather_init_states(bank, indices)
    assert gathered.fullphysics.device == device
    target = torch.full((3, 5), -1.0, dtype=torch.float32, device=device)
    scatter_init_states(target, bank, indices, indices)
    torch.testing.assert_close(target[2], bank.fullphysics[2])
    with pytest.raises(ValueError, match="Upload once"):
        gather_init_states(_init_state_bank(), indices)


@pytest.mark.official_integration
def test_task_compiler_pilot_uses_exact_official_model_and_metadata() -> None:
    import mujoco
    import numpy as np

    CameraConfig, EnvConfig, _, TaskCompiler, *_ = _runtime_compiler_api()
    config = EnvConfig(
        suite="libero_spatial",
        task_index=0,
        cameras=[CameraConfig("agentview", 48, 64)],
        horizon=1000,
    )
    compiled = TaskCompiler().compile(config)
    try:
        metadata = compiled.metadata
        assert metadata.task_name == LIBERO_TASK_MAP["libero_spatial"][0]
        assert isinstance(metadata.model, mujoco.MjModel)
        assert metadata.model is compiled.official_env._env.sim.model._model
        assert metadata.timestep == pytest.approx(0.002)
        assert metadata.control_frequency == 20
        assert metadata.control_substeps == 25
        assert (metadata.nq, metadata.nv, metadata.nu, metadata.na) == (
            metadata.model.nq,
            metadata.model.nv,
            metadata.model.nu,
            metadata.model.na,
        )
        assert metadata.body_ids
        assert metadata.site_ids
        assert metadata.geom_ids
        assert metadata.joint_ids
        assert metadata.camera_ids
        assert metadata.actuator_ids
        assert metadata.requested_camera_ids == {
            "agentview": metadata.camera_ids["agentview"]
        }
        with pytest.raises(TypeError):
            metadata.body_ids["unexpected"] = 0

        bank = compiled.init_state_bank
        assert bank.size > 0
        assert bank.fullphysics.shape == bank.legacy_flattened.shape
        assert bank.fullphysics.shape[1] == metadata.fullphysics_state_size
        assert bank.qpos.shape == (bank.size, metadata.nq)
        assert bank.qvel.shape == (bank.size, metadata.nv)
        assert bank.act.shape == (bank.size, metadata.na)
        assert bank.fullphysics.device.type == "cpu"
        assert torch.isfinite(bank.fullphysics).all()

        for state_index in (0, bank.size - 1):
            legacy_row = bank.legacy_flattened[state_index]
            compiled.official_env.reset(init_state=legacy_row)
            data = compiled.official_env._env.sim.data._data
            independently_read_fullphysics = np.empty(
                metadata.fullphysics_state_size, dtype=np.float64
            )
            mujoco.mj_getState(
                metadata.model,
                data,
                independently_read_fullphysics,
                mujoco.mjtState.mjSTATE_FULLPHYSICS,
            )
            torch.testing.assert_close(
                bank.fullphysics[state_index],
                torch.from_numpy(independently_read_fullphysics),
            )
            torch.testing.assert_close(
                bank.qpos[state_index], torch.from_numpy(data.qpos.copy())
            )
            torch.testing.assert_close(
                bank.qvel[state_index], torch.from_numpy(data.qvel.copy())
            )
            assert bank.time[state_index].item() == pytest.approx(data.time)

        compiled.official_env.reset(init_state=bank.fullphysics[0])
        assert compiled.get_state().shape == (1, metadata.fullphysics_state_size)
        assert compiled.get_proprio().shape[0] == 1
        assert compiled.get_sim_time().shape == (1,)
    finally:
        compiled.close()


@pytest.mark.official_integration
def test_task_compiler_uses_one_official_world_for_a_warp_target() -> None:
    import mujoco

    CameraConfig, EnvConfig, _, TaskCompiler, *_ = _runtime_compiler_api()
    target_config = EnvConfig(
        suite="libero_spatial",
        task_index=0,
        cameras=[CameraConfig("agentview", 32, 48)],
        backend="warp",
        num_worlds=128,
    )
    compiled = TaskCompiler().compile(target_config)
    try:
        assert target_config.backend == "warp"
        assert target_config.num_worlds == 128
        assert compiled.official_env.config.backend == "official"
        assert compiled.official_env.config.num_worlds == 1
        assert compiled.metadata.target_backend == "warp"
        assert compiled.metadata.target_num_worlds == 128
        assert isinstance(compiled.metadata.model, mujoco.MjModel)
        assert compiled.metadata.model is compiled.official_env._env.sim.model._model
    finally:
        compiled.close()


@pytest.mark.official_integration
def test_seed_zero_reset_sequence_has_a_stable_in_memory_mjb_hash() -> None:
    from libero.libero.runtime import make_env

    CameraConfig, EnvConfig, _, TaskCompiler, *_ = _runtime_compiler_api()
    config = EnvConfig(
        suite="libero_spatial",
        task_index=0,
        cameras=[CameraConfig("agentview", 32, 32)],
        seed=0,
    )
    caller_random_state = np.random.get_state()
    first = make_env(config)
    second = make_env(config)
    try:
        reset_hashes = []
        reset_array_hashes = []
        for _ in range(2):
            first.reset()
            second.reset()
            first_hash, first_size = model_mjb_fingerprint(first._env.sim.model._model)
            second_hash, second_size = model_mjb_fingerprint(
                second._env.sim.model._model
            )
            first_arrays = model_arrays_fingerprint(first._env.sim.model._model)
            second_arrays = model_arrays_fingerprint(second._env.sim.model._model)
            assert first_hash == second_hash
            assert first_size == second_size
            assert first_arrays["sha256"] == second_arrays["sha256"]
            reset_hashes.append(first_hash)
            reset_array_hashes.append(first_arrays["sha256"])

        # The retained legacy wrapper API must accept a seed argument and use
        # it locally: resetting after each explicit re-seed is reproducible.
        first._env.seed(0)
        first.reset()
        legacy_mjb_hash, legacy_mjb_size = model_mjb_fingerprint(
            first._env.sim.model._model
        )
        legacy_arrays_hash = model_arrays_fingerprint(first._env.sim.model._model)[
            "sha256"
        ]
        first._env.seed(0)
        first.reset()
        repeated_mjb_hash, repeated_mjb_size = model_mjb_fingerprint(
            first._env.sim.model._model
        )
        repeated_arrays_hash = model_arrays_fingerprint(first._env.sim.model._model)[
            "sha256"
        ]
        assert (repeated_mjb_hash, repeated_mjb_size) == (
            legacy_mjb_hash,
            legacy_mjb_size,
        )
        assert repeated_arrays_hash == legacy_arrays_hash

        # A compiler creates its single official environment at the same first
        # seeded construction boundary as reset_hashes[0].  Comparing it to the
        # second reset would incorrectly compare the next deterministic random
        # placement sequence with a fresh compiler.
        compiled = TaskCompiler().compile(config)
        second_compiled = TaskCompiler().compile(config)
        try:
            compiled_hash, compiled_size = model_mjb_fingerprint(compiled.model)
            compiled_arrays = model_arrays_fingerprint(compiled.model)
            second_compiled_hash, second_compiled_size = model_mjb_fingerprint(
                second_compiled.model
            )
            second_compiled_arrays = model_arrays_fingerprint(second_compiled.model)
            assert compiled_hash == reset_hashes[0]
            assert compiled_size == first_size
            assert compiled_arrays["sha256"] == reset_array_hashes[0]
            assert second_compiled_hash == compiled_hash
            assert second_compiled_size == compiled_size
            assert second_compiled_arrays["sha256"] == compiled_arrays["sha256"]
        finally:
            second_compiled.close()
            compiled.close()
    finally:
        second.close()
        first.close()
    _assert_random_state_equal(np.random.get_state(), caller_random_state)
