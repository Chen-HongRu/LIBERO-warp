"""Frozen public runtime API for official and future Warp LIBERO backends."""

from __future__ import annotations

from .compiler import (
    CompiledTask,
    InitStateBank,
    PhysicsStateBatch,
    TaskCompiler,
    TaskRuntimeMetadata,
    gather_init_states,
    reset_all_worlds,
    scatter_init_states,
)
from .official import OfficialBatchEnv
from .types import (
    CameraConfig,
    EnvConfig,
    ObservationBatch,
    RenderExactState,
    RuntimeEnv,
    StepBatch,
)


def make_env(config: EnvConfig) -> RuntimeEnv:
    """Create one configured runtime environment.

    The official adapter accepts exactly one world. ``backend='warp'`` is
    deliberately reserved until the MJWarp backend exists rather than falling
    back to an implementation with different tensor/device semantics.
    """
    if not isinstance(config, EnvConfig):
        raise TypeError("make_env requires an EnvConfig instance.")
    if config.backend == "warp":
        raise NotImplementedError("The Warp runtime backend is not available yet.")
    return OfficialBatchEnv(config)


__all__ = [
    "CameraConfig",
    "CompiledTask",
    "EnvConfig",
    "InitStateBank",
    "ObservationBatch",
    "OfficialBatchEnv",
    "PhysicsStateBatch",
    "RenderExactState",
    "RuntimeEnv",
    "StepBatch",
    "TaskCompiler",
    "TaskRuntimeMetadata",
    "gather_init_states",
    "make_env",
    "reset_all_worlds",
    "scatter_init_states",
]
