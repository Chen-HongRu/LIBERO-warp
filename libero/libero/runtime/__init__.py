"""Frozen public runtime API for official and Warp LIBERO backends."""

from __future__ import annotations

from .compiler import (
    CompiledTask,
    InitStateBank,
    PhysicsStateBatch,
    TaskCompiler,
    TaskRuntimeMetadata,
    gather_init_states,
    remap_demo_model_xml_assets,
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

    Both adapters currently accept exactly one world.  Importing the Warp
    implementation is deferred so an official-only installation does not need
    Torch CUDA, Warp, or MJWarp at module import time.
    """
    if not isinstance(config, EnvConfig):
        raise TypeError("make_env requires an EnvConfig instance.")
    if config.backend == "warp":
        from .warp import WarpBatchEnv

        return WarpBatchEnv(config)
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
    "remap_demo_model_xml_assets",
    "reset_all_worlds",
    "scatter_init_states",
]
