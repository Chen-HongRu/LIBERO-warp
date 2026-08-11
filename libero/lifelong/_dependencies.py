"""Dependency guard for the restored legacy training surface."""

from __future__ import annotations

from importlib.util import find_spec


def require_legacy_dependencies() -> None:
    required = (
        "einops",
        "hydra",
        "robomimic",
        "thop",
        "torch",
        "torchvision",
        "transformers",
        "wandb",
    )
    missing = [name for name in required if find_spec(name) is None]
    if missing:
        names = ", ".join(missing)
        raise ImportError(
            f"libero.lifelong requires the legacy profile; missing: {names}. "
            "Install with `pip install libero-warp[legacy]`."
        )
