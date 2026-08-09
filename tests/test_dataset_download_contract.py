"""Offline contracts for the retained public dataset downloader."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.official_integration
def test_huggingface_download_uses_the_locked_hub_127_signature(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep the optional downloader compatible without issuing a network request.

    ``huggingface-hub==1.27`` removed ``local_dir_use_symlinks``.  This strict
    stand-in also rejects an unconditional ``force_download`` so normal runs
    preserve the Hub cache and avoid needless transfers.
    """

    from libero.libero.utils import download_utils

    calls: dict[str, object] = {}

    def snapshot_download(
        repo_id: str,
        *,
        repo_type: str,
        local_dir: str,
        allow_patterns: str,
    ) -> Path:
        calls.update(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
        )
        return Path(local_dir)

    monkeypatch.setattr(download_utils, "HUGGINGFACE_AVAILABLE", True)
    monkeypatch.setattr(download_utils, "snapshot_download", snapshot_download)

    destination = tmp_path / "datasets"
    download_utils.download_from_huggingface(
        "libero_spatial", str(destination), check_overwrite=False
    )

    assert calls == {
        "repo_id": download_utils.HF_REPO_ID,
        "repo_type": "dataset",
        "local_dir": str(destination),
        "allow_patterns": "libero_spatial/*",
    }
