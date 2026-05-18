from __future__ import annotations

from pathlib import Path

from vlm_micro.cli import summarize_artifacts


def test_summarize_existing_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"abc")

    summary = summarize_artifacts([str(artifact)])[0]

    assert summary.exists is True
    assert summary.bytes == 3
    assert summary.sha256 == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_summarize_missing_artifact() -> None:
    summary = summarize_artifacts(["missing.bin"])[0]

    assert summary.exists is False
    assert summary.bytes is None
    assert summary.sha256 is None
