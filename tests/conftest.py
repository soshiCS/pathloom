"""Keep test runs from writing into the real logs/ and evidence/ folders."""
import pytest

from src.cua import artifact, evidence, replay


@pytest.fixture(autouse=True)
def isolated_output_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(evidence, "EVIDENCE_DIR", tmp_path / "evidence")
    monkeypatch.setattr(artifact, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(replay, "CHECKPOINT_TIMEOUT_S", 0.0)   # the fake surface never needs to settle
    monkeypatch.setattr(replay, "GUARD_TIMEOUT_S", 0.0)
    yield tmp_path
