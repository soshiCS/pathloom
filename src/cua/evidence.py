"""Structured run logging and evidence collection.

Every run (discovery or replay) gets one JSONL log under logs/ and one screenshot
folder. Every event passes through redaction before it touches disk. When the run
finishes, the log and screenshots are copied into evidence/ as a self-contained bundle.
"""
from __future__ import annotations

import json
import secrets as _secrets
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .policy import redact

LOGS_DIR = Path("logs")
EVIDENCE_DIR = Path("evidence")


class RunLog:
    """Per-run JSONL log plus screenshot collector."""

    run_id: str
    path: Path

    def __init__(self, kind: str, secrets: tuple[str, ...] = (), echo: bool = False):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.kind = kind
        self.run_id = f"{kind}-{stamp}-{_secrets.token_hex(2)}"
        self.secrets = tuple(s for s in secrets if s)
        self.echo = echo
        self.seq = 0
        LOGS_DIR.mkdir(exist_ok=True)
        self.path = LOGS_DIR / f"{self.run_id}.jsonl"
        self.shots_dir = LOGS_DIR / self.run_id
        self.started = time.time()

    def event(self, name: str, **data) -> None:
        """Append one redacted, timestamped event."""
        self.seq += 1
        record = {
            "seq": self.seq,
            "t": round(time.time() - self.started, 3),
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "event": name,
            **redact(data, self.secrets),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.echo:
            summary = {k: v for k, v in record.items() if k not in ("seq", "t", "ts", "run_id", "event")}
            print(f"[{record['t']:7.3f}s] {name} {json.dumps(summary, ensure_ascii=False)[:300]}",
                  file=sys.stderr)

    def screenshot(self, surface, label: str) -> str:
        """Capture the current surface state; returns the file path (or "" if unavailable)."""
        self.shots_dir.mkdir(parents=True, exist_ok=True)
        path = self.shots_dir / f"{self.seq:03d}-{label}.png"
        try:
            surface.screenshot(str(path))
        except Exception as error:  # evidence capture must never break the run
            self.event("screenshot_failed", label=label, error=str(error))
            return ""
        self.event("screenshot", label=label, path=str(path))
        return str(path)

    def copy_to_evidence(self) -> Path:
        """Copy the completed log and screenshots into evidence/<run_id>/."""
        destination = EVIDENCE_DIR / self.run_id
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.path, destination / "run.jsonl")
        if self.shots_dir.exists():
            for shot in sorted(self.shots_dir.glob("*.png")):
                shutil.copy2(shot, destination / shot.name)
        return destination
