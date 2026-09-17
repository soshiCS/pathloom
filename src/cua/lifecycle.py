"""The artifact lifecycle around replay: loading, the unattended gate, evidence bundles.

    draft -> supervised testing / stability runs -> human approval -> approved -> unattended replay

A draft may be replayed with a person watching (--operator console) or by the stability
command, which is the verification workflow. Unattended replay (--operator none) requires an
approved artifact and otherwise stops before any surface exists, with a structured failure and
an evidence bundle like any other run. Approval is a local review record (approval.py), not a
signature: anyone with write access to artifacts/ could produce one.

Every replay states its purpose at this boundary: "supervised" (a person is watching, drafts
allowed), "stability" (the verification workflow, drafts allowed, irreversible policy forced to
deny), "unattended" (approved artifacts only; a draft is refused before the engine runs, so a
surface the caller already opened receives no action) or "discovery_reuse" (an approved prefix
replayed at the start of a discovery: approved only, irreversible policy forced to deny). The CLI
checks the same rule earlier, before it opens a browser at all.

Every replay leaves a reproducible bundle: run.jsonl and screenshots (RunLog), result.json
(redacted with the run's secret values, like the log), a byte-exact copy of the artifact that
ran, and manifest.json naming the artifact, its schema and capability versions, its status and
its SHA-256 digest. Parameter values are never written; only their names.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from . import artifact as artifact_module
from .escalation import Escalator
from .evidence import RunLog
from .models import Artifact, ReplayResult
from .policy import Policy, redact
from .replay import DEFAULT_IRREVERSIBLE_POLICY, failure, replay
from .surface import Surface

APPROVED = "approved"
DRAFT = "draft"
MANIFEST_SCHEMA_VERSION = "1.0"
SUPERVISED, STABILITY, UNATTENDED, DISCOVERY_REUSE = "supervised", "stability", "unattended", "discovery_reuse"
PURPOSES = (SUPERVISED, STABILITY, UNATTENDED, DISCOVERY_REUSE)
STABILITY_IRREVERSIBLE_POLICY = "deny"


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_artifact(path: str | Path) -> Artifact:
    """A validated capability graph whose file name agrees with the capability and revision it holds."""
    loaded = artifact_module.load_artifact(path)
    artifact_module.check_path_names_artifact(path, loaded)
    return loaded


def is_approved(loaded: Artifact) -> bool:
    return loaded.status == APPROVED


def not_approved_result(loaded: Artifact) -> ReplayResult:
    """The structured failure an unattended replay of a draft returns, before any surface exists."""
    return failure("artifact_not_approved", None, f"an artifact with status {APPROVED!r}",
                   f"{loaded.name} v{loaded.version} has status {loaded.status!r}; run it with --operator console "
                   f"to test it supervised, or approve it after stability runs")


def run_replay(loaded: Artifact, params: dict, surface: Surface, policy: Policy, escalator: Escalator, log: RunLog,
               purpose: str, irreversible_policy: str = DEFAULT_IRREVERSIBLE_POLICY) -> ReplayResult:
    """Replay for an explicit purpose (see PURPOSES); the one entry point every lifecycle operation uses.

    An unattended replay of a draft returns artifact_not_approved without touching the surface.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    if purpose in (UNATTENDED, DISCOVERY_REUSE) and not is_approved(loaded):
        result = not_approved_result(loaded)
        log.event("replay_blocked", capability=loaded.name, version=loaded.version, status=loaded.status,
                  purpose=purpose, outcome_code=result.outcome_code, reason=result.observed)
        return result
    if purpose in (STABILITY, DISCOVERY_REUSE):
        irreversible_policy = STABILITY_IRREVERSIBLE_POLICY   # never repeat an irreversible or unknown action
    return replay(loaded, params, surface, policy, escalator, log, irreversible_policy=irreversible_policy)


def policy_for(loaded: Artifact, allowed_hosts: list[str] | None = None) -> Policy:
    """The allowlist a replay runs under: an explicit list, else the artifact's, else its entry host."""
    from urllib.parse import urlparse
    hosts = allowed_hosts or list(loaded.surface.get("allowed_hosts") or []) \
        or [urlparse(loaded.surface["entry_url"]).hostname or ""]
    return Policy(allowed_hosts=hosts)


def write_bundle(log: RunLog, loaded: Artifact, artifact_path: str | Path, result: ReplayResult, params: dict,
                 sensitive: list[str] | tuple[str, ...] = ()) -> Path:
    """Copy the run into evidence/ with result.json, the exact artifact that ran, and a manifest."""
    evidence_dir = log.copy_to_evidence()
    # The persisted copy is scrubbed with the run's real secret values (outputs, error text, human actions);
    # the caller keeps the unredacted result.
    persisted = redact(asdict(result), log.secrets)
    (evidence_dir / "result.json").write_text(json.dumps(persisted, indent=2), encoding="utf-8")
    source = Path(artifact_path)
    shutil.copyfile(source, evidence_dir / source.name)
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": log.run_id,
        "artifact_path": str(source),
        "artifact_file": source.name,
        "sha256": sha256_of(source),
        "capability": loaded.name,
        "capability_version": loaded.version,
        "schema_version": loaded.schema_version,
        "status": loaded.status,
        "result_status": result.status,
        "outcome_code": result.outcome_code,
        "param_names": sorted(params),
        "sensitive_params": sorted(name for name in sensitive if name in params),
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return evidence_dir
