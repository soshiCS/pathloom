"""The artifact lifecycle around replay: loading either schema, the unattended gate, evidence bundles.

    draft -> supervised testing / stability runs -> human approval -> approved -> unattended replay

A draft may be replayed with a person watching (--operator console) or by the stability
command, which is the verification workflow. Unattended replay (--operator none) requires an
approved artifact and otherwise stops before any surface exists, with a structured failure and
an evidence bundle like any other run. Approval is a local review record (approval.py), not a
signature: anyone with write access to artifacts/ could produce one.

Every replay states its purpose at this boundary: "supervised" (a person is watching, drafts
allowed), "stability" (the verification workflow, drafts allowed, irreversible policy forced to
deny) or "unattended" (approved artifacts only; a draft is refused before either engine runs, so
a surface the caller already opened receives no action). The CLI checks the same rule earlier,
before it opens a browser at all.

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
from . import graph as graph_module
from .escalation import Escalator
from .evidence import RunLog
from .graph_replay import DEFAULT_IRREVERSIBLE_POLICY, replay_graph
from .models import Artifact, ArtifactV2, ReplayResult
from .policy import Policy, redact
from .replay import failure, replay
from .surface import Surface

APPROVED = "approved"
DRAFT = "draft"
MANIFEST_SCHEMA_VERSION = "1.0"
SUPERVISED, STABILITY, UNATTENDED = "supervised", "stability", "unattended"
PURPOSES = (SUPERVISED, STABILITY, UNATTENDED)
STABILITY_IRREVERSIBLE_POLICY = "deny"


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_any_version(path: str | Path) -> Artifact | ArtifactV2:
    """A 1.0 artifact for the linear engine, or a 2.0 graph for the graph engine; never converted."""
    data = artifact_module.read_json(path)
    version = data.get("schema_version") if isinstance(data, dict) else None
    if version == graph_module.GRAPH_SCHEMA_VERSION:
        return graph_module.load_graph(path)
    return artifact_module.load(path)


def is_approved(loaded: Artifact | ArtifactV2) -> bool:
    return loaded.status == APPROVED


def not_approved_result(loaded: Artifact | ArtifactV2) -> ReplayResult:
    """The structured failure an unattended replay of a draft returns, before any surface exists."""
    return failure("artifact_not_approved", None, f"an artifact with status {APPROVED!r}",
                   f"{loaded.name} v{loaded.version} has status {loaded.status!r}; run it with --operator console "
                   f"to test it supervised, or approve it after stability runs")


def replay_any(loaded: Artifact | ArtifactV2, params: dict, surface: Surface, policy: Policy, escalator: Escalator,
               log: RunLog, purpose: str, irreversible_policy: str = DEFAULT_IRREVERSIBLE_POLICY) -> ReplayResult:
    """Run the engine that matches the artifact's schema, for an explicit purpose (see PURPOSES).

    An unattended replay of a draft returns artifact_not_approved without touching the surface.
    """
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    if purpose == UNATTENDED and not is_approved(loaded):
        result = not_approved_result(loaded)
        log.event("replay_blocked", capability=loaded.name, version=loaded.version, status=loaded.status,
                  purpose=purpose, outcome_code=result.outcome_code, reason=result.observed)
        return result
    if purpose == STABILITY:
        irreversible_policy = STABILITY_IRREVERSIBLE_POLICY   # never repeat an irreversible or unknown action
    if isinstance(loaded, ArtifactV2):
        return replay_graph(loaded, params, surface, policy, escalator, log, irreversible_policy=irreversible_policy)
    return replay(loaded, params, surface, policy, escalator, log)


def policy_for(loaded: Artifact | ArtifactV2, allowed_hosts: list[str] | None = None) -> Policy:
    """The allowlist a replay runs under: an explicit list, else the artifact's, else its entry host."""
    from urllib.parse import urlparse
    hosts = allowed_hosts or list(loaded.surface.get("allowed_hosts") or []) \
        or [urlparse(loaded.surface["entry_url"]).hostname or ""]
    return Policy(allowed_hosts=hosts)


def write_bundle(log: RunLog, loaded: Artifact | ArtifactV2, artifact_path: str | Path, result: ReplayResult,
                 params: dict, sensitive: list[str] | tuple[str, ...] = ()) -> Path:
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
