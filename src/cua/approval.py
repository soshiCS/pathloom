"""Human approval of a draft artifact, backed by stability reports.

Approval is a local review record, not a signature or an authentication step: it proves that
a named reviewer saw eligible stability evidence for exactly this artifact file, and it turns
the draft into the next immutable version with status "approved". The draft file is never
modified. Nothing here touches a surface or a model.

Every report must be well formed, reference the draft's exact SHA-256 digest, and be eligible
by recomputation from its own run records (stability.verify_report): the eligibility flag and
every summary field are re-derived, so an edited flag or total is refused. A report rewritten
consistently end to end would still pass; that is the limit of a local review record.

Coverage is derived from the graph itself, not from provenance. A schema 2.0 artifact with
selector inputs must start with an entry decision gate whose edges each carry exactly one
input_equals guard per selector; those guard values are the declared assignments, and one
eligible report is required per assignment, with no duplicates and none undeclared. Campaign
provenance, when present, must agree with the graph; stripping it never lowers the bar. Any
other artifact needs at least one eligible report.
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from . import artifact as artifact_module
from .artifact import ArtifactError
from .graph import nodes_by_id, outgoing, save_graph
from .lifecycle import APPROVED, DRAFT, load_any_version, sha256_of
from .models import ArtifactV2
from .stability import REPORT_SCHEMA_VERSION, ReportError, verify_report

REPORT_KEYS = {"report_schema_version", "stability_id", "artifact", "runs_requested", "runs_completed",
               "selector_assignment", "eligible_for_approval", "ineligible_reasons", "runs"}


class ApprovalError(ValueError):
    """The artifact or its evidence does not qualify for approval."""


def approve(artifact_path: str | Path, report_paths: list[str | Path], reviewer: str) -> Path:
    """Create the approved next version of a draft; returns the new file's path."""
    if not reviewer or not reviewer.strip():
        raise ApprovalError("a reviewer name is required")
    if not report_paths:
        raise ApprovalError("at least one stability report is required")
    draft = load_any_version(artifact_path)
    if draft.status != DRAFT:
        raise ApprovalError(f"{artifact_path} has status {draft.status!r}; only a draft can be approved")
    digest = sha256_of(artifact_path)
    declared = declared_assignments(draft)          # the artifact's own gate first: a bad graph fails fast
    reports = [load_report(path, draft, digest) for path in report_paths]
    tested = check_coverage(declared, reports)

    approval = {
        "reviewer": reviewer.strip(),
        "approved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_version": draft.version,
        "source_sha256": digest,
        "source_path": str(artifact_path),
        "reports": [{"path": str(path), "sha256": sha256_of(path), "stability_id": report["stability_id"],
                     "runs_completed": report["runs_completed"], "selector_assignment": report["selector_assignment"]}
                    for path, report in zip(report_paths, reports)],
        "tested_selector_assignments": tested,
    }
    provenance = {**copy.deepcopy(draft.provenance), "approval": approval}
    # The approved copy is a later version than its source, and never collides with a saved one.
    version = max(artifact_module.next_version(draft.name), draft.version + 1)
    approved = replace(draft, version=version, status=APPROVED, provenance=provenance)
    if isinstance(approved, ArtifactV2):
        return save_graph(approved)
    return artifact_module.save(approved)


def load_report(path: str | Path, draft, digest: str) -> dict:
    """A well-formed, eligible report about exactly this artifact file."""
    where = f"report {path}"
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ApprovalError(f"cannot read {where}: {error}") from error
    if not isinstance(report, dict):
        raise ApprovalError(f"{where}: must be a JSON object")
    missing = REPORT_KEYS - set(report)
    if missing:
        raise ApprovalError(f"{where}: missing keys {sorted(missing)}")
    if report["report_schema_version"] != REPORT_SCHEMA_VERSION:
        raise ApprovalError(f"{where}: unsupported report_schema_version {report['report_schema_version']!r}")
    about = report["artifact"]
    if not isinstance(about, dict) or not {"name", "version", "sha256"} <= set(about):
        raise ApprovalError(f"{where}: artifact section must carry name, version and sha256")
    if about["sha256"] != digest:
        raise ApprovalError(f"{where}: was produced for artifact digest {about['sha256'][:12]}..., but the draft's "
                            f"digest is {digest[:12]}...; the report is stale or about another artifact")
    if (about["name"], about["version"]) != (draft.name, draft.version):
        raise ApprovalError(f"{where}: is about {about['name']} v{about['version']}, not {draft.name} v{draft.version}")
    try:
        recomputed = verify_report(report)        # the summary and the verdict must follow from the run records
    except ReportError as error:
        raise ApprovalError(f"{where}: is inconsistent and cannot be trusted: {error}") from error
    if not recomputed["eligible_for_approval"]:
        raise ApprovalError(f"{where}: is not eligible for approval ({'; '.join(recomputed['ineligible_reasons'])})")
    if report["selector_assignment"] is not None and not isinstance(report["selector_assignment"], dict):
        raise ApprovalError(f"{where}: selector_assignment must be an object or null")
    return report


def check_coverage(declared: list[dict], reports: list[dict]) -> list[dict | None]:
    """Gated graphs need every declared selector assignment tested exactly once; others need one report."""
    tested = [report["selector_assignment"] for report in reports]
    if not declared:
        return tested
    seen: list[dict] = []
    for assignment in tested:
        if assignment is None:
            raise ApprovalError("a report without a selector assignment cannot cover a campaign graph")
        if assignment not in declared:
            raise ApprovalError(f"report tests an undeclared selector assignment {assignment}")
        if assignment in seen:
            raise ApprovalError(f"two reports cover the same selector assignment {assignment}; each declared "
                                f"assignment needs exactly one eligible report")
        seen.append(assignment)
    uncovered = [assignment for assignment in declared if assignment not in seen]
    if uncovered:
        raise ApprovalError(f"no stability report for selector assignment(s) {uncovered}")
    return tested


def declared_assignments(draft) -> list[dict]:
    """The selector assignments the artifact admits, read from its entry gate; empty when it has no selectors.

    Provenance is only cross-checked: it can never add to or remove from what the graph enforces.
    """
    selectors = [name for name, spec in draft.inputs.items() if spec.get("selector")] \
        if isinstance(draft, ArtifactV2) else []
    assignments = gate_assignments(draft, selectors) if selectors else []
    provenance = draft.provenance or {}
    recorded = [scenario.get("selectors") for scenario in provenance.get("scenarios") or []
                if isinstance(scenario, dict) and scenario.get("selectors")]
    if recorded or provenance.get("selectors"):
        if sorted(provenance.get("selectors") or []) != sorted(selectors):
            raise ApprovalError(f"provenance lists selectors {provenance.get('selectors')!r} but the artifact's "
                                f"selector inputs are {selectors!r}")
        if sorted(map(json.dumps, recorded)) != sorted(map(json.dumps, assignments)):
            raise ApprovalError(f"provenance records selector assignments {recorded} but the entry gate admits "
                                f"{assignments}")
    return assignments


def gate_assignments(graph: ArtifactV2, selectors: list[str]) -> list[dict]:
    """One complete selector assignment per edge leaving the entry decision gate."""
    entry = nodes_by_id(graph)[graph.entry_node]
    if entry.kind != "decision":
        raise ApprovalError(f"the artifact declares selector inputs {selectors} but its entry node {entry.id!r} is "
                            f"not a decision gate")
    assignments: list[dict] = []
    for edge in outgoing(graph, entry.id):
        where = f"entry gate edge {edge.source}->{edge.target} (priority {edge.priority})"
        if any(guard.kind != "input_equals" for guard in edge.guards):
            raise ApprovalError(f"{where}: only input_equals guards belong on the entry gate")
        inputs = [guard.input for guard in edge.guards]
        if len(set(inputs)) != len(inputs):
            raise ApprovalError(f"{where}: a selector is guarded twice")
        if set(inputs) != set(selectors):
            raise ApprovalError(f"{where}: guards {inputs} do not cover exactly the selectors {selectors}")
        assignment = {selector: next(g.value for g in edge.guards if g.input == selector) for selector in selectors}
        if assignment in assignments:
            raise ApprovalError(f"{where}: selector assignment {assignment} is admitted twice")
        assignments.append(assignment)
    return assignments
