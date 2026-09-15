"""Multi-run stability verification: replay one invocation N times, unattended, and report.

Each run gets a fresh surface, session control, run log, replay context and evidence bundle.
Nothing is shared between runs but the artifact file and the parameters. The irreversible
policy is always "deny": a stability run must never repeat an irreversible action, and an
unknown effect is refused the same way, so a flow with such a node fails safely and stays
ineligible. No planner or LLM is involved.

Deliberate limitation: this verifies prepare-only or read-only flows that finish before any
irreversible effect. Repeating a flow that really commits (an order, a transfer) would need a
sandbox, an idempotency key, or a rollback mechanism, none of which exist here.

report.json (report_schema_version 1.0) records the artifact identity and SHA-256 digest, the
requested and completed runs, parameter and sensitive-parameter names (never values), the
non-sensitive selector assignment when the artifact declares selectors, counts by status and
outcome, success and clean-run rates, totals of recoveries, interventions and locator drift
signals, one record per run linking its evidence, and eligible_for_approval.

Every summary field is derived from the run records by `summarize`, and `verify_report`
re-derives them the same way, so approval never trusts a summary or the eligibility flag on
its own. This is consistency checking, not a signature: whoever rewrites the whole report
consistently can still forge it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .escalation import Escalator, NoOperator, SessionControl
from .evidence import RunLog
from .lifecycle import (STABILITY, STABILITY_IRREVERSIBLE_POLICY, load_any_version, policy_for, replay_any, sha256_of,
                        write_bundle)
from .models import Artifact, ArtifactV2, ReplayResult
from .policy import redact
from .surface import Surface

REPORT_SCHEMA_VERSION = "1.0"
MIN_RUNS_FOR_APPROVAL = 3
RUN_KEYS = {"run", "run_id", "status", "outcome_code", "step_id", "evidence", "artifact_sha256", "recoveries",
            "interventions", "drift_signals", "clean", "error"}
COUNTED = ("recoveries", "interventions", "drift_signals")
SUMMARY_KEYS = ("runs_completed", "status_counts", "outcome_counts", "success_rate", "clean_run_rate",
                "total_recoveries", "total_interventions", "total_drift_signals", "eligible_for_approval",
                "ineligible_reasons")


class ReportError(ValueError):
    """A stability report is malformed or its summary does not follow from its run records."""


def run_stability(
    artifact_path: str | Path,
    params: dict,
    sensitive: list[str],
    runs: int,
    surface_factory: Callable[[tuple[str, ...]], Surface],
    allowed_hosts: list[str] | None = None,
    echo: bool = False,
) -> Path:
    """Replay the invocation `runs` times and write evidence/<stability id>/report.json; returns its path."""
    if runs < 1:
        raise ValueError("runs must be at least 1")
    secrets = tuple(str(params[name]) for name in sensitive if name in params)
    log = RunLog("stability", secrets=secrets, echo=echo)
    loaded = load_any_version(artifact_path)          # validated once up front; every run reloads it
    log.event("stability_started", artifact=str(artifact_path), capability=loaded.name, version=loaded.version,
              schema_version=loaded.schema_version, status=loaded.status, runs=runs, params=sorted(params),
              irreversible_policy=STABILITY_IRREVERSIBLE_POLICY)
    records = [one_run(index, artifact_path, params, secrets, surface_factory, allowed_hosts, echo, log)
               for index in range(1, runs + 1)]
    report = build_report(log.run_id, artifact_path, loaded, params, sensitive, runs, records)
    log.event("stability_finished", runs_completed=report["runs_completed"], success_rate=report["success_rate"],
              eligible_for_approval=report["eligible_for_approval"], reasons=report["ineligible_reasons"])
    evidence_dir = log.copy_to_evidence()
    path = evidence_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def one_run(index: int, artifact_path, params: dict, secrets: tuple[str, ...], surface_factory, allowed_hosts,
            echo: bool, log: RunLog) -> dict:
    """One fresh, unattended replay. An error in the run is recorded, never raised, so the others still run."""
    run_log = RunLog("replay", secrets=secrets, echo=echo)
    record = {"run": index, "run_id": run_log.run_id, "status": "error", "outcome_code": None, "step_id": None,
              "evidence": None, "artifact_sha256": None, "recoveries": 0, "interventions": 0, "drift_signals": 0,
              "clean": False, "error": None}
    log.event("run_started", run=index, run_id=run_log.run_id)
    surface = None
    try:
        loaded = load_any_version(artifact_path)
        record["artifact_sha256"] = sha256_of(artifact_path)
        surface = surface_factory(secrets)
        escalator = Escalator(NoOperator(), SessionControl(), run_log)
        result = replay_any(loaded, dict(params), surface, policy_for(loaded, allowed_hosts), escalator, run_log,
                            purpose=STABILITY)
    except Exception as error:
        record["error"] = redact(f"{type(error).__name__}: {error}", secrets)
        run_log.event("replay_crashed", error=record["error"])
        record["evidence"] = str(run_log.copy_to_evidence())
    else:
        record.update(status=result.status, outcome_code=result.outcome_code, step_id=result.step_id,
                      recoveries=len(result.recoveries), interventions=len(result.interventions),
                      drift_signals=count_drift_signals(run_log))
        record["clean"] = is_clean(record)
        record["evidence"] = str(write_bundle(run_log, loaded, artifact_path, result, params,
                                              secrets_names(params, secrets)))
    finally:
        close = getattr(surface, "close", None)
        if close is not None:
            try:
                close()
            except Exception as error:   # a session that will not close must not lose the run's record
                log.event("surface_close_failed", run=index, error=redact(str(error), secrets))
    log.event("run_finished", **{key: record[key] for key in ("run", "run_id", "status", "outcome_code",
                                                             "recoveries", "interventions", "drift_signals", "error")})
    return record


def secrets_names(params: dict, secrets: tuple[str, ...]) -> list[str]:
    return [name for name, value in params.items() if str(value) in secrets]


def count_drift_signals(run_log: RunLog) -> int:
    """Locator resolutions that needed a fallback rung, as logged by the engine."""
    count = 0
    for line in run_log.path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") == "target_resolved" and event.get("drift_signal"):
            count += 1
    return count


def build_report(stability_id: str, artifact_path, loaded: Artifact | ArtifactV2, params: dict, sensitive: list[str],
                 runs: int, records: list[dict]) -> dict:
    digest = sha256_of(artifact_path)
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "stability_id": stability_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": {"name": loaded.name, "version": loaded.version, "schema_version": loaded.schema_version,
                     "status": loaded.status, "path": str(artifact_path), "sha256": digest},
        "irreversible_policy": STABILITY_IRREVERSIBLE_POLICY,
        "runs_requested": runs,
        "param_names": sorted(params),
        "sensitive_params": sorted(name for name in sensitive if name in params),
        "selector_assignment": selector_assignment(loaded, params),
        **summarize(records, digest),
        "runs": records,
    }


def is_clean(record: dict) -> bool:
    return record["status"] == "success" and not any(record[key] for key in COUNTED)


def summarize(records: list[dict], digest: str) -> dict:
    """Every summary field and the eligibility verdict, derived from the run records alone.

    Used to write a report and, unchanged, to check one at approval time.
    """
    completed = [r for r in records if r["error"] is None]
    successes = [r for r in completed if r["status"] == "success"]
    digests = {r["artifact_sha256"] for r in records if r["artifact_sha256"]}
    reasons = []
    if len(completed) < MIN_RUNS_FOR_APPROVAL:
        reasons.append(f"only {len(completed)} of the required {MIN_RUNS_FOR_APPROVAL} runs completed")
    if len(successes) != len(completed) or len(records) != len(completed):
        reasons.append("not every run returned success")
    if any(r["interventions"] for r in records):
        reasons.append("a run required human intervention")
    if digests != {digest}:
        reasons.append("the artifact changed between runs")
    return {
        "runs_completed": len(completed),
        "status_counts": counts(r["status"] for r in records),
        "outcome_counts": counts(r["outcome_code"] or r["status"] for r in records),
        "success_rate": len(successes) / len(records),
        "clean_run_rate": sum(is_clean(r) for r in records) / len(records),
        "total_recoveries": sum(r["recoveries"] for r in records),
        "total_interventions": sum(r["interventions"] for r in records),
        "total_drift_signals": sum(r["drift_signals"] for r in records),
        "eligible_for_approval": not reasons,
        "ineligible_reasons": reasons,
    }


def verify_report(report: dict) -> dict:
    """Check a report's shape and that its summary follows from its run records; returns the recomputation.

    Raises ReportError naming the first field that does not add up. The eligibility flag and the
    reasons are among the fields re-derived, so flipping them alone is caught.
    """
    if not isinstance(report, dict):
        raise ReportError("report must be a JSON object")
    missing = ({"artifact", "irreversible_policy", "runs_requested", "runs"} | set(SUMMARY_KEYS)) - set(report)
    if missing:
        raise ReportError(f"missing keys {sorted(missing)}")
    if report["irreversible_policy"] != STABILITY_IRREVERSIBLE_POLICY:
        raise ReportError(f"irreversible_policy must be {STABILITY_IRREVERSIBLE_POLICY!r}, "
                          f"got {report['irreversible_policy']!r}")
    about = report["artifact"]
    if not isinstance(about, dict) or not isinstance(about.get("sha256"), str):
        raise ReportError("artifact section must carry a sha256 digest")
    records = report["runs"]
    if not isinstance(records, list) or not records:
        raise ReportError("runs must be a non-empty list")
    for index, record in enumerate(records, start=1):
        where = f"run record {index}"
        if not isinstance(record, dict) or set(record) != RUN_KEYS:
            raise ReportError(f"{where} must carry exactly the keys {sorted(RUN_KEYS)}")
        bad_error = record["error"] is not None and not isinstance(record["error"], str)
        if not isinstance(record["status"], str) or bad_error:
            raise ReportError(f"{where}: status must be a string and error a string or null")
        if any(not isinstance(record[key], int) or isinstance(record[key], bool) or record[key] < 0 for key in COUNTED):
            raise ReportError(f"{where}: recoveries, interventions and drift_signals must be non-negative integers")
        if record["error"] is None and record["artifact_sha256"] != about["sha256"]:
            raise ReportError(f"{where}: artifact digest {record['artifact_sha256']!r} is not the report's "
                              f"{about['sha256']!r}")
        if record["clean"] != is_clean(record):
            raise ReportError(f"{where}: clean flag does not follow from its status and counts")
    if report["runs_requested"] != len(records):
        raise ReportError(f"runs_requested {report['runs_requested']!r} does not match {len(records)} run records")
    recomputed = summarize(records, about["sha256"])
    for key in SUMMARY_KEYS:
        if report[key] != recomputed[key]:
            raise ReportError(f"{key} does not follow from the run records: {report[key]!r} vs {recomputed[key]!r}")
    return recomputed


def selector_assignment(loaded: Artifact | ArtifactV2, params: dict) -> dict | None:
    """The values of the artifact's selector inputs for this invocation; sensitive ones are never written."""
    selectors = [name for name, spec in loaded.inputs.items() if spec.get("selector") and not spec.get("sensitive")]
    assignment = {name: str(params[name]) for name in selectors if name in params}
    return assignment or None


def counts(values) -> dict:
    result: dict = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return dict(sorted(result.items()))
