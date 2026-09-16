"""Human-declared multi-scenario discovery.

A campaign specification (JSON) names one capability, one goal, the selector inputs whose
values pick a path, and the scenarios a person wants recorded: nothing else is explored.
Every scenario is discovered on its own (own planner session, own surface, own run log and
evidence, the usual sensitive-parameter protections), in specification order. Only when
every scenario succeeded are the verified traces merged (merge.py) and one draft graph
artifact saved. A failed scenario ends the campaign with its evidence kept and nothing saved.

    {"name": "lookup_member", "goal": "...", "url": "https://...", "selectors": ["lookup_method"],
     "scenarios": [{"name": "by_member_id", "params": {"lookup_method": "member_id", "member_id": "1001"},
                    "sensitive": []}, ...]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from . import artifact as artifact_module
from .agent import DEFAULT_MAX_STEPS, DiscoveryFailed, discover
from .artifact import ArtifactError, validate_outcome
from .escalation import Escalator, Operator, SessionControl
from .evidence import RunLog
from .graph import save_graph
from .merge import MergeError, merge_traces
from .models import CampaignResult, CampaignSpec, Scenario, ScenarioTrace
from .planner import Planner
from .policy import Policy, redact
from .surface import Surface

SPEC_KEYS = {"name", "goal", "url", "selectors", "scenarios"}
OPTIONAL_SPEC_KEYS = {"outputs", "outcomes"}
OUTCOME_SPEC_KEYS = {"code", "kind", "detect", "recover", "source"}
SCENARIO_KEYS = {"name", "params", "sensitive"}
OUTPUT_KEYS = {"type", "required", "pattern", "description"}
OUTPUT_TYPES = {"string", "number", "integer"}
IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*")


class CampaignError(ValueError):
    """The campaign specification is malformed."""


class CampaignFailed(Exception):
    """A scenario's discovery failed, or the traces could not be merged. Nothing was saved."""

    def __init__(self, message: str, scenario: str | None, summary_path: Path):
        super().__init__(message)
        self.scenario = scenario
        self.summary_path = summary_path


# ---------- specification ----------

def load_spec(path: str | Path) -> CampaignSpec:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read campaign spec {path}: {error}") from error
    return spec_from_dict(data)


def spec_from_dict(data) -> CampaignSpec:
    """Check the specification the way validate() checks an artifact: precise errors, no guessing."""
    if not isinstance(data, dict):
        raise CampaignError("campaign spec must be a JSON object")
    missing = SPEC_KEYS - set(data)
    if missing:
        raise CampaignError(f"campaign spec is missing keys: {sorted(missing)}")
    unknown = set(data) - SPEC_KEYS - OPTIONAL_SPEC_KEYS
    if unknown:
        raise CampaignError(f"campaign spec has unknown keys: {sorted(unknown)}")
    if not isinstance(data["name"], str) or not IDENTIFIER.fullmatch(data["name"]):
        raise CampaignError("name must be a snake_case identifier")
    for key in ("goal", "url"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise CampaignError(f"{key} must be a non-empty string")
    selectors = data["selectors"]
    if not isinstance(selectors, list) or not all(isinstance(s, str) and IDENTIFIER.fullmatch(s) for s in selectors):
        raise CampaignError("selectors must be a list of snake_case input names")
    if len(set(selectors)) != len(selectors):
        raise CampaignError("selectors must be unique")
    if not isinstance(data["scenarios"], list) or not data["scenarios"]:
        raise CampaignError("scenarios must be a non-empty list")
    scenarios = [scenario_from_dict(raw, index, selectors) for index, raw in enumerate(data["scenarios"])]
    names = [scenario.name for scenario in scenarios]
    if len(set(names)) != len(names):
        raise CampaignError(f"scenario names must be unique, got {names}")
    seen: dict[tuple, str] = {}
    for scenario in scenarios:
        combination = tuple(str(scenario.params[selector]) for selector in selectors)
        if combination in seen:
            raise CampaignError(f"scenarios {seen[combination]!r} and {scenario.name!r} have the same selector values "
                                f"{dict(zip(selectors, combination))}; selectors must tell every scenario apart")
        seen[combination] = scenario.name
    return CampaignSpec(name=data["name"], goal=data["goal"], url=data["url"], selectors=list(selectors),
                        scenarios=scenarios, outputs=outputs_from_dict(data.get("outputs", {})),
                        outcomes=outcomes_from_dict(data.get("outcomes", [])))


def outcomes_from_dict(raw) -> list[dict]:
    """Reviewer-declared outcomes, checked by the artifact's own outcome rules; source is always 'reviewer'."""
    if not isinstance(raw, list):
        raise CampaignError("outcomes must be a list of outcome objects")
    outcomes: list[dict] = []
    for index, item in enumerate(raw):
        where = f"outcome {index}"
        if not isinstance(item, dict):
            raise CampaignError(f"{where} must be an object")
        unknown = set(item) - OUTCOME_SPEC_KEYS
        if unknown:
            raise CampaignError(f"{where} has unknown keys: {sorted(unknown)}")
        if item.get("source", "reviewer") != "reviewer":
            raise CampaignError(f"{where}: source must be 'reviewer' for an outcome declared in the spec")
        outcome = {**item, "source": "reviewer"}
        try:
            validate_outcome(outcome)
        except ArtifactError as error:
            raise CampaignError(f"{where}: {error}") from error
        if any(existing["code"] == outcome["code"] for existing in outcomes):
            raise CampaignError(f"{where}: outcome code {outcome['code']!r} is declared twice")
        outcomes.append(outcome)
    return outcomes


def reconcile_outcomes(recorded: list[dict], declared: list[dict], log: RunLog | None = None) -> list[dict]:
    """Reviewer-declared outcomes are authoritative for their codes.

    A planner outcome whose code the reviewer did not declare is kept unchanged. For a code the
    reviewer declared, every planner copy is dropped and exactly one reviewer version is kept: an
    identical definition (apart from source) is deduplicated silently, a different one is logged
    as `reviewer_outcome_overrode_planner` with both definitions, and never fails the scenario.
    """
    definition = lambda outcome: {key: value for key, value in outcome.items() if key != "source"}
    owned = {outcome["code"] for outcome in declared}
    merged = [outcome for outcome in recorded if outcome["code"] not in owned]
    for outcome in declared:
        for planner_version in (o for o in recorded if o["code"] == outcome["code"]):
            if definition(planner_version) != definition(outcome) and log is not None:
                log.event("reviewer_outcome_overrode_planner", code=outcome["code"],
                          planner=definition(planner_version), reviewer=definition(outcome))
        merged.append(outcome)
    return merged


def outputs_from_dict(raw) -> dict:
    """The declared output contract every scenario must record: name -> {type, required, pattern}."""
    if not isinstance(raw, dict) or not all(isinstance(k, str) and IDENTIFIER.fullmatch(k) for k in raw):
        raise CampaignError("outputs must be an object keyed by snake_case output names")
    outputs = {}
    for name, spec in raw.items():
        where = f"output {name!r}"
        if not isinstance(spec, dict):
            raise CampaignError(f"{where} must be an object")
        unknown = set(spec) - OUTPUT_KEYS
        if unknown:
            raise CampaignError(f"{where} has unknown keys: {sorted(unknown)}")
        if spec.get("type", "string") not in OUTPUT_TYPES:
            raise CampaignError(f"{where}: type must be one of {sorted(OUTPUT_TYPES)}")
        if not isinstance(spec.get("required", True), bool):
            raise CampaignError(f"{where}: required must be true or false")
        pattern = spec.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise CampaignError(f"{where}: a regex pattern is required")
        try:
            re.compile(pattern)
        except re.error as error:
            raise CampaignError(f"{where}: pattern is not a valid regex: {error}") from error
        outputs[name] = {"type": spec.get("type", "string"), "required": spec.get("required", True),
                         "pattern": pattern}
    return outputs


def scenario_from_dict(raw, index: int, selectors: list[str]) -> Scenario:
    where = f"scenario {index}"
    if not isinstance(raw, dict):
        raise CampaignError(f"{where} must be a JSON object")
    if not isinstance(raw.get("name"), str) or not IDENTIFIER.fullmatch(raw["name"]):
        raise CampaignError(f"{where}: name must be a snake_case identifier")
    where = f"scenario {raw['name']!r}"
    unknown = set(raw) - SCENARIO_KEYS
    if unknown:
        raise CampaignError(f"{where} has unknown keys: {sorted(unknown)}")
    params = raw.get("params")
    if not isinstance(params, dict) or not all(isinstance(k, str) and IDENTIFIER.fullmatch(k) for k in params):
        raise CampaignError(f"{where}: params must be an object keyed by snake_case input names")
    if not all(isinstance(value, str) for value in params.values()):
        raise CampaignError(f"{where}: every param value must be a string")
    absent = [selector for selector in selectors if selector not in params]
    if absent:
        raise CampaignError(f"{where}: missing a value for selector(s) {absent}")
    sensitive = raw.get("sensitive", [])
    if not isinstance(sensitive, list) or not all(isinstance(name, str) for name in sensitive):
        raise CampaignError(f"{where}: sensitive must be a list of param names")
    undeclared = [name for name in sensitive if name not in params]
    if undeclared:
        raise CampaignError(f"{where}: sensitive names {undeclared} are not params")
    secret_selectors = [name for name in sensitive if name in selectors]
    if secret_selectors:
        raise CampaignError(f"{where}: selectors {secret_selectors} cannot be sensitive; their values are written "
                            f"into the artifact's guards")
    return Scenario(name=raw["name"], params=dict(params), sensitive=list(sensitive))


def selector_assignment(scenario: Scenario, selectors: list[str]) -> dict:
    return {selector: str(scenario.params[selector]) for selector in selectors}


# ---------- orchestration ----------

def run_campaign(
    spec: CampaignSpec,
    surface_factory: Callable[[tuple[str, ...]], Surface],
    planner_factory: Callable[[Scenario], Planner],
    operator: Operator,
    allowed_hosts: list[str],
    max_steps: int = DEFAULT_MAX_STEPS,
    echo: bool = False,
    spec_path: str = "",
) -> CampaignResult:
    """Discover every scenario in order, each on a fresh session, then merge and save one draft graph.

    Raises CampaignFailed (with the summary written and all evidence kept) if any scenario fails
    or the traces cannot be merged; previously saved artifacts are never touched.
    """
    secrets = tuple(str(scenario.params[name]) for scenario in spec.scenarios for name in scenario.sensitive)
    log = RunLog("campaign", secrets=secrets, echo=echo)
    log.event("campaign_started", capability=spec.name, goal=spec.goal, entry_url=spec.url, spec=spec_path,
              selectors=spec.selectors, scenarios=[s.name for s in spec.scenarios], allowed_hosts=allowed_hosts,
              outputs=sorted(spec.outputs))
    records: list[dict] = []
    traces: list[ScenarioTrace] = []
    for scenario in spec.scenarios:
        record = {"name": scenario.name, "selectors": selector_assignment(scenario, spec.selectors),
                  "params": sorted(scenario.params), "sensitive": list(scenario.sensitive), "status": "pending",
                  "run_id": None, "evidence": None, "trace": None, "node_path": None, "error": None}
        records.append(record)
        log.event("scenario_started", scenario=scenario.name, selectors=record["selectors"], params=record["params"])
        try:
            traces.append(discover_scenario(spec, scenario, record, surface_factory, planner_factory, operator,
                                            allowed_hosts, max_steps, echo))
        except Exception as error:   # DiscoveryFailed, or anything the provider, planner, or browser threw
            # Error text may quote a typed value: redact before it reaches the summary or the exception.
            why = redact(str(error), secrets)
            if not isinstance(error, DiscoveryFailed):
                why = f"unexpected {type(error).__name__}: {why}"
            record.update(status="failed", error=why)
            log.event("scenario_failed", scenario=scenario.name, run_id=record["run_id"], error=why)
            summary = finish(log, spec, records, "failed", None, spec_path)
            raise CampaignFailed(f"scenario {scenario.name!r} failed: {why}", scenario.name, summary) from error
        record["status"] = "succeeded"
        log.event("scenario_succeeded", scenario=scenario.name, run_id=record["run_id"],
                  steps=len(traces[-1].artifact.steps))

    try:
        graph = merge_traces(traces, spec.selectors, campaign_id=log.run_id,
                             version=artifact_module.next_version(spec.name))
        for record, entry in zip(records, graph.provenance["scenarios"]):
            record["node_path"] = entry["node_path"]
        path = save_graph(graph, secrets=secrets)
    except Exception as error:   # MergeError, a refused save, or anything unexpected
        why = redact(str(error), secrets)
        log.event("campaign_failed", error=why)
        summary = finish(log, spec, records, "merge_failed", None, spec_path)
        raise CampaignFailed(f"traces could not be merged: {why}", None, summary) from error
    log.event("traces_merged", nodes=len(graph.nodes), edges=len(graph.edges), entry_node=graph.entry_node)
    log.event("artifact_saved", path=str(path), version=graph.version)
    summary = finish(log, spec, records, "succeeded", path, spec_path)
    return CampaignResult(campaign_id=log.run_id, artifact_path=str(path), summary_path=str(summary), graph=graph,
                          scenarios=records)


def discover_scenario(spec: CampaignSpec, scenario: Scenario, record: dict, surface_factory, planner_factory,
                      operator: Operator, allowed_hosts: list[str], max_steps: int, echo: bool) -> ScenarioTrace:
    """One ordinary discovery run: its own log, session, planner, policy and escalation.

    Any failure (a stopped discovery, a provider error, a browser that would not start) leaves
    its evidence behind, closes the session if one was opened, and propagates. Interrupts are not
    swallowed, but the session is still closed.
    """
    secrets = tuple(str(scenario.params[name]) for name in scenario.sensitive)
    run_log = RunLog("discovery", secrets=secrets, echo=echo)
    record["run_id"] = run_log.run_id
    surface = None
    try:
        planner = planner_factory(scenario)
        surface = surface_factory(secrets)
        built = discover(goal=spec.goal, name=spec.name, params=dict(scenario.params), surface=surface,
                         planner=planner, policy=Policy(allowed_hosts=list(allowed_hosts)),
                         escalator=Escalator(operator, SessionControl(), run_log), log=run_log, entry_url=spec.url,
                         sensitive=set(scenario.sensitive), max_steps=max_steps, output_contract=spec.outputs,
                         selectors=set(spec.selectors), extra_outcomes=[dict(o) for o in spec.outcomes])
        if spec.outcomes:
            # discover() appends the reviewer's outcomes as given; reconcile with what the planner declared.
            recorded = [o for o in built.outcomes if not (o.get("source") == "reviewer" and o in spec.outcomes)]
            built.outcomes = reconcile_outcomes(recorded, spec.outcomes, run_log)
            artifact_module.validate(built)
        if spec.outputs and set(built.outputs) != set(spec.outputs):
            raise DiscoveryFailed(f"recorded outputs {sorted(built.outputs)} do not match the declared contract "
                                  f"{sorted(spec.outputs)}")
    except Exception as error:
        run_log.event("discovery_failed", error=str(error), kind=type(error).__name__)
        if surface is not None:
            run_log.screenshot(surface, "failed")
        close_quietly(surface, run_log)
        record["evidence"] = str(run_log.copy_to_evidence())
        raise
    except BaseException:
        close_quietly(surface, run_log)
        raise
    close_quietly(surface, run_log)
    evidence_dir = run_log.copy_to_evidence()
    trace_path = evidence_dir / f"{scenario.name}.trace.json"
    trace_path.write_text(artifact_module.dumps(built, secrets), encoding="utf-8")
    record.update(evidence=str(evidence_dir), trace=str(trace_path))
    return ScenarioTrace(scenario=scenario, artifact=built, run_id=run_log.run_id, planner=planner.name)


def close_quietly(surface, run_log: RunLog) -> None:
    """Close a session if one was opened; a failing close is logged and never hides the real failure."""
    close = getattr(surface, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception as error:
        run_log.event("surface_close_failed", error=str(error), kind=type(error).__name__)


def finish(log: RunLog, spec: CampaignSpec, records: list[dict], status: str, artifact_path, spec_path: str) -> Path:
    """Copy the campaign log to evidence and write the summary mapping every scenario to its evidence."""
    log.event("campaign_finished", status=status, artifact=str(artifact_path) if artifact_path else None)
    evidence_dir = log.copy_to_evidence()
    summary = {"campaign_id": log.run_id, "status": status, "spec": spec_path, "capability": spec.name,
               "selectors": spec.selectors, "artifact": str(artifact_path) if artifact_path else None,
               "scenario_count": len(spec.scenarios),
               "successful_scenario_count": sum(r["status"] == "succeeded" for r in records),
               "scenarios": records}
    path = evidence_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
