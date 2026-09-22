"""The discovery session owner: forced-prefix reuse, the automatic library, and clean restarts.

Two kinds of reuse open or extend a discovery, both deterministic and both inlined into the new
artifact so it is self-contained:

  - a forced prefix (`--reuse-capability NAME` / `--reuse-artifact PATH`): one named approved
    artifact replayed whole at the very start, before the planner sees anything;
  - the automatic library (`library.py`, on by default): every approved artifact in the artifact
    directory, cut into verified segments and offered to the planner before each decision.

This module resolves and preflights the forced prefix, builds the library, and owns the live
session(s): a forced prefix that fails closes its session and discovery starts afresh; an automatic
segment that leaves the session unverified makes discovery restart on a fresh session with that
segment excluded, a bounded number of times. Every surface opened here is closed exactly once.
Imported nodes come from the engine's `executed_path` (action nodes completed once each on the
branch actually taken); `performed_attempts` is diagnostic and never imported.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Callable

from . import artifact as artifact_module
from .agent import DiscoveryFailed, RestartDiscovery, discover
from .artifact import ArtifactError, missing_inputs
from .escalation import Escalator, NoOperator, SessionControl
from .evidence import RunLog
from .library import DEFAULT_MAX_AUTO_REUSES, Library, build_library
from .lifecycle import DISCOVERY_REUSE, is_approved, load_artifact, run_replay, sha256_of
from .models import Artifact, GraphNode, ReplayResult, ReusePlan, ReusePrefix
from .policy import Policy
from .replay import failure
from .surface import Surface

VERSIONED = re.compile(r"\.v(\d+)\.json$")
MAX_DISCOVERY_RESTARTS = 2


class ReuseError(ValueError):
    """The requested reuse is misconfigured or unsafe; raised before any surface action."""


# ---------- resolution ----------

def resolve_reuse(capability: str | None, artifact_path: str | None, log: RunLog | None = None) -> ReusePlan | None:
    """The approved artifact to reuse, or None when a capability name has no approved version yet.

    By name, the candidates are the files `NAME.v<N>.json` in the artifacts directory that load,
    hold capability NAME at revision N and are approved; the greatest N wins, so the choice depends
    on nothing but the files present.
    """
    if capability and artifact_path:
        raise ReuseError("give either a capability name to reuse or an exact artifact path, not both")
    if artifact_path:
        try:
            loaded = load_artifact(artifact_path)
        except ArtifactError as error:
            raise ReuseError(f"cannot reuse {artifact_path}: {error}") from error
        if not is_approved(loaded):
            raise ReuseError(f"cannot reuse {artifact_path}: status is {loaded.status!r}, only an approved "
                             f"artifact may open a discovery")
        return plan_for(artifact_path, loaded, log)
    if not capability:
        return None
    candidates = []
    for path in sorted(artifact_module.ARTIFACTS_DIR.glob(f"{capability}.v*.json")):
        match = VERSIONED.search(path.name)
        if match is None:
            continue
        try:
            loaded = load_artifact(path)     # also refuses a file whose name and contents disagree
        except ArtifactError:
            continue                         # a broken file is not a candidate
        if is_approved(loaded):
            candidates.append((loaded.version, path, loaded))
    if not candidates:
        if log is not None:
            log.event("reuse_not_found", capability=capability, reason="no approved artifact with that name")
        return None
    version, path, loaded = max(candidates, key=lambda item: item[0])
    return plan_for(str(path), loaded, log)


def plan_for(path: str | Path, loaded: Artifact, log: RunLog | None) -> ReusePlan:
    plan = ReusePlan(path=str(path), digest=sha256_of(path), artifact=loaded, name=loaded.name,
                     version=loaded.version, schema_version=loaded.schema_version)
    if log is not None:
        log.event("reuse_resolved", capability=plan.name, version=plan.version, schema_version=plan.schema_version,
                  path=plan.path, digest=plan.digest)
    return plan


# ---------- preflight ----------

def preflight(plan: ReusePlan, params: dict, sensitive: set[str], entry_url: str, allowed_hosts: list[str]) -> None:
    """Refuse an unsafe or incompatible reuse before any browser action. Messages never carry values."""
    artifact = plan.artifact
    if not is_approved(artifact):
        raise ReuseError(f"{plan.name} v{plan.version} is {artifact.status!r}; only an approved artifact may be reused")
    if sha256_of(plan.path) != plan.digest:
        raise ReuseError(f"{plan.path} changed since it was resolved")
    missing = missing_inputs(artifact.inputs, params)
    if missing:
        raise ReuseError(f"{plan.name} v{plan.version} needs inputs {missing} that this discovery does not supply")
    wrongly_visible = sorted(name for name, spec in artifact.inputs.items()
                             if spec.get("sensitive") and name in params and name not in sensitive)
    if wrongly_visible:
        raise ReuseError(f"inputs {wrongly_visible} are sensitive in {plan.name} v{plan.version} but not marked "
                         f"sensitive for this discovery")
    if artifact.surface.get("entry_url") != entry_url:
        raise ReuseError(f"{plan.name} v{plan.version} starts at {artifact.surface.get('entry_url')!r}, "
                         f"this discovery at {entry_url!r}")
    foreign = sorted(set(artifact.surface.get("allowed_hosts") or []) - set(allowed_hosts))
    if foreign:
        raise ReuseError(f"{plan.name} v{plan.version} may visit hosts {foreign} that this discovery does not allow")
    unsafe = unsafe_actions(artifact)
    if unsafe:
        raise ReuseError(f"{plan.name} v{plan.version} contains actions that must never run unattended: {unsafe}")


def unsafe_actions(artifact: Artifact) -> list[str]:
    return [f"{n.id} ({n.effect})" for n in artifact.nodes if n.kind == "action"
            and n.effect in ("irreversible", "unknown")]


# ---------- execution and import ----------

def execute_reuse(plan: ReusePlan, params: dict, sensitive: set[str], surface: Surface, allowed_hosts: list[str],
                  log: RunLog) -> ReplayResult:
    """Replay the approved prefix on the live surface with no operator and no model."""
    log.event("reuse_started", capability=plan.name, version=plan.version, params=sorted(params),
              purpose=DISCOVERY_REUSE)
    escalator = Escalator(NoOperator(), SessionControl(), log)
    try:
        result = run_replay(plan.artifact, dict(params), surface, Policy(allowed_hosts=list(allowed_hosts)), escalator,
                            log, purpose=DISCOVERY_REUSE)
    except Exception as error:                       # a crash while reusing is a failed reuse, not a crash
        result = failure("reuse_crashed", None, "the reusable prefix to replay", f"{type(error).__name__}: {error}")
    for entry in result.executed_path:
        log.event("reuse_step_executed", node_id=entry["id"], action=entry["action"]["action"],
                  checkpoint=entry["action"].get("checkpoint"), effect=entry["effect"],
                  retry_safety=entry["retry_safety"])
    return result


def prefix_from(result: ReplayResult, plan: ReusePlan, output_contract: dict | None, reuse_run_id: str) -> ReusePrefix:
    """Turn a successful replay's executed path into the opening nodes of a new artifact.

    Nodes are renumbered and otherwise copied exactly (action, effect, retry safety); extractions
    are imported only when the new discovery declares the same output contract; recoverable
    outcomes come along, business outcomes do not (the new discovery declares its own).
    """
    if result.status != "success":
        raise ReuseError(f"only a successful replay can be imported, got {result.status!r}")
    contract = output_contract or {}
    source_outputs = plan.artifact.outputs
    nodes: list[GraphNode] = []
    outputs: dict = {}
    for entry in result.executed_path:
        action = artifact_module.action_from_dict(entry["action"], f"executed node {entry['id']}")
        if action.action in ("extract", "extract_many"):
            declared, recorded = contract.get(action.value), source_outputs.get(action.value, {})
            same = declared is not None and all(declared.get(k) == recorded.get(k)
                                                for k in ("type", "required", "pattern", "items"))
            if not same:
                continue
            outputs[action.value] = copy.deepcopy(recorded)
        nodes.append(GraphNode(id=f"s{len(nodes) + 1}", kind="action", action=action, effect=entry["effect"],
                               retry_safety=entry["retry_safety"]))
    outcomes = [dict(o) for o in plan.artifact.outcomes if o.get("kind") == "recoverable"]
    provenance = {"mode": "forced_prefix", "candidate_id": None, "source_name": plan.name,
                  "source_version": plan.version, "source_digest": plan.digest, "source_path": plan.path,
                  "start_node": plan.artifact.entry_node,
                  "executed_path": [entry["id"] for entry in result.executed_path], "imported_nodes": len(nodes),
                  "imported_as": [node.id for node in nodes], "reuse_run_id": reuse_run_id, "planner_turn": 0,
                  "entry_condition": {"entry_url": plan.artifact.surface.get("entry_url")},
                  "ending_checkpoint": dict(plan.artifact.success), "result": "succeeded"}
    return ReusePrefix(nodes=nodes, outcomes=outcomes, outputs=outputs, provenance=provenance)


# ---------- the orchestrator ----------

def discover_with_reuse(surface_factory: Callable[[], Surface], reuse: ReusePlan | None, *, goal: str, name: str,
                        params: dict, planner, policy: Policy, escalator: Escalator, log: RunLog, entry_url: str,
                        sensitive: set[str] | None = None, auto_reuse: bool = True,
                        max_auto_reuses: int = DEFAULT_MAX_AUTO_REUSES, library_dir: str | Path | None = None,
                        **discover_options) -> Artifact:
    """Discovery that composes verified work: an optional forced prefix, then the automatic library.

    Owns every surface it opens and closes each exactly once. A forced prefix that does not succeed
    or import closes its session and discovery starts on a fresh one. A discovery that raises
    RestartDiscovery (a reused segment left the session unverified) is rerun on a fresh session with
    that segment excluded, at most MAX_DISCOVERY_RESTARTS times; the forced prefix, if any, runs
    again on the new session.
    """
    sensitive = sensitive or set()
    if reuse is not None:
        preflight(reuse, params, sensitive, entry_url, policy.allowed_hosts)   # before any surface exists
        log.event("reuse_preflight_passed", capability=reuse.name, version=reuse.version)
    library = None
    if auto_reuse:
        library = build_library(library_dir, entry_url, policy.allowed_hosts, params, sensitive, log,
                                max_reuses=max_auto_reuses)
    excluded: set = set()
    for attempt in range(MAX_DISCOVERY_RESTARTS + 1):
        if library is not None:
            library.new_session(excluded)
        surface, prefix = open_session(surface_factory, reuse, params, sensitive, policy, log,
                                       discover_options.get("output_contract"))
        try:
            return run_discovery(surface, goal=goal, name=name, params=params, planner=planner, policy=policy,
                                 escalator=escalator, log=log, entry_url=entry_url, sensitive=sensitive,
                                 prefix=prefix, library=library, **discover_options)
        except RestartDiscovery as restart:
            excluded |= restart.excluded
            log.event("discovery_restarted", attempt=attempt + 1, max_restarts=MAX_DISCOVERY_RESTARTS,
                      reason=str(restart),
                      excluded=sorted(f"{e.name}.v{e.version}" for e in (library.entries if library else [])
                                      if e.digest in excluded))
    raise DiscoveryFailed(f"discovery restarted {MAX_DISCOVERY_RESTARTS} times after reused segments left the "
                          f"session unverified; giving up")


def open_session(surface_factory, reuse: ReusePlan | None, params: dict, sensitive: set[str], policy: Policy,
                 log: RunLog, output_contract: dict | None) -> tuple[Surface, ReusePrefix | None]:
    """A live session positioned for discovery: after the forced prefix when there is one and it imported,
    otherwise fresh."""
    surface = surface_factory()
    if reuse is None:
        return surface, None
    prefix = reuse_prefix(reuse, params, sensitive, surface, policy, log, output_contract)
    if prefix is not None:
        return surface, prefix
    close_quietly(surface, log)
    log.event("reuse_fallback_started", reason="the reusable prefix did not succeed; fresh session")
    return surface_factory(), None


def reuse_prefix(reuse: ReusePlan, params: dict, sensitive: set[str], surface: Surface, policy: Policy, log: RunLog,
                 output_contract: dict | None) -> ReusePrefix | None:
    """Replay the prefix on `surface` and import it; None when the session must not be continued."""
    result = execute_reuse(reuse, params, sensitive, surface, policy.allowed_hosts, log)
    if result.status != "success":
        log.event("reuse_failed", capability=reuse.name, version=reuse.version, status=result.status,
                  outcome_code=result.outcome_code, step_id=result.step_id,
                  executed_path=[entry["id"] for entry in result.executed_path],
                  performed_attempts=result.performed_attempts)
        return None
    try:
        prefix = prefix_from(result, reuse, output_contract, log.run_id)
    except (ArtifactError, ReuseError, KeyError, TypeError, ValueError) as error:
        log.event("reuse_import_failed", capability=reuse.name, version=reuse.version,
                  error=f"{type(error).__name__}: {error}")
        return None
    log.event("reuse_succeeded", capability=reuse.name, version=reuse.version, executed_path=len(result.executed_path),
              performed_attempts=result.performed_attempts, imported_nodes=len(prefix.nodes))
    return prefix


def run_discovery(surface: Surface, **kwargs) -> Artifact:
    """Ordinary discovery on a surface this module opened: a failure screenshot, then the session is closed."""
    try:
        return discover(surface=surface, **kwargs)
    except RestartDiscovery as restart:
        kwargs["log"].event("discovery_restart_requested", reason=str(restart))
        kwargs["log"].screenshot(surface, "before-restart")
        raise
    except Exception as error:
        kwargs["log"].event("discovery_failed", error=str(error), kind=type(error).__name__)
        kwargs["log"].screenshot(surface, "failed")
        raise
    finally:
        close_quietly(surface, kwargs["log"])


def close_quietly(surface, log: RunLog) -> None:
    close = getattr(surface, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception as error:
        log.event("surface_close_failed", error=str(error), kind=type(error).__name__)
