"""Automatic verified composition: the approved artifacts are a library of UI knowledge.

    scan artifacts/ once   -> catalog: approved, valid, same surface, safe to consider
    before every planner decision:
        observe -> find segments whose entry state holds on this screen -> offer them to the planner
    planner picks a candidate -> recheck digest and entry state -> deterministic replay of the
        segment on the live session (no model, irreversible policy deny) -> verified nodes inlined
        into the new artifact -> discovery continues

A candidate is one verified action node of an approved graph, offered in the state it is applicable
to: the state proven by the nearest checkpoint before it (the entry url, or a preceding action's
verified checkpoint), with decisions on the way resolved from the parameters or the current screen.
It is offered only when that state holds now, the action is neither irreversible nor unknown, and
every input it needs is available; it is executed and verified on its own, after which the planner
is asked again. Matching a screen is necessary, not sufficient: the planner decides whether the
action advances the goal, and it never runs the rest of an artifact automatically. This is
composition of individually verified actions, never a cache of action results.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from . import artifact as artifact_module
from .artifact import (CARDINALITY_KEYS, SCHEMA_VERSION, SUCCESS_NODE_ID, ArtifactError, copy_node, locator_placeholders,
                       nodes_by_id, outgoing, parameterize, placeholders_in, validate)
from .escalation import Escalator, NoOperator, SessionControl
from .evidence import RunLog
from .lifecycle import DISCOVERY_REUSE, is_approved, load_artifact, run_replay, sha256_of
from .models import Artifact, GraphEdge, GraphNode, Guard, Observation, ReplayResult, ReuseCandidate
from .policy import Policy
from .replay import checkpoint_met, failure, guard_holds
from .surface import Surface

DEFAULT_MAX_AUTO_REUSES = 5
UNSAFE_EFFECTS = {"irreversible", "unknown"}
PARAM_ONLY_GUARDS = {"always", "input_equals"}
PHYSICAL_ACTIONS = {"navigate", "click", "type"}


@dataclass
class CatalogEntry:
    path: str
    digest: str
    artifact: Artifact

    @property
    def name(self) -> str:
        return self.artifact.name

    @property
    def version(self) -> int:
        return self.artifact.version


@dataclass
class Library:
    """The catalog plus one discovery's reuse bookkeeping.

    Discovery-wide fields live for the whole discovery run, across every live session a clean
    restart opens: `count` (automatic reuses started so far, capped by `max_reuses`; a forced
    prefix never counts), `exhausted_logged` and `excluded` (source digests that failed
    uncertainly: never run again). The one session-local field is `attempted`, the
    (digest, start node, screen key) triples tried on the current session; a fresh session may
    need the same safe segments again to rebuild the application state.
    """

    entries: list[CatalogEntry]
    max_reuses: int = DEFAULT_MAX_AUTO_REUSES
    rejected: list[dict] = field(default_factory=list)        # catalog-time rejections, value-free
    attempted: set = field(default_factory=set)               # session-local
    excluded: set = field(default_factory=set)                # discovery-wide
    count: int = 0                                            # discovery-wide
    exhausted_logged: bool = False                            # discovery-wide

    def new_session(self, excluded: set | None = None) -> None:
        """A fresh live session after a clean restart: clear only the session-local attempts and record
        the sources the failed session excluded; the budget and its exhaustion are discovery-wide."""
        self.attempted = set()
        if excluded:
            self.excluded |= set(excluded)

    def entry_for(self, candidate: ReuseCandidate) -> CatalogEntry | None:
        return next((e for e in self.entries if e.digest == candidate.source_digest), None)


# ---------- the catalog ----------

def build_library(directory: str | Path | None, entry_url: str, allowed_hosts: list[str], params: dict,
                  sensitive: set[str], log: RunLog, max_reuses: int = DEFAULT_MAX_AUTO_REUSES) -> Library:
    """Scan the artifact directory once. Anything that cannot be reused safely is skipped with a reason."""
    directory = Path(directory) if directory else artifact_module.ARTIFACTS_DIR
    library = Library(entries=[], max_reuses=max_reuses)
    for path in sorted(directory.glob("*.v*.json")) if directory.exists() else []:
        reason = catalog_rejection(path, entry_url, allowed_hosts, params, sensitive)
        if reason is not None:
            library.rejected.append({"path": str(path), "reason": reason})
            continue
        library.entries.append(CatalogEntry(path=str(path), digest=sha256_of(path), artifact=load_artifact(path)))
    log.event("artifact_catalog_built", directory=str(directory), max_auto_reuses=max_reuses,
              cataloged=[{"name": e.name, "version": e.version, "path": e.path, "digest": e.digest}
                         for e in library.entries],
              rejected=library.rejected)
    return library


def catalog_rejection(path: Path, entry_url: str, allowed_hosts: list[str], params: dict,
                      sensitive: set[str]) -> str | None:
    """Why a file is not in the catalog, or None. Never raises, never mentions a parameter value."""
    try:
        loaded = load_artifact(path)             # canonical schema, validated, file name agrees with contents
    except ArtifactError as error:
        return f"invalid: {error}"
    except Exception as error:                   # unreadable in some other way
        return f"unreadable: {type(error).__name__}"
    if not is_approved(loaded):
        return f"not approved: status is {loaded.status!r}"
    if loaded.surface.get("entry_url") != entry_url:
        return f"different entry url: {loaded.surface.get('entry_url')!r}"
    foreign = sorted(set(loaded.surface.get("allowed_hosts") or []) - set(allowed_hosts))
    if foreign:
        return f"hosts outside this discovery's allowlist: {foreign}"
    wrongly_visible = sorted(name for name, spec in loaded.inputs.items()
                             if spec.get("sensitive") and name in params and name not in sensitive)
    if wrongly_visible:
        return f"inputs {wrongly_visible} are sensitive there but not marked sensitive here"
    return None


# ---------- segments and candidates ----------

def screen_key(observation: Observation) -> str:
    """A hash of the structured screen: which url and which controls. Never stores text."""
    digest = hashlib.sha256(observation.url.encode())
    for element in observation.elements:
        digest.update(f"|{element.role}|{element.name}|{element.text}".encode())
    digest.update(f"|dialog={observation.dialog}".encode())
    return digest.hexdigest()[:16]


def applicable_actions(artifact: Artifact, params: dict, observation: Observation,
                       surface: Surface) -> list[tuple[GraphNode, dict]]:
    """Every action node whose applicability state holds on the current screen, with that condition.

    Conditions propagate forward through the graph from the entry (proven by being at the entry url):
    an action with a checkpoint proves that checkpoint for what follows, an action without one passes
    its own condition on (nothing better is known), and a decision passes its condition to the one
    branch that resolves now (from the parameters alone anywhere, from the live screen only when the
    decision's own condition holds, that is when this is the screen being looked at). Irreversible and
    unknown actions are never applicable; a navigation whose destination is already proven on screen is
    skipped as pointless. Each node is a candidate on its own, never a suffix.
    """
    nodes = nodes_by_id(artifact)
    ctx = SimpleNamespace(params=params, surface=surface)
    conditions: dict[str, dict] = {artifact.entry_node: {"entry_url": artifact.surface["entry_url"]}}
    order = [artifact.entry_node]
    found: list[tuple[GraphNode, dict]] = []
    for node_id in order:                                   # acyclic: every node is reached once
        node, condition = nodes[node_id], conditions[node_id]
        if node.kind == "terminal":
            continue
        if node.kind == "action":
            passes_on = dict(node.action.checkpoint) if node.action.checkpoint else condition
            if node.effect not in UNSAFE_EFFECTS and condition_holds(condition, observation, params):
                ending = dict(node.action.checkpoint) if node.action.checkpoint else condition
                if not (node.action.action == "navigate" and condition_holds(ending, observation, params)):
                    found.append((node, condition))
            successors = [edge.target for edge in outgoing(artifact, node.id)]
        else:
            passes_on = condition
            here = condition_holds(condition, observation, params)
            chosen = resolve_edges(outgoing(artifact, node.id), ctx, observation, screen_ok=here)
            successors = [chosen.target] if chosen is not None else []
        for target in successors:
            if target not in conditions:
                conditions[target] = passes_on
                order.append(target)
    return found


def condition_holds(condition: dict, observation: Observation, params: dict) -> bool:
    if "entry_url" in condition:
        return observation.url.rstrip("/") == condition["entry_url"].rstrip("/")
    return checkpoint_met(condition, observation, params)


def resolve_edges(edges: list[GraphEdge], ctx, observation: Observation, screen_ok: bool) -> GraphEdge | None:
    """The first edge (by priority) whose guards all hold, judged now; None when a guard needs a screen
    that is not the current one."""
    for edge in edges:
        holds = True
        for guard in edge.guards:
            if guard.kind not in PARAM_ONLY_GUARDS and not screen_ok:
                return None
            if not guard_holds(ctx, guard, observation):
                holds = False
                break
        if holds:
            return edge
    return None


def required_inputs(path: list[GraphNode], ending: dict) -> list[str]:
    names: set[str] = set()
    for node in path:
        action = node.action
        names |= placeholders_in(action.value)
        for ladder in ([action.target] if action.target else []) + list(action.targets or []):
            for strategy in ladder.strategies:
                names |= locator_placeholders(strategy)
        for value in (action.checkpoint or {}).values():
            names |= placeholders_in(str(value))
    for value in ending.values():
        names |= placeholders_in(str(value))
    return sorted(names)


def describe_node(node: GraphNode) -> str:
    action = node.action
    if action.action == "navigate":
        return f"navigate to {action.value}"
    if action.action == "back":
        return "go back to the previous page"
    strategy = action.target.strategies[0] if action.target and action.target.strategies else {}
    what = (f"{strategy.get('role', 'control')} '{strategy.get('name', '')}'" if strategy.get("kind") == "role"
            else f"text '{strategy.get('text', '')}'" if strategy.get("kind") == "text" else "a control")
    if action.action == "type":
        return f"type {action.value} into {what}"
    if action.action == "select":
        return f"select {action.value} in {what}"
    how = "append to" if action.mode == "append" else "read"
    if action.action == "extract":
        return f"{how} output '{action.value}' from {what}"
    if action.action == "extract_many":
        return f"{how} list output '{action.value}' from {len(action.targets or [])} controls"
    return f"click {what}"


def candidate_from(entry: CatalogEntry, node: GraphNode, condition: dict) -> ReuseCandidate:
    """One verified action as a candidate: it starts and ends at `node`, proven by `condition` before and by the
    node's checkpoint (or, for an action without one, the same condition) after."""
    ending = dict(node.action.checkpoint) if node.action.checkpoint else dict(condition)
    return ReuseCandidate(
        candidate_id=f"{entry.name}.v{entry.version}@{node.id}", source_name=entry.name, source_version=entry.version,
        source_digest=entry.digest, source_path=entry.path, start_node=node.id, description=entry.artifact.description,
        required_inputs=required_inputs([node], ending),
        outputs=[node.action.value] if node.action.action in ("extract", "extract_many") else [],
        outline=[describe_node(node)], entry_condition=dict(condition), ending_condition=ending,
        effects=[node.effect], action_count=1, node_ids=[node.id],
    )


def candidate_summary(candidate: ReuseCandidate) -> dict:
    """What the planner and the log see: names, never values."""
    return {"candidate_id": candidate.candidate_id, "source": candidate.source_name,
            "revision": candidate.source_version, "digest": candidate.source_digest, "start_node": candidate.start_node,
            "description": candidate.description, "required_inputs": candidate.required_inputs,
            "outputs": candidate.outputs, "outline": candidate.outline, "entry_condition": candidate.entry_condition,
            "ending_condition": candidate.ending_condition, "effects": candidate.effects,
            "action_count": candidate.action_count}


def find_candidates(library: Library, observation: Observation, params: dict, surface: Surface,
                    log: RunLog, turn: int) -> list[ReuseCandidate]:
    """Every verified action applicable to this screen, one candidate each, before a planner decision."""
    if library.count >= library.max_reuses:
        if not library.exhausted_logged:
            log.event("reuse_exhausted", reuses=library.count, max_auto_reuses=library.max_reuses, turn=turn)
            library.exhausted_logged = True
        return []
    key = screen_key(observation)
    found: list[ReuseCandidate] = []
    for entry in library.entries:
        for node, condition in applicable_actions(entry.artifact, params, observation, surface):
            candidate = candidate_from(entry, node, condition)
            reason = offer_rejection(library, candidate, params, key)
            if reason is not None:
                log.event("reuse_candidate_rejected", candidate_id=candidate.candidate_id, turn=turn, reason=reason)
                continue
            found.append(candidate)
    log.event("reuse_candidates_found", turn=turn, screen=key, candidates=[c.candidate_id for c in found])
    return found


def offer_rejection(library: Library, candidate: ReuseCandidate, params: dict, key: str) -> str | None:
    missing = [name for name in candidate.required_inputs if name not in params]
    if missing:
        return f"missing inputs {missing}"
    if candidate.source_digest in library.excluded:
        return "excluded after an uncertain failure earlier in this discovery"
    if (candidate.source_digest, candidate.start_node, key) in library.attempted:
        return "already attempted on this screen"
    return None


# ---------- execution ----------

def segment_outputs(source: Artifact, candidate: ReuseCandidate) -> dict:
    """Only the outputs the segment's own nodes write, without whole-flow cardinality bounds: a segment reads one
    step of a list that the whole flow accumulates, so its success is judged on that step alone."""
    outputs = {}
    for name in candidate.outputs:
        spec = copy.deepcopy(source.outputs[name])
        for key in CARDINALITY_KEYS:
            spec.pop(key, None)
        outputs[name] = spec
    return outputs


def segment_success(candidate: ReuseCandidate, path: list[GraphNode]) -> dict:
    """What a reused segment must show to count as done: its own last action's checkpoint.

    Never the source artifact's global success condition, and never a condition borrowed from the screen
    the action started on: one action out of a longer flow cannot be expected to complete that flow, and
    the screen it began on is exactly what it may have changed. An action with no checkpoint of its own
    asserts nothing here; whether running it was enough is decided by the existing atomic progress rule,
    which is what already governs a checkpoint-less reused action.
    """
    last = path[-1]
    if last.action and last.action.checkpoint:
        return dict(last.action.checkpoint)
    if candidate.action_count > 1 and candidate.ending_condition:
        return dict(candidate.ending_condition)            # a multi-action segment keeps its ending proof
    return {"url_contains": ""}                            # holds on any screen: the action itself is the proof


def build_segment(entry: CatalogEntry, candidate: ReuseCandidate) -> Artifact:
    """A self-contained linear artifact for the segment: the source's nodes copied verbatim, chained by
    always edges into a success terminal that verifies the segment's own ending checkpoint (never the
    source's global success condition, see `segment_success`)."""
    source = entry.artifact
    nodes = nodes_by_id(source)
    path = [copy_node(nodes[node_id], node_id) for node_id in candidate.node_ids]
    terminal = GraphNode(id=SUCCESS_NODE_ID, kind="terminal", effect="none", retry_safety="safe", status="success")
    chain = path + [terminal]
    edges = [GraphEdge(source=a.id, target=b.id, guards=[Guard(kind="always")], priority=0)
             for a, b in zip(chain, chain[1:])]
    used = set(candidate.required_inputs)
    inputs = {}
    for name, spec in source.inputs.items():
        spec = {k: v for k, v in spec.items() if k not in ("required_when", "selector")}
        spec["required"] = name in used
        inputs[name] = spec
    segment = Artifact(
        schema_version=SCHEMA_VERSION, name=source.name, version=source.version, status="approved",
        description=f"{source.description} (segment from {candidate.start_node})", surface=copy.deepcopy(source.surface),
        inputs=inputs, outputs=segment_outputs(source, candidate), entry_node=path[0].id, nodes=chain, edges=edges,
        outcomes=copy.deepcopy(source.outcomes), success=segment_success(candidate, path),
        provenance={"segment_of": {"name": source.name, "version": source.version, "digest": entry.digest,
                                   "start_node": candidate.start_node}},
    )
    validate(segment)
    return segment


@dataclass
class Execution:
    """What running a candidate produced: a replay result, or a value-free reason it never started."""
    candidate: ReuseCandidate
    result: ReplayResult | None = None
    rejected: str | None = None
    segment: Artifact | None = None


def execute_candidate(library: Library, candidate: ReuseCandidate, params: dict, surface: Surface,
                      policy: Policy, log: RunLog, turn: int) -> Execution:
    """Recheck, then replay the segment on the live session with no operator and no model."""
    entry = library.entry_for(candidate)
    if entry is None or sha256_of(entry.path) != candidate.source_digest:
        return Execution(candidate, rejected="source artifact changed since the catalog was built")
    try:
        observation = surface.observe()
    except Exception as error:
        return Execution(candidate, rejected=f"screen could not be observed: {type(error).__name__}")
    if not condition_holds(candidate.entry_condition, observation, params):
        return Execution(candidate, rejected="entry condition no longer holds")
    try:
        segment = build_segment(entry, candidate)
    except ArtifactError as error:
        return Execution(candidate, rejected=f"segment is not a valid artifact: {error}")
    log.event("reuse_started", candidate_id=candidate.candidate_id, source=candidate.source_name,
              revision=candidate.source_version, start_node=candidate.start_node, actions=candidate.action_count,
              turn=turn, purpose=DISCOVERY_REUSE)
    library.count += 1
    escalator = Escalator(NoOperator(), SessionControl(), log)
    try:
        result = run_replay(segment, dict(params), surface, policy, escalator, log, purpose=DISCOVERY_REUSE)
    except Exception as error:                     # a crash mid-segment: the session state is unknown
        result = failure("reuse_crashed", None, "the segment to replay", f"{type(error).__name__}: {error}")
    for entry_ in result.executed_path:
        log.event("reuse_step_executed", candidate_id=candidate.candidate_id, node_id=entry_["id"],
                  action=entry_["action"]["action"], checkpoint=entry_["action"].get("checkpoint"),
                  effect=entry_["effect"], retry_safety=entry_["retry_safety"])
    return Execution(candidate, result=result, segment=segment)


def classify(result: ReplayResult) -> str:
    """success | business_outcome | before_acting | partial_verified | uncertain.

    A failure is `before_acting` when nothing physical was attempted, `partial_verified` when
    every attempt belongs to a completed node and the last completed node has a checkpoint (the
    session provably stands at that node's state), and `uncertain` otherwise: a physical action
    whose result is unproven, which must never be repeated blindly or handed on silently.
    """
    if result.status in ("success", "business_outcome"):
        return result.status
    if result.outcome_code == "reuse_crashed":
        return "uncertain"
    if result.performed_attempts == 0:
        return "before_acting"
    physical = sum(1 for e in result.executed_path if e["action"]["action"] in PHYSICAL_ACTIONS)
    if result.performed_attempts == physical and result.executed_path and result.executed_path[-1]["action"].get("checkpoint"):
        return "partial_verified"
    return "uncertain"


def verified_entries(result: ReplayResult, complete: bool) -> list[dict]:
    """The executed entries safe to import: all of them after a verified success, else up to and
    including the last completed node with a checkpoint."""
    entries = list(result.executed_path)
    if complete:
        return entries
    while entries and not entries[-1]["action"].get("checkpoint"):
        entries.pop()
    return entries


CONTRACT_FIELDS = ("type", "required", "pattern", "items", "min_items", "max_items")


def importable_nodes(entries: list[dict], source: Artifact, output_contract: dict | None, produced: dict | None = None,
                     params: dict | None = None) -> tuple[list[GraphNode], dict]:
    """Copies of the executed nodes (action, effect, retry safety exact) and the output definitions they carry.

    An extraction node is imported only when its output was really produced by the replay (`produced`:
    the replay's outputs; an optional output that came back absent or null is never imported). Its
    definition is the approved source's own (type, requiredness, pattern or items rule, description),
    with the example refreshed from the value just read, parameterized like any recorded sample. When the
    discovery declares an output contract that contract is authoritative: the output is imported only if
    the source's definition agrees with it, and an incompatible one is left out rather than adopted.
    """
    contract = output_contract or {}
    nodes: list[GraphNode] = []
    outputs: dict = {}
    for entry in entries:
        action = artifact_module.action_from_dict(entry["action"], f"executed node {entry['id']}")
        if action.action in ("extract", "extract_many"):
            name = action.value
            recorded = source.outputs.get(name)
            if recorded is None or (produced is not None and produced.get(name) is None):
                continue                                   # never for a value the replay did not deliver
            if contract:
                declared = contract.get(name)
                if declared is None or any(declared.get(k) != recorded.get(k) for k in CONTRACT_FIELDS):
                    continue                               # the discovery's contract wins; nothing is adopted
            spec = copy.deepcopy(recorded)
            if produced is not None and params is not None:
                spec["example"] = example_of(produced[name], params)
            outputs[name] = spec
        nodes.append(GraphNode(id=entry["id"], kind="action", action=action, effect=entry["effect"],
                               retry_safety=entry["retry_safety"]))
    return nodes, outputs


def example_of(value, params: dict):
    """A value just read, recorded the way discovery records samples: literal inputs become placeholders."""
    if isinstance(value, list):
        return [parameterize(str(item), params) for item in value]
    return parameterize(str(value), params)
