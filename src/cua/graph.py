"""Capability graphs (schema 2.0): validation, serialization, and the linear-artifact adapter.

A version-2 artifact keeps the version-1 contract (inputs, outputs, outcomes, success,
provenance) and replaces the ordered step list with a graph. Action nodes carry what a
version-1 step does (GraphAction) with the step's id and risk expressed once, as the
node's id and effect; decision nodes only branch; terminal nodes end the flow with a
replay status. Edges carry typed guards and a priority: leaving a node, the
lowest-priority edge whose guards all hold is the one to take, and an "always" edge is the
fallback, tried last. The graph is acyclic;
retries belong to a node's retry policy, not to edges, and an action whose effect is
irreversible or unknown is never retried automatically.

Nothing here executes a graph. Replay still runs version-1 artifacts (replay.py), and
discovery still records them. `from_linear` and `load_graph` are the entry points a
graph replay will use; `nodes_by_id` and `outgoing` are its lookups.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict
from pathlib import Path

from . import artifact as artifact_module
from .artifact import (LOCATOR_TEXT_FIELDS, ArtifactError, read_json, validate_action, validate_contract,
                       validate_placeholders)
from .models import Artifact, ArtifactV2, GraphAction, GraphEdge, GraphNode, Guard, Locator, Step
from .policy import redact

GRAPH_SCHEMA_VERSION = "2.0"
NODE_KINDS = {"action", "decision", "terminal"}
EFFECTS = {"none", "reversible", "irreversible", "unknown"}
RETRY_SAFETIES = {"safe", "verify_before_retry", "never_retry"}
TERMINAL_STATUSES = {"success", "business_outcome", "failure"}
# The payload fields each guard kind carries; every other payload field must be absent.
GUARD_FIELDS = {
    "always": (),
    "input_equals": ("input", "value"),
    "text_visible": ("value",),
    "url_matches": ("pattern",),
    "element_present": ("target",),
    "dialog_contains": ("value",),
}
GUARD_PAYLOAD = ("input", "value", "pattern", "target")

GRAPH_KEYS = {"schema_version", "name", "version", "status", "description", "surface", "inputs", "outputs",
              "entry_node", "nodes", "edges", "outcomes", "success", "provenance"}
NO_AUTOMATIC_RETRY = {"irreversible", "unknown"}   # effects that never_retry is the only safe answer for
NODE_KEYS = {"id", "kind", "action", "effect", "retry_safety", "status", "outcome_code"}
ACTION_KEYS = {"action", "target", "value", "checkpoint"}
EDGE_KEYS = {"source", "target", "guards", "priority"}
GUARD_KEYS = {"kind", *GUARD_PAYLOAD}


# ---------- lookups ----------

def nodes_by_id(graph: ArtifactV2) -> dict[str, GraphNode]:
    return {node.id: node for node in graph.nodes}


def outgoing(graph: ArtifactV2, node_id: str) -> list[GraphEdge]:
    """The edges leaving a node in the order they are to be tried."""
    return sorted((edge for edge in graph.edges if edge.source == node_id), key=lambda edge: edge.priority)


# ---------- validate ----------

def validate_graph(graph: ArtifactV2) -> None:
    """Reject graphs a deterministic replay could not run. Raises ArtifactError with a precise reason."""
    if graph.schema_version != GRAPH_SCHEMA_VERSION:
        raise ArtifactError(f"unsupported schema_version {graph.schema_version!r}, expected {GRAPH_SCHEMA_VERSION!r}")
    validate_contract(graph)
    validate_inputs(graph.inputs)
    if not graph.nodes:
        raise ArtifactError("graph has no nodes")
    nodes = _validate_nodes(graph)
    if graph.entry_node not in nodes:
        raise ArtifactError(f"entry_node {graph.entry_node!r} is not a node")
    if nodes[graph.entry_node].kind == "terminal":
        raise ArtifactError(f"entry_node {graph.entry_node!r} is a terminal node; the flow would do nothing")
    _validate_edges(graph, nodes)
    _validate_reachable(graph, nodes)
    _validate_acyclic(graph, nodes)


def validate_inputs(inputs: dict) -> None:
    """Version-2 input rules: a selector is always required; required_when lists non-empty AND-groups
    (OR-ed together) over selector inputs only, and only on inputs that are not always required."""
    selectors = {name for name, spec in inputs.items() if spec.get("selector")}
    for name, spec in inputs.items():
        if spec.get("selector") and not spec.get("required"):
            raise ArtifactError(f"input {name!r}: a selector input must be required")
        clauses = spec.get("required_when")
        if clauses is None:
            continue
        if spec.get("required"):
            raise ArtifactError(f"input {name!r}: required_when is meaningless on an input that is always required")
        if not isinstance(clauses, list) or not clauses:
            raise ArtifactError(f"input {name!r}: required_when must be a non-empty list of conditions")
        for clause in clauses:
            if not isinstance(clause, dict) or not clause:
                raise ArtifactError(f"input {name!r}: each required_when condition must be a non-empty object")
            for key, value in clause.items():
                if key not in selectors:
                    raise ArtifactError(f"input {name!r}: required_when refers to {key!r}, "
                                        f"which is not a selector input")
                if not isinstance(value, str):
                    raise ArtifactError(f"input {name!r}: required_when value for {key!r} must be a string")


def missing_inputs(inputs: dict, params: dict) -> list[str]:
    """Declared inputs the caller did not supply but must: always-required ones, and conditional ones
    whose required_when has a condition that the supplied parameters satisfy."""
    def satisfied(clause: dict) -> bool:
        return all(key in params and str(params[key]) == value for key, value in clause.items())

    return [name for name, spec in inputs.items() if name not in params
            and (spec.get("required") or any(satisfied(clause) for clause in spec.get("required_when") or []))]


def _validate_nodes(graph: ArtifactV2) -> dict[str, GraphNode]:
    declared_inputs = set(graph.inputs)
    outcome_codes = {outcome.get("code") for outcome in graph.outcomes}
    nodes: dict[str, GraphNode] = {}
    for node in graph.nodes:
        if not node.id or node.id in nodes:
            raise ArtifactError(f"node id {node.id!r} is missing or duplicated")
        nodes[node.id] = node
        if node.kind not in NODE_KINDS:
            raise ArtifactError(f"node {node.id}: unknown kind {node.kind!r}, expected one of {sorted(NODE_KINDS)}")
        if node.effect not in EFFECTS:
            raise ArtifactError(f"node {node.id}: effect must be one of {sorted(EFFECTS)}, got {node.effect!r}")
        if node.retry_safety not in RETRY_SAFETIES:
            raise ArtifactError(f"node {node.id}: retry_safety must be one of {sorted(RETRY_SAFETIES)}, "
                                f"got {node.retry_safety!r}")
        if node.kind == "action":
            _validate_action_node(node, declared_inputs, graph.outputs)
        elif node.kind == "decision":
            _validate_decision_node(node)
        else:
            _validate_terminal_node(node, outcome_codes)
    return nodes


def _validate_action_node(node: GraphNode, declared_inputs: set[str], outputs: dict) -> None:
    if node.action is None:
        raise ArtifactError(f"node {node.id}: an action node needs an action")
    if node.status is not None or node.outcome_code is not None:
        raise ArtifactError(f"node {node.id}: only terminal nodes carry a status or outcome_code")
    if node.effect in NO_AUTOMATIC_RETRY and node.retry_safety != "never_retry":
        raise ArtifactError(f"node {node.id}: an action with effect {node.effect!r} must have "
                            f"retry_safety 'never_retry', got {node.retry_safety!r}")
    validate_action(node.action, declared_inputs, outputs, f"node {node.id}")


def _validate_decision_node(node: GraphNode) -> None:
    if node.action is not None:
        raise ArtifactError(f"node {node.id}: a decision node does not carry an action")
    if node.status is not None or node.outcome_code is not None:
        raise ArtifactError(f"node {node.id}: only terminal nodes carry a status or outcome_code")
    _validate_passive(node)


def _validate_terminal_node(node: GraphNode, outcome_codes: set) -> None:
    if node.action is not None:
        raise ArtifactError(f"node {node.id}: a terminal node does not carry an action")
    _validate_passive(node)
    if node.status not in TERMINAL_STATUSES:
        raise ArtifactError(f"node {node.id}: a terminal node needs a status in {sorted(TERMINAL_STATUSES)}, "
                            f"got {node.status!r}")
    if node.status == "success" and node.outcome_code is not None:
        raise ArtifactError(f"node {node.id}: a success terminal has no outcome_code")
    if node.status != "success" and not node.outcome_code:
        raise ArtifactError(f"node {node.id}: a {node.status} terminal needs an outcome_code")
    if node.status == "business_outcome" and node.outcome_code not in outcome_codes:
        raise ArtifactError(f"node {node.id}: outcome_code {node.outcome_code!r} is not a declared outcome")


def _validate_passive(node: GraphNode) -> None:
    """Decision and terminal nodes do nothing to the surface, so effect and retry policy do not apply."""
    if node.effect != "none":
        raise ArtifactError(f"node {node.id}: a {node.kind} node has effect 'none', got {node.effect!r}")
    if node.retry_safety != "safe":
        raise ArtifactError(f"node {node.id}: a {node.kind} node has retry_safety 'safe', got {node.retry_safety!r}")


def _validate_edges(graph: ArtifactV2, nodes: dict[str, GraphNode]) -> None:
    declared_inputs = set(graph.inputs)
    priorities: dict[str, set[int]] = {}
    fallbacks: dict[str, GraphEdge] = {}
    for edge in graph.edges:
        where = f"edge {edge.source}->{edge.target}"
        for end in (edge.source, edge.target):
            if end not in nodes:
                raise ArtifactError(f"{where}: {end!r} is not a node")
        if nodes[edge.source].kind == "terminal":
            raise ArtifactError(f"{where}: a terminal node has no outgoing edges")
        if not isinstance(edge.priority, int) or isinstance(edge.priority, bool):
            raise ArtifactError(f"{where}: priority must be an integer, got {edge.priority!r}")
        if edge.priority in priorities.setdefault(edge.source, set()):
            raise ArtifactError(f"{where}: priority {edge.priority} is used twice on edges leaving {edge.source}")
        priorities[edge.source].add(edge.priority)
        if not edge.guards:
            raise ArtifactError(f"{where}: an edge needs at least one guard (use an 'always' guard)")
        if len(edge.guards) > 1 and any(guard.kind == "always" for guard in edge.guards):
            raise ArtifactError(f"{where}: an 'always' guard must be the only guard on its edge")
        for guard in edge.guards:
            validate_guard(guard, declared_inputs, where)
        if edge.guards[0].kind == "always":
            if edge.source in fallbacks:
                raise ArtifactError(f"{where}: node {edge.source} already has an 'always' edge to "
                                    f"{fallbacks[edge.source].target}; only one fallback edge per node")
            fallbacks[edge.source] = edge
    for node in graph.nodes:
        if node.kind != "terminal" and node.id not in priorities:
            raise ArtifactError(f"node {node.id}: {node.kind} nodes need at least one outgoing edge")
    for source, fallback in fallbacks.items():
        if fallback.priority < max(priorities[source]):
            raise ArtifactError(f"edge {source}->{fallback.target}: an 'always' edge must have the greatest priority "
                                f"among the edges leaving {source}, or it would hide the guarded ones")


def validate_guard(guard: Guard, declared_inputs: set[str], where: str) -> None:
    if guard.kind not in GUARD_FIELDS:
        raise ArtifactError(f"{where}: unsupported guard kind {guard.kind!r}, expected one of {sorted(GUARD_FIELDS)}")
    expected = GUARD_FIELDS[guard.kind]
    for name in GUARD_PAYLOAD:
        present = getattr(guard, name) is not None
        if name in expected and not present:
            raise ArtifactError(f"{where}: {guard.kind} guard needs {name}")
        if name not in expected and present:
            raise ArtifactError(f"{where}: {guard.kind} guard does not take {name}")
    if guard.kind == "input_equals" and guard.input not in declared_inputs:
        raise ArtifactError(f"{where}: input_equals refers to undeclared input {guard.input!r}")
    if guard.kind == "url_matches":
        try:
            re.compile(guard.pattern)
        except re.error as error:
            raise ArtifactError(f"{where}: url_matches pattern {guard.pattern!r} is not a valid regex: "
                                f"{error}") from error
    if guard.kind == "element_present" and not guard.target.strategies:
        raise ArtifactError(f"{where}: element_present needs a locator with at least one strategy")
    texts = {"value": guard.value, "pattern": guard.pattern}
    if guard.target:
        for index, strategy in enumerate(guard.target.strategies):
            texts.update({f"target strategy {index} {key}": str(strategy.get(key, "")) for key in LOCATOR_TEXT_FIELDS})
    validate_placeholders({key: text for key, text in texts.items() if text}, declared_inputs, f"{where} guard")


def _validate_reachable(graph: ArtifactV2, nodes: dict[str, GraphNode]) -> None:
    seen = {graph.entry_node}
    frontier = [graph.entry_node]
    while frontier:
        current = frontier.pop()
        for edge in outgoing(graph, current):
            if edge.target not in seen:
                seen.add(edge.target)
                frontier.append(edge.target)
    unreachable = [node_id for node_id in nodes if node_id not in seen]
    if unreachable:
        raise ArtifactError(f"nodes {unreachable} are unreachable from entry_node {graph.entry_node!r}")


def _validate_acyclic(graph: ArtifactV2, nodes: dict[str, GraphNode]) -> None:
    """Depth-first search from the entry; a back edge is a cycle, reported as the path that closes it."""
    on_path: list[str] = []
    finished: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in on_path:
            cycle = on_path[on_path.index(node_id):] + [node_id]
            raise ArtifactError(f"graph has a cycle: {' -> '.join(cycle)}")
        if node_id in finished:
            return
        on_path.append(node_id)
        for edge in outgoing(graph, node_id):
            visit(edge.target)
        on_path.pop()
        finished.add(node_id)

    for node_id in nodes:
        visit(node_id)


# ---------- serialize ----------

def to_dict(graph: ArtifactV2) -> dict:
    """Plain JSON-ready data in a fixed key order; guards carry only the fields their kind uses."""
    data = asdict(graph)
    data["edges"] = [edge_to_dict(edge) for edge in graph.edges]
    return data


def edge_to_dict(edge: GraphEdge) -> dict:
    return {"source": edge.source, "target": edge.target,
            "guards": [guard_to_dict(guard) for guard in edge.guards], "priority": edge.priority}


def guard_to_dict(guard: Guard) -> dict:
    return {key: value for key, value in asdict(guard).items() if value is not None}


def dumps(graph: ArtifactV2, secrets: tuple[str, ...] = ()) -> str:
    """Stable, reviewable JSON: dataclass field order, two-space indent, secrets redacted."""
    return json.dumps(redact(to_dict(graph), secrets), indent=2, ensure_ascii=False)


def from_dict(data) -> ArtifactV2:
    """Rebuild the graph dataclasses from plain JSON, naming the exact node, edge, or guard at fault."""
    if not isinstance(data, dict):
        raise ArtifactError("artifact must be a JSON object")
    _require_keys(data, GRAPH_KEYS, "artifact")
    nodes = [node_from_dict(raw, f"node {index}") for index, raw in enumerate(_list(data["nodes"], "nodes"))]
    edges = [edge_from_dict(raw, f"edge {index}") for index, raw in enumerate(_list(data["edges"], "edges"))]
    try:
        version = int(data["version"])
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"artifact version must be an integer, got {data['version']!r}") from error
    return ArtifactV2(
        schema_version=str(data["schema_version"]), name=data["name"], version=version, status=data["status"],
        description=data["description"], surface=data["surface"], inputs=data["inputs"], outputs=data["outputs"],
        entry_node=data["entry_node"], nodes=nodes, edges=edges, outcomes=data["outcomes"],
        success=data["success"], provenance=data["provenance"],
    )


def node_from_dict(raw, where: str) -> GraphNode:
    raw = _object(raw, where)
    where = f"{where} ({raw['id']!r})" if isinstance(raw.get("id"), str) else where
    _require_keys(raw, {"id", "kind"}, where)
    _reject_unknown_keys(raw, NODE_KEYS, where)
    action = None
    if raw.get("action") is not None:
        action = action_from_dict(raw["action"], f"{where} action")
    return GraphNode(id=raw["id"], kind=raw["kind"], action=action, effect=raw.get("effect", "unknown"),
                     retry_safety=raw.get("retry_safety", "never_retry"), status=raw.get("status"),
                     outcome_code=raw.get("outcome_code"))


def action_from_dict(raw, where: str) -> GraphAction:
    raw = _object(raw, where)
    _require_keys(raw, {"action", "target"}, where)
    _reject_unknown_keys(raw, ACTION_KEYS, where)
    target = None
    if raw["target"] is not None:
        strategies = _object(raw["target"], f"{where} target").get("strategies")
        if not isinstance(strategies, list):
            raise ArtifactError(f"{where} target needs a strategies list")
        target = Locator(strategies=strategies)
    return GraphAction(action=raw["action"], target=target, value=raw.get("value"), checkpoint=raw.get("checkpoint"))


def edge_from_dict(raw, where: str) -> GraphEdge:
    raw = _object(raw, where)
    _require_keys(raw, {"source", "target", "guards"}, where)
    _reject_unknown_keys(raw, EDGE_KEYS, where)
    where = f"{where} ({raw['source']!r}->{raw['target']!r})"
    guards = [guard_from_dict(guard, f"{where} guard {index}")
              for index, guard in enumerate(_list(raw["guards"], f"{where} guards"))]
    return GraphEdge(source=raw["source"], target=raw["target"], guards=guards, priority=raw.get("priority", 0))


def guard_from_dict(raw, where: str) -> Guard:
    raw = _object(raw, where)
    _require_keys(raw, {"kind"}, where)
    _reject_unknown_keys(raw, GUARD_KEYS, where)
    target = None
    if raw.get("target") is not None:
        strategies = _object(raw["target"], f"{where} target").get("strategies")
        if not isinstance(strategies, list):
            raise ArtifactError(f"{where} target needs a strategies list")
        target = Locator(strategies=strategies)
    return Guard(kind=raw["kind"], input=raw.get("input"), value=raw.get("value"), pattern=raw.get("pattern"),
                 target=target)


def _object(raw, where: str) -> dict:
    if not isinstance(raw, dict):
        raise ArtifactError(f"{where} must be a JSON object, got {type(raw).__name__}")
    return raw


def _list(raw, where: str) -> list:
    if not isinstance(raw, list):
        raise ArtifactError(f"{where} must be a JSON list, got {type(raw).__name__}")
    return raw


def _require_keys(raw: dict, required: set[str], where: str) -> None:
    missing = required - set(raw)
    if missing:
        raise ArtifactError(f"{where} is missing keys: {sorted(missing)}")


def _reject_unknown_keys(raw: dict, known: set[str], where: str) -> None:
    unknown = set(raw) - known
    if unknown:
        raise ArtifactError(f"{where} has unknown keys: {sorted(unknown)}")


# ---------- save / load ----------

def save_graph(graph: ArtifactV2, secrets: tuple[str, ...] = (), path: str | Path | None = None) -> Path:
    """Persist a graph as JSON. Never overwrites: a converted version-1 artifact keeps its version
    number, so writing it next to the original would replace the file replay runs today."""
    validate_graph(graph)
    raw_text = json.dumps(to_dict(graph), ensure_ascii=False)
    for secret in secrets:
        if secret and secret in raw_text:
            raise ArtifactError("refusing to save: a sensitive input value was not parameterized")
    path = Path(path) if path else artifact_module.ARTIFACTS_DIR / f"{graph.name}.v{graph.version}.json"
    if path.exists():
        raise ArtifactError(f"refusing to overwrite {path}: artifacts are immutable once saved; "
                            f"pass a new path or bump the version")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(graph, secrets), encoding="utf-8")
    return path


def load_graph(path: str | Path) -> ArtifactV2:
    """Load either schema version as a validated in-memory graph.

    A version-1 file is converted with `from_linear`; the file itself is never touched.
    """
    data = read_json(path)
    version = data.get("schema_version") if isinstance(data, dict) else None
    if version == artifact_module.SCHEMA_VERSION:
        linear = artifact_module.from_dict(data)
        artifact_module.validate(linear)
        return from_linear(linear)
    if version == GRAPH_SCHEMA_VERSION:
        graph = from_dict(data)
        validate_graph(graph)
        return graph
    raise ArtifactError(f"unsupported schema_version {version!r}, expected {artifact_module.SCHEMA_VERSION!r} "
                        f"or {GRAPH_SCHEMA_VERSION!r}")


# ---------- version-1 adapter ----------

SUCCESS_NODE_ID = "success"
# What a version-1 step's action and risk say about it, in version-2 terms: a safe navigate or
# extract changes nothing, a safe click or type can be undone, a risky action is irreversible.
# Retry policy follows: non-risky steps are re-verified before a retry (what replay does today),
# irreversible ones are never retried automatically.
LINEAR_EFFECTS = {"navigate": "none", "extract": "none", "click": "reversible", "type": "reversible"}


def from_linear(artifact: Artifact) -> ArtifactV2:
    """Convert a linear artifact into the equivalent graph, in memory.

    step 1 -> step 2 -> ... -> step N -> success terminal, chained by unconditional edges in
    the recorded order. Step.id becomes the node id, Step.risk and the action kind become the
    node's effect, and the rest of the step is the node's action. Metadata is carried over
    unchanged (plus a provenance note); the graph owns deep copies, so the original artifact
    is never modified.
    """
    artifact_module.validate(artifact)
    if any(step.id == SUCCESS_NODE_ID for step in artifact.steps):
        raise ArtifactError(f"cannot convert: step id {SUCCESS_NODE_ID!r} collides with the success terminal")
    nodes = [action_node(step) for step in artifact.steps]
    nodes.append(GraphNode(id=SUCCESS_NODE_ID, kind="terminal", effect="none", retry_safety="safe", status="success"))
    edges = [GraphEdge(source=before.id, target=after.id, guards=[Guard(kind="always")], priority=0)
             for before, after in zip(nodes, nodes[1:])]
    graph = ArtifactV2(
        schema_version=GRAPH_SCHEMA_VERSION, name=artifact.name, version=artifact.version, status=artifact.status,
        description=artifact.description, surface=copy.deepcopy(artifact.surface),
        inputs=copy.deepcopy(artifact.inputs), outputs=copy.deepcopy(artifact.outputs),
        entry_node=nodes[0].id, nodes=nodes, edges=edges, outcomes=copy.deepcopy(artifact.outcomes),
        success=copy.deepcopy(artifact.success),
        provenance={**copy.deepcopy(artifact.provenance), "converted_from_schema": artifact.schema_version},
    )
    validate_graph(graph)
    return graph


def action_node(step: Step) -> GraphNode:
    action = GraphAction(action=step.action, target=copy.deepcopy(step.target), value=step.value,
                         checkpoint=copy.deepcopy(step.checkpoint))
    return GraphNode(id=step.id, kind="action", action=action, effect=linear_effect(step),
                     retry_safety=linear_retry_safety(step))


def linear_effect(step: Step) -> str:
    return "irreversible" if step.risk == "risky" else LINEAR_EFFECTS[step.action]


def linear_retry_safety(step: Step) -> str:
    return "never_retry" if step.risk == "risky" else "verify_before_retry"
