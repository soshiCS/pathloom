"""Conservative merge of verified linear graphs into one capability graph: a prefix tree.

Each scenario's discovery run recorded a linear graph. Traces are merged by walking their
action nodes in specification order and sharing a node only while the *complete* normalized
meaning of the node matches (action kind, parameterized value, locator ladder, checkpoint,
effect, retry safety; node ids are ignored because every run numbers its own). At the first
difference a decision node is inserted and each branch is guarded by the complete selector
assignment of the scenarios that took it. Merging continues inside each branch. Nothing is
merged after paths diverge: a common suffix stays duplicated, because two screens reached by
different routes are not assumed to be the same state. A scenario that ends where another
continues becomes a guarded edge to the success terminal next to the continuing action.
Scenarios with identical complete traces share one path; provenance names them all. When the
campaign declares selectors, the graph starts with a decision node that admits only the declared
selector assignments, so an undeclared combination stops before any action runs. A merged node
keeps its source node's effect and retry safety exactly.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .artifact import SCHEMA_VERSION, SUCCESS_NODE_ID, ArtifactError, copy_node, linear_path, validate
from .models import Artifact, GraphEdge, GraphNode, Guard, ScenarioTrace

MERGE_STRATEGY = "prefix_tree"
COMPATIBILITY_FIELDS = ("name", "description", "surface.kind", "surface.entry_url", "surface.allowed_hosts", "success")
OUTPUT_CONTRACT = ("type", "required", "pattern", "items", "min_items", "max_items")   # what replay checks; description and example are free


class MergeError(ArtifactError):
    """The traces do not describe one capability, or cannot be merged without guessing."""


@dataclass
class TreeNode:
    """One shared action in the prefix tree, and which scenarios pass through or stop here."""
    key: str
    node: GraphNode | None
    scenarios: list[str] = field(default_factory=list)
    ends: list[str] = field(default_factory=list)
    children: list[TreeNode] = field(default_factory=list)


def merge_traces(traces: list[ScenarioTrace], selectors: list[str], campaign_id: str, version: int = 1) -> Artifact:
    """Merge the traces, in the given order, into a validated draft graph with campaign provenance."""
    if not traces:
        raise MergeError("nothing to merge: no successful traces")
    check_compatible(traces)
    paths = {trace.scenario.name: trace_path(trace) for trace in traces}
    first = traces[0].artifact
    builder = GraphBuilder(traces, selectors)
    entry = builder.emit(build_tree(traces, paths), source=None)
    builder.nodes.append(builder.success)   # the one terminal, listed last for readability
    graph = Artifact(
        schema_version=SCHEMA_VERSION, name=first.name, version=version, status="draft",
        description=first.description, surface=copy.deepcopy(first.surface),
        inputs=merge_inputs(traces, selectors), outputs=copy.deepcopy(first.outputs),
        entry_node=entry, nodes=builder.nodes, edges=builder.sorted_edges(), outcomes=merge_outcomes(traces),
        success=copy.deepcopy(first.success), provenance=provenance(traces, selectors, campaign_id, builder, paths),
    )
    validate(graph)
    return graph


def trace_path(trace: ScenarioTrace) -> list[GraphNode]:
    """The action nodes a scenario recorded, in order; a trace must be a linear graph."""
    try:
        return linear_path(trace.artifact)
    except ArtifactError as error:
        raise MergeError(f"scenario {trace.scenario.name!r} did not record a linear graph: {error}") from error


# ---------- compatibility ----------

def check_compatible(traces: list[ScenarioTrace]) -> None:
    """Every trace must describe the same capability contract; the first trace is the reference."""
    reference = traces[0]
    for trace in traces[1:]:
        for field_name in COMPATIBILITY_FIELDS:
            expected, actual = contract_field(reference, field_name), contract_field(trace, field_name)
            if expected != actual:
                raise MergeError(f"scenarios {reference.scenario.name!r} and {trace.scenario.name!r} disagree on "
                                 f"{field_name}: {expected!r} vs {actual!r}")
        check_outputs_compatible(reference, trace)


def check_outputs_compatible(reference: ScenarioTrace, trace: ScenarioTrace) -> None:
    """Same output names, and for each the same type, requiredness and parsing pattern."""
    who = f"scenarios {reference.scenario.name!r} and {trace.scenario.name!r}"
    expected, actual = reference.artifact.outputs, trace.artifact.outputs
    if set(expected) != set(actual):
        raise MergeError(f"{who} disagree on outputs: {sorted(expected)} vs {sorted(actual)}")
    for name in expected:
        for field_name in OUTPUT_CONTRACT:
            default = True if field_name == "required" else None
            left, right = expected[name].get(field_name, default), actual[name].get(field_name, default)
            if left != right:
                raise MergeError(f"{who} disagree on outputs: output {name!r} {field_name} {left!r} vs {right!r}")


def contract_field(trace: ScenarioTrace, field_name: str):
    artifact = trace.artifact
    if field_name == "surface.allowed_hosts":
        return sorted(artifact.surface.get("allowed_hosts") or [])
    if field_name.startswith("surface."):
        return artifact.surface.get(field_name.split(".", 1)[1])
    return getattr(artifact, field_name)


def merge_outcomes(traces: list[ScenarioTrace]) -> list[dict]:
    """Union by code; the same code must mean the same thing everywhere. The first source is kept."""
    merged: dict[str, tuple[dict, str]] = {}
    for trace in traces:
        for outcome in trace.artifact.outcomes:
            definition = {key: value for key, value in outcome.items() if key != "source"}
            code = outcome["code"]
            if code in merged:
                known, owner = merged[code]
                if {key: value for key, value in known.items() if key != "source"} != definition:
                    raise MergeError(f"outcome {code!r} is defined differently by scenarios {owner!r} and "
                                     f"{trace.scenario.name!r}")
                continue
            merged[code] = (copy.deepcopy(outcome), trace.scenario.name)
    return [outcome for outcome, _ in merged.values()]


def merge_inputs(traces: list[ScenarioTrace], selectors: list[str]) -> dict:
    """Selectors are always required; an input every scenario has stays required; a path-specific
    input is required only when the selector assignment of a scenario that used it holds."""
    names: list[str] = []
    for trace in traces:
        names += [name for name in trace.artifact.inputs if name not in names]
    merged = {}
    for name in names:
        users = [trace for trace in traces if name in trace.artifact.inputs]
        specs = [trace.artifact.inputs[name] for trace in users]
        types = sorted({spec.get("type", "string") for spec in specs})
        if len(types) > 1:
            raise MergeError(f"input {name!r} has conflicting types across scenarios: {types}")
        spec = {"type": types[0], "required": True, "sensitive": any(s.get("sensitive") for s in specs),
                "description": specs[0].get("description", "")}
        if name in selectors:
            spec["selector"] = True
        elif len(users) < len(traces):
            spec["required"] = False
            spec["required_when"] = unique([selector_assignment(trace, selectors) for trace in users])
        merged[name] = spec
    return merged


def selector_assignment(trace: ScenarioTrace, selectors: list[str]) -> dict:
    return {selector: str(trace.scenario.params[selector]) for selector in selectors}


def unique(clauses: list[dict]) -> list[dict]:
    seen: list[dict] = []
    for clause in clauses:
        if clause not in seen:
            seen.append(clause)
    return seen


# ---------- prefix tree ----------

def node_key(node: GraphNode) -> str:
    """The complete execution meaning of an action node, without its id."""
    return json.dumps({"action": asdict(node.action), "effect": node.effect, "retry_safety": node.retry_safety},
                      sort_keys=True)


def build_tree(traces: list[ScenarioTrace], paths: dict[str, list[GraphNode]]) -> TreeNode:
    root = TreeNode(key="", node=None)
    for trace in traces:
        tree = root
        tree.scenarios.append(trace.scenario.name)
        for node in paths[trace.scenario.name]:
            key = node_key(node)
            child = next((c for c in tree.children if c.key == key), None)
            if child is None:
                child = TreeNode(key=key, node=copy.deepcopy(node))
                tree.children.append(child)
            child.scenarios.append(trace.scenario.name)
            tree = child
        tree.ends.append(trace.scenario.name)
    return root


class GraphBuilder:
    """Turns the prefix tree into nodes and edges with deterministic ids, priorities and scenario paths."""

    def __init__(self, traces: list[ScenarioTrace], selectors: list[str]):
        self.selectors = selectors
        self.order = {trace.scenario.name: index for index, trace in enumerate(traces)}
        self.assignment = {trace.scenario.name: selector_assignment(trace, selectors) for trace in traces}
        self.nodes: list[GraphNode] = []
        self.edges: list[GraphEdge] = []
        self.paths: dict[str, list[str]] = {trace.scenario.name: [] for trace in traces}
        self.actions = 0
        self.decisions = 0
        self.success = GraphNode(id=SUCCESS_NODE_ID, kind="terminal", effect="none", retry_safety="safe",
                                 status="success")

    def emit(self, tree: TreeNode, source: str | None) -> str:
        """Emit the edges leaving `source` for this tree node (the root when source is None); returns the entry id.

        At the root of a campaign with selectors the decision is the entry gate: one guarded edge
        per declared scenario, all pointing at the shared first action.
        """
        branches = [(child.scenarios, child) for child in tree.children]
        if tree.ends:
            branches.append((tree.ends, None))
        branches.sort(key=lambda branch: min(self.order[name] for name in branch[0]))
        gate = source is None and bool(self.selectors)
        if len(branches) == 1 and not gate:
            scenarios, child = branches[0]
            target = self.emit_action(child) if child is not None else self.emit_success(scenarios)
            if source is not None:
                self.edges.append(GraphEdge(source=source, target=target, guards=[Guard(kind="always")], priority=0))
            return target
        decision = self.emit_decision(tree.scenarios)
        if source is not None:
            self.edges.append(GraphEdge(source=source, target=decision, guards=[Guard(kind="always")], priority=0))
        for scenarios, child in branches:
            target = self.emit_action(child) if child is not None else self.emit_success(scenarios)
            for name in scenarios:
                self.edges.append(GraphEdge(source=decision, target=target, guards=self.guards_for(name),
                                            priority=self.order[name]))
        return decision

    def emit_action(self, tree: TreeNode) -> str:
        self.actions += 1
        node = copy_node(tree.node, f"s{self.actions}")   # effect and retry safety travel unchanged
        self.nodes.append(node)
        for name in tree.scenarios:
            self.paths[name].append(node.id)
        self.emit(tree, source=node.id)
        return node.id

    def emit_decision(self, scenarios: list[str]) -> str:
        self.decisions += 1
        node = GraphNode(id=f"d{self.decisions}", kind="decision", effect="none", retry_safety="safe")
        self.nodes.append(node)
        for name in scenarios:
            self.paths[name].append(node.id)
        return node.id

    def emit_success(self, scenarios: list[str]) -> str:
        for name in scenarios:
            self.paths[name].append(SUCCESS_NODE_ID)
        return SUCCESS_NODE_ID

    def guards_for(self, scenario: str) -> list[Guard]:
        """The complete selector assignment, so an undeclared combination never matches a declared path."""
        return [Guard(kind="input_equals", input=selector, value=value)
                for selector, value in self.assignment[scenario].items()]

    def sorted_edges(self) -> list[GraphEdge]:
        position = {node.id: index for index, node in enumerate(self.nodes)}
        return sorted(self.edges, key=lambda edge: (position[edge.source], edge.priority))


# ---------- provenance ----------

def provenance(traces: list[ScenarioTrace], selectors: list[str], campaign_id: str, builder: GraphBuilder,
               paths: dict[str, list[GraphNode]]) -> dict:
    scenarios = []
    for trace in traces:
        origin = trace.artifact.provenance
        scenarios.append({"name": trace.scenario.name, "selectors": builder.assignment[trace.scenario.name],
                          "run_id": trace.run_id, "planner": trace.planner, "recorded_at": origin.get("recorded_at"),
                          "action_count": len(paths[trace.scenario.name]),
                          "interventions": origin.get("interventions", 0),
                          "node_path": list(builder.paths[trace.scenario.name]), "verified": True})
    return {
        "campaign_id": campaign_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "merge_strategy": MERGE_STRATEGY,
        "planners": sorted({trace.planner for trace in traces}),
        "selectors": list(selectors),
        "scenario_count": len(traces),
        "successful_scenario_count": len(traces),
        "node_count": len(builder.nodes),
        "edge_count": len(builder.edges),
        "scenarios": scenarios,
    }
