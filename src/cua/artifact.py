"""The capability artifact: one schema, a guarded acyclic graph. Build, validate, serialize, save, load.

An artifact is the contract between discovery and replay. It carries the typed inputs a caller
must supply, the typed outputs it gets back, the declared non-success outcomes, the success
checkpoint, and the flow as a graph: action nodes (what to do, with an effect and a retry
policy), decision nodes (typed guards choose the next edge) and terminal nodes (the result).
A plain discovery yields a linear graph: action -> action -> ... -> success, joined by `always`
edges; campaigns merge such graphs into branching ones. Never the model transcript, never an
input value.

`schema_version` is the file format and is always "2.0". `version` in `name.v<N>.json` is the
capability revision: a re-recording or an approval gets the next number.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .models import MAX_EXTRACT_TARGETS, OUTPUT_MODES, Artifact, GraphAction, GraphEdge, GraphNode, Guard, Locator
from .policy import redact

SCHEMA_VERSION = "2.0"
ARTIFACTS_DIR = Path("artifacts")
ACTIONS = {"navigate", "back", "click", "type", "select", "extract", "extract_many"}
READ_ACTIONS = {"extract", "extract_many"}
OUTPUT_TYPES = {"string", "number", "integer", "list"}
ITEM_TYPES = {"string", "number", "integer"}
CARDINALITY_KEYS = ("min_items", "max_items")   # optional bounds on a list output's length
OUTCOME_KINDS = {"business", "recoverable"}
NODE_KINDS = {"action", "decision", "terminal"}
EFFECTS = {"none", "reversible", "irreversible", "unknown"}
RETRY_SAFETIES = {"safe", "verify_before_retry", "never_retry"}
NO_AUTOMATIC_RETRY = {"irreversible", "unknown"}   # effects for which never_retry is the only safe answer
TERMINAL_STATUSES = {"success", "business_outcome", "failure"}
SUCCESS_NODE_ID = "success"
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
PLACEHOLDER_TOKEN = re.compile(r"(\{\{\w+\}\})")   # split-friendly: keeps the placeholders as segments
DETECT_KEYS = ("text_contains", "text_missing", "dialog_contains")
# What a step checkpoint may assert. Closed: replay refuses a condition it cannot verify rather than
# skipping it, so an unknown key can never pass vacuously.
CHECKPOINT_KEYS = ("text_contains", "url_contains", "selection_present", "target_selected", "target_absent")
RECOVER_ACTIONS = ("click",)
VERSIONED = re.compile(r"\.v(\d+)\.json$")
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
ARTIFACT_KEYS = {"schema_version", "name", "version", "status", "description", "surface", "inputs", "outputs",
                 "entry_node", "nodes", "edges", "outcomes", "success", "provenance"}
NODE_KEYS = {"id", "kind", "action", "effect", "retry_safety", "status", "outcome_code"}
ACTION_KEYS = {"action", "target", "value", "checkpoint", "targets", "mode"}
EDGE_KEYS = {"source", "target", "guards", "priority"}
GUARD_KEYS = {"kind", *GUARD_PAYLOAD}
LOCATOR_TEXT_FIELDS = ("name", "text", "context")


class ArtifactError(ValueError):
    """The artifact is malformed, unsupported, or unsafe to run."""


# ---------- parameters ----------

def parameterize(value: str | None, params: dict) -> str | None:
    """Replace discovery-time literals with {{name}} placeholders.

    Only whole tokens are replaced ("User" never matches inside "standard_user"), longer
    values go first so one value cannot clobber a longer one containing it, and values
    shorter than two characters are skipped: they would match almost anything. Text that
    is already a placeholder is left alone, so a value that happens to equal another
    input's name (lookup_method = "member_id" next to {{member_id}}) cannot corrupt it.
    """
    if value is None:
        return None
    segments = PLACEHOLDER_TOKEN.split(value)          # [text, {{placeholder}}, text, ...]
    for index in range(0, len(segments), 2):
        segments[index] = parameterize_text(segments[index], params)
    return "".join(segments)


def parameterize_text(text: str, params: dict) -> str:
    for name, literal in sorted(params.items(), key=lambda item: -len(str(item[1]))):
        literal = str(literal)
        if len(literal) < 2:
            continue
        bounded = r"(?<!\w)" + re.escape(literal) + r"(?!\w)"
        text = re.sub(bounded, "{{" + name + "}}", text)
    return text


def parameterize_locator(locator: Locator, params: dict, keep_structure: bool = False) -> Locator:
    """Locators can depend on inputs too: "the Add to cart button in the {{product_name}} card".

    When a rung depends on an input, coordinates are dropped: they point at whatever happened
    to be there during discovery. A structural path is retained after the semantic rungs, but
    replay may use it only to disambiguate controls that already matched a parameterized semantic
    rung. It can therefore distinguish duplicate rendered copies without becoming a fallback for
    a different input value.

    `keep_structure` remains accepted for callers recording ordered lists; structural rungs now
    survive for both lists and single targets under the guarded replay rule above.
    """
    strategies = []
    for strategy in locator.strategies:
        strategies.append({key: (parameterize(value, params) if key in LOCATOR_TEXT_FIELDS else value)
                           for key, value in strategy.items()})
    parameterized = [s for s in strategies if locator_placeholders(s)]
    if not parameterized:
        return Locator(strategies=strategies)
    parameterized += [s for s in strategies if s.get("kind") == "css" and s not in parameterized]
    return Locator(strategies=parameterized)


def locator_placeholders(strategy: dict) -> set[str]:
    names: set[str] = set()
    for key in LOCATOR_TEXT_FIELDS:
        names |= placeholders_in(str(strategy.get(key, "")))
    return names


def substitute_locator(locator: Locator, params: dict) -> Locator:
    strategies = []
    for strategy in locator.strategies:
        strategies.append({key: (substitute(value, params) if key in LOCATOR_TEXT_FIELDS else value)
                           for key, value in strategy.items()})
    return Locator(strategies=strategies)


def substitute(value: str | None, params: dict) -> str | None:
    """Fill {{name}} placeholders with concrete values (replay direction)."""
    if value is None:
        return None
    return PLACEHOLDER.sub(lambda match: str(params[match.group(1)]), value)


def placeholders_in(value: str | None) -> set[str]:
    return set(PLACEHOLDER.findall(value or ""))


# ---------- classification defaults for discovered actions ----------

def classify_effect(action: str, risk) -> str:
    """What a discovered action does to the world, from the policy's risk verdict ("safe", "risky" or
    "unknown"; a bare True means risky) and the action kind. An ambiguous verdict on a click is an unknown
    effect: never repeated automatically, and always confirmed by a person at replay."""
    if risk is True or risk == "risky":
        return "irreversible"
    if action in ("navigate", "back", "extract", "extract_many"):
        return "none"
    if action in ("click", "type", "select"):
        return "unknown" if risk == "unknown" else "reversible"
    return "unknown"


def classify_retry_safety(action: str, effect: str, checkpoint: dict | None) -> str:
    """The conservative retry rule for a discovered action.

    An irreversible or unknown effect is never repeated automatically. A read-only extraction
    is safe. An action with no lasting effect or a reversible one whose checkpoint can be
    verified first is retried only after that verification. A click, type or back without a
    checkpoint is never repeated: nothing could tell whether it already took effect (a second
    "back" would leave the page it should have arrived at). A navigation without a checkpoint is
    safe: reloading a page changes nothing.
    """
    if effect in NO_AUTOMATIC_RETRY:
        return "never_retry"
    if action in READ_ACTIONS:
        return "safe"
    if checkpoint:
        return "verify_before_retry"
    if action == "navigate":
        return "safe"
    return "never_retry"


# ---------- build ----------

def next_version(name: str) -> int:
    """Artifacts are immutable once saved; a re-recording gets the next revision number."""
    existing = [int(m.group(1)) for path in ARTIFACTS_DIR.glob(f"{name}.v*.json")
                if (m := VERSIONED.search(path.name))]
    return max(existing, default=0) + 1


def linear_nodes(actions: list[GraphAction], effects: list[str] | None = None,
                 retry_safeties: list[str] | None = None) -> list[GraphNode]:
    """Action nodes s1..sN for a straight path, classified by the defaults unless told otherwise."""
    nodes = []
    for index, action in enumerate(actions):
        effect = effects[index] if effects else classify_effect(action.action, False)
        retry = retry_safeties[index] if retry_safeties else classify_retry_safety(action.action, effect, action.checkpoint)
        nodes.append(GraphNode(id=f"s{index + 1}", kind="action", action=action, effect=effect, retry_safety=retry))
    return nodes


def linear_graph(nodes: list[GraphNode]) -> tuple[str, list[GraphNode], list[GraphEdge]]:
    """A chain of action nodes joined by always edges, ending in the success terminal."""
    if not nodes:
        raise ArtifactError("a linear graph needs at least one action node")
    terminal = GraphNode(id=SUCCESS_NODE_ID, kind="terminal", effect="none", retry_safety="safe", status="success")
    chain = list(nodes) + [terminal]
    edges = [GraphEdge(source=before.id, target=after.id, guards=[Guard(kind="always")], priority=0)
             for before, after in zip(chain, chain[1:])]
    return nodes[0].id, chain, edges


def build_linear(
    name: str,
    goal: str,
    surface_meta: dict,
    params: dict,
    nodes: list[GraphNode],
    outputs: dict,
    outcomes: list[dict],
    success: dict,
    run_id: str,
    sensitive: set[str] | None = None,
    planner_name: str = "",
) -> Artifact:
    """Build a typed, versioned draft artifact from one discovered path (a linear graph)."""
    sensitive = sensitive or set()
    inputs = {
        param: {"type": "string", "required": True, "sensitive": param in sensitive,
                "description": f"Value typed where '{param}' was used during discovery"}
        for param in params
    }
    entry, all_nodes, edges = linear_graph(nodes)
    artifact = Artifact(
        schema_version=SCHEMA_VERSION, name=name, version=next_version(name), status="draft", description=goal,
        surface=surface_meta, inputs=inputs, outputs=outputs, entry_node=entry, nodes=all_nodes, edges=edges,
        outcomes=outcomes, success=success,
        provenance={"run_id": run_id, "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "planner": planner_name, "node_count": len(all_nodes)},
    )
    validate(artifact)
    return artifact


# ---------- lookups ----------

def nodes_by_id(artifact: Artifact) -> dict[str, GraphNode]:
    return {node.id: node for node in artifact.nodes}


def outgoing(artifact: Artifact, node_id: str) -> list[GraphEdge]:
    """The edges leaving a node in the order they are to be tried."""
    return sorted((edge for edge in artifact.edges if edge.source == node_id), key=lambda edge: edge.priority)


def action_nodes(artifact: Artifact) -> list[GraphNode]:
    return [node for node in artifact.nodes if node.kind == "action"]


def linear_path(artifact: Artifact) -> list[GraphNode]:
    """The action nodes of a linear artifact in flow order (entry to terminal, always edges only)."""
    nodes = nodes_by_id(artifact)
    path, current = [], nodes[artifact.entry_node]
    while current.kind != "terminal":
        edges = outgoing(artifact, current.id)
        if current.kind != "action" or len(edges) != 1 or edges[0].guards[0].kind != "always":
            raise ArtifactError(f"{artifact.name} is not a linear artifact at node {current.id!r}")
        path.append(current)
        current = nodes[edges[0].target]
    return path


# ---------- validate ----------

def validate(artifact: Artifact) -> None:
    """Reject artifacts a deterministic replay could not run safely. Raises ArtifactError with a precise reason."""
    if artifact.schema_version != SCHEMA_VERSION:
        raise ArtifactError(f"unsupported schema_version {artifact.schema_version!r}, expected {SCHEMA_VERSION!r}")
    validate_contract(artifact)
    validate_inputs(artifact.inputs)
    if not artifact.nodes:
        raise ArtifactError("artifact has no nodes")
    nodes = _validate_nodes(artifact)
    if artifact.entry_node not in nodes:
        raise ArtifactError(f"entry_node {artifact.entry_node!r} is not a node")
    if nodes[artifact.entry_node].kind == "terminal":
        raise ArtifactError(f"entry_node {artifact.entry_node!r} is a terminal node; the flow would do nothing")
    _validate_edges(artifact, nodes)
    _validate_reachable(artifact, nodes)
    _validate_acyclic(artifact, nodes)


def validate_contract(artifact) -> None:
    """Identity, surface, outcomes, outputs and the success checkpoint."""
    if not artifact.name or not re.fullmatch(r"[a-z0-9_]+", artifact.name):
        raise ArtifactError("name must be a non-empty snake_case identifier")
    if artifact.status not in ("draft", "approved"):
        raise ArtifactError(f"unknown status {artifact.status!r}")
    if not artifact.surface.get("entry_url"):
        raise ArtifactError("surface.entry_url is required")
    if not artifact.success:
        raise ArtifactError("success checkpoint is required")
    validate_checkpoint(artifact.success, set(artifact.inputs), "success checkpoint")
    for outcome in artifact.outcomes:
        validate_outcome(outcome)
    for name, spec in artifact.outputs.items():
        validate_output_spec(name, spec)


def validate_output_spec(name: str, spec) -> None:
    """A scalar output (string, number, integer, with an optional regex) or an ordered list output whose
    `items` say how every element is parsed (a scalar type and an optional regex)."""
    if not isinstance(spec, dict) or "type" not in spec:
        raise ArtifactError(f"output {name!r} has no type")
    if spec["type"] not in OUTPUT_TYPES:
        raise ArtifactError(f"output {name!r}: type must be one of {sorted(OUTPUT_TYPES)}, got {spec['type']!r}")
    if spec["type"] == "list":
        items = spec.get("items")
        if not isinstance(items, dict) or items.get("type") not in ITEM_TYPES:
            raise ArtifactError(f"output {name!r}: a list output needs items with a type in {sorted(ITEM_TYPES)}")
        _validate_pattern(items.get("pattern"), f"output {name!r} items")
        validate_cardinality(spec, f"output {name!r}")
    elif "items" in spec:
        raise ArtifactError(f"output {name!r}: only a list output carries items")
    elif any(key in spec for key in CARDINALITY_KEYS):
        raise ArtifactError(f"output {name!r}: only a list output carries min_items or max_items")
    _validate_pattern(spec.get("pattern"), f"output {name!r}")


def validate_cardinality(spec: dict, where: str) -> None:
    """min_items and max_items, when present, are non-negative integers with min <= max and max >= 1."""
    for key in CARDINALITY_KEYS:
        value = spec.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ArtifactError(f"{where}: {key} must be a non-negative integer")
    low, high = spec.get("min_items"), spec.get("max_items")
    if high is not None and high < 1:
        raise ArtifactError(f"{where}: max_items must be at least 1")
    if low is not None and high is not None and low > high:
        raise ArtifactError(f"{where}: min_items {low} exceeds max_items {high}")


def output_problems(outputs: dict, values: dict) -> dict[str, str]:
    """Which declared outputs the final values do not satisfy, and why (names and counts only, never a value).

    A required output must be present and non-null; an optional one may be absent or null. A scalar must fit its
    declared type; a list must be a list whose every item fits the items type and whose length lies within
    min_items and max_items when they are declared. Patterns are enforced where a value is read (the value is the
    pattern's own capture), so they are not re-applied here.
    """
    problems: dict[str, str] = {}
    for name, spec in outputs.items():
        value = values.get(name)
        if value is None:
            if spec.get("required", True):
                problems[name] = "missing"
            continue
        if spec.get("type") == "list":
            problem = list_problem(spec, value)
        else:
            problem = None if fits_type(value, spec.get("type", "string")) else f"not a {spec.get('type')}"
        if problem:
            problems[name] = problem
    return problems


def list_problem(spec: dict, value) -> str | None:
    if not isinstance(value, list):
        return "not a list"
    item_type = (spec.get("items") or {}).get("type", "string")
    bad = [index for index, item in enumerate(value) if not fits_type(item, item_type)]
    if bad:
        return f"item {bad[0]} is not a {item_type}"
    low, high = spec.get("min_items"), spec.get("max_items")
    if low is not None and len(value) < low:
        return f"{len(value)} item(s), at least {low} required"
    if high is not None and len(value) > high:
        return f"{len(value)} item(s), at most {high} allowed"
    return None


def fits_type(value, kind: str) -> bool:
    """A typed value, or the text it was read as (discovery keeps examples as text), fits the declared type.
    An example that carries an input placeholder (`{{name}}`, possibly inside surrounding text such as a
    currency format: the value read came from that input) is typed by the input, not judged here."""
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        if PLACEHOLDER.search(value):
            return True
        plain = re.sub(r"^[$€£]", "", value.replace(",", "").replace(" ", ""))
        if kind == "integer":
            return bool(re.fullmatch(r"-?\d+", plain))
        if kind == "number":
            return bool(re.fullmatch(r"-?\d+(\.\d+)?", plain))
        return bool(value.strip())
    if kind == "integer":
        return isinstance(value, int)
    if kind == "number":
        return isinstance(value, (int, float))
    return False


def _validate_pattern(pattern, where: str) -> None:
    if pattern is None:
        return
    if not isinstance(pattern, str):
        raise ArtifactError(f"{where}: pattern must be a string")
    try:
        re.compile(pattern)
    except re.error as error:
        raise ArtifactError(f"{where}: pattern is not a valid regex: {error}") from error


def validate_inputs(inputs: dict) -> None:
    """A selector is always required; required_when lists non-empty AND-groups (OR-ed together) over
    selector inputs only, and only on inputs that are not always required."""
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


def validate_action(action: GraphAction, declared_inputs: set[str], outputs: dict, where: str) -> None:
    """What an action does: a known kind, a target where one is needed, declared outputs and placeholders."""
    if action.action not in ACTIONS:
        raise ArtifactError(f"{where}: unknown action {action.action!r}")
    if action.action == "navigate" and not action.value:
        raise ArtifactError(f"{where}: navigate needs a url value")
    if action.action == "back" and (action.target is not None or action.value is not None):
        raise ArtifactError(f"{where}: back carries no target and no value (it returns to the previous page)")
    if action.action in ("click", "type", "select", "extract") and not (action.target and action.target.strategies):
        raise ArtifactError(f"{where}: {action.action} needs a target locator")
    if action.action in ("type", "select") and action.value is None:
        raise ArtifactError(f"{where}: {action.action} needs a value")
    if action.action in READ_ACTIONS and action.value not in outputs:
        raise ArtifactError(f"{where}: {action.action} writes undeclared output {action.value!r}")
    if action.mode is not None and action.mode not in OUTPUT_MODES:
        raise ArtifactError(f"{where}: unknown output mode {action.mode!r}")
    if action.mode is not None and action.action not in READ_ACTIONS:
        raise ArtifactError(f"{where}: only extract and extract_many carry an output mode")
    if action.action in READ_ACTIONS:
        _validate_output_mode(action, outputs, where)
    if action.action == "extract_many":
        _validate_many_targets(action, outputs, where)
    elif action.targets is not None:
        raise ArtifactError(f"{where}: only extract_many carries targets")
    unknown = placeholders_in(action.value) - declared_inputs
    if unknown:
        raise ArtifactError(f"{where}: placeholders {sorted(unknown)} are not declared inputs")
    for ladder in ([action.target] if action.target else []) + list(action.targets or []):
        for strategy in ladder.strategies:
            for key in LOCATOR_TEXT_FIELDS:
                unknown = placeholders_in(str(strategy.get(key, ""))) - declared_inputs
                if unknown:
                    raise ArtifactError(f"{where}: locator placeholders {sorted(unknown)} are not declared inputs")
    if action.checkpoint:
        validate_checkpoint(action.checkpoint, declared_inputs, f"{where} checkpoint")


def _validate_output_mode(action: GraphAction, outputs: dict, where: str) -> None:
    """set assigns a scalar (extract) or a whole list (extract_many) once; append adds to a list output."""
    kind = outputs[action.value].get("type")
    if action.mode == "append":
        if kind != "list":
            raise ArtifactError(f"{where}: append adds to a list, but output {action.value!r} is {kind!r}")
        return
    if action.action == "extract" and kind == "list":
        raise ArtifactError(f"{where}: extract reads one value, but output {action.value!r} is a list "
                            f"(use extract_many, or mode append to add one item)")


def _validate_many_targets(action: GraphAction, outputs: dict, where: str) -> None:
    """extract_many: an ordered, bounded, duplicate-free list of ladders into one list output."""
    if action.target is not None:
        raise ArtifactError(f"{where}: extract_many uses targets, not a single target")
    if not isinstance(action.targets, list) or not action.targets:
        raise ArtifactError(f"{where}: extract_many needs a non-empty list of targets")
    if len(action.targets) > MAX_EXTRACT_TARGETS:
        raise ArtifactError(f"{where}: extract_many may read at most {MAX_EXTRACT_TARGETS} targets, "
                            f"got {len(action.targets)}")
    seen: list = []
    for index, ladder in enumerate(action.targets):
        if not isinstance(ladder, Locator) or not ladder.strategies:
            raise ArtifactError(f"{where}: target {index} needs a locator with at least one strategy")
        if ladder.strategies in seen:
            raise ArtifactError(f"{where}: target {index} duplicates an earlier target")
        seen.append(ladder.strategies)
    if outputs[action.value].get("type") != "list":
        raise ArtifactError(f"{where}: extract_many writes output {action.value!r}, which is not a list")


def validate_placeholders(checkpoint: dict, declared_inputs: set[str], where: str) -> None:
    for value in checkpoint.values():
        unknown = placeholders_in(str(value)) - declared_inputs
        if unknown:
            raise ArtifactError(f"{where}: placeholders {sorted(unknown)} are not declared inputs")


def validate_checkpoint(checkpoint: dict, declared_inputs: set[str], where: str) -> None:
    """Every condition in a checkpoint is one this engine can verify, on declared inputs only.

    The vocabulary is closed on purpose: a condition replay does not understand would otherwise sit in an
    artifact and be silently skipped, which is the same as having no checkpoint while looking like proof.
    """
    if not isinstance(checkpoint, dict):
        raise ArtifactError(f"{where}: must be an object of conditions")
    unknown_keys = sorted(set(checkpoint) - set(CHECKPOINT_KEYS))
    if unknown_keys:
        raise ArtifactError(f"{where}: unknown condition(s) {unknown_keys}; supported: {sorted(CHECKPOINT_KEYS)}")
    for key, value in checkpoint.items():
        if not isinstance(value, str):
            raise ArtifactError(f"{where}: condition {key!r} needs a string")
    validate_placeholders(checkpoint, declared_inputs, where)


def validate_outcome(outcome: dict) -> None:
    """One declared outcome, whoever declared it: a code, a kind, a typed detection, and for a
    recoverable one a recovery replay can perform (a click on a locator ladder)."""
    if not isinstance(outcome, dict) or not outcome.get("code") or outcome.get("kind") not in OUTCOME_KINDS:
        raise ArtifactError(f"outcome {outcome!r} needs a code and a kind in {sorted(OUTCOME_KINDS)}")
    code = outcome["code"]
    detect = outcome.get("detect")
    if not isinstance(detect, dict) or not detect:
        raise ArtifactError(f"outcome {code}: detect needs text_contains, text_missing, or dialog_contains")
    unknown = set(detect) - set(DETECT_KEYS)
    if unknown:
        raise ArtifactError(f"outcome {code}: detect has unknown keys {sorted(unknown)}")
    if not all(isinstance(value, str) and value for value in detect.values()):
        raise ArtifactError(f"outcome {code}: every detect value must be a non-empty string")
    recover = outcome.get("recover")
    if outcome["kind"] == "recoverable":
        if not isinstance(recover, dict) or recover.get("action") not in RECOVER_ACTIONS:
            raise ArtifactError(f"outcome {code}: recoverable outcomes need a recover action in {RECOVER_ACTIONS}")
        target = recover.get("target")
        strategies = target.get("strategies") if isinstance(target, dict) else None
        if not isinstance(strategies, list) or not strategies or not all(isinstance(s, dict) for s in strategies):
            raise ArtifactError(f"outcome {code}: recover needs a target with a non-empty list of locator strategies")
    elif recover is not None:
        raise ArtifactError(f"outcome {code}: only recoverable outcomes carry a recover action")


def _validate_nodes(artifact: Artifact) -> dict[str, GraphNode]:
    declared_inputs = set(artifact.inputs)
    outcome_codes = {outcome.get("code") for outcome in artifact.outcomes}
    nodes: dict[str, GraphNode] = {}
    for node in artifact.nodes:
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
            _validate_action_node(node, declared_inputs, artifact.outputs)
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


def _validate_edges(artifact: Artifact, nodes: dict[str, GraphNode]) -> None:
    declared_inputs = set(artifact.inputs)
    priorities: dict[str, set[int]] = {}
    fallbacks: dict[str, GraphEdge] = {}
    for edge in artifact.edges:
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
    for node in artifact.nodes:
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


def _validate_reachable(artifact: Artifact, nodes: dict[str, GraphNode]) -> None:
    seen = {artifact.entry_node}
    frontier = [artifact.entry_node]
    while frontier:
        current = frontier.pop()
        for edge in outgoing(artifact, current):
            if edge.target not in seen:
                seen.add(edge.target)
                frontier.append(edge.target)
    unreachable = [node_id for node_id in nodes if node_id not in seen]
    if unreachable:
        raise ArtifactError(f"nodes {unreachable} are unreachable from entry_node {artifact.entry_node!r}")


def _validate_acyclic(artifact: Artifact, nodes: dict[str, GraphNode]) -> None:
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
        for edge in outgoing(artifact, node_id):
            visit(edge.target)
        on_path.pop()
        finished.add(node_id)

    for node_id in nodes:
        visit(node_id)


# ---------- serialize ----------

def to_dict(artifact: Artifact) -> dict:
    """Plain JSON-ready data in a fixed key order; guards carry only the fields their kind uses."""
    data = asdict(artifact)
    data["edges"] = [edge_to_dict(edge) for edge in artifact.edges]
    return data


def edge_to_dict(edge: GraphEdge) -> dict:
    return {"source": edge.source, "target": edge.target,
            "guards": [guard_to_dict(guard) for guard in edge.guards], "priority": edge.priority}


def guard_to_dict(guard: Guard) -> dict:
    return {key: value for key, value in asdict(guard).items() if value is not None}


def dumps(artifact: Artifact, secrets: tuple[str, ...] = ()) -> str:
    """Stable, reviewable JSON: dataclass field order, two-space indent, secrets redacted."""
    return json.dumps(redact(to_dict(artifact), secrets), indent=2, ensure_ascii=False)


def from_dict(data) -> Artifact:
    """Rebuild the dataclasses from plain JSON, naming the exact node, edge, or guard at fault."""
    if not isinstance(data, dict):
        raise ArtifactError("artifact must be a JSON object")
    _require_keys(data, ARTIFACT_KEYS, "artifact")
    _reject_unknown_keys(data, ARTIFACT_KEYS, "artifact")
    nodes = [node_from_dict(raw, f"node {index}") for index, raw in enumerate(_list(data["nodes"], "nodes"))]
    edges = [edge_from_dict(raw, f"edge {index}") for index, raw in enumerate(_list(data["edges"], "edges"))]
    try:
        version = int(data["version"])
    except (TypeError, ValueError) as error:
        raise ArtifactError(f"artifact version must be an integer, got {data['version']!r}") from error
    return Artifact(
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
        target = _locator_from(raw["target"], f"{where} target")
    targets = None
    if raw.get("targets") is not None:
        targets = [_locator_from(item, f"{where} target {index}")
                   for index, item in enumerate(_list(raw["targets"], f"{where} targets"))]
    return GraphAction(action=raw["action"], target=target, value=raw.get("value"), checkpoint=raw.get("checkpoint"),
                       targets=targets, mode=raw.get("mode"))


def _locator_from(raw, where: str) -> Locator:
    strategies = _object(raw, where).get("strategies")
    if not isinstance(strategies, list) or not all(isinstance(s, dict) for s in strategies):
        raise ArtifactError(f"{where} needs a strategies list")
    return Locator(strategies=strategies)


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

def save_artifact(artifact: Artifact, secrets: tuple[str, ...] = (), path: str | Path | None = None) -> Path:
    """Persist an artifact as JSON. Refuses to write if a known secret value is present, and never
    overwrites: artifacts are immutable once saved, a new recording or approval gets the next revision."""
    validate(artifact)
    raw_text = json.dumps(to_dict(artifact), ensure_ascii=False)
    for secret in secrets:
        # A secret inside the artifact means parameterization failed; masking it would only
        # produce a flow that types "[REDACTED]" on replay, so refuse instead.
        if secret and secret in raw_text:
            raise ArtifactError("refusing to save: a sensitive input value was not parameterized")
    path = Path(path) if path else ARTIFACTS_DIR / f"{artifact.name}.v{artifact.version}.json"
    if path.exists():
        raise ArtifactError(f"refusing to overwrite {path}: artifacts are immutable once saved; "
                            f"pass a new path or bump the version")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(artifact, secrets), encoding="utf-8")
    return path


def load_artifact(path: str | Path) -> Artifact:
    """Load and validate an artifact; only the capability graph schema is accepted."""
    data = read_json(path)
    version = data.get("schema_version") if isinstance(data, dict) else None
    if version != SCHEMA_VERSION:
        raise ArtifactError(f"unsupported schema_version {version!r} in {path}: only {SCHEMA_VERSION!r} capability "
                            f"graphs are supported")
    artifact = from_dict(data)
    validate(artifact)
    return artifact


def check_path_names_artifact(path: str | Path, artifact: Artifact) -> None:
    """A `NAME.vN.json` file must hold capability NAME at revision N; anything else is a mix-up."""
    match = VERSIONED.search(Path(path).name)
    if match is None:
        return
    expected_name = Path(path).name[: -len(match.group(0))]
    if artifact.name != expected_name or artifact.version != int(match.group(1)):
        raise ArtifactError(f"{path} holds {artifact.name} v{artifact.version}, not {expected_name} "
                            f"v{match.group(1)} as its name says")


def read_json(path: str | Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"cannot read artifact {path}: {error}") from error


def copy_node(node: GraphNode, node_id: str) -> GraphNode:
    """A deep copy of an action node under a new id; effect and retry safety travel with it unchanged."""
    return GraphNode(id=node_id, kind=node.kind, action=copy.deepcopy(node.action), effect=node.effect,
                     retry_safety=node.retry_safety, status=node.status, outcome_code=node.outcome_code)
