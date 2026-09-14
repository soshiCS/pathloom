"""Deterministic execution of a capability graph (schema 2.0).

    load graph -> enter node -> execute (action) or evaluate (decision) -> select an edge
               -> move to the next node -> return the terminal's result

No planner and no LLM: every decision comes from the graph (guards, effects, retry
policies, terminals), from the artifact's declared outcomes, or from a fixed rule here.
The surface-level mechanics (dialog recovery, locator ladders, checkpoints, output
extraction, escalation) are the version-1 helpers in replay.py, applied to a Step view of
each action node; this module adds only what a graph needs: edge selection, effect-based
confirmation, and per-node retry safety.

Effect policy (--irreversible-policy deny|confirm|allow, default confirm):
  - none / reversible: no effect-based confirmation
  - irreversible:      deny -> policy failure; confirm -> a human approves; allow -> run
  - unknown:           deny -> policy failure (it may well be irreversible); confirm and allow ->
                       a human approves, never automatic
The global allowlist and blocked-control policy is checked first and cannot be overridden;
when it flags an action the graph calls reversible, the stricter view wins.

Retry safety after the action was performed and its result is uncertain (transient error,
human handoff):
  - safe:                repeat within the bounded transient retry budget
  - verify_before_retry: skip if the checkpoint already holds, else repeat; without a
                         checkpoint a human decides
  - never_retry:         skip if the checkpoint already holds, else a human decides
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .artifact import PLACEHOLDER, locator_placeholders, substitute, substitute_locator
from .escalation import AUTOMATION, Escalator
from .evidence import RunLog
from .graph import missing_inputs, nodes_by_id, outgoing
from .models import (Action, ArtifactV2, Element, GraphEdge, GraphNode, Guard, Locator, Observation, ReplayResult,
                     Step, TransientError)
from .policy import Policy
from .replay import (MAX_DIALOG_RECOVERIES_PER_STEP, MAX_TRANSIENT_ATTEMPTS, POLL_INTERVAL_S, ReplayContext,
                     RestartFlow, RetryStep, StopReplay, act, business_result, checkpoint_met, confirmed_by_human,
                     contains, describe_checkpoint, element_hint, escalate, extract_output, failure,
                     match_absence_outcome, match_business_outcome, match_dialog, output_is_optional, recover,
                     record_missing_output, reload_after_transient, resolve_target, run_with_restarts, settle)
from .surface import Surface, visible_text
from . import replay as linear

IRREVERSIBLE_POLICIES = ("deny", "confirm", "allow")
DEFAULT_IRREVERSIBLE_POLICY = "confirm"
GUARD_TIMEOUT_S = 6.0     # how long a node waits for one of its guarded edges to match
STATE_CHANGING = {"navigate", "click", "type"}


@dataclass
class GraphContext(ReplayContext):
    """A replay context over a graph: the same seams, plus the effect policy and node lookups."""

    artifact: ArtifactV2
    irreversible_policy: str = DEFAULT_IRREVERSIBLE_POLICY
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    acted: bool = False       # the current attempt has performed its node's action on the surface


def replay_graph(
    graph: ArtifactV2,
    params: dict,
    surface: Surface,
    policy: Policy,
    escalator: Escalator,
    log: RunLog,
    irreversible_policy: str = DEFAULT_IRREVERSIBLE_POLICY,
) -> ReplayResult:
    """Replay a capability graph with concrete input parameters."""
    if irreversible_policy not in IRREVERSIBLE_POLICIES:
        raise ValueError(f"irreversible_policy must be one of {IRREVERSIBLE_POLICIES}, got {irreversible_policy!r}")
    missing = missing_inputs(graph.inputs, params)
    if missing:
        result = failure("missing_inputs", None, f"inputs {sorted(graph.inputs)}", f"missing {missing}")
        log.event("replay_finished", status=result.status, outcome_code=result.outcome_code, observed=result.observed)
        return result
    log.event("replay_started", capability=graph.name, version=graph.version, status=graph.status, params=params,
              schema_version=graph.schema_version, nodes=len(graph.nodes), edges=len(graph.edges),
              entry_node=graph.entry_node, irreversible_policy=irreversible_policy)
    ctx = GraphContext(graph, params, surface, policy, escalator, log, irreversible_policy=irreversible_policy,
                       nodes=nodes_by_id(graph))
    return run_with_restarts(ctx, run_graph)


def run_graph(ctx: GraphContext) -> ReplayResult:
    """Walk from the entry node to a terminal; stop early at a failure or business outcome."""
    ctx.escalator.control.require(AUTOMATION)
    node = ctx.nodes[ctx.artifact.entry_node]
    while True:
        ctx.log.event("node_entered", node_id=node.id, kind=node.kind)
        if node.kind == "terminal":
            return terminal_result(ctx, node)
        if node.kind == "action":
            stop = run_action(ctx, node)
            if stop is not None:
                return stop
        try:
            edge = select_edge(ctx, node)
        except StopReplay as stop:
            return stop.result
        node = ctx.nodes[edge.target]


# ---------- action nodes ----------

def node_step(node: GraphNode) -> Step:
    """The version-1 view of a node, so the version-1 helpers can act on it and name it in logs."""
    if node.action is None:
        return Step(id=node.id, action=node.kind, target=None)
    return Step(id=node.id, action=node.action.action, target=node.action.target, value=node.action.value,
                checkpoint=node.action.checkpoint, risk="risky" if node.effect == "irreversible" else "safe")


def run_action(ctx: GraphContext, node: GraphNode) -> ReplayResult | None:
    """Run one action node with transient retries, retry safety, and human-resume handling."""
    step = node_step(node)
    attempts = 0
    performed = False   # the action has been carried out at least once: repeating it is a retry
    while True:
        attempts += 1
        ctx.acted = False
        try:
            return attempt_action(ctx, node, step, performed)
        except RetryStep:
            performed = performed or ctx.acted
        except StopReplay as stop:
            return stop.result
        except TransientError as error:
            performed = performed or ctx.acted
            if attempts >= MAX_TRANSIENT_ATTEMPTS:
                return failure("transient_retries_exhausted", node.id, "page to load", str(error))
            ctx.recoveries.append(f"{node.id}: transient error ({error}); retry {attempts}")
            ctx.log.event("recovered", step_id=node.id, kind="transient_retry", detail=str(error), attempt=attempts,
                          performed=performed, retry_safety=node.retry_safety)
            time.sleep(attempts)  # linear backoff: 1s, then 2s
            reload_after_transient(ctx, error)


def attempt_action(ctx: GraphContext, node: GraphNode, step: Step, performed: bool) -> ReplayResult | None:
    concrete_value = substitute(step.value, ctx.params)
    observation = settle(ctx, step)
    if performed:
        if step.checkpoint and checkpoint_met(step.checkpoint, observation, ctx.params):
            ctx.log.event("step_already_satisfied", step_id=node.id)
            return None
        if not may_repeat(node, step):
            return uncertain_result(ctx, node, step, observation)
        ctx.log.event("action_repeated", step_id=node.id, retry_safety=node.retry_safety)

    # 1. Policy: the allowlist first, then the node's effect against the run's irreversible policy.
    stop = check_effect_policy(ctx, node, step, concrete_value, observation.url)
    if stop is not None:
        return stop

    # 2. Targeting: the locator ladder; an unresolvable target is a stuck state, not a crash.
    element = None
    if step.action != "navigate":
        element = resolve_target(ctx, step)
        if element is None and step.action == "extract" and output_is_optional(ctx, step):
            business = match_business_outcome(ctx.artifact, observation)
            if business is not None:
                return business_result(ctx, step, business)
            return record_missing_output(ctx, step, "control not on screen")
        if element is None:
            absence = match_absence_outcome(ctx, step, observation)
            if absence is not None:
                return business_result(ctx, step, absence)
            escalate(ctx, step, "target_not_found", f"a control matching {step.target.strategies[0]}",
                     visible_text(observation)[:300])

    # 3. Act, then wait for the checkpoint if the node has one; the edges judge the rest.
    if step.action == "extract":
        return extract_output(ctx, step, element)
    ctx.acted = True
    act(ctx, step, element, concrete_value)
    return settle_after_node_action(ctx, node, step, before=observation)


def may_repeat(node: GraphNode, step: Step) -> bool:
    """Whether an already-performed action may be carried out again without asking anyone."""
    if node.retry_safety == "safe":
        return True
    return node.retry_safety == "verify_before_retry" and bool(step.checkpoint)


def uncertain_result(ctx: GraphContext, node: GraphNode, step: Step, observation: Observation) -> ReplayResult | None:
    """The action ran once and must not run again blindly: a human looks at the live session.

    resume: the human made the screen right; carry on to this node's edges without repeating.
    """
    expected = describe_checkpoint(step) if step.checkpoint else "evidence that the action took effect"
    try:
        escalate(ctx, step, "action_result_uncertain", expected, visible_text(observation)[:300])
    except RetryStep:
        ctx.log.event("action_assumed_done", step_id=node.id, retry_safety=node.retry_safety)
        return None


def check_effect_policy(ctx: GraphContext, node: GraphNode, step: Step, value: str | None,
                        url: str) -> ReplayResult | None:
    """Global policy first (never overridable), then the node's effect under the irreversible policy."""
    verdict = ctx.policy.check(Action(kind=step.action, target=element_hint(step), value=value), url)
    ctx.log.event("policy_checked", step_id=node.id, decision=verdict.decision, reason=verdict.reason,
                  effect=node.effect, irreversible_policy=ctx.irreversible_policy)
    if verdict.decision == "deny":
        return failure("policy_denied", node.id, "an allowlisted action", verdict.reason)
    effect = node.effect
    if verdict.decision == "confirm" and effect in ("none", "reversible"):
        # The graph and the policy disagree; the stricter classification wins and a human decides.
        ctx.log.event("effect_conflict", step_id=node.id, graph_effect=effect, policy_reason=verdict.reason)
        if ctx.irreversible_policy == "deny":
            return failure("irreversible_denied", node.id, "a reversible action",
                           f"policy classifies it as risky ({verdict.reason}) and the irreversible policy is deny")
        return require_approval(ctx, node, step, "irreversible_not_confirmed",
                                f"policy classifies it as risky ({verdict.reason}) but the graph calls it {effect}")
    if effect == "unknown":
        if ctx.irreversible_policy == "deny":
            return failure("unknown_effect_denied", node.id, "an action with a known, reversible effect",
                           "effect is unknown and the irreversible policy is deny")
        return require_approval(ctx, node, step, "unknown_effect_not_confirmed", "effect is unknown")
    if effect == "irreversible":
        if ctx.irreversible_policy == "deny":
            return failure("irreversible_denied", node.id, "a reversible action",
                           "effect is irreversible and the irreversible policy is deny")
        if ctx.irreversible_policy == "allow":
            ctx.log.event("irreversible_allowed", step_id=node.id, reason="irreversible policy is allow")
            return None
        return require_approval(ctx, node, step, "irreversible_not_confirmed", "effect is irreversible")
    return None


def require_approval(ctx: GraphContext, node: GraphNode, step: Step, code: str, why: str) -> ReplayResult | None:
    """A human approves the pending action on the live session, or the run stops with a policy failure."""
    if confirmed_by_human(ctx, step, why):
        ctx.log.event("action_approved", step_id=node.id, reason=why)
        return None
    return failure(code, node.id, "human approval", f"{why}; not approved")


def settle_after_node_action(ctx: GraphContext, node: GraphNode, step: Step,
                             before: Observation) -> ReplayResult | None:
    """Wait for the node's checkpoint. A node without one passes at once: its edges judge the screen.

    While waiting, a declared business outcome on a changed screen ends the run, as in
    version-1 replay, so a converted linear artifact keeps its outcome classification.
    """
    if not step.checkpoint:
        ctx.surface.observe()   # a load that fails right after acting belongs to this node's retry policy
        ctx.log.event("checkpoint_passed", step_id=node.id, checkpoint=None)
        return None
    deadline = time.time() + linear.CHECKPOINT_TIMEOUT_S
    text_before = visible_text(before)
    while True:
        observation = settle(ctx, step)
        if checkpoint_met(step.checkpoint, observation, ctx.params):
            ctx.log.event("checkpoint_passed", step_id=node.id, checkpoint=step.checkpoint)
            return None
        if visible_text(observation) != text_before:
            business = match_business_outcome(ctx.artifact, observation)
            if business is not None:
                return business_result(ctx, step, business)
        if time.time() >= deadline:
            escalate(ctx, step, "checkpoint_not_met", describe_checkpoint(step), visible_text(observation)[:300])
        time.sleep(POLL_INTERVAL_S)


# ---------- edges and guards ----------

def select_edge(ctx: GraphContext, node: GraphNode) -> GraphEdge:
    """Take the first edge (ascending priority) whose guards all hold on the live screen.

    Guarded edges are polled for a bounded time; the node's "always" edge, if any, is the
    fallback once that time is up. A node whose only edge is "always" leaves at once. A
    known dialog is cleared while waiting. When nothing matches, a human takes over.
    """
    edges = outgoing(ctx.artifact, node.id)
    guarded = [edge for edge in edges if not is_fallback(edge)]
    fallback = next((edge for edge in edges if is_fallback(edge)), None)
    step = node_step(node)
    deadline = time.time() + GUARD_TIMEOUT_S
    recoveries = 0
    while True:
        observation = observe_with_recovery(ctx, node)
        matched = first_matching_edge(ctx, node, guarded, observation)
        if matched is not None:
            return selected(ctx, node, matched, fallback=False)
        if observation.dialog and recoveries < MAX_DIALOG_RECOVERIES_PER_STEP:
            known = match_dialog(ctx.artifact, observation.dialog)
            if known is not None:
                recover(ctx, step, known)
                recoveries += 1
                continue
        if fallback is not None and (not guarded or time.time() >= deadline):
            return selected(ctx, node, fallback, fallback=bool(guarded))
        if time.time() >= deadline:
            try:
                escalate(ctx, step, "no_matching_edge", describe_edges(guarded),
                         f"dialog: {observation.dialog}" if observation.dialog else visible_text(observation)[:300])
            except RetryStep:
                deadline = time.time() + GUARD_TIMEOUT_S   # the human changed the screen: look again
                continue
        time.sleep(POLL_INTERVAL_S)


def observe_with_recovery(ctx: GraphContext, node: GraphNode) -> Observation:
    """Observe the screen for a node that performs no action, riding out a bounded number of failed loads.

    Raises StopReplay with a transient_retries_exhausted failure once the budget is spent.
    """
    attempts = 0
    while True:
        try:
            return ctx.surface.observe()
        except TransientError as error:
            attempts += 1
            if attempts >= MAX_TRANSIENT_ATTEMPTS:
                raise StopReplay(failure("transient_retries_exhausted", node.id, "page to load", str(error)))
            ctx.recoveries.append(f"{node.id}: transient error ({error}); retry {attempts}")
            ctx.log.event("recovered", step_id=node.id, kind="transient_retry", detail=str(error), attempt=attempts)
            time.sleep(attempts)  # linear backoff: 1s, then 2s
            reload_after_transient(ctx, error)


def is_fallback(edge: GraphEdge) -> bool:
    return edge.guards[0].kind == "always"


def first_matching_edge(ctx: GraphContext, node: GraphNode, edges: list[GraphEdge],
                        observation: Observation) -> GraphEdge | None:
    for edge in edges:
        holds = True
        for guard in edge.guards:
            holds = guard_holds(ctx, guard, observation)
            ctx.log.event("guard_evaluated", node_id=node.id, target=edge.target, priority=edge.priority,
                          guard=guard.kind, holds=holds)
            if not holds:
                break
        if holds:
            return edge
    return None


def selected(ctx: GraphContext, node: GraphNode, edge: GraphEdge, fallback: bool) -> GraphEdge:
    ctx.log.event("edge_selected", node_id=node.id, target=edge.target, priority=edge.priority, fallback=fallback)
    return edge


def guard_holds(ctx: GraphContext, guard: Guard, observation: Observation) -> bool:
    """Deterministic evaluation of one guard against the parameters and the current screen."""
    if guard.kind == "always":
        return True
    if guard.kind == "input_equals":
        return str(ctx.params.get(guard.input, "")) == substitute(guard.value, ctx.params)
    if guard.kind == "text_visible":
        return contains(visible_text(observation), substitute(guard.value, ctx.params))
    if guard.kind == "url_matches":
        return re.search(substitute_pattern(guard.pattern, ctx.params), observation.url) is not None
    if guard.kind == "element_present":
        return guard_element(ctx, guard.target) is not None
    if guard.kind == "dialog_contains":
        return bool(observation.dialog) and contains(observation.dialog, substitute(guard.value, ctx.params))
    raise ValueError(f"unsupported guard kind {guard.kind!r}")   # validation rejects these before replay


def substitute_pattern(pattern: str, params: dict) -> str:
    """Fill placeholders in a regular expression with the literal (escaped) parameter values."""
    return PLACEHOLDER.sub(lambda match: re.escape(str(params[match.group(1)])), pattern)


def guard_element(ctx: GraphContext, locator: Locator) -> Element | None:
    """Resolve a guard's ladder with the same rule as targets: a parameterized ladder never
    falls back to a rung that ignores the input."""
    concrete = substitute_locator(locator, ctx.params)
    parameterized = any(locator_placeholders(recorded) for recorded in locator.strategies)
    strategies = [strategy for strategy, recorded in zip(concrete.strategies, locator.strategies)
                  if not parameterized or locator_placeholders(recorded)]
    return ctx.surface.resolve(Locator(strategies=strategies))


def describe_edges(edges: list[GraphEdge]) -> str:
    def describe_guard(guard: Guard) -> str:
        detail = guard.input or guard.value or guard.pattern or (guard.target.strategies[0] if guard.target else "")
        return f"{guard.kind} {detail!r}" if detail else guard.kind
    return "one of: " + "; ".join(f"-> {edge.target} when {' and '.join(describe_guard(g) for g in edge.guards)}"
                                 for edge in edges)


# ---------- terminal nodes ----------

def terminal_result(ctx: GraphContext, node: GraphNode) -> ReplayResult:
    try:
        observation = observe_with_recovery(ctx, node)
    except StopReplay as stop:
        return stop.result
    ctx.log.event("terminal_reached", node_id=node.id, status=node.status, outcome_code=node.outcome_code)
    screen = visible_text(observation)[:300]
    if node.status == "success":
        if checkpoint_met(ctx.artifact.success, observation, ctx.params):
            ctx.log.event("success_verified", checkpoint=ctx.artifact.success)
            return ReplayResult(status="success", outputs=dict(ctx.outputs))
        business = match_business_outcome(ctx.artifact, observation)
        if business is not None:
            return business_result(ctx, node_step(node), business)
        return failure("success_condition_not_met", node.id, str(ctx.artifact.success), screen)
    if node.status == "business_outcome":
        ctx.log.event("business_outcome", step_id=node.id, code=node.outcome_code)
        return ReplayResult(status="business_outcome", outcome_code=node.outcome_code, step_id=node.id,
                            expected=f"terminal {node.id}", observed=screen)
    return failure(node.outcome_code, node.id, "a declared success or business outcome", screen)
