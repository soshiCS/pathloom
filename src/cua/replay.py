"""Deterministic execution of a capability artifact.

    load artifact -> enter node -> execute (action) or evaluate (decision) -> select an edge
                  -> move to the next node -> return the terminal's result

Replay never consults a planner or an LLM. Every decision it makes comes from the artifact
(guards, effects, retry policies, terminals, declared outcomes) or from a fixed rule in this
file (retry a transient load, escalate anything unknown).

Result contract (ReplayResult.status):
  - success:           the success terminal was reached and its checkpoint verified, outputs extracted
  - business_outcome:  a declared, legitimate non-success state was reached (outcome_code)
  - failure:           a hard failure; step_id / expected / observed say where and why
Recoverable conditions never surface as a status; they are listed in `recoveries`. `executed_path`
lists the action nodes completed and traversed, once each; `performed_attempts` counts every
physical action attempted, including ones whose checkpoint then failed (diagnostic only).

Effect policy (--irreversible-policy deny|confirm|allow, default confirm):
  - none / reversible: no effect-based confirmation
  - irreversible:      deny -> policy failure; confirm -> a human approves; allow -> run
  - unknown:           deny -> policy failure (it may well be irreversible); confirm and allow ->
                       a human approves, never automatic
The global allowlist and blocked-control policy is checked first and cannot be overridden;
when it flags an action the artifact calls reversible, the stricter view wins.

Retry safety after the action was performed and its result is uncertain (transient error,
human handoff):
  - safe:                repeat within the bounded transient retry budget
  - verify_before_retry: skip if the checkpoint already holds, else repeat; without a
                         checkpoint a human decides
  - never_retry:         skip if the checkpoint already holds, else a human decides

Human handoff is expressed as three exceptions, one per disposition the operator can
choose: RetryStep (resume), RestartFlow (restart), StopReplay (abort).
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field, replace
from typing import NoReturn

from .artifact import (PLACEHOLDER, locator_placeholders, missing_inputs, nodes_by_id, outgoing, output_problems,
                       substitute, substitute_locator)
from .escalation import AUTOMATION, Escalator
from .evidence import RunLog
from .models import (MAX_TOKEN_TEXT, Action, ActionError, Artifact, Element, GraphAction, GraphEdge, GraphNode, Guard,
                     InterventionRequest, Locator, Observation, ReplayResult, TransientError)
from .policy import Policy
from .surface import PATH_SEPARATOR, Surface, ref_encloses, visible_text

MAX_TRANSIENT_ATTEMPTS = 3
MAX_RESTARTS = 1
MAX_DIALOG_RECOVERIES_PER_STEP = 3
CHECKPOINT_TIMEOUT_S = 6.0
GUARD_TIMEOUT_S = 6.0     # how long a node waits for one of its guarded edges to match
POLL_INTERVAL_S = 0.3
IRREVERSIBLE_POLICIES = ("deny", "confirm", "allow")
DEFAULT_IRREVERSIBLE_POLICY = "confirm"
STATE_CHANGING = {"navigate", "click", "type"}


class RetryStep(Exception):
    """The human resolved the situation; re-check and re-run the current node."""


class RestartFlow(Exception):
    """The human reset the session; start the flow over from the entry node."""


class StopReplay(Exception):
    """The human aborted (or no operator was available); carries the failure result."""

    def __init__(self, result: ReplayResult):
        super().__init__(result.outcome_code)
        self.result = result


@dataclass
class ReplayContext:
    """Everything one replay run carries between nodes."""

    artifact: Artifact
    params: dict
    surface: Surface
    policy: Policy
    escalator: Escalator
    log: RunLog
    irreversible_policy: str = DEFAULT_IRREVERSIBLE_POLICY
    nodes: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    recoveries: list[str] = field(default_factory=list)
    executed_path: list[dict] = field(default_factory=list)   # action nodes completed, once each, in order
    appended: set = field(default_factory=set)               # ids of append nodes whose items are already in
    performed_attempts: int = 0                              # physical actions attempted, never cleared
    acted: bool = False       # the current attempt has performed its node's action on the surface


def executed_entry(node: GraphNode) -> dict:
    """One executed_path entry: what ran, as recorded in the artifact (placeholders, never values)."""
    return {"id": node.id, "action": asdict(node.action), "effect": node.effect, "retry_safety": node.retry_safety}


def replay(
    artifact: Artifact,
    params: dict,
    surface: Surface,
    policy: Policy,
    escalator: Escalator,
    log: RunLog,
    irreversible_policy: str = DEFAULT_IRREVERSIBLE_POLICY,
) -> ReplayResult:
    """Replay an artifact with concrete input parameters."""
    if irreversible_policy not in IRREVERSIBLE_POLICIES:
        raise ValueError(f"irreversible_policy must be one of {IRREVERSIBLE_POLICIES}, got {irreversible_policy!r}")
    missing = missing_inputs(artifact.inputs, params)
    if missing:
        result = failure("missing_inputs", None, f"inputs {sorted(artifact.inputs)}", f"missing {missing}")
        log.event("replay_finished", status=result.status, outcome_code=result.outcome_code, observed=result.observed)
        return result
    log.event("replay_started", capability=artifact.name, version=artifact.version, status=artifact.status,
              params=params, schema_version=artifact.schema_version, nodes=len(artifact.nodes),
              edges=len(artifact.edges), entry_node=artifact.entry_node, irreversible_policy=irreversible_policy)
    ctx = ReplayContext(artifact, params, surface, policy, escalator, log, irreversible_policy=irreversible_policy,
                        nodes=nodes_by_id(artifact))
    return run_with_restarts(ctx, run_graph)


def run_with_restarts(ctx: ReplayContext, flow) -> ReplayResult:
    """Run the flow, honoring a bounded human-requested restart, then finish the run."""
    restarts = 0
    while True:
        try:
            result = flow(ctx)
            break
        except RestartFlow:
            restarts += 1
            if restarts > MAX_RESTARTS:
                result = failure("restart_limit_reached", None, f"at most {MAX_RESTARTS} restart", "human asked again")
                break
            ctx.outputs.clear()
            ctx.appended.clear()
            ctx.executed_path.clear()          # the path starts over; performed_attempts keeps counting
            ctx.log.event("flow_restarted", restart=restarts, performed_attempts=ctx.performed_attempts)
    result.recoveries = list(ctx.recoveries)
    result.interventions = list(ctx.escalator.interventions)
    result.executed_path = list(ctx.executed_path)
    result.performed_attempts = ctx.performed_attempts
    ctx.log.screenshot(ctx.surface, f"final-{result.status}")
    ctx.log.event("replay_finished", status=result.status, outcome_code=result.outcome_code, step_id=result.step_id,
                  outputs=result.outputs, recoveries=result.recoveries, executed_path=[e["id"] for e in result.executed_path],
                  performed_attempts=result.performed_attempts)
    return result


def run_graph(ctx: ReplayContext) -> ReplayResult:
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
            ctx.executed_path.append(executed_entry(node))
        try:
            edge = select_edge(ctx, node)
        except StopReplay as stop:
            return stop.result
        node = ctx.nodes[edge.target]


# ---------- action nodes ----------

def run_action(ctx: ReplayContext, node: GraphNode) -> ReplayResult | None:
    """Run one action node with transient retries, retry safety, and human-resume handling."""
    attempts = 0
    performed = False   # the action has been carried out at least once: repeating it is a retry
    while True:
        attempts += 1
        ctx.acted = False
        try:
            return attempt_action(ctx, node, performed)
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


def reload_after_transient(ctx: ReplayContext, error: TransientError) -> None:
    """Reload the page that failed. If the reload fails too, the next attempt will notice."""
    if not error.url:
        return
    try:
        ctx.surface.navigate(error.url)
    except TransientError as again:
        ctx.log.event("reload_failed", detail=str(again))


def attempt_action(ctx: ReplayContext, node: GraphNode, performed: bool) -> ReplayResult | None:
    action = node.action
    concrete_value = substitute(action.value, ctx.params)
    observation = settle(ctx, node)
    if performed:
        if action.checkpoint and checkpoint_met(action.checkpoint, observation, ctx.params, ctx, node):
            ctx.log.event("step_already_satisfied", step_id=node.id)
            return None
        if not may_repeat(node):
            return uncertain_result(ctx, node, observation)
        ctx.log.event("action_repeated", step_id=node.id, retry_safety=node.retry_safety)

    # 1. Policy, part one: the allowlist, before anything is resolved or touched.
    stop = check_reach_policy(ctx, node, concrete_value, observation.url)
    if stop is not None:
        return stop

    # 2. Targeting: the locator ladder; an unresolvable target is a stuck state, not a crash. Resolution is
    # read-only, and it comes before the contextual verdict so the policy judges the live control.
    if action.action == "extract_many":
        stop = check_effect_policy(ctx, node, concrete_value, observation.url)
        if stop is not None:
            return stop
        return extract_many_output(ctx, node, observation)
    element = None
    if action.action not in ("navigate", "back"):
        element, status = resolve_ladder_detail(ctx, node, action.target, reading=action.action == "extract")
        if element is None and status == "ambiguous":
            # Several controls fit and nothing narrows it down: acting on any of them would be a guess.
            ctx.log.event("ambiguous_target", step_id=node.id, action=action.action)
            return failure("ambiguous_target", node.id, f"exactly one control matching {action.target.strategies[0]}",
                           "several controls match the locator's rungs and none resolves uniquely")
        if element is None and action.action == "extract" and output_is_optional(ctx, node):
            business = match_business_outcome(ctx.artifact, observation)
            if business is not None:
                return business_result(ctx, node, business)
            return record_missing_output(ctx, node, "control not on screen")
        if element is None:
            absence = match_absence_outcome(ctx, node, observation)
            if absence is not None:
                return business_result(ctx, node, absence)
            escalate(ctx, node, "target_not_found", f"a control matching {action.target.strategies[0]}",
                     visible_text(observation)[:300])

    # 3. Policy, part two: the control itself, judged on what the live page shows about it.
    stop = check_effect_policy(ctx, node, concrete_value, observation.url, element=element)
    if stop is not None:
        return stop

    # 4. Act, then wait for the checkpoint if the node has one; the edges judge the rest.
    if action.action == "extract":
        return extract_output(ctx, node, element)
    ctx.acted = True
    try:
        act(ctx, node, element, concrete_value)
    except ActionError as error:
        if error.performed == "no":
            ctx.acted = False          # the surface proved nothing physical happened: not an attempt
        else:
            ctx.performed_attempts += 1
        if error.cause == "ambiguous_selection":
            # Retrying cannot make one option out of several: a structured failure, never a blind choice.
            ctx.log.event("ambiguous_selection", step_id=node.id, performed=error.performed, detail=str(error))
            return failure("ambiguous_selection", node.id, "exactly one option matching the value", str(error))
        if error.cause == "stale_target":
            # The control this step needs was there a moment ago and is gone; nothing was dispatched, and
            # waiting or repeating cannot bring it back. A structured failure, zero actions performed.
            ctx.log.event("stale_target", step_id=node.id, action=node.action.action)
            return failure("stale_target", node.id, "the control the step targets, still on screen",
                           "the control was present when the screen was read and had gone before acting")
        raise
    except TransientError:
        ctx.performed_attempts += 1    # a load failure during the action: it may have happened
        raise
    ctx.performed_attempts += 1
    return settle_after_action(ctx, node, before=observation)


def may_repeat(node: GraphNode) -> bool:
    """Whether an already-performed action may be carried out again without asking anyone."""
    if node.retry_safety == "safe":
        return True
    return node.retry_safety == "verify_before_retry" and bool(node.action.checkpoint)


def uncertain_result(ctx: ReplayContext, node: GraphNode, observation: Observation) -> ReplayResult | None:
    """The action ran once and must not run again blindly: a human looks at the live session.

    resume: the human made the screen right; carry on to this node's edges without repeating.
    """
    expected = describe_checkpoint(node) if node.action.checkpoint else "evidence that the action took effect"
    try:
        escalate(ctx, node, "action_result_uncertain", expected, visible_text(observation)[:300])
    except RetryStep:
        ctx.log.event("action_assumed_done", step_id=node.id, retry_safety=node.retry_safety)
        return None


def check_reach_policy(ctx: ReplayContext, node: GraphNode, value: str | None, url: str) -> ReplayResult | None:
    """The allowlist, before anything is resolved or touched. Never overridable by an artifact."""
    action = node.action
    verdict = ctx.policy.check_reach(Action(kind=action.action, target=element_hint(action), value=value), url)
    if verdict.decision == "deny":
        ctx.log.event("policy_checked", step_id=node.id, decision=verdict.decision, reason=verdict.reason,
                      effect=node.effect, irreversible_policy=ctx.irreversible_policy, stage="reach")
        return failure("policy_denied", node.id, "an allowlisted action", verdict.reason)
    return None


def check_effect_policy(ctx: ReplayContext, node: GraphNode, value: str | None, url: str,
                        element: Element | None = None) -> ReplayResult | None:
    """The control's own risk, then the node's effect under the irreversible policy.

    `element` is the control as it was just resolved on the live page, carrying the runtime-only evidence
    (native kind, enclosing landmark, dismissal metadata) that a stored locator cannot. Resolving first is
    read-only: nothing has been performed when this runs. The stricter verdict still wins, so live evidence
    that makes a stored effect look less safe is honoured, never the other way round.
    """
    action = node.action
    target = element if element is not None else element_hint(action)
    verdict = ctx.policy.check(Action(kind=action.action, target=target, value=value), url)
    ctx.log.event("policy_checked", step_id=node.id, decision=verdict.decision, reason=verdict.reason,
                  effect=node.effect, irreversible_policy=ctx.irreversible_policy,
                  evidence="resolved" if element is not None else "locator")
    if verdict.decision == "deny":
        return failure("policy_denied", node.id, "an allowlisted action", verdict.reason)
    effect = node.effect
    if verdict.decision == "confirm" and effect in ("none", "reversible"):
        # The artifact and the policy disagree; the stricter classification wins and a human decides.
        ctx.log.event("effect_conflict", step_id=node.id, graph_effect=effect, policy_reason=verdict.reason)
        if ctx.irreversible_policy == "deny":
            return failure("irreversible_denied", node.id, "a reversible action",
                           f"policy classifies it as risky ({verdict.reason}) and the irreversible policy is deny")
        return require_approval(ctx, node, "irreversible_not_confirmed",
                                f"policy classifies it as risky ({verdict.reason}) but the artifact calls it {effect}")
    if effect == "unknown":
        if ctx.irreversible_policy == "deny":
            return failure("unknown_effect_denied", node.id, "an action with a known, reversible effect",
                           "effect is unknown and the irreversible policy is deny")
        return require_approval(ctx, node, "unknown_effect_not_confirmed", "effect is unknown")
    if effect == "irreversible":
        if ctx.irreversible_policy == "deny":
            return failure("irreversible_denied", node.id, "a reversible action",
                           "effect is irreversible and the irreversible policy is deny")
        if ctx.irreversible_policy == "allow":
            ctx.log.event("irreversible_allowed", step_id=node.id, reason="irreversible policy is allow")
            return None
        return require_approval(ctx, node, "irreversible_not_confirmed", "effect is irreversible")
    return None


def require_approval(ctx: ReplayContext, node: GraphNode, code: str, why: str) -> ReplayResult | None:
    """A human approves the pending action on the live session, or the run stops with a policy failure."""
    if confirmed_by_human(ctx, node, why):
        ctx.log.event("action_approved", step_id=node.id, reason=why)
        return None
    return failure(code, node.id, "human approval", f"{why}; not approved: would {describe_action(node.action)}")


def describe_action(action: GraphAction) -> str:
    """'click button 'Finish'', for handoff requests and policy failures."""
    hint = element_hint(action)
    if action.action == "back":
        return "back to the previous page"
    if hint is None:
        return f"{action.action} {action.value!r}"
    return f"{action.action} {hint.role} {hint.name!r}"


def settle_after_action(ctx: ReplayContext, node: GraphNode, before: Observation) -> ReplayResult | None:
    """Wait for the node's checkpoint. A node without one passes at once: its edges judge the screen.

    While waiting, a declared business outcome on a changed screen ends the run; a met checkpoint
    always wins (outcome text can legitimately appear elsewhere on a good screen), and outcome
    text is only judged once the screen has changed, so a slow load is never mistaken for an
    error shown on the previous screen.
    """
    action = node.action
    if not action.checkpoint:
        ctx.surface.observe()   # a load that fails right after acting belongs to this node's retry policy
        # This node never carried proof; say so plainly rather than calling an absent checkpoint verified.
        ctx.log.event("no_checkpoint_recorded", step_id=node.id, action=action.action)
        return None
    deadline = time.time() + CHECKPOINT_TIMEOUT_S
    text_before = visible_text(before)
    while True:
        observation = settle(ctx, node)
        if checkpoint_met(action.checkpoint, observation, ctx.params, ctx, node):
            ctx.log.event("checkpoint_passed", step_id=node.id, checkpoint=action.checkpoint)
            return None
        if visible_text(observation) != text_before:
            business = match_business_outcome(ctx.artifact, observation)
            if business is not None:
                return business_result(ctx, node, business)
        if time.time() >= deadline:
            escalate(ctx, node, "checkpoint_not_met", describe_checkpoint(node), visible_text(observation)[:300])
        time.sleep(POLL_INTERVAL_S)


# ---------- perception, dialogs, outcomes ----------

def settle(ctx: ReplayContext, node: GraphNode) -> Observation:
    """Observe the screen, clearing any dialog the artifact knows how to clear first."""
    observation = ctx.surface.observe()
    recoveries = 0
    while observation.dialog:
        outcome = match_dialog(ctx.artifact, observation.dialog)
        if outcome is None or recoveries >= MAX_DIALOG_RECOVERIES_PER_STEP:
            escalate(ctx, node, "unknown_dialog", "no dialog, or a declared recoverable one",
                     f"dialog: {observation.dialog}")
        recover(ctx, node, outcome)
        recoveries += 1
        observation = ctx.surface.observe()
    return observation


def match_dialog(artifact: Artifact, dialog_text: str) -> dict | None:
    for outcome in artifact.outcomes:
        needle = outcome.get("detect", {}).get("dialog_contains")
        if outcome["kind"] == "recoverable" and needle and contains(dialog_text, needle):
            return outcome
    return None


def match_business_outcome(artifact: Artifact, observation: Observation) -> dict | None:
    text = visible_text(observation)
    for outcome in artifact.outcomes:
        needle = outcome.get("detect", {}).get("text_contains")
        if outcome["kind"] == "business" and needle and contains(text, needle):
            return outcome
    return None


def contains(text: str, needle: str) -> bool:
    """Outcome texts are declared from memory, so they are matched case-insensitively."""
    return needle.lower() in text.lower()


def recover(ctx: ReplayContext, node: GraphNode, outcome: dict) -> None:
    """Apply a declared recovery (currently: click a control) and log it."""
    action = outcome["recover"]
    element = ctx.surface.resolve(Locator(strategies=action["target"]["strategies"]))
    if element is None:
        escalate(ctx, node, "recovery_failed", f"the control that clears '{outcome['code']}'", "control not found")
    ctx.surface.click(element)
    ctx.recoveries.append(f"{node.id}: {outcome['code']}")
    ctx.log.event("recovered", step_id=node.id, kind=outcome["code"], detail=outcome["detect"])


# What a control that takes a selected thing away calls itself. Stems only, so "Remove", "removing",
# "Deselect" and "Unselect" all match without listing every inflection. Matched against the control's own
# accessible name, never against page copy or markup conventions.
REMOVAL_WORDS = ("remov", "delet", "clear", "deselect", "unselect", "dismiss", "×", "✕", "✖")


def checkpoint_met(checkpoint: dict, observation: Observation, params: dict,
                   ctx: "ReplayContext | None" = None, node: GraphNode | None = None) -> bool:
    """All conditions in a checkpoint must hold against what is currently perceived.

    An unknown condition is never treated as satisfied: a predicate this engine cannot check is a
    predicate that did not hold, so a checkpoint from a newer writer fails loudly instead of passing
    vacuously. `target_selected` asks about the node's own control, so it needs the node and the live
    surface; without them it cannot be checked and therefore does not hold.
    """
    text = visible_text(observation)
    for kind, raw in checkpoint.items():
        expected = substitute(str(raw), params)
        if kind == "text_contains":
            if expected not in text:
                return False
        elif kind == "url_contains":
            if expected not in observation.url:
                return False
        elif kind == "selection_present":
            if not selection_present(expected, observation):
                return False
        elif kind == "target_selected":
            if not target_selected(expected, observation, ctx, node):
                return False
        elif kind == "target_absent":
            if not target_absent(expected, observation, ctx, node):
                return False
        else:
            return False
    return True


def target_absent(expected: str, observation: Observation, ctx: "ReplayContext | None",
                  node: GraphNode | None) -> bool:
    """Whether the node's own control is no longer on the screen: the dismissal it performed took effect.

    The step itself was resolved and clicked by the ordinary path before this runs, so this asks only
    about the aftermath. It fails safely and for a stated reason: a control that still resolves, one whose
    ladder now resolves ambiguously, and one replaced by another control the ladder also matches are all
    simply "not met". Nothing is clicked and nothing is changed here.
    """
    wanted = str(expected).strip().lower() != "false"
    if ctx is None or node is None or node.action is None or node.action.target is None:
        return False
    # Absence is asked of the semantic rungs only. A bare coordinate rung answers "whatever is drawn at
    # this point", which is not the recorded control and would report it present after anything at all
    # moved into its place, so it can never show that a control is still there.
    semantic = replace(node.action.target,
                       strategies=[s for s in node.action.target.strategies if s.get("kind") != "coords"])
    if not semantic.strategies:
        ctx.log.event("target_absent_unmet", step_id=node.id,
                      reason="the control was only ever identified by its position")
        return not wanted
    element, status = resolve_ladder_detail(ctx, node, semantic, quiet=True, reading=True)
    if status == "missing":
        return wanted                  # nothing matches the recorded control any more: it is gone
    if status != "ok" or element is None:
        # Ambiguous: several controls now match, so which one was dismissed cannot be told.
        ctx.log.event("target_absent_unmet", step_id=node.id, reason=f"the control is {status}")
        return not wanted
    ctx.log.event("target_absent_unmet", step_id=node.id, reason="the control is still on screen")
    return not wanted


def target_selected(expected: str, observation: Observation, ctx: "ReplayContext | None",
                    node: GraphNode | None) -> bool:
    """Whether the node's own control now reads as selected, by resolving its recorded locator.

    Fails safely and for a stated reason: the control being absent, the ladder resolving to more than one
    element, a surface that cannot report selectedness, and a control that reports "not selected" are all
    simply "not met". Nothing is clicked and nothing is changed.
    """
    wanted = str(expected).strip().lower() != "false"
    if ctx is None or node is None or node.action is None or node.action.target is None:
        return False
    reader = getattr(ctx.surface, "selected_state", None)
    if reader is None:
        return False
    element, status = resolve_ladder_detail(ctx, node, node.action.target, quiet=True, reading=True)
    if status != "ok" or element is None:
        ctx.log.event("target_selected_unmet", step_id=node.id, reason=f"the control is {status}")
        return False
    try:
        state = reader(element)
    except Exception:
        state = None
    if not state or not state.get("known"):
        ctx.log.event("target_selected_unmet", step_id=node.id, reason="the control reports no selected state")
        return False
    return bool(state.get("selected")) is wanted


def selection_present(expected: str, observation: Observation) -> bool:
    """Whether a selected token carrying this value is on screen, by ordinary perception.

    The token is an element whose own visible text is the committed value and which reports a selected
    state or offers a removal control of its own, exactly what recorded it. Page copy that merely mentions
    the value is not a token, so this is strictly stronger than `text_contains`.

    Canonical, the same way discovery is: several nested elements around one chip all satisfy the test, so
    a candidate that merely contains another candidate is not counted as a token of its own. Existence is
    all this predicate reports, but applying the same rule keeps the two sides agreeing about what a token
    is, and keeps an outer wrapper whose text runs several chips together from ever standing in for one.
    """
    wanted = " ".join(expected.split())
    if not wanted:
        return False
    return bool(selected_tokens(wanted, observation))


def selected_tokens(wanted: str, observation: Observation) -> list[Element]:
    """The canonical selected tokens whose own text carries `wanted`, innermost first, in document order."""
    candidates = []
    for element in observation.elements:
        own = " ".join((element.text or element.name or "").split())
        if wanted not in own or len(own) > MAX_TOKEN_TEXT:
            continue
        if element.states.get("selected") == "true" or element.states.get("checked") == "true":
            candidates.append(element)
        elif removal_control_for(element, observation):
            candidates.append(element)
    # An ancestor that merely contains another candidate is a wrapper, not a second token. The same
    # nesting test discovery canonicalizes with, so both sides agree about what one token is.
    return [element for element in candidates
            if not any(other is not element and ref_encloses(element.ref, other.ref) for other in candidates)]


def removal_control_for(token: Element, observation: Observation) -> bool:
    """Whether a control that removes this token belongs to the same chip, structurally.

    Perception lists the label and the controls, never the container that holds them, so the removal
    control is rarely *inside* the element carrying the text: in the usual chip it is that element's
    sibling. The test is therefore containment by the token's own reference (a control nested inside it)
    or a shared immediate parent (the container discovery recorded as the canonical token). Nothing here
    reads a class name or any other markup convention.
    """
    ref = token.ref or ""
    if not ref:
        return False
    parent = enclosing_ref(ref)
    for element in observation.elements:
        other = element.ref or ""
        if element is token or not other or other == ref:
            continue
        if element.role not in ("button", "link") and not element.native.startswith("button"):
            continue
        if ref_encloses(ref, other):
            return True                    # a control the token itself contains: scoped by containment
        # A sibling control is only a removal control when it says so. Without the containment that
        # scopes the nested case, any button sharing a parent with any text would otherwise qualify.
        if parent and enclosing_ref(other) == parent and removal_wording(element):
            return True
    return False


def removal_wording(control: Element) -> bool:
    """Whether a control's own accessible name reads as removing or deselecting something."""
    label = f"{control.name} {control.text}".lower()
    return any(word in label for word in REMOVAL_WORDS)


def enclosing_ref(ref: str) -> str:
    """The structural reference of the element that directly contains this one, or "" at the top."""
    head, separator, _ = ref.rpartition(PATH_SEPARATOR)
    return head if separator else ""


def business_result(ctx: ReplayContext, node: GraphNode, outcome: dict) -> ReplayResult:
    ctx.log.event("business_outcome", step_id=node.id, code=outcome["code"])
    detect = outcome["detect"]
    observed = detect.get("text_contains") or f"missing: {substitute(detect.get('text_missing'), ctx.params)}"
    return ReplayResult(status="business_outcome", outcome_code=outcome["code"], step_id=node.id,
                        expected=describe_checkpoint(node), observed=observed)


def match_absence_outcome(ctx: ReplayContext, node: GraphNode, observation: Observation) -> dict | None:
    """Outcomes defined by something *missing* apply only to the node that looks for that thing.

    "The requested product is not listed" is judged when the node targeting the product
    cannot find it, never when some unrelated node is stuck on a screen without the product.
    """
    text = visible_text(observation)
    concrete = substitute_locator(node.action.target, ctx.params)
    target_text = " ".join(str(strategy.get(key, "")) for strategy in concrete.strategies
                           for key in ("name", "text", "context"))
    for outcome in ctx.artifact.outcomes:
        needle = substitute(outcome.get("detect", {}).get("text_missing"), ctx.params)
        if outcome["kind"] != "business" or not needle:
            continue
        if contains(target_text, needle) and not contains(text, needle):
            return outcome
    return None


def describe_checkpoint(node: GraphNode) -> str:
    checkpoint = node.action.checkpoint if node.action else None
    return str(checkpoint) if checkpoint else "no checkpoint"


# ---------- targeting, policy, acting ----------

def resolve_target(ctx: ReplayContext, node: GraphNode) -> Element | None:
    return resolve_ladder(ctx, node, node.action.target)


def resolve_ladder(ctx: ReplayContext, node: GraphNode, target: Locator, index: int | None = None,
                   exclude: frozenset[str] = frozenset(), quiet: bool = False) -> Element | None:
    return resolve_ladder_detail(ctx, node, target, index, exclude, quiet)[0]


def resolve_ladder_detail(ctx: ReplayContext, node: GraphNode, target: Locator, index: int | None = None,
                          exclude: frozenset[str] = frozenset(), quiet: bool = False,
                          reading: bool = False) -> tuple[Element | None, str]:
    """Walk one locator ladder a rung at a time so the log shows which rung matched (and which target of an
    extract_many it was); returns the element and a status.

    A role or text rung resolves only when exactly one perceived element matches all its fields; several
    matches make the rung `ambiguous` and the walk continues to the next, more specific rung, so document
    order never decides. A rung whose unique result is in `exclude` (taken by an earlier item of the same
    list) is passed over the same way. For a read (`reading`), a bare point the coordinate rung hands back when
    nothing is drawn there is not a resolution. Statuses: "ok"; "ambiguous" when no rung resolved and at least
    one was ambiguous or taken; "missing" when nothing matched at all.
    """
    concrete = substitute_locator(target, ctx.params)   # "{{product_name}}" -> the requested product
    parameterized = any(locator_placeholders(s) for s in target.strategies)
    where = {"step_id": node.id} if index is None else {"step_id": node.id, "target_index": index}
    ambiguous = False
    semantic_candidates: set[str] | None = None
    for rung, (strategy, recorded) in enumerate(zip(concrete.strategies, target.strategies)):
        if parameterized and not locator_placeholders(recorded) and index is None:
            # A structural rung may disambiguate duplicate controls only when it resolves to one of the
            # controls that already matched the parameterized semantic rung. It is never a fallback when the
            # requested input matched nothing, so a changed parameter cannot silently select the old position.
            if strategy.get("kind") == "css" and semantic_candidates:
                element = ctx.surface.resolve(Locator(strategies=[strategy]))
                if element and element.ref in semantic_candidates:
                    if not quiet:
                        ctx.log.event("target_resolved", rung=rung, strategy=strategy["kind"],
                                      drift_signal=True, **where)
                    return element, "ok"
            if not quiet:
                ctx.log.event("rung_skipped", rung=rung, strategy=strategy["kind"],
                              reason="structural rung did not disambiguate a parameterized semantic match", **where)
            continue
        if strategy.get("kind") in ("role", "text"):
            found = ctx.surface.matches(strategy)
            if len(found) > 1:
                ambiguous = True
                semantic_candidates = {element.ref for element in found if element.ref}
                if not quiet:
                    ctx.log.event("rung_ambiguous", rung=rung, strategy=strategy["kind"], matches=len(found), **where)
                continue
            element = found[0] if found else None
        else:
            element = ctx.surface.resolve(Locator(strategies=[strategy]))
        if element is not None and element.ref and element.ref in exclude:
            ambiguous = True
            if not quiet:
                ctx.log.event("rung_taken", rung=rung, strategy=strategy["kind"], **where)
            continue
        if element is not None and reading and not (element.ref or element.text or element.name):
            continue                     # a bare point with nothing drawn there: no value can be read
        if element is not None:
            if not quiet:
                ctx.log.event("target_resolved", rung=rung, strategy=strategy["kind"], drift_signal=rung > 0, **where)
            return element, "ok"
    if not quiet:
        ctx.log.event("target_unresolved", strategies=[s["kind"] for s in target.strategies], ambiguous=ambiguous, **where)
    return None, "ambiguous" if ambiguous else "missing"


def readable(element: Element | None) -> Element | None:
    """A resolved element a value can be read from: one with a page identity or text. A bare point the
    coordinate rung hands back when nothing is drawn there is not."""
    if element is None or not (element.ref or element.text or element.name):
        return None
    return element


def identity_of(element: Element) -> str:
    """What makes two resolved elements the same page element: the structural reference, never text."""
    return element.ref or f"box:{element.box}"


def element_hint(action: GraphAction) -> Element | None:
    """What the artifact says the target is, before resolving it (used for policy checks)."""
    if action.target is None:
        return None
    for strategy in action.target.strategies:
        if strategy.get("kind") == "role":
            return Element(role=strategy["role"], name=strategy["name"])
    return Element(role="unknown", name="")


def confirmed_by_human(ctx: ReplayContext, node: GraphNode, reason: str) -> bool:
    request = InterventionRequest(run_id=ctx.log.run_id, capability=ctx.artifact.name,
                                  goal=ctx.artifact.description, step_id=node.id,
                                  kind="confirm", reason=f"confirmation required: {reason}",
                                  observed=f"about to {describe_action(node.action)}", screenshot=None)
    result = ctx.escalator.request(request, ctx.surface)
    return result.disposition == "approve"


def act(ctx: ReplayContext, node: GraphNode, element: Element | None, value: str | None) -> None:
    action = node.action
    if action.action == "navigate":
        ctx.surface.navigate(value or "")
    elif action.action == "back":
        ctx.surface.back()
    elif action.action == "click":
        ctx.surface.click(element)
    elif action.action == "type":
        ctx.surface.type(element, value or "")
    elif action.action == "select":
        ctx.surface.select(element, value or "")
        ctx.log.event("selection_accepted", step_id=node.id, how=getattr(ctx.surface, "last_selection_proof", ""))
    ctx.log.event("acted", step_id=node.id, action=action.action,
                  value="[typed]" if action.action in ("type", "select") else value)


def extract_output(ctx: ReplayContext, node: GraphNode, element: Element) -> ReplayResult | None:
    """Read a declared output from the resolved control.

    A pattern that does not match is a hard failure for a required output and a null for
    an optional one (a discount line that not every order shows, for example).
    """
    name = node.action.value
    spec = ctx.artifact.outputs[name]
    text = element.text or element.name
    value = text.strip()
    pattern = (spec.get("items") or {}).get("pattern") if node.action.mode == "append" else spec.get("pattern")
    if pattern:
        match = re.search(pattern, text)
        if match is None and output_is_optional(ctx, node):
            return record_missing_output(ctx, node, f"pattern did not match {text[:80]!r}")
        if match is None:
            return failure("output_not_found", node.id, f"text matching {pattern!r}", text[:200])
        value = (match.group(1) if match.groups() else match.group(0)).strip()
    if node.action.mode == "append":
        items = spec.get("items") or {}
        store_items(ctx, node, [typed_value(value, items.get("type", "string"))])
        return None
    ctx.outputs[name] = typed_value(value, spec.get("type", "string"))
    ctx.log.event("output_extracted", step_id=node.id, output=name, value=ctx.outputs[name])
    return None


def store_items(ctx: ReplayContext, node: GraphNode, values: list) -> None:
    """Land parsed list items: a set node assigns the whole list; an append node adds them, in order, to the
    list built so far, exactly once per node however many times the node is attempted."""
    name = node.action.value
    if node.action.mode != "append":
        ctx.outputs[name] = values
        ctx.log.event("output_extracted", step_id=node.id, output=name, value=values, items=len(values))
        return
    if node.id in ctx.appended:
        ctx.log.event("output_already_appended", step_id=node.id, output=name)
        return
    current = ctx.outputs.get(name)
    ctx.outputs[name] = (list(current) if isinstance(current, list) else []) + values
    ctx.appended.add(node.id)
    ctx.log.event("output_extracted", step_id=node.id, output=name, value=values, items=len(values),
                  mode="append", total=len(ctx.outputs[name]))


def extract_many_output(ctx: ReplayContext, node: GraphNode, observation: Observation) -> ReplayResult | None:
    """Read an ordered list output, all or nothing: every ladder resolved on its own, in declared order, each
    value parsed by the output's item contract. Every declared target is a position in the list, so a single
    item that cannot be resolved or parsed means the list cannot be emitted: a required output takes the usual
    business-outcome / handoff / failure path; an optional output is absent as a whole (not in the outputs,
    one `optional_output_absent` event naming the item) and the graph continues. A list is never compressed.
    """
    name = node.action.value
    spec = ctx.artifact.outputs[name]
    items, optional = spec.get("items") or {}, not spec.get("required", True)
    values: list = []
    taken: set[str] = set()          # the list is an ordered one-to-one mapping onto distinct page elements
    for index, ladder in enumerate(node.action.targets):
        found, status = resolve_ladder_detail(ctx, node, ladder, index, exclude=frozenset(taken), reading=True)
        element = readable(found)
        if element is None and status == "ambiguous":
            # Several controls fit, or every reachable one is already another item's: no distinct assignment.
            ctx.log.event("ambiguous_target", step_id=node.id, output=name, target_index=index, taken=sorted(taken))
            return failure("ambiguous_target", node.id, f"a distinct control for item {index} of {name!r}",
                           "several controls match the item's rungs, or every one it reaches is an earlier item's")
        if element is not None:
            taken.add(identity_of(element))
        if element is None:
            if optional:
                return optional_list_absent(ctx, node, index, "control not on screen")
            business = match_business_outcome(ctx.artifact, observation)
            if business is not None:
                return business_result(ctx, node, business)
            escalate(ctx, node, "target_not_found", f"a control matching {ladder.strategies[0]} (item {index})",
                     visible_text(observation)[:300])
        text = element.text or element.name
        value = parse_item(text, items.get("pattern"))
        if value is None:
            if optional:
                return optional_list_absent(ctx, node, index, "pattern did not match the item's text")
            return failure("output_not_found", node.id, f"item {index} text matching {items.get('pattern')!r}", text[:200])
        values.append(typed_value(value, items.get("type", "string")))
    store_items(ctx, node, values)
    return None


def optional_list_absent(ctx: ReplayContext, node: GraphNode, index: int, reason: str) -> None:
    """An optional ordered list with one unusable position is absent as a whole (an append node then adds
    nothing and leaves the items already collected alone); the reason never quotes text."""
    if node.action.mode != "append":
        ctx.outputs.pop(node.action.value, None)
    ctx.log.event("optional_output_absent", step_id=node.id, output=node.action.value, target_index=index, reason=reason)
    return None


def parse_item(text: str, pattern: str | None) -> str | None:
    """One list item through its parsing rule: the whole text, or the pattern's group 1 / whole match; None
    when a pattern is given and does not match."""
    if not pattern:
        return text.strip()
    match = re.search(pattern, text)
    if match is None:
        return None
    return (match.group(1) if match.groups() else match.group(0)).strip()


def output_is_optional(ctx: ReplayContext, node: GraphNode) -> bool:
    return not ctx.artifact.outputs[node.action.value].get("required", True)


def record_missing_output(ctx: ReplayContext, node: GraphNode, why: str) -> None:
    """An optional output that is absent is a null in the result, not an error."""
    ctx.outputs[node.action.value] = None
    ctx.log.event("output_missing", step_id=node.id, output=node.action.value, reason=why)
    return None


def typed_value(value: str, kind: str):
    """Coerce to the declared output type; leaves the text alone if it does not fit."""
    plain = re.sub(r"^[$€£]", "", value.replace(",", "").replace(" ", ""))
    if kind == "integer" and plain.lstrip("-").isdigit():
        return int(plain)
    if kind == "number" and re.fullmatch(r"-?\d+(\.\d+)?", plain):
        return float(plain)
    return value


# ---------- edges and guards ----------

def select_edge(ctx: ReplayContext, node: GraphNode) -> GraphEdge:
    """Take the first edge (ascending priority) whose guards all hold on the live screen.

    Guarded edges are polled for a bounded time; the node's "always" edge, if any, is the
    fallback once that time is up. A node whose only edge is "always" leaves at once. A
    known dialog is cleared while waiting. When nothing matches, a human takes over.
    """
    edges = outgoing(ctx.artifact, node.id)
    guarded = [edge for edge in edges if not is_fallback(edge)]
    fallback = next((edge for edge in edges if is_fallback(edge)), None)
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
                recover(ctx, node, known)
                recoveries += 1
                continue
        if fallback is not None and (not guarded or time.time() >= deadline):
            return selected(ctx, node, fallback, fallback=bool(guarded))
        if time.time() >= deadline:
            try:
                escalate(ctx, node, "no_matching_edge", describe_edges(guarded),
                         f"dialog: {observation.dialog}" if observation.dialog else visible_text(observation)[:300])
            except RetryStep:
                deadline = time.time() + GUARD_TIMEOUT_S   # the human changed the screen: look again
                continue
        time.sleep(POLL_INTERVAL_S)


def observe_with_recovery(ctx: ReplayContext, node: GraphNode) -> Observation:
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


def first_matching_edge(ctx: ReplayContext, node: GraphNode, edges: list[GraphEdge],
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


def selected(ctx: ReplayContext, node: GraphNode, edge: GraphEdge, fallback: bool) -> GraphEdge:
    ctx.log.event("edge_selected", node_id=node.id, target=edge.target, priority=edge.priority, fallback=fallback)
    return edge


def guard_holds(ctx: ReplayContext, guard: Guard, observation: Observation) -> bool:
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


def guard_element(ctx: ReplayContext, locator: Locator) -> Element | None:
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


# ---------- terminal nodes, escalation and results ----------

def terminal_result(ctx: ReplayContext, node: GraphNode) -> ReplayResult:
    try:
        observation = observe_with_recovery(ctx, node)
    except StopReplay as stop:
        return stop.result
    ctx.log.event("terminal_reached", node_id=node.id, status=node.status, outcome_code=node.outcome_code)
    screen = visible_text(observation)[:300]
    if node.status == "success":
        if checkpoint_met(ctx.artifact.success, observation, ctx.params):
            ctx.log.event("success_verified", checkpoint=ctx.artifact.success)
            problems = output_problems(ctx.artifact.outputs, ctx.outputs)
            ctx.log.event("outputs_validated", problems=problems)
            if problems:
                return failure("outputs_incomplete", node.id, f"outputs {sorted(problems)} satisfying the declared "
                               f"contract", "; ".join(f"{name}: {why}" for name, why in problems.items()))
            return ReplayResult(status="success", outputs=dict(ctx.outputs))
        business = match_business_outcome(ctx.artifact, observation)
        if business is not None:
            return business_result(ctx, node, business)
        return failure("success_condition_not_met", node.id, str(ctx.artifact.success), screen)
    if node.status == "business_outcome":
        ctx.log.event("business_outcome", step_id=node.id, code=node.outcome_code)
        return ReplayResult(status="business_outcome", outcome_code=node.outcome_code, step_id=node.id,
                            expected=f"terminal {node.id}", observed=screen)
    return failure(node.outcome_code, node.id, "a declared success or business outcome", screen)


def escalate(ctx: ReplayContext, node: GraphNode, code: str, expected: str, observed: str) -> NoReturn:
    """Hand the live session to a human; translate their disposition into control flow."""
    request = InterventionRequest(run_id=ctx.log.run_id, capability=ctx.artifact.name,
                                  goal=ctx.artifact.description, step_id=node.id,
                                  kind="stuck", reason=f"{code}: expected {expected}", observed=observed,
                                  screenshot=None)
    result = ctx.escalator.request(request, ctx.surface)
    if result.disposition == "resume":
        # A person who acted on the session may have done this very step: the retry re-checks the node's
        # checkpoint first and repeats nothing that is already satisfied (the node's retry safety decides
        # the rest). A resume without any manual action retries the step as before.
        ctx.acted = ctx.acted or bool(result.human_actions)
        raise RetryStep
    if result.disposition == "restart":
        raise RestartFlow
    raise StopReplay(failure(code, node.id, expected, observed))


def failure(code: str, step_id: str | None, expected: str, observed: str) -> ReplayResult:
    return ReplayResult(status="failure", outcome_code=code, step_id=step_id, expected=expected, observed=observed)
