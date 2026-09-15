"""Deterministic execution of a saved capability artifact.

Replay never consults a planner or an LLM. Every decision it makes comes from the
artifact (steps, locator ladders, checkpoints, declared outcomes) or from a fixed
rule in this file (retry a transient load, escalate anything unknown).

Result contract (ReplayResult.status):
  - success:           every step passed its checkpoint, outputs extracted
  - business_outcome:  a declared, legitimate non-success state was reached (outcome_code)
  - failure:           a hard failure; step_id / expected / observed say where and why
Recoverable conditions never surface as a status; they are listed in `recoveries`.

Human handoff is expressed as three exceptions, one per disposition the operator can
choose: RetryStep (resume), RestartFlow (restart), StopReplay (abort).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import NoReturn

from .artifact import locator_placeholders, substitute, substitute_locator
from .escalation import AUTOMATION, Escalator
from .evidence import RunLog
from .models import (Action, Artifact, Element, InterventionRequest, Locator, Observation, ReplayResult, Step,
                     TransientError)
from .policy import Policy, Verdict
from .surface import Surface, visible_text

MAX_TRANSIENT_ATTEMPTS = 3
MAX_RESTARTS = 1
MAX_DIALOG_RECOVERIES_PER_STEP = 3
CHECKPOINT_TIMEOUT_S = 6.0
POLL_INTERVAL_S = 0.3


class RetryStep(Exception):
    """The human resolved the situation; re-check and re-run the current step."""


class RestartFlow(Exception):
    """The human reset the session; start the flow over from the first step."""


class StopReplay(Exception):
    """The human aborted (or no operator was available); carries the failure result."""

    def __init__(self, result: ReplayResult):
        super().__init__(result.outcome_code)
        self.result = result


@dataclass
class ReplayContext:
    """Everything one replay run carries between steps."""

    artifact: Artifact
    params: dict
    surface: Surface
    policy: Policy
    escalator: Escalator
    log: RunLog
    outputs: dict = field(default_factory=dict)
    recoveries: list[str] = field(default_factory=list)


def replay(
    artifact: Artifact,
    params: dict,
    surface: Surface,
    policy: Policy,
    escalator: Escalator,
    log: RunLog,
) -> ReplayResult:
    """Replay an artifact with concrete input parameters."""
    missing = [name for name, spec in artifact.inputs.items() if spec.get("required") and name not in params]
    if missing:
        return failure("missing_inputs", None, f"inputs {sorted(artifact.inputs)}", f"missing {missing}")

    log.event("replay_started", capability=artifact.name, version=artifact.version, status=artifact.status,
              params=params, steps=len(artifact.steps))
    ctx = ReplayContext(artifact, params, surface, policy, escalator, log)
    return run_with_restarts(ctx, run_flow)


def run_with_restarts(ctx: ReplayContext, flow) -> ReplayResult:
    """Run a flow (linear steps or a graph), honoring a bounded human-requested restart, then finish the run."""
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
            ctx.log.event("flow_restarted", restart=restarts)
    result.recoveries = list(ctx.recoveries)
    result.interventions = list(ctx.escalator.interventions)
    ctx.log.screenshot(ctx.surface, f"final-{result.status}")
    ctx.log.event("replay_finished", status=result.status, outcome_code=result.outcome_code, step_id=result.step_id,
                  outputs=result.outputs, recoveries=result.recoveries)
    return result


def run_flow(ctx: ReplayContext) -> ReplayResult:
    """Execute every step in order; stop at the first business outcome or failure."""
    ctx.escalator.control.require(AUTOMATION)
    for step in ctx.artifact.steps:
        stop = run_step(ctx, step)
        if stop is not None:
            return stop
    return verify_success(ctx)


def run_step(ctx: ReplayContext, step: Step) -> ReplayResult | None:
    """Run one step with transient retries and human-resume handling."""
    attempts = 0
    verify_first = False  # after a retry or a handoff, the step may already be satisfied
    while True:
        attempts += 1
        try:
            return attempt_step(ctx, step, verify_first)
        except RetryStep:
            verify_first = True
        except StopReplay as stop:
            return stop.result
        except TransientError as error:
            if attempts >= MAX_TRANSIENT_ATTEMPTS:
                return failure("transient_retries_exhausted", step.id, "page to load", str(error))
            ctx.recoveries.append(f"{step.id}: transient error ({error}); retry {attempts}")
            ctx.log.event("recovered", step_id=step.id, kind="transient_retry", detail=str(error), attempt=attempts)
            time.sleep(attempts)  # linear backoff: 1s, then 2s
            reload_after_transient(ctx, error)
            verify_first = True


def reload_after_transient(ctx: ReplayContext, error: TransientError) -> None:
    """Reload the page that failed. If the reload fails too, the next attempt will notice."""
    if not error.url:
        return
    try:
        ctx.surface.navigate(error.url)
    except TransientError as again:
        ctx.log.event("reload_failed", detail=str(again))


def attempt_step(ctx: ReplayContext, step: Step, verify_first: bool) -> ReplayResult | None:
    concrete_value = substitute(step.value, ctx.params)
    observation = settle(ctx, step)
    if verify_first and step.checkpoint and checkpoint_met(step.checkpoint, observation, ctx.params):
        ctx.log.event("step_already_satisfied", step_id=step.id)
        return None

    # 1. Policy: the allowlist is re-checked at replay time, so an edited artifact cannot escape it.
    verdict = check_policy(ctx, step, concrete_value, observation.url)
    if verdict.decision == "deny":
        return failure("policy_denied", step.id, "an allowlisted action", verdict.reason)
    if verdict.decision == "confirm" and not confirmed_by_human(ctx, step, verdict.reason):
        return failure("risky_step_not_confirmed", step.id, "human approval", verdict.reason)

    # 2. Targeting: walk the locator ladder; an unresolvable target is a stuck state, not a crash.
    element = None
    if step.action != "navigate":
        element = resolve_target(ctx, step)
        if element is None and step.action == "extract" and output_is_optional(ctx, step):
            # Absence is only acceptable on a screen that is not a declared outcome (e.g. a 404).
            business = match_business_outcome(ctx.artifact, observation)
            if business is not None:
                return business_result(ctx, step, business)
            return record_missing_output(ctx, step, "control not on screen")
        if element is None:
            # A required control that is not there may itself be a declared outcome
            # ("the requested product is not listed") before it is a reason to call a human.
            absence = match_absence_outcome(ctx, step, observation)
            if absence is not None:
                return business_result(ctx, step, absence)
            escalate(ctx, step, "target_not_found", f"a control matching {step.target.strategies[0]}",
                     visible_text(observation)[:300])

    # 3. Act, then classify what the screen shows next.
    if step.action == "extract":
        return extract_output(ctx, step, element)
    act(ctx, step, element, concrete_value)
    return settle_after_action(ctx, step, before=observation)


# ---------- perception, dialogs, outcomes ----------

def settle(ctx: ReplayContext, step: Step) -> Observation:
    """Observe the screen, clearing any dialog the artifact knows how to clear first."""
    observation = ctx.surface.observe()
    recoveries = 0
    while observation.dialog:
        outcome = match_dialog(ctx.artifact, observation.dialog)
        if outcome is None or recoveries >= MAX_DIALOG_RECOVERIES_PER_STEP:
            escalate(ctx, step, "unknown_dialog", "no dialog, or a declared recoverable one",
                     f"dialog: {observation.dialog}")
        recover(ctx, step, outcome)
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


def recover(ctx: ReplayContext, step: Step, outcome: dict) -> None:
    """Apply a declared recovery (currently: click a control) and log it."""
    action = outcome["recover"]
    element = ctx.surface.resolve(Locator(strategies=action["target"]["strategies"]))
    if element is None:
        escalate(ctx, step, "recovery_failed", f"the control that clears '{outcome['code']}'", "control not found")
    ctx.surface.click(element)
    ctx.recoveries.append(f"{step.id}: {outcome['code']}")
    ctx.log.event("recovered", step_id=step.id, kind=outcome["code"], detail=outcome["detect"])


def checkpoint_met(checkpoint: dict, observation: Observation, params: dict) -> bool:
    """All conditions in a checkpoint must hold against what is currently perceived."""
    text = visible_text(observation)
    for kind, raw in checkpoint.items():
        expected = substitute(str(raw), params)
        if kind == "text_contains" and expected not in text:
            return False
        if kind == "url_contains" and expected not in observation.url:
            return False
    return True


def settle_after_action(ctx: ReplayContext, step: Step, before: Observation) -> ReplayResult | None:
    """Wait for the checkpoint; meanwhile classify whatever else shows up.

    Order matters: a met checkpoint always wins (outcome text can legitimately appear
    elsewhere on a good screen, e.g. in help text); outcome text is only judged once
    the screen has actually changed from what it was before the action, so a slow
    navigation is never mistaken for an error shown on the previous screen. A step
    without a checkpoint passes after that single outcome check.
    """
    deadline = time.time() + CHECKPOINT_TIMEOUT_S
    text_before = visible_text(before)
    while True:
        observation = settle(ctx, step)
        if step.checkpoint and checkpoint_met(step.checkpoint, observation, ctx.params):
            ctx.log.event("checkpoint_passed", step_id=step.id, checkpoint=step.checkpoint)
            return None
        screen_changed = visible_text(observation) != text_before
        business = match_business_outcome(ctx.artifact, observation) if screen_changed else None
        if business is not None:
            return business_result(ctx, step, business)
        if not step.checkpoint:
            ctx.log.event("checkpoint_passed", step_id=step.id, checkpoint=None)
            return None
        if time.time() >= deadline:
            escalate(ctx, step, "checkpoint_not_met", describe_checkpoint(step), visible_text(observation)[:300])
        time.sleep(POLL_INTERVAL_S)


def business_result(ctx: ReplayContext, step: Step, outcome: dict) -> ReplayResult:
    ctx.log.event("business_outcome", step_id=step.id, code=outcome["code"])
    detect = outcome["detect"]
    observed = detect.get("text_contains") or f"missing: {substitute(detect.get('text_missing'), ctx.params)}"
    return ReplayResult(status="business_outcome", outcome_code=outcome["code"], step_id=step.id,
                        expected=describe_checkpoint(step), observed=observed)


def match_absence_outcome(ctx: ReplayContext, step: Step, observation: Observation) -> dict | None:
    """Outcomes defined by something *missing* apply only to the step that looks for that thing.

    "The requested product is not listed" is judged when the step targeting the product
    cannot find it, never when some unrelated step is stuck on a screen without the product.
    """
    text = visible_text(observation)
    target_text = " ".join(str(strategy.get(key, "")) for strategy in substitute_locator(step.target, ctx.params).strategies
                           for key in ("name", "text", "context"))
    for outcome in ctx.artifact.outcomes:
        needle = substitute(outcome.get("detect", {}).get("text_missing"), ctx.params)
        if outcome["kind"] != "business" or not needle:
            continue
        if contains(target_text, needle) and not contains(text, needle):
            return outcome
    return None


def describe_checkpoint(step: Step) -> str:
    return str(step.checkpoint) if step.checkpoint else "no checkpoint"


# ---------- targeting, policy, acting ----------

def resolve_target(ctx: ReplayContext, step: Step) -> Element | None:
    """Walk the locator ladder one rung at a time so the log shows which rung matched."""
    concrete = substitute_locator(step.target, ctx.params)   # "{{product_name}}" -> the requested product
    parameterized = any(locator_placeholders(s) for s in step.target.strategies)
    for rung, (strategy, recorded) in enumerate(zip(concrete.strategies, step.target.strategies)):
        if parameterized and not locator_placeholders(recorded):
            # Never fall back to a rung that ignores the input: it would act on the wrong item.
            ctx.log.event("rung_skipped", step_id=step.id, rung=rung, strategy=strategy["kind"],
                          reason="does not depend on the input the locator is parameterized by")
            continue
        element = ctx.surface.resolve(Locator(strategies=[strategy]))
        if element is not None:
            ctx.log.event("target_resolved", step_id=step.id, rung=rung, strategy=strategy["kind"],
                          drift_signal=rung > 0)
            return element
    ctx.log.event("target_unresolved", step_id=step.id, strategies=[s["kind"] for s in step.target.strategies])
    return None


def element_hint(step: Step) -> Element | None:
    """What the artifact says the target is, before resolving it (used for policy checks)."""
    if step.target is None:
        return None
    for strategy in step.target.strategies:
        if strategy.get("kind") == "role":
            return Element(role=strategy["role"], name=strategy["name"])
    return Element(role="unknown", name="")


def check_policy(ctx: ReplayContext, step: Step, value: str | None, url: str) -> Verdict:
    action = Action(kind=step.action, target=element_hint(step), value=value)
    verdict = ctx.policy.check(action, url)
    if verdict.decision == "allow" and step.risk == "risky":
        hint = element_hint(step)
        verdict = Verdict("confirm", f"step {step.id} ('{hint.name if hint else value}') was recorded as risky")
    ctx.log.event("policy_checked", step_id=step.id, decision=verdict.decision, reason=verdict.reason)
    return verdict


def confirmed_by_human(ctx: ReplayContext, step: Step, reason: str) -> bool:
    hint = element_hint(step)
    request = InterventionRequest(run_id=ctx.log.run_id, capability=ctx.artifact.name,
                                  goal=ctx.artifact.description, step_id=step.id,
                                  kind="confirm", reason=f"confirmation required: {reason}",
                                  observed=f"about to {step.action} {hint.role if hint else ''} "
                                           f"'{hint.name if hint else step.value}'", screenshot=None)
    result = ctx.escalator.request(request, ctx.surface)
    return result.disposition == "approve"


def act(ctx: ReplayContext, step: Step, element: Element | None, value: str | None) -> None:
    if step.action == "navigate":
        ctx.surface.navigate(value or "")
    elif step.action == "click":
        ctx.surface.click(element)
    elif step.action == "type":
        ctx.surface.type(element, value or "")
    ctx.log.event("acted", step_id=step.id, action=step.action, value="[typed]" if step.action == "type" else value)


def extract_output(ctx: ReplayContext, step: Step, element: Element) -> ReplayResult | None:
    """Read a declared output from the resolved control.

    A pattern that does not match is a hard failure for a required output and a null for
    an optional one (a discount line that not every order shows, for example).
    """
    name = step.value
    spec = ctx.artifact.outputs[name]
    text = element.text or element.name
    value = text.strip()
    pattern = spec.get("pattern")
    if pattern:
        match = re.search(pattern, text)
        if match is None and output_is_optional(ctx, step):
            return record_missing_output(ctx, step, f"pattern did not match {text[:80]!r}")
        if match is None:
            return failure("output_not_found", step.id, f"text matching {pattern!r}", text[:200])
        value = (match.group(1) if match.groups() else match.group(0)).strip()
    ctx.outputs[name] = typed_value(value, spec.get("type", "string"))
    ctx.log.event("output_extracted", step_id=step.id, output=name, value=ctx.outputs[name])
    return None


def output_is_optional(ctx: ReplayContext, step: Step) -> bool:
    return not ctx.artifact.outputs[step.value].get("required", True)


def record_missing_output(ctx: ReplayContext, step: Step, why: str) -> None:
    """An optional output that is absent is a null in the result, not an error."""
    ctx.outputs[step.value] = None
    ctx.log.event("output_missing", step_id=step.id, output=step.value, reason=why)
    return None


def typed_value(value: str, kind: str):
    """Coerce to the declared output type; leaves the text alone if it does not fit."""
    plain = re.sub(r"^[$€£]", "", value.replace(",", "").replace(" ", ""))
    if kind == "integer" and plain.lstrip("-").isdigit():
        return int(plain)
    if kind == "number" and re.fullmatch(r"-?\d+(\.\d+)?", plain):
        return float(plain)
    return value


# ---------- escalation and results ----------

def escalate(ctx: ReplayContext, step: Step, code: str, expected: str, observed: str) -> NoReturn:
    """Hand the live session to a human; translate their disposition into control flow."""
    request = InterventionRequest(run_id=ctx.log.run_id, capability=ctx.artifact.name,
                                  goal=ctx.artifact.description, step_id=step.id,
                                  kind="stuck", reason=f"{code}: expected {expected}", observed=observed,
                                  screenshot=None)
    result = ctx.escalator.request(request, ctx.surface)
    if result.disposition == "resume":
        raise RetryStep
    if result.disposition == "restart":
        raise RestartFlow
    raise StopReplay(failure(code, step.id, expected, observed))


def verify_success(ctx: ReplayContext) -> ReplayResult:
    observation = ctx.surface.observe()
    if checkpoint_met(ctx.artifact.success, observation, ctx.params):
        ctx.log.event("success_verified", checkpoint=ctx.artifact.success)
        return ReplayResult(status="success", outputs=dict(ctx.outputs))
    business = match_business_outcome(ctx.artifact, observation)
    if business is not None:
        return business_result(ctx, ctx.artifact.steps[-1], business)
    return failure("success_condition_not_met", None, str(ctx.artifact.success), visible_text(observation)[:300])


def failure(code: str, step_id: str | None, expected: str, observed: str) -> ReplayResult:
    return ReplayResult(status="failure", outcome_code=code, step_id=step_id, expected=expected, observed=observed)
