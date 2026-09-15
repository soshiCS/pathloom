"""One-time LLM-driven discovery.

The loop is:  observe -> ask planner for ONE action -> policy check -> act -> verify -> record

What gets recorded is not the transcript but the reusable flow: every performed action
becomes a step (with a checkpoint only when the planner's expectation was actually seen),
dialog dismissals become recoverable outcomes, and every literal input value is replaced
by a named placeholder.
"""
from __future__ import annotations

import re
import time
from dataclasses import replace

from . import artifact as artifact_module
from .artifact import parameterize, parameterize_locator, substitute
from .escalation import AUTOMATION, Escalator
from .evidence import RunLog
from .models import Action, Artifact, InterventionRequest, Observation, Step, TransientError
from .planner import Planner
from .policy import Policy
from .surface import Surface, is_ambiguous, locator_for, visible_text

DEFAULT_MAX_STEPS = 15
MAX_CONSECUTIVE_DENIALS = 3
EXPECT_TIMEOUT_S = 6.0
POLL_INTERVAL_S = 0.3


class DiscoveryFailed(Exception):
    """Discovery hit a stopping condition without reaching the goal."""


class Recorder:
    """Accumulates the pieces of the artifact while discovery runs."""

    def __init__(self, params: dict):
        self.params = params
        self.steps: list[Step] = []
        self.outputs: dict = {}
        self.outcomes: list[dict] = []
        self.success: dict | None = None  # set when the planner's "done" claim is verified

    def next_id(self) -> str:
        return f"s{len(self.steps) + 1}"

    def record_step(self, action: Action, risk: str, seen: Observation) -> None:
        # Extraction verifies itself (the pattern must match), so it carries no checkpoint.
        checkpoint = None
        if action.expect and action.kind != "extract":
            checkpoint = {"text_contains": parameterize(action.expect, self.params)}
        target = None
        if action.target:
            # Qualify the locator by its item only when the name alone was ambiguous on screen,
            # then parameterize it: "Add to cart" in "{{product_name}}".
            target = parameterize_locator(locator_for(action.target, is_ambiguous(action.target, seen)), self.params)
        value = action.output_name if action.kind == "extract" else parameterize(action.value, self.params)
        self.steps.append(Step(id=self.next_id(), action=action.kind, target=target,
                               value=value, checkpoint=checkpoint, risk=risk))

    def record_output(self, action: Action, sample: str) -> None:
        self.outputs[action.output_name] = {
            "type": infer_type(sample),
            "required": not action.optional,   # optional outputs come back as null when absent
            "pattern": action.pattern,
            "description": f"Text read from {action.target.role} '{action.target.name}'",
            "example": parameterize(sample, self.params),
        }

    def record_recoverable_dialog(self, dialog_text: str, action: Action) -> None:
        """A dialog the model dismissed may or may not appear next time: record how to clear it."""
        snippet = dialog_text[:60].strip()
        code = "dismiss_" + re.sub(r"[^a-z0-9]+", "_", snippet.lower()).strip("_")[:40]
        if any(outcome["code"] == code for outcome in self.outcomes):
            return
        self.outcomes.append({
            "code": code, "kind": "recoverable", "source": "observed",
            "detect": {"dialog_contains": snippet},
            "recover": {"action": "click", "target": {"strategies": locator_for(action.target).strategies}},
        })

    def record_business_outcomes(self, declared: list[dict]) -> None:
        """Outcomes the planner declares were not necessarily seen; the source tag says so to reviewers."""
        for outcome in declared:
            detect = {}
            if outcome.get("text_contains"):
                detect["text_contains"] = outcome["text_contains"]
            if outcome.get("text_missing"):
                detect["text_missing"] = parameterize(outcome["text_missing"], self.params)
            if detect:
                self.outcomes.append({"code": outcome["code"], "kind": "business", "source": "planner",
                                      "detect": detect})

    def last_checkpoint(self) -> dict | None:
        for step in reversed(self.steps):
            if step.checkpoint:
                return step.checkpoint
        return None


def discover(
    goal: str,
    name: str,
    params: dict,
    surface: Surface,
    planner: Planner,
    policy: Policy,
    escalator: Escalator,
    log: RunLog,
    entry_url: str,
    sensitive: set[str] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    extra_outcomes: list[dict] | None = None,
) -> Artifact:
    """Discover a capability by driving the live surface with the planner; return its artifact."""
    sensitive = sensitive or set()
    # Sensitive values are never shown to the model; it sees (and types) the placeholder.
    visible_params = {key: ("{{" + key + "}}" if key in sensitive else value) for key, value in params.items()}
    recorder = Recorder(params)
    history: list[Action] = []
    log.event("discovery_started", goal=goal, capability=name, params=visible_params, entry_url=entry_url,
              planner=planner.name, allowed_hosts=policy.allowed_hosts)

    escalator.control.require(AUTOMATION)
    navigate_with_retry(surface, entry_url, log)
    observation = observe_with_retry(surface, log)
    app_name = getattr(surface, "title", "") or "unknown"  # captured at the entry screen, before navigating away
    recorder.steps.append(Step(id="s1", action="navigate", target=None, value=entry_url,
                               checkpoint=entry_checkpoint(observation), risk="safe"))
    log.screenshot(surface, "entry")

    consecutive_denials = 0
    turn = 0
    while turn < max_steps:
        turn += 1
        observation = observe_with_retry(surface, log)
        action = planner.decide(goal, visible_params, observation, history)
        log.event("planner_decided", turn=turn, kind=action.kind, target=describe(action.target),
                  value=action.value, expect=action.expect, output_name=action.output_name,
                  optional=action.optional, reason=action.reason)

        if action.kind == "done":
            if finish(action, observation, params, recorder, log):
                log.screenshot(surface, "goal-reached")
                break
            history.append(action)
            continue
        if action.kind == "stuck":
            handle_stuck(action, observation, name, goal, escalator, surface, log)
            history.append(action)
            continue

        # Policy check happens before anything touches the surface, on the concrete value the surface
        # would receive: a navigation to "{{cart_url}}" is judged by the host it really goes to.
        verdict = policy.check(replace(action, value=substitute(action.value, params)), observation.url)
        log.event("policy_checked", decision=verdict.decision, reason=verdict.reason)
        if verdict.decision == "confirm":
            verdict = confirm_with_human(action, observation, name, goal, escalator, surface, log, verdict.reason)
        if verdict.decision == "deny":
            action.result = f"denied: {verdict.reason}"
            history.append(action)
            consecutive_denials += 1
            if consecutive_denials >= MAX_CONSECUTIVE_DENIALS:
                raise DiscoveryFailed(f"planner kept choosing disallowed actions ({consecutive_denials} in a row)")
            continue
        consecutive_denials = 0

        perform(action, params, surface, log)
        verify_and_record(action, observation, params, policy, recorder, surface, log)
        history.append(action)
    else:
        log.screenshot(surface, "max-steps")
        raise DiscoveryFailed(f"goal not reached within {max_steps} steps")

    recorder.outcomes.extend(extra_outcomes or [])
    success = recorder.success or recorder.last_checkpoint() or {"url_contains": entry_url}
    built = artifact_module.build(
        name=name, goal=goal,
        surface_meta={"kind": "web", "app": app_name, "entry_url": entry_url,
                      "allowed_hosts": list(policy.allowed_hosts)},
        params=params, steps=recorder.steps, outputs=recorder.outputs, outcomes=recorder.outcomes,
        success=success, run_id=log.run_id, sensitive=sensitive, planner_name=planner.name,
    )
    built.provenance["interventions"] = len(escalator.interventions)
    log.event("artifact_built", capability=built.name, version=built.version, steps=len(built.steps),
              outputs=list(built.outputs), outcomes=[o["code"] for o in built.outcomes])
    return built


# ---------- loop pieces ----------

def describe(element) -> str | None:
    return f"{element.role} '{element.name or element.text}'" if element else None


def entry_checkpoint(observation: Observation) -> dict | None:
    """The first heading on the entry page is a cheap 'we are on the right app' check."""
    for element in observation.elements:
        if element.role == "heading" and element.text:
            return {"text_contains": element.text}
    return None


def navigate_with_retry(surface: Surface, url: str, log: RunLog, attempts: int = 3) -> None:
    """The entry page can be slow like any other; a transient load failure is retried, not fatal."""
    for attempt in range(1, attempts + 1):
        try:
            surface.navigate(url)
            return
        except TransientError as error:
            log.event("transient_error", attempt=attempt, error=str(error))
            if attempt == attempts:
                raise DiscoveryFailed(f"entry page kept failing to load: {error}") from error
            time.sleep(attempt)


def observe_with_retry(surface: Surface, log: RunLog, attempts: int = 3) -> Observation:
    """Discovery tolerates a slow/failed load by waiting, the same way replay will."""
    for attempt in range(1, attempts + 1):
        try:
            return surface.observe()
        except TransientError as error:
            log.event("transient_error", attempt=attempt, error=str(error))
            if attempt == attempts:
                raise DiscoveryFailed(f"surface kept failing to load: {error}") from error
            time.sleep(attempt)
            if error.url:
                surface.navigate(error.url)
    raise AssertionError("unreachable")


def perform(action: Action, params: dict, surface: Surface, log: RunLog) -> None:
    """Execute one allowed action. Extract reads; the others drive the surface."""
    concrete_value = substitute(action.value, params)
    if action.kind == "navigate":
        surface.navigate(concrete_value or "")
    elif action.kind == "click":
        surface.click(action.target)
    elif action.kind == "type":
        surface.type(action.target, concrete_value or "")
    # "extract" touches nothing; its value is read during verification below.
    log.event("acted", kind=action.kind, target=describe(action.target), value=concrete_value)


def verify_and_record(action: Action, before: Observation, params: dict, policy: Policy,
                      recorder: Recorder, surface: Surface, log: RunLog) -> None:
    """Record what was actually done. The planner's expectation, if met, becomes the checkpoint.

    A performed action changed the session even if the planner guessed the resulting text
    wrong, so it is always recorded; only the checkpoint is dropped, and the planner is told.
    """
    if action.kind == "extract":
        record_extraction(action, recorder, log, before)
        return
    expected = substitute(action.expect, params)
    if expected and not wait_for_text(surface, expected):
        observed = visible_text(surface.observe())[:300]
        action.result = f"done, but expected to see {expected!r} and the screen shows: {observed}"
        action.expect = None  # no verified checkpoint for this step
        log.event("expectation_failed", expected=expected, observed=observed)
        log.screenshot(surface, "expectation-failed")
    else:
        action.result = "ok"
    if before.dialog and action.kind == "click":
        # Clicking while a modal is up is a dismissal, not part of the main flow.
        recorder.record_recoverable_dialog(before.dialog, action)
        log.event("recorded_recoverable_outcome", dialog=before.dialog[:60])
    else:
        recorder.record_step(action, risk=policy.risk_of(action), seen=before)
        log.event("recorded_step", step_id=recorder.steps[-1].id, action=action.kind, checkpoint=action.expect)


def record_extraction(action: Action, recorder: Recorder, log: RunLog, before: Observation) -> None:
    if not action.output_name or action.target is None:
        action.result = "extract needs output_name and a target control"
        return
    value = apply_pattern(action.target.text or action.target.name, action.pattern)
    if not value:
        action.result = f"pattern {action.pattern!r} matched nothing in {action.target.text!r}"
        log.event("extraction_failed", output_name=action.output_name, text=action.target.text)
        return
    action.result = f"ok, extracted {value!r}"
    recorder.record_output(action, value)
    recorder.record_step(action, risk="safe", seen=before)
    log.event("recorded_output", output_name=action.output_name, value=value, step_id=recorder.steps[-1].id)


def infer_type(sample: str) -> str:
    """Declare the output type from the discovery-time sample so callers get numbers as numbers.

    Currency signs and thousands separators are tolerated ("$ 1,299.99" is a number).
    """
    plain = re.sub(r"^[$€£]\s*", "", sample.strip()).replace(",", "")
    if plain.isdigit():
        return "integer"
    if re.fullmatch(r"\d+\.\d+", plain):
        return "number"
    return "string"


def apply_pattern(text: str, pattern: str | None) -> str:
    if not pattern:
        return text.strip()
    match = re.search(pattern, text)
    if not match:
        return ""
    return (match.group(1) if match.groups() else match.group(0)).strip()


def wait_for_text(surface: Surface, expected: str, timeout: float = EXPECT_TIMEOUT_S) -> bool:
    """Poll perception until the expected text shows up. This is the discovery-time checkpoint."""
    deadline = time.time() + timeout
    while True:
        try:
            if expected in visible_text(surface.observe()):
                return True
        except TransientError:
            pass
        if time.time() >= deadline:
            return False
        time.sleep(POLL_INTERVAL_S)


def finish(action: Action, observation: Observation, params: dict, recorder: Recorder, log: RunLog) -> bool:
    """The planner says the goal is met. Trust but verify against the screen."""
    expected = substitute(action.expect, params)
    if expected and expected not in visible_text(observation):
        action.result = f"claimed done but {expected!r} is not on screen"
        log.event("done_rejected", expected=expected)
        return False
    recorder.record_business_outcomes(action.outcomes)
    if expected and not depends_on_extracted_values(expected, recorder):
        recorder.success = {"text_contains": parameterize(expected, params)}
    else:
        log.event("success_checkpoint_fallback", reason="done text was empty or was an extracted value")
    log.event("goal_reached", outcomes=[o["code"] for o in action.outcomes])
    return True


def depends_on_extracted_values(expected: str, recorder: Recorder) -> bool:
    """A success check must hold for every input; a value read from this run (a balance) does not."""
    return any(spec.get("example") and spec["example"] in expected for spec in recorder.outputs.values())


def handle_stuck(action: Action, observation: Observation, name: str, goal: str, escalator: Escalator,
                 surface: Surface, log: RunLog) -> None:
    """The planner cannot proceed: bring a human in on the same session, then let the planner look again."""
    request = InterventionRequest(run_id=log.run_id, capability=name, goal=goal, step_id=None, kind="stuck",
                                  reason=f"planner is stuck: {action.reason}",
                                  observed=visible_text(observation)[:400], screenshot=None)
    result = escalator.request(request, surface)
    if result.disposition == "abort":
        raise DiscoveryFailed(f"human aborted discovery: {result.note}")
    action.result = f"human intervened ({len(result.human_actions)} manual actions); re-observe the screen"


def confirm_with_human(action: Action, observation: Observation, name: str, goal: str, escalator: Escalator,
                       surface: Surface, log: RunLog, reason: str):
    """Risky actions are never taken on the model's say-so; a person approves or denies."""
    from .policy import Verdict

    request = InterventionRequest(run_id=log.run_id, capability=name, goal=goal, step_id=None, kind="confirm",
                                  reason=f"confirmation required: {reason}",
                                  observed=f"planner wants to {action.kind} {describe(action.target)}; {action.reason}",
                                  screenshot=None)
    result = escalator.request(request, surface)
    if result.disposition == "approve":
        return Verdict("allow", "approved by human operator")
    return Verdict("deny", f"human operator did not approve ({result.disposition})")
