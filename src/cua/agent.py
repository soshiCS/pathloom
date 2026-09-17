"""One-time LLM-driven discovery.

The loop is:  observe -> ask planner for ONE action -> policy check -> act -> verify -> record

What gets recorded is not the transcript but the reusable flow as a linear capability graph:
every performed action becomes an action node (with a checkpoint only when the planner's
expectation was actually seen, an effect from the policy's risk verdict, and a retry policy
derived from both), the nodes are chained by unconditional edges into a success terminal,
dialog dismissals become recoverable outcomes, and every literal input value is replaced by
a named placeholder.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import replace

from .artifact import (build_linear, classify_effect, classify_retry_safety, parameterize, parameterize_locator,
                       placeholders_in, substitute)
from .escalation import AUTOMATION, Escalator
from .evidence import RunLog
from .models import (Action, Artifact, Element, GraphAction, GraphNode, InterventionRequest, Locator, Observation,
                     ReusePrefix, ScreenshotFrame, TransientError)
from .planner import Planner, VisionPlanner
from .policy import Policy
from .surface import Surface, is_ambiguous, locator_for, visible_text

DEFAULT_MAX_STEPS = 15
DEFAULT_MAX_VISION_ATTEMPTS = 2
MAX_CONSECUTIVE_DENIALS = 3
EXPECT_TIMEOUT_S = 6.0
POLL_INTERVAL_S = 0.3


class DiscoveryFailed(Exception):
    """Discovery hit a stopping condition without reaching the goal."""


class Recorder:
    """Accumulates the pieces of the artifact while discovery runs: the action nodes of a linear graph."""

    def __init__(self, params: dict, output_contract: dict | None = None):
        self.params = params
        self.output_contract = output_contract or {}   # declared outputs: their spec wins over the planner's
        self.nodes: list[GraphNode] = []
        self.outputs: dict = {}
        self.outcomes: list[dict] = []
        self.success: dict | None = None  # set when the planner's "done" claim is verified

    def next_id(self) -> str:
        return f"s{len(self.nodes) + 1}"

    def record_step(self, action: Action, risk: str, seen: Observation, locator: Locator | None = None) -> GraphNode:
        """Record one performed action as the next action node; returns it."""
        # Extraction verifies itself (the pattern must match), so it carries no checkpoint.
        checkpoint = None
        if action.expect and action.kind != "extract":
            checkpoint = {"text_contains": parameterize(action.expect, self.params)}
        target = None
        if action.target:
            # Qualify the locator by its item only when the name alone was ambiguous on screen,
            # then parameterize it: "Add to cart" in "{{product_name}}". A caller may supply the
            # ladder itself (a vision target records exact coordinates instead).
            ladder = locator or locator_for(action.target, is_ambiguous(action.target, seen))
            target = parameterize_locator(ladder, self.params)
        value = action.output_name if action.kind == "extract" else parameterize(action.value, self.params)
        effect = classify_effect(action.kind, risky=risk == "risky")
        node = GraphNode(id=self.next_id(), kind="action",
                         action=GraphAction(action=action.kind, target=target, value=value, checkpoint=checkpoint),
                         effect=effect, retry_safety=classify_retry_safety(action.kind, effect, checkpoint))
        self.nodes.append(node)
        return node

    def record_output(self, action: Action, sample: str) -> None:
        declared = self.output_contract.get(action.output_name, {})
        self.outputs[action.output_name] = {
            "type": declared.get("type", infer_type(sample)),
            "required": declared.get("required", not action.optional),   # optional outputs come back as null
            "pattern": self.output_pattern(action),
            "description": f"Text read from {action.target.role} '{action.target.name}'",
            "example": parameterize(sample, self.params),
        }

    def output_pattern(self, action: Action) -> str | None:
        """The declared pattern for a contracted output, else what the planner proposed."""
        declared = self.output_contract.get(action.output_name)
        return declared["pattern"] if declared else action.pattern

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
        for node in reversed(self.nodes):
            if node.action.checkpoint:
                return node.action.checkpoint
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
    output_contract: dict | None = None,
    selectors: set[str] | None = None,
    vision: VisionPlanner | None = None,
    max_vision_attempts: int = DEFAULT_MAX_VISION_ATTEMPTS,
    prefix: ReusePrefix | None = None,
) -> Artifact:
    """Discover a capability by driving the live surface with the planner; return its artifact.

    `output_contract` (name -> {type, required, pattern}) fixes how declared outputs are read,
    whatever regex the planner proposes, so independent runs record the same contract.
    `selectors` are inputs that choose a path rather than data the flow uses: their values are
    recorded literally, never as placeholders, so a route called "details" cannot rewrite the
    "View details" link a path happens to click.
    `vision`, when given, is the bounded screenshot fallback (see `Vision`): tried only when the
    planner is stuck, at most `max_vision_attempts` times per run and once per unchanged screen.
    `prefix`, when given, is a verified reusable path that already ran on this surface (reuse.py):
    its action nodes open the recording and the surface is already positioned, so the entry
    navigation is skipped and the planner's first look is the screen the prefix ended on.
    """
    sensitive = sensitive or set()
    # Sensitive values are never shown to the model; it sees (and types) the placeholder.
    visible_params = {key: ("{{" + key + "}}" if key in sensitive else value) for key, value in params.items()}
    recorder = Recorder({key: value for key, value in params.items() if key not in (selectors or set())},
                        output_contract)
    history: list[Action] = []
    log.event("discovery_started", goal=goal, capability=name, params=visible_params, entry_url=entry_url,
              planner=planner.name, allowed_hosts=policy.allowed_hosts)

    escalator.control.require(AUTOMATION)
    if prefix is None:
        navigate_with_retry(surface, entry_url, log)
        observation = observe_with_retry(surface, log)
        app_name = getattr(surface, "title", "") or "unknown"  # captured at the entry screen, before navigating away
        checkpoint = entry_checkpoint(observation)
        recorder.nodes.append(GraphNode(
            id="s1", kind="action", action=GraphAction(action="navigate", target=None, value=entry_url,
                                                        checkpoint=checkpoint),
            effect="none", retry_safety=classify_retry_safety("navigate", "none", checkpoint)))
        log.screenshot(surface, "entry")
    else:
        recorder.nodes.extend(prefix.nodes)
        recorder.outcomes.extend(prefix.outcomes)
        recorder.outputs.update(prefix.outputs)
        observation = observe_with_retry(surface, log)
        app_name = getattr(surface, "title", "") or "unknown"
        log.event("discovery_resumed_after_reuse", imported_nodes=len(prefix.nodes),
                  source=prefix.provenance.get("source_name"), source_version=prefix.provenance.get("source_version"))
        log.screenshot(surface, "after-reuse")

    consecutive_denials = 0
    fallback = Vision(vision, max_vision_attempts, goal, visible_params, params, surface, log) if vision else None
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
            # The structured planner cannot see a control it needs: the bounded vision fallback may
            # propose one visual action; everything else about it (policy, acting, verification) is
            # the normal path below.
            visual = fallback.propose(action, observation, history) if fallback else None
            if visual is None:
                handle_stuck(action, observation, name, goal, escalator, surface, log)
                history.append(action)
                continue
            action = visual

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
        if action.target is not None and action.target.source == "vision":
            verify_vision_action(action, observation, params, policy, recorder, surface, log, fallback, name, goal,
                                 escalator)
        else:
            verify_and_record(action, observation, params, policy, recorder, surface, log)
        history.append(action)
    else:
        log.screenshot(surface, "max-steps")
        raise DiscoveryFailed(f"goal not reached within {max_steps} steps")

    recorder.outcomes.extend(extra_outcomes or [])
    success = recorder.success or recorder.last_checkpoint() or {"url_contains": entry_url}
    built = build_linear(
        name=name, goal=goal,
        surface_meta={"kind": "web", "app": app_name, "entry_url": entry_url,
                      "allowed_hosts": list(policy.allowed_hosts)},
        params=params, nodes=recorder.nodes, outputs=recorder.outputs, outcomes=recorder.outcomes,
        success=success, run_id=log.run_id, sensitive=sensitive, planner_name=planner.name,
    )
    built.provenance["interventions"] = len(escalator.interventions)
    if fallback and fallback.attempts:
        built.provenance["vision_fallback"] = fallback.provenance()
    if prefix is not None:
        built.provenance["reuse"] = {**prefix.provenance, "planner_decisions": turn}
    log.event("artifact_built", capability=built.name, version=built.version, nodes=len(built.nodes),
              edges=len(built.edges), entry_node=built.entry_node, outputs=list(built.outputs),
              outcomes=[o["code"] for o in built.outcomes])
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
        node = recorder.record_step(action, risk=policy.risk_of(action), seen=before)
        log.event("recorded_step", step_id=node.id, action=action.kind, checkpoint=action.expect,
                  effect=node.effect, retry_safety=node.retry_safety)


def record_extraction(action: Action, recorder: Recorder, log: RunLog, before: Observation) -> None:
    if not action.output_name or action.target is None:
        action.result = "extract needs output_name and a target control"
        return
    pattern = recorder.output_pattern(action)
    value = apply_pattern(action.target.text or action.target.name, pattern)
    if not value:
        action.result = f"pattern {pattern!r} matched nothing in {action.target.text!r}"
        log.event("extraction_failed", output_name=action.output_name, text=action.target.text)
        return
    action.result = f"ok, extracted {value!r}"
    recorder.record_output(action, value)
    node = recorder.record_step(action, risk="safe", seen=before)
    log.event("recorded_output", output_name=action.output_name, value=value, step_id=node.id)


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
        recorder.success = {"text_contains": parameterize(expected, recorder.params)}
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


# ---------- the vision fallback ----------

class Vision:
    """Bounded screenshot fallback for discovery: one visual proposal when the planner is stuck.

    Budget: at most `max_attempts` attempts per run, and never twice for the same screen (a
    fingerprint of the structured observation plus the masked screenshot bytes; the bytes are
    hashed, never stored). A proposal is turned into an ordinary click or type Action whose
    target carries `source="vision"` and no structural reference, so it goes through the normal
    policy check and is acted on by coordinates.
    """

    def __init__(self, planner: VisionPlanner, max_attempts: int, goal: str, visible_params: dict, params: dict,
                 surface: Surface, log: RunLog):
        self.planner, self.max_attempts = planner, max_attempts
        self.goal, self.visible_params, self.params, self.surface, self.log = goal, visible_params, params, surface, log
        self.attempts = 0
        self.last_fingerprint: str | None = None
        self.exhausted_logged = False
        self.recorded_steps: list[str] = []
        self.frame: ScreenshotFrame | None = None

    def propose(self, stuck: Action, observation: Observation, history: list[Action]) -> Action | None:
        if stuck.stuck_cause != "perception":
            # Provider failures and ordinary uncertainty are not perception gaps: straight to a person.
            self.log.event("vision_fallback_skipped", reason=f"stuck cause is {stuck.stuck_cause or 'unknown'!r}, "
                                                             f"not a missing control")
            return None
        capture = getattr(self.surface, "viewport_screenshot", None)
        decide = getattr(self.planner, "decide_visually", None)
        if capture is None or decide is None:
            self.log.event("vision_fallback_unavailable", reason="surface or planner cannot provide vision")
            return None
        if self.attempts >= self.max_attempts:
            if not self.exhausted_logged:
                self.log.event("vision_budget_exhausted", attempts=self.attempts, max_attempts=self.max_attempts)
                self.exhausted_logged = True
            return None
        self.log.shots_dir.mkdir(parents=True, exist_ok=True)
        path = self.log.shots_dir / f"{self.log.seq:03d}-vision-attempt-{self.attempts + 1}.png"
        try:
            frame = capture(str(path))
        except Exception as error:   # an unusable capture (scale mismatch, browser gone) is not a proposal
            self.log.event("vision_fallback_unavailable", reason=f"{type(error).__name__}: {str(error)[:200]}")
            return None
        fingerprint = screen_fingerprint(observation, frame.png)
        if fingerprint == self.last_fingerprint:
            self.log.event("vision_fallback_skipped", reason="screen unchanged since the last visual attempt",
                           fingerprint=fingerprint)
            return None
        self.attempts += 1
        self.last_fingerprint = fingerprint
        self.frame = frame
        remaining = self.max_attempts - self.attempts
        self.log.event("vision_fallback_requested", attempt=self.attempts, reason=stuck.reason, fingerprint=fingerprint,
                       screenshot=str(path), width=frame.width, height=frame.height, scroll=[frame.scroll_x,
                       frame.scroll_y], remaining=remaining)
        decision = decide(self.goal, self.visible_params, observation, history, frame, remaining)
        target = decision.target
        self.log.event("vision_fallback_decided", attempt=self.attempts, kind=decision.kind,
                       target=describe_visual(target), confidence=target.confidence if target else None,
                       expect=target.expect if target else None, reason=decision.reason[:200])
        if decision.kind == "unavailable":
            self.log.event("vision_fallback_unavailable", reason=decision.reason[:200])
            return None
        if decision.rejected:
            self.log.event("vision_target_rejected", attempt=self.attempts, reason=decision.rejected)
            return None
        if decision.kind == "no_target" or target is None:
            return None
        rejection = self.unsafe_to_act(decision.value, target.expect, observation)
        if rejection:
            self.log.event("vision_target_rejected", attempt=self.attempts, reason=rejection)
            return None
        element = Element(role=target.role, name=target.name, text="", box=(target.x, target.y, target.width,
                                                                              target.height), source="vision")
        kind = "type" if decision.kind == "visual_type" else "click"
        return Action(kind=kind, target=element, value=decision.value, expect=target.expect,
                      reason=f"vision fallback: {target.reason}")

    def unsafe_to_act(self, value: str | None, expect: str, observation: Observation) -> str | None:
        """Why an accepted-looking proposal must still not run: an undeclared placeholder in what would be typed or
        checked, or an expected text that is already on screen (the action could then never be verified)."""
        unknown = (placeholders_in(value) | placeholders_in(expect)) - set(self.params)
        if unknown:
            return f"undeclared placeholders {sorted(unknown)} in the proposed value or expected text"
        expected = substitute(expect, self.params)
        if expected in visible_text(observation):
            return f"expected text {expected!r} is already on screen before acting; the action could not be verified"
        return None

    def provenance(self) -> dict:
        return {"attempts": self.attempts, "max_attempts": self.max_attempts, "planner": self.planner.name,
                "steps": list(self.recorded_steps)}


def screen_fingerprint(observation: Observation, png: bytes) -> str:
    """A hash of what is on screen: structured controls plus the masked image bytes. Nothing sensitive is kept."""
    digest = hashlib.sha256()
    digest.update(observation.url.encode())
    for element in observation.elements:
        digest.update(f"{element.role}|{element.name}|{element.text}|{element.states}".encode())
    digest.update(png)
    return digest.hexdigest()[:16]


def describe_visual(target) -> dict | None:
    if target is None:
        return None
    return {"role": target.role, "name": target.name[:80], "box": [target.x, target.y, target.width, target.height]}


def verify_vision_action(action: Action, before: Observation, params: dict, policy: Policy, recorder: Recorder,
                         surface: Surface, log: RunLog, fallback: "Vision", name: str, goal: str,
                         escalator: Escalator) -> None:
    """A visual action is kept only when structured perception proves its expected text appeared.

    Otherwise it is neither repeated nor recorded: a person is asked, with the screenshot in the log.
    """
    expected = substitute(action.expect, params)
    verified = wait_for_text(surface, expected)
    log.event("vision_action_verified", verified=verified, kind=action.kind, target=describe_visual_element(action),
              expect=action.expect)
    if not verified:
        action.result = f"visual {action.kind} did not lead to {expected!r}"
        stuck = Action(kind="stuck", reason=f"visual action could not be verified: expected {expected!r}")
        handle_stuck(stuck, surface.observe(), name, goal, escalator, surface, log)
        return
    action.result = "ok"
    node = recorder.record_step(action, risk=policy.risk_of(action), seen=before,
                                locator=vision_locator(action.target, before, fallback.frame))
    fallback.recorded_steps.append(node.id)
    log.event("recorded_step", step_id=node.id, action=action.kind, checkpoint=action.expect,
              effect=node.effect, retry_safety=node.retry_safety, targeting="vision")


def vision_locator(target: Element, seen: Observation, frame: ScreenshotFrame) -> Locator:
    """The ladder recorded for a visual target: exact coordinates, bound to the viewport and scroll position
    they were captured at, and a role/name rung only when a structured element with that identity really
    sits under the box (grounded); a guessed name alone must never let replay click a different control."""
    x, y, w, h = target.box
    rungs: list[dict] = []
    for element in seen.elements:
        if (element.role, element.name) == (target.role, target.name) and boxes_overlap(element.box, target.box):
            rungs.append({"kind": "role", "role": element.role, "name": element.name})
            break
    rungs.append({"kind": "coords", "x": x + w // 2, "y": y + h // 2, "exact": True,
                  "viewport": {"width": frame.width, "height": frame.height},
                  "scroll": {"x": frame.scroll_x, "y": frame.scroll_y}})
    return Locator(strategies=rungs)


def boxes_overlap(a: tuple, b: tuple) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return aw > 0 and ah > 0 and ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def describe_visual_element(action: Action) -> dict | None:
    target = action.target
    if target is None:
        return None
    return {"role": target.role, "name": target.name[:80], "box": list(target.box)}
