"""Planner interface and planner adapters.

The planner is used only during discovery. Deterministic replay must not use it.

ClaudePlanner and OpenAIPlanner are interchangeable LLM adapters: one API call per
decision, one action per call. They are stateless across calls; the agent passes a
compact history instead of the raw transcript, which keeps the recorded artifact
independent of the provider. Tests use a scripted stand-in that lives under tests/
and implements the same protocol.
"""
from __future__ import annotations

import base64
import json
import math
import os
import time
from typing import Callable, Protocol

from .models import (MAX_EXTRACT_TARGETS, Action, Element, Observation, ReuseCandidate, ScreenshotFrame,
                     VisualDecision, VisualTarget)
from .surface import is_ambiguous

DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-astra")
# Optional: a different model for the vision fallback; the configured model is used otherwise.
DEFAULT_ANTHROPIC_VISION_MODEL = os.environ.get("ANTHROPIC_VISION_MODEL") or None
DEFAULT_OPENAI_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL") or None
MAX_ELEMENTS_IN_PROMPT = 500
HISTORY_IN_PROMPT = 8
MAX_PROVIDER_ATTEMPTS = 3            # one request plus at most two retries of a transient provider failure
PROVIDER_BACKOFF_S = (1.0, 4.0)      # waits before the second and third attempt
RATE_LIMIT_STATUS = 429
# Exception classes (by name, so no SDK import is needed offline) that both SDKs raise for a request that never
# got an answer: the network, a timeout. Everything else is judged by its HTTP status, never by its text.
CONNECTION_ERROR_CLASSES = {"APIConnectionError", "APITimeoutError"}
MIN_VISION_CONFIDENCE = 0.6
VISUAL_KINDS = ("visual_click", "visual_type", "visual_extract", "no_target")
MAX_VISUAL_BOXES = MAX_EXTRACT_TARGETS


class Planner(Protocol):
    """Chooses one next action from the current observation, or one of the verified segments offered."""

    name: str

    def decide(
        self,
        goal: str,
        params: dict,
        observation: Observation,
        history: list[Action],
        candidates: list[ReuseCandidate] = (),
    ) -> Action: ...


class VisionPlanner(Protocol):
    """Proposes exactly one visual action from a masked viewport screenshot. Discovery only."""

    name: str

    def decide_visually(
        self,
        goal: str,
        params: dict,
        observation: Observation,
        history: list[Action],
        frame: ScreenshotFrame,
        remaining_attempts: int,
    ) -> VisualDecision: ...


# ---------- shared helpers ----------

def find_element(observation: Observation, role: str, name: str) -> Element | None:
    """Locate a control by role and accessible name (case-insensitive, name may be a prefix)."""
    for element in observation.elements:
        if element.role == role and element.name.lower().startswith(name.lower()):
            return element
    return None


STUCK_CAUSES = ("perception", "missing_data", "planner", "provider")


class PlannerUnavailable(Exception):
    """The model provider could not answer: transient failures were retried and kept failing, or the failure
    was permanent (authentication, a bad request, quota). Carries the classification, never the provider's text."""

    def __init__(self, provider: str, category: str, attempts: int, error: Exception):
        self.provider, self.category, self.attempts = provider, category, attempts
        self.error_type = type(error).__name__
        self.status = provider_status(error)
        detail = f"HTTP {self.status}" if self.status is not None else self.error_type
        super().__init__(f"{provider} planner unavailable: {category} ({detail}) after {attempts} attempt(s)")


def provider_status(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def failure_category(error: Exception) -> str:
    """rate_limited | server_error | connection (all transient) | permanent, from the status or the class."""
    status = provider_status(error)
    if status == RATE_LIMIT_STATUS:
        return "rate_limited"
    if status is not None:
        return "server_error" if status >= 500 else "permanent"
    if isinstance(error, (ConnectionError, TimeoutError)):
        return "connection"
    if CONNECTION_ERROR_CLASSES & {cls.__name__ for cls in type(error).__mro__}:
        return "connection"
    return "permanent"


TRANSIENT_CATEGORIES = {"rate_limited", "server_error", "connection"}


def call_provider(provider: str, request: Callable, retries: list[dict]):
    """Run one provider request with bounded retries of transient failures.

    A rate limit, a 5xx or a connection/timeout error is retried after a short backoff, at most
    MAX_PROVIDER_ATTEMPTS times in all, and every retry is appended to `retries` as a structured record
    (provider, attempt, category, error type, status; never the message). A permanent failure, or a transient
    one that outlives the budget, raises PlannerUnavailable so discovery can end cleanly.
    """
    for attempt in range(1, MAX_PROVIDER_ATTEMPTS + 1):
        try:
            return request()
        except PlannerUnavailable:
            raise
        except Exception as error:
            category = failure_category(error)
            if category not in TRANSIENT_CATEGORIES or attempt >= MAX_PROVIDER_ATTEMPTS:
                raise PlannerUnavailable(provider, category, attempt, error) from error
            retries.append({"provider": provider, "attempt": attempt, "category": category,
                            "error_type": type(error).__name__, "status": provider_status(error)})
            time.sleep(PROVIDER_BACKOFF_S[min(attempt, len(PROVIDER_BACKOFF_S)) - 1])


def unavailable_reason(error: Exception) -> str:
    """Why a vision request failed, from the provider's own error when the retry boundary wrapped it."""
    cause = error.__cause__ if isinstance(error, PlannerUnavailable) and error.__cause__ is not None else error
    return f"{type(cause).__name__}: {str(cause)[:200]}"


def retry_records(planner) -> list[dict]:
    """The planner's own list of transient provider failures retried since the last drain."""
    return planner.__dict__.setdefault("retries", [])


def drain_retries(planner) -> list[dict]:
    """The retry records a planner accumulated since the last drain (empty for planners that keep none)."""
    retries = getattr(planner, "retries", None)
    if not retries:
        return []
    records, retries[:] = list(retries), []
    return records


def provider_stuck(reason: str) -> Action:
    """A stuck state the provider caused (refusal, malformed output): never a reason to look at pixels."""
    return Action(kind="stuck", reason=reason, stuck_cause="provider")


def action_from_tool_input(data: dict, observation: Observation) -> Action:
    """Turn the model's tool call into an Action bound to real observed elements.

    A malformed extraction (no target, no output name, an empty, duplicated or oversized index list) is
    a provider failure: it never reaches execution, and the reason goes to the next decision.
    """
    target = None
    index = data.get("element_index")
    if index is not None:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(observation.elements):
            return provider_stuck(f"model referenced element {index!r}, which does not exist")
        target = observation.elements[index]
    kind = data.get("kind")
    output_name = data.get("output_name")
    if kind in ("extract", "extract_many") and (not isinstance(output_name, str) or not output_name.strip()):
        return provider_stuck(f"model asked to {kind} without an output_name")
    if kind == "extract" and target is None:
        return provider_stuck("model asked to extract without exactly one target element")
    targets: list[Element] = []
    if kind == "extract_many":
        problem = many_targets_problem(data.get("element_indexes"), observation)
        if problem:
            return provider_stuck(f"model asked to extract_many but {problem}")
        targets = [observation.elements[i] for i in data["element_indexes"]]
    outcomes = [{"code": o["code"], "text_contains": o.get("text_contains"), "text_missing": o.get("text_missing")}
                for o in data.get("outcomes") or []]
    stuck_cause = ""
    if data.get("kind") == "stuck":
        # Only an explicit "the control I need is not in the list" or "the value I need is not in the list"
        # may lead to the vision fallback.
        stuck_cause = {"missing_control": "perception", "missing_data": "missing_data"}.get(
            data.get("stuck_cause"), "planner")
    candidate_id = data.get("candidate_id") if data.get("kind") == "reuse_candidate" else None
    output_mode = data.get("output_mode") if kind in ("extract", "extract_many") else None
    return Action(
        kind=data["kind"], target=target, value=data.get("value"), output_name=data.get("output_name"),
        targets=targets, pattern=data.get("pattern"), optional=bool(data.get("optional")), expect=data.get("expect"),
        reason=data.get("reason", ""), outcomes=outcomes, stuck_cause=stuck_cause,
        candidate_id=str(candidate_id) if candidate_id is not None else None,
        output_mode="append" if output_mode == "append" else "set",
    )


def many_targets_problem(indexes, observation: Observation) -> str | None:
    """Why an extract_many index list is unusable, or None: a non-empty, bounded list of distinct valid indexes."""
    if not isinstance(indexes, list) or not indexes:
        return "gave no element_indexes"
    if len(indexes) > MAX_EXTRACT_TARGETS:
        return f"selected {len(indexes)} elements; at most {MAX_EXTRACT_TARGETS} are allowed"
    for index in indexes:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(observation.elements):
            return f"referenced element {index!r}, which does not exist"
    if len(set(indexes)) != len(indexes):
        return "selected the same element more than once"
    return None


def render_observation(observation: Observation) -> str:
    lines = [f"url: {observation.url}"]
    if observation.dialog:
        lines.append(f"A DIALOG IS COVERING THE SCREEN: {observation.dialog}")
    for index, element in enumerate(observation.elements[:MAX_ELEMENTS_IN_PROMPT]):
        detail = f' text="{element.text}"' if element.text and element.text != element.name else ""
        # The enclosing item is shown only when the name alone would be ambiguous.
        where = f' (in: {element.context})' if element.context and is_ambiguous(element, observation) else ""
        # Computed states (checked, expanded, ...) let the model tell a ticked box from an empty one.
        states = f" ({', '.join(f'{k}={v}' for k, v in element.states.items())})" if element.states else ""
        lines.append(f'[{index}] {element.role} "{element.name}"{detail}{states}{where}')
    return "\n".join(lines)


def render_candidates(candidates: list[ReuseCandidate]) -> str:
    """The verified segments the planner may choose instead of a novel action; names only, never values."""
    if not candidates:
        return ""
    lines = ["VERIFIED REUSABLE SEGMENTS (each already proven on this application; choose one with kind "
             "\"reuse_candidate\" and its candidate_id ONLY if it directly advances what remains of the goal):"]
    for candidate in candidates:
        lines.append(f"- candidate_id: {candidate.candidate_id}")
        lines.append(f"  from capability '{candidate.source_name}' revision {candidate.source_version}: "
                     f"{candidate.description}")
        lines.append(f"  does ({candidate.action_count} verified actions): " + "; ".join(candidate.outline))
        lines.append(f"  needs inputs: {', '.join(candidate.required_inputs) or 'none'}; "
                     f"produces outputs: {', '.join(candidate.outputs) or 'none'}")
        lines.append(f"  ends when: {candidate.ending_condition}")
    return "\n".join(lines) + "\n\n"


def render_history(history: list[Action]) -> str:
    if not history:
        return "(no actions yet)"
    lines = []
    for action in history[-HISTORY_IN_PROMPT:]:
        target = f' on {action.target.role} "{action.target.name}"' if action.target else ""
        value = f" value={action.value!r}" if action.value else ""
        lines.append(f"- {action.kind}{target}{value} -> {action.result or 'ok'}")
    return "\n".join(lines)


# ---------- the LLM adapter ----------

SYSTEM_PROMPT = """You are operating a web application on behalf of an automation system.
You see the screen as a numbered list of controls (role and label), not raw HTML. When several
controls share a label, each shows the item it belongs to, e.g. button "Add to cart" (in: Backpack).
Each turn, choose exactly ONE next action by calling the choose_action tool.

Rules:
- Work toward the goal with the fewest actions. Do not explore unrelated screens.
- Do only what the goal asks. Never create accounts, purchase, pay, or complete an order unless
  the goal explicitly requires it. Stop conditions in the goal are binding: if the goal says to
  stop before an action, do not take it; respond with kind "done" instead.
- If a dialog is covering the screen, deal with it first (usually by clicking its button).
- Input parameters are given by name. Whenever you type a parameter value or put it in a
  URL, write the placeholder {{name}} instead of the value; the system substitutes the real
  value. Example: navigate to https://example.com/profile/{{username}}.
  A parameter shown as {{name}} is sensitive; you never see its value, use the placeholder.
- Set `expect` to a short piece of text you expect to see on screen after the action
  succeeds. It becomes the checkpoint that verifies the step during replay, so choose text
  that will be present for any input, not something specific to this run, and that is NOT
  already on the screen before the action: text that was there before proves nothing.
- Use `select` with `element_index` and `value` to choose an entry in a dropdown, combobox or
  autocomplete field: the system types the value if needed, waits for the suggestions, picks the
  matching one and verifies it was accepted. Typing into such a field with `type` does not select
  anything; never type the same value again hoping it will be accepted.
- Use `back` to return to the previous page through the browser history (after reading a
  detail page, for example). Do not navigate to a URL you guessed to get back.
- Use `extract` with `element_index`, `output_name` and a regex `pattern` to read ONE value the
  goal asks for: exactly one control that contains the value; use a capture group for the
  number/text you want. Set `optional` to true when that value may be absent for other inputs
  (e.g. a discount line that not every order shows). One extract per output; when the goal names
  the values to return, use those names (snake_case).
- Use `extract_many` with `element_indexes` (an ordered list of the controls, in the order the
  goal wants them) and `output_name` when ONE output is an ordered list assembled from several
  visible items (the steps of a route, the lines of an address). Select only the requested items,
  each exactly once, in the requested order; the same `pattern` is applied to every item. Do not
  read a whole page into one value, and never say "stuck" to ask for a screenshot when the items
  are already in the list of controls.
- When the goal asks for the first N displayed results as several parallel lists (titles, ids,
  dates, ...), fix WHICH results they are before you leave the list page: apply and verify the
  requested filtering and sorting on screen, then read one whole column with `extract_many` in
  displayed order (the identity the goal names, a title, name or id). Only then open each result
  and append the remaining values. Do not open a result first: lists reorder while you are away,
  and an append-only list cannot be repaired afterwards. After returning, open the next result by
  the identity you recorded, not by its position in the list.
- An output is recorded once: a second extract into the same name is refused. To build ONE list
  from several pages or several reads (one value per detail page), set `output_mode` to "append"
  on every extract or extract_many that adds to that list, in the order the goal wants; each
  append adds its value(s) to the end. Never re-read values you already recorded.
- When the goal is fully achieved, respond with kind "done" with `expect` set to text that
  identifies the final screen for ANY input (a heading or label), never a value you extracted
  or an input-specific detail. In `outcomes`, list the known
  non-success results this flow can produce that a caller should be told about (for example
  "invalid credentials", "item not listed"), each identified either by distinctive on-screen
  text (`text_contains`) or by something that would be missing (`text_missing`, e.g. the
  requested item's name written as its {{placeholder}}). Do not list dialogs you already
  dismissed; those are recorded automatically.
- If you cannot make progress, respond with kind "stuck" and explain why in `reason`. Set
  `stuck_cause` to "missing_control" ONLY when a control you need is clearly visible on the
  page but absent from the list (for example drawn on a canvas or an unlabeled icon); set it to
  "missing_data" with `output_name` ONLY when a value the goal asks for is visibly on screen but
  absent from the list, so the system can look at the screen for it; do not click around to
  reveal a value that is already displayed; use "other" for every other reason.
- Sometimes VERIFIED REUSABLE SEGMENTS are listed: sequences of actions already proven to work
  on this application from the current screen. Choose one with kind "reuse_candidate" and its
  exact `candidate_id` ONLY when it directly advances what remains of the goal. Do not choose a
  segment merely because it applies to this screen, and never one that repeats work already done.
  When a relevant segment exists, prefer it over inventing the same actions yourself. When none
  is relevant, or none is listed, respond with an ordinary action."""

CHOOSE_ACTION_TOOL = {
    "name": "choose_action",
    "description": "Choose the single next action to take on the screen.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "element_index", "element_indexes", "value", "output_name", "pattern", "optional",
                     "output_mode", "expect", "reason", "outcomes", "stuck_cause", "candidate_id"],
        "properties": {
            "kind": {"type": "string", "enum": ["navigate", "back", "click", "type", "select", "extract",
                                                "extract_many", "done", "stuck", "reuse_candidate"]},
            "output_mode": {"type": ["string", "null"], "enum": ["set", "append", None],
                            "description": "For extract / extract_many: set (default) records the output once; "
                                           "append adds the value(s) to a list output, in order."},
            "element_indexes": {"type": ["array", "null"], "items": {"type": "integer"},
                                "description": "For extract_many: the indexes of the controls to read, in the "
                                               "order the list should have; each at most once."},
            "candidate_id": {"type": ["string", "null"],
                             "description": "For reuse_candidate: the candidate_id of one listed verified segment."},
            "stuck_cause": {"type": ["string", "null"], "enum": ["missing_control", "missing_data", "other", None],
                            "description": "For stuck: missing_control when a needed visible control is not in "
                                           "the list; missing_data when a value the goal asks for is visibly on "
                                           "screen but not in the list (name it in output_name); other otherwise."},
            "element_index": {"type": ["integer", "null"], "description": "Index of the control to act on."},
            "value": {"type": ["string", "null"], "description": "Text to type, or URL to navigate to."},
            "output_name": {"type": ["string", "null"], "description": "For extract: name of the output."},
            "pattern": {"type": ["string", "null"],
                        "description": "For extract: regex applied to the control's text; group 1 or whole match."},
            "optional": {"type": "boolean", "description": "For extract: true if the value may be absent for other inputs."},
            "expect": {"type": ["string", "null"], "description": "Text expected on screen after the action."},
            "reason": {"type": "string", "description": "One sentence explaining the choice."},
            "outcomes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["code", "text_contains", "text_missing"],
                    "properties": {"code": {"type": "string"},
                                   "text_contains": {"type": ["string", "null"],
                                                     "description": "On-screen text that identifies this outcome."},
                                   "text_missing": {"type": ["string", "null"],
                                                    "description": "Text whose absence identifies this outcome."}},
                },
            },
        },
    },
}


class ClaudePlanner:
    """LLM-backed discovery planner using the Anthropic Messages API with one tool."""

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL):
        import anthropic  # imported lazily so offline runs never need the SDK

        self.client = anthropic.Anthropic()
        self.model = model
        self.vision_model = DEFAULT_ANTHROPIC_VISION_MODEL or model
        self.name = f"claude:{model}"

    def decide(self, goal: str, params: dict, observation: Observation, history: list[Action],
               candidates: list[ReuseCandidate] = ()) -> Action:
        prompt = self.render_prompt(goal, params, observation, history, candidates)
        response = call_provider("anthropic", lambda: self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            tools=[CHOOSE_ACTION_TOOL],
            messages=[{"role": "user", "content": prompt}],
        ), retry_records(self))
        if response.stop_reason == "refusal":
            return provider_stuck("the model declined to act on this screen")
        tool_use = next((block for block in response.content if block.type == "tool_use"), None)
        if tool_use is None:
            text = " ".join(block.text for block in response.content if block.type == "text")
            return provider_stuck(f"model returned no action: {text[:200]}")
        return action_from_tool_input(dict(tool_use.input), observation)

    def decide_visually(self, goal: str, params: dict, observation: Observation, history: list[Action],
                        frame: ScreenshotFrame, remaining_attempts: int) -> VisualDecision:
        """One visual proposal from a masked viewport screenshot (Anthropic image block + one tool)."""
        prompt = render_visual_prompt(goal, params, observation, history, frame, remaining_attempts)
        try:
            response = call_provider("anthropic", lambda: self.client.messages.create(
                model=self.vision_model, max_tokens=2000, system=VISION_SYSTEM_PROMPT,
                tools=[CHOOSE_VISUAL_ACTION_TOOL], messages=anthropic_visual_message(prompt, frame)), retry_records(self))
        except Exception as error:   # a model without image support, or any API failure: vision is unavailable
            return VisualDecision(kind="unavailable", reason=unavailable_reason(error))
        if getattr(response, "stop_reason", None) == "refusal":
            return VisualDecision(kind="no_target", reason="the model declined to act on this screenshot")
        tool_use = next((block for block in response.content if block.type == "tool_use"), None)
        if tool_use is None:
            return VisualDecision(kind="no_target", reason="model returned no visual action")
        return visual_decision_from_tool_input(dict(tool_use.input), frame.width, frame.height)

    @staticmethod
    def render_prompt(goal: str, params: dict, observation: Observation, history: list[Action],
                      candidates: list[ReuseCandidate] = ()) -> str:
        return (
            f"GOAL: {goal}\n\n"
            f"INPUT PARAMETERS:\n{json.dumps(params, indent=2)}\n\n"
            f"ACTIONS SO FAR:\n{render_history(history)}\n\n"
            f"{render_candidates(list(candidates))}"
            f"CURRENT SCREEN:\n{render_observation(observation)}\n\n"
            "Call choose_action with the single next action."
        )


OPENAI_CHOOSE_ACTION_TOOL = {
    "type": "function",
    "name": CHOOSE_ACTION_TOOL["name"],
    "description": CHOOSE_ACTION_TOOL["description"],
    "parameters": CHOOSE_ACTION_TOOL["input_schema"],
    "strict": True,
}


class OpenAIPlanner:
    """LLM-backed discovery planner using the OpenAI Responses API."""

    def __init__(self, model: str = DEFAULT_OPENAI_MODEL, client=None):
        if client is None:
            from openai import OpenAI  # lazy: replay and offline tests never need the SDK

            client = OpenAI()
        self.client = client
        self.model = model
        self.vision_model = DEFAULT_OPENAI_VISION_MODEL or model
        self.name = f"openai:{model}"

    def decide(self, goal: str, params: dict, observation: Observation, history: list[Action],
               candidates: list[ReuseCandidate] = ()) -> Action:
        prompt = ClaudePlanner.render_prompt(goal, params, observation, history, candidates)
        response = call_provider("openai", lambda: self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            tools=[OPENAI_CHOOSE_ACTION_TOOL],
            tool_choice={"type": "function", "name": "choose_action"},
            parallel_tool_calls=False,
        ), retry_records(self))
        tool_call = next(
            (item for item in response.output
             if item.type == "function_call" and item.name == "choose_action"),
            None,
        )
        if tool_call is None:
            text = getattr(response, "output_text", "")
            return provider_stuck(f"model returned no action: {text[:200]}")
        try:
            data = json.loads(tool_call.arguments)
        except (TypeError, json.JSONDecodeError) as error:
            return provider_stuck(f"model returned invalid action JSON: {error}")
        return action_from_tool_input(data, observation)

    def decide_visually(self, goal: str, params: dict, observation: Observation, history: list[Action],
                        frame: ScreenshotFrame, remaining_attempts: int) -> VisualDecision:
        """One visual proposal from a masked viewport screenshot (OpenAI input_image + one function tool)."""
        prompt = render_visual_prompt(goal, params, observation, history, frame, remaining_attempts)
        try:
            response = call_provider("openai", lambda: self.client.responses.create(
                model=self.vision_model, instructions=VISION_SYSTEM_PROMPT,
                input=openai_visual_input(prompt, frame), tools=[OPENAI_CHOOSE_VISUAL_ACTION_TOOL],
                tool_choice={"type": "function", "name": "choose_visual_action"}, parallel_tool_calls=False),
                retry_records(self))
        except Exception as error:
            return VisualDecision(kind="unavailable", reason=unavailable_reason(error))
        tool_call = next((item for item in response.output
                          if item.type == "function_call" and item.name == "choose_visual_action"), None)
        if tool_call is None:
            return VisualDecision(kind="no_target", reason="model returned no visual action")
        try:
            data = json.loads(tool_call.arguments)
        except (TypeError, json.JSONDecodeError) as error:
            return VisualDecision(kind="no_target", rejected=f"model returned invalid visual action JSON: {error}")
        return visual_decision_from_tool_input(data, frame.width, frame.height)


# ---------- the vision fallback (discovery only) ----------

VISION_SYSTEM_PROMPT = """You are helping an automation system that could not find a control it needs.
You see the structured list of controls the system perceives AND a screenshot of the current viewport.
The screenshot is the truth about what is drawn; the list is what the system can already target. Sensitive
values in the screenshot are masked as bullets.

Propose exactly ONE visual action by calling choose_visual_action:
- visual_click: click a control that is visible in the screenshot but missing from the list (drawn on a
  canvas, an icon without a label, a purely visual widget). Give its bounding box in screenshot pixels.
- visual_type: click such a control and type a value; write parameter values as {{name}} placeholders.
- visual_extract: ONLY when asked for a missing value: give `output_name` and one box per value in `boxes`,
  in the order the goal wants them, each tightly around the text as drawn. The system reads the actual page
  text under each box itself and parses it by the declared contract; whatever you read is never used as the
  value, so never guess or type values.
- no_target: you cannot identify a control that would make progress toward the goal.
Set `expect` to short text that will appear on screen after the action succeeds; the system only keeps an
action it can verify that way. Never propose extracting a value, finishing, navigating elsewhere, or anything
outside the goal. Give a confidence between 0 and 1."""

CHOOSE_VISUAL_ACTION_TOOL = {
    "name": "choose_visual_action",
    "description": "Propose the single visual action that would make progress, or no_target.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "role", "name", "x", "y", "width", "height", "value", "expect", "reason", "confidence",
                     "output_name", "boxes"],
        "properties": {
            "kind": {"type": "string", "enum": list(VISUAL_KINDS)},
            "output_name": {"type": ["string", "null"], "description": "For visual_extract: the declared output."},
            "boxes": {"type": "array",
                      "description": "For visual_extract: one box per value in output order; empty otherwise.",
                      "items": {"type": "object", "additionalProperties": False,
                                "required": ["x", "y", "width", "height", "reason", "confidence"],
                                "properties": {"x": {"type": "number"}, "y": {"type": "number"},
                                               "width": {"type": "number"}, "height": {"type": "number"},
                                               "reason": {"type": "string", "description": "What is drawn there."},
                                               "confidence": {"type": "number", "description": "0 to 1."}}}},
            "role": {"type": "string", "description": "Best guess at the control's role: button, link, textbox, ..."},
            "name": {"type": "string", "description": "Short human description of the control."},
            "x": {"type": "number", "description": "Left edge of the control in screenshot pixels."},
            "y": {"type": "number", "description": "Top edge of the control in screenshot pixels."},
            "width": {"type": "number"},
            "height": {"type": "number"},
            "value": {"type": ["string", "null"],
                      "description": "For visual_type: text to type ({{name}} placeholders)."},
            "expect": {"type": ["string", "null"], "description": "Text expected on screen after the action."},
            "reason": {"type": "string"},
            "confidence": {"type": "number", "description": "0 to 1."},
        },
    },
}

OPENAI_CHOOSE_VISUAL_ACTION_TOOL = {
    "type": "function",
    "name": CHOOSE_VISUAL_ACTION_TOOL["name"],
    "description": CHOOSE_VISUAL_ACTION_TOOL["description"],
    "parameters": CHOOSE_VISUAL_ACTION_TOOL["input_schema"],
    "strict": True,
}


def render_visual_prompt(goal: str, params: dict, observation: Observation, history: list[Action],
                         frame: ScreenshotFrame, remaining_attempts: int) -> str:
    return (
        f"GOAL: {goal}\n\n"
        f"INPUT PARAMETERS:\n{json.dumps(params, indent=2)}\n\n"
        f"ACTIONS SO FAR:\n{render_history(history)}\n\n"
        f"CONTROLS THE SYSTEM CAN ALREADY TARGET:\n{render_observation(observation)}\n\n"
        f"The attached screenshot shows the viewport, {frame.width} x {frame.height} pixels; coordinates you "
        f"return are relative to it. Visual attempts remaining after this one: {remaining_attempts}.\n"
        "Call choose_visual_action once."
    )


def visual_decision_from_tool_input(data: dict, width: int, height: int) -> VisualDecision:
    """Validate the model's visual proposal against the screenshot it saw; refuse anything unsafe to act on."""
    kind = data.get("kind")
    if kind not in VISUAL_KINDS:
        return VisualDecision(kind="no_target", rejected=f"unknown kind {kind!r}")
    if kind == "no_target":
        return VisualDecision(kind="no_target", reason=str(data.get("reason", "")))
    if kind == "visual_extract":
        return visual_extraction_from_tool_input(data, width, height)
    numbers = {}
    for key in ("x", "y", "width", "height", "confidence"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return VisualDecision(kind="no_target", rejected=f"{key} is missing or not a finite number")
        numbers[key] = value
    x, y, w, h = (int(round(numbers[k])) for k in ("x", "y", "width", "height"))
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        return VisualDecision(kind="no_target", rejected=f"target box ({x}, {y}, {w}, {h}) is not inside the "
                                                          f"{width} x {height} viewport")
    if numbers["confidence"] < MIN_VISION_CONFIDENCE:
        return VisualDecision(kind="no_target", rejected=f"confidence {numbers['confidence']:.2f} is below "
                                                          f"{MIN_VISION_CONFIDENCE}")
    expect = data.get("expect")
    if not isinstance(expect, str) or not expect.strip():
        return VisualDecision(kind="no_target",
                              rejected="no expected post-action text: the action could not be verified")
    value = data.get("value")
    if kind == "visual_type" and (not isinstance(value, str) or not value):
        return VisualDecision(kind="no_target", rejected="visual_type needs a value")
    target = VisualTarget(role=str(data.get("role") or "unknown"), name=str(data.get("name") or ""), x=x, y=y,
                          width=w, height=h, expect=expect.strip(), reason=str(data.get("reason", "")),
                          confidence=float(numbers["confidence"]))
    return VisualDecision(kind=kind, target=target, value=value if kind == "visual_type" else None,
                          reason=target.reason)


def visual_box(raw, width: int, height: int, where: str) -> VisualTarget | str:
    """One screenshot box: finite numbers, inside the viewport, confident enough; else the reason it is not."""
    if not isinstance(raw, dict):
        return f"{where} is not an object"
    numbers = {}
    for key in ("x", "y", "width", "height", "confidence"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return f"{where}: {key} is missing or not a finite number"
        numbers[key] = value
    x, y, w, h = (int(round(numbers[k])) for k in ("x", "y", "width", "height"))
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        return f"{where}: box ({x}, {y}, {w}, {h}) is not inside the {width} x {height} viewport"
    if numbers["confidence"] < MIN_VISION_CONFIDENCE:
        return f"{where}: confidence {numbers['confidence']:.2f} is below {MIN_VISION_CONFIDENCE}"
    return VisualTarget(role="text", name="", x=x, y=y, width=w, height=h, expect=None,
                        reason=str(raw.get("reason", "")), confidence=float(numbers["confidence"]))


def visual_extraction_from_tool_input(data: dict, width: int, height: int) -> VisualDecision:
    """A visual_extract proposal: a declared output name and an ordered, bounded list of valid boxes. The
    model's own reading of the text, if it sent one, is dropped here and never travels further."""
    name = data.get("output_name")
    if not isinstance(name, str) or not name.strip():
        return VisualDecision(kind="no_target", rejected="visual_extract needs an output_name")
    raw_boxes = data.get("boxes")
    if not isinstance(raw_boxes, list) or not raw_boxes:
        return VisualDecision(kind="no_target", rejected="visual_extract needs at least one box")
    if len(raw_boxes) > MAX_VISUAL_BOXES:
        return VisualDecision(kind="no_target", rejected=f"{len(raw_boxes)} boxes exceed the limit of {MAX_VISUAL_BOXES}")
    boxes: list[VisualTarget] = []
    for index, raw in enumerate(raw_boxes):
        box = visual_box(raw, width, height, f"box {index}")
        if isinstance(box, str):
            return VisualDecision(kind="no_target", rejected=box)
        boxes.append(box)
    return VisualDecision(kind="visual_extract", output_name=name.strip(), boxes=boxes, reason=str(data.get("reason", "")))


def anthropic_visual_message(prompt: str, frame: ScreenshotFrame) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": base64.b64encode(frame.png).decode("ascii")}},
        {"type": "text", "text": prompt},
    ]}]


def openai_visual_input(prompt: str, frame: ScreenshotFrame) -> list[dict]:
    data_url = "data:image/png;base64," + base64.b64encode(frame.png).decode("ascii")
    return [{"role": "user", "content": [{"type": "input_text", "text": prompt},
                                         {"type": "input_image", "image_url": data_url}]}]
