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
from typing import Protocol

from .models import Action, Element, Observation, ScreenshotFrame, VisualDecision, VisualTarget
from .surface import is_ambiguous

DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-astra")
# Optional: a different model for the vision fallback; the configured model is used otherwise.
DEFAULT_ANTHROPIC_VISION_MODEL = os.environ.get("ANTHROPIC_VISION_MODEL") or None
DEFAULT_OPENAI_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL") or None
MAX_ELEMENTS_IN_PROMPT = 200
HISTORY_IN_PROMPT = 8
MIN_VISION_CONFIDENCE = 0.6
VISUAL_KINDS = ("visual_click", "visual_type", "no_target")


class Planner(Protocol):
    """Chooses one next action from the current observation."""

    name: str

    def decide(
        self,
        goal: str,
        params: dict,
        observation: Observation,
        history: list[Action],
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


STUCK_CAUSES = ("perception", "planner", "provider")


def provider_stuck(reason: str) -> Action:
    """A stuck state the provider caused (refusal, malformed output): never a reason to look at pixels."""
    return Action(kind="stuck", reason=reason, stuck_cause="provider")


def action_from_tool_input(data: dict, observation: Observation) -> Action:
    """Turn the model's tool call into an Action bound to a real observed element."""
    target = None
    index = data.get("element_index")
    if index is not None:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(observation.elements):
            return provider_stuck(f"model referenced element {index!r}, which does not exist")
        target = observation.elements[index]
    outcomes = [{"code": o["code"], "text_contains": o.get("text_contains"), "text_missing": o.get("text_missing")}
                for o in data.get("outcomes") or []]
    stuck_cause = ""
    if data.get("kind") == "stuck":
        # Only an explicit "the control I need is not in the list" may lead to the vision fallback.
        stuck_cause = "perception" if data.get("stuck_cause") == "missing_control" else "planner"
    return Action(
        kind=data["kind"], target=target, value=data.get("value"), output_name=data.get("output_name"),
        pattern=data.get("pattern"), optional=bool(data.get("optional")), expect=data.get("expect"),
        reason=data.get("reason", ""), outcomes=outcomes, stuck_cause=stuck_cause,
    )


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
  that will be present for any input, not something specific to this run.
- Use `extract` with `output_name` and a regex `pattern` to read a value the goal asks for.
  Pick the control that contains the value; use a capture group for the number/text you want.
  Set `optional` to true when that value may be absent for other inputs (e.g. a discount line
  that not every order shows). One extract per output; when the goal names
  the values to return, use those names (snake_case).
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
  page but absent from the list (for example drawn on a canvas or an unlabeled icon); use
  "other" for every other reason."""

CHOOSE_ACTION_TOOL = {
    "name": "choose_action",
    "description": "Choose the single next action to take on the screen.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "element_index", "value", "output_name", "pattern", "optional", "expect", "reason",
                     "outcomes", "stuck_cause"],
        "properties": {
            "kind": {"type": "string", "enum": ["navigate", "click", "type", "extract", "done", "stuck"]},
            "stuck_cause": {"type": ["string", "null"], "enum": ["missing_control", "other", None],
                            "description": "For stuck: missing_control when a needed visible control is not in "
                                           "the list; other otherwise."},
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

    def decide(self, goal: str, params: dict, observation: Observation, history: list[Action]) -> Action:
        prompt = self.render_prompt(goal, params, observation, history)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            tools=[CHOOSE_ACTION_TOOL],
            messages=[{"role": "user", "content": prompt}],
        )
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
            response = self.client.messages.create(
                model=self.vision_model, max_tokens=2000, system=VISION_SYSTEM_PROMPT,
                tools=[CHOOSE_VISUAL_ACTION_TOOL], messages=anthropic_visual_message(prompt, frame))
        except Exception as error:   # a model without image support, or any API failure: vision is unavailable
            return VisualDecision(kind="unavailable", reason=f"{type(error).__name__}: {str(error)[:200]}")
        if getattr(response, "stop_reason", None) == "refusal":
            return VisualDecision(kind="no_target", reason="the model declined to act on this screenshot")
        tool_use = next((block for block in response.content if block.type == "tool_use"), None)
        if tool_use is None:
            return VisualDecision(kind="no_target", reason="model returned no visual action")
        return visual_decision_from_tool_input(dict(tool_use.input), frame.width, frame.height)

    @staticmethod
    def render_prompt(goal: str, params: dict, observation: Observation, history: list[Action]) -> str:
        return (
            f"GOAL: {goal}\n\n"
            f"INPUT PARAMETERS:\n{json.dumps(params, indent=2)}\n\n"
            f"ACTIONS SO FAR:\n{render_history(history)}\n\n"
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

    def decide(self, goal: str, params: dict, observation: Observation, history: list[Action]) -> Action:
        prompt = ClaudePlanner.render_prompt(goal, params, observation, history)
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            tools=[OPENAI_CHOOSE_ACTION_TOOL],
            tool_choice={"type": "function", "name": "choose_action"},
            parallel_tool_calls=False,
        )
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
            response = self.client.responses.create(
                model=self.vision_model, instructions=VISION_SYSTEM_PROMPT,
                input=openai_visual_input(prompt, frame), tools=[OPENAI_CHOOSE_VISUAL_ACTION_TOOL],
                tool_choice={"type": "function", "name": "choose_visual_action"}, parallel_tool_calls=False)
        except Exception as error:
            return VisualDecision(kind="unavailable", reason=f"{type(error).__name__}: {str(error)[:200]}")
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
        "required": ["kind", "role", "name", "x", "y", "width", "height", "value", "expect", "reason", "confidence"],
        "properties": {
            "kind": {"type": "string", "enum": list(VISUAL_KINDS)},
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
