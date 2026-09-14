"""Planner interface and planner adapters.

The planner is used only during discovery. Deterministic replay must not use it.

ClaudePlanner and OpenAIPlanner are interchangeable LLM adapters: one API call per
decision, one action per call. They are stateless across calls; the agent passes a
compact history instead of the raw transcript, which keeps the recorded artifact
independent of the provider. Tests use a scripted stand-in that lives under tests/
and implements the same protocol.
"""
from __future__ import annotations

import json
import os
from typing import Protocol

from .models import Action, Element, Observation
from .surface import is_ambiguous

DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-astra")
MAX_ELEMENTS_IN_PROMPT = 200
HISTORY_IN_PROMPT = 8


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


# ---------- shared helpers ----------

def find_element(observation: Observation, role: str, name: str) -> Element | None:
    """Locate a control by role and accessible name (case-insensitive, name may be a prefix)."""
    for element in observation.elements:
        if element.role == role and element.name.lower().startswith(name.lower()):
            return element
    return None


def action_from_tool_input(data: dict, observation: Observation) -> Action:
    """Turn the model's tool call into an Action bound to a real observed element."""
    target = None
    index = data.get("element_index")
    if index is not None:
        if not 0 <= index < len(observation.elements):
            return Action(kind="stuck", reason=f"model referenced element {index}, which does not exist")
        target = observation.elements[index]
    outcomes = [{"code": o["code"], "text_contains": o.get("text_contains"), "text_missing": o.get("text_missing")}
                for o in data.get("outcomes") or []]
    return Action(
        kind=data["kind"], target=target, value=data.get("value"), output_name=data.get("output_name"),
        pattern=data.get("pattern"), optional=bool(data.get("optional")), expect=data.get("expect"),
        reason=data.get("reason", ""), outcomes=outcomes,
    )


def render_observation(observation: Observation) -> str:
    lines = [f"url: {observation.url}"]
    if observation.dialog:
        lines.append(f"A DIALOG IS COVERING THE SCREEN: {observation.dialog}")
    for index, element in enumerate(observation.elements[:MAX_ELEMENTS_IN_PROMPT]):
        detail = f' text="{element.text}"' if element.text and element.text != element.name else ""
        # The enclosing item is shown only when the name alone would be ambiguous.
        where = f' (in: {element.context})' if element.context and is_ambiguous(element, observation) else ""
        lines.append(f'[{index}] {element.role} "{element.name}"{detail}{where}')
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
- If you cannot make progress, respond with kind "stuck" and explain why in `reason`."""

CHOOSE_ACTION_TOOL = {
    "name": "choose_action",
    "description": "Choose the single next action to take on the screen.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "element_index", "value", "output_name", "pattern", "optional", "expect", "reason",
                     "outcomes"],
        "properties": {
            "kind": {"type": "string", "enum": ["navigate", "click", "type", "extract", "done", "stuck"]},
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
            return Action(kind="stuck", reason="the model declined to act on this screen")
        tool_use = next((block for block in response.content if block.type == "tool_use"), None)
        if tool_use is None:
            text = " ".join(block.text for block in response.content if block.type == "text")
            return Action(kind="stuck", reason=f"model returned no action: {text[:200]}")
        return action_from_tool_input(dict(tool_use.input), observation)

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
            return Action(kind="stuck", reason=f"model returned no action: {text[:200]}")
        try:
            data = json.loads(tool_call.arguments)
        except (TypeError, json.JSONDecodeError) as error:
            return Action(kind="stuck", reason=f"model returned invalid action JSON: {error}")
        return action_from_tool_input(data, observation)
