"""A scripted Planner: the test double for the LLM at the Planner protocol boundary.

It follows a fixed decision list and binds each decision to a real element from the
observation at decision time, exactly as the model's tool call is bound. It never runs
in production and replay never sees it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.cua.artifact import substitute
from src.cua.models import ScreenshotFrame, VisualDecision, VisualTarget

from .context import Action, Element, Observation


@dataclass
class ScriptedStep:
    kind: str
    role: str = ""
    name: str = ""            # accessible name to match (may contain {{placeholders}})
    context: str = ""         # enclosing item to match, for ambiguous names
    text: str = ""            # text prefix to match, for unnamed text elements
    value: str | None = None
    output_name: str | None = None
    pattern: str | None = None
    optional: bool = False
    expect: str | None = None
    outcomes: list[dict] = field(default_factory=list)
    skip_if_absent: bool = False  # skip silently if the control is not on screen (e.g. a notice)
    stuck_cause: str = "other"    # for "stuck": "missing_control" is the only cause that may lead to vision


def find_element(observation: Observation, step: ScriptedStep, params: dict) -> Element | None:
    name, context, text = (substitute(step.name, params), substitute(step.context, params), substitute(step.text, params))
    for element in observation.elements:
        if element.role != step.role:
            continue
        if name and not element.name.lower().startswith(name.lower()):
            continue
        if context and element.context != context:
            continue
        if text and not element.text.startswith(text):
            continue
        return element
    return None


class ScriptedPlanner:
    name = "scripted"

    def __init__(self, script: list[ScriptedStep]):
        self.script = list(script)

    def decide(self, goal: str, params: dict, observation: Observation, history: list[Action]) -> Action:
        while self.script:
            step = self.script[0]
            if step.kind in ("done", "stuck", "navigate"):
                self.script.pop(0)
                cause = ""
                if step.kind == "stuck":
                    cause = "perception" if step.stuck_cause == "missing_control" else "planner"
                return Action(kind=step.kind, value=step.value, expect=step.expect, outcomes=step.outcomes,
                              reason="scripted", stuck_cause=cause)
            element = find_element(observation, step, params)
            if element is None and step.skip_if_absent:
                self.script.pop(0)
                continue
            if element is None:
                return Action(kind="stuck", reason=f'scripted step expected {step.role} "{step.name or step.text}"',
                              stuck_cause="planner")
            self.script.pop(0)
            return Action(kind=step.kind, target=element, value=step.value, output_name=step.output_name,
                          pattern=step.pattern, optional=step.optional, expect=step.expect, reason="scripted")
        return Action(kind="stuck", reason="script exhausted before the goal was reached")


MONEY = r"\$\s*([\d.]+)"


def checkout_script() -> list[ScriptedStep]:
    """The decisions a model makes for the shop's checkout-review goal."""
    return [
        ScriptedStep("click", "button", "Accept", expect="Swag Labs", skip_if_absent=True),
        ScriptedStep("type", "textbox", "Username", value="{{username}}"),
        ScriptedStep("type", "textbox", "Password", value="{{password}}"),
        ScriptedStep("click", "button", "Login", expect="Products"),
        ScriptedStep("click", "button", "Add to cart", context="{{product_name}}", expect="Remove"),
        ScriptedStep("click", "link", "cart", expect="Your Cart"),
        ScriptedStep("click", "button", "Checkout", expect="Checkout: Your Information"),
        ScriptedStep("type", "textbox", "First Name", value="{{first_name}}"),
        ScriptedStep("type", "textbox", "Last Name", value="{{last_name}}"),
        ScriptedStep("type", "textbox", "Zip/Postal Code", value="{{postal_code}}"),
        ScriptedStep("click", "button", "Continue", expect="Checkout: Overview"),
        ScriptedStep("extract", "link", "View details for {{product_name}}", output_name="product_name", pattern=r"(.+)"),
        ScriptedStep("extract", "text", text="Item total:", output_name="subtotal", pattern=MONEY),
        ScriptedStep("extract", "text", text="Tax:", output_name="tax", pattern=MONEY),
        ScriptedStep("extract", "text", text="Total:", output_name="total", pattern=MONEY),
        ScriptedStep("done", expect="Checkout: Overview", outcomes=[
            {"code": "user_locked_out", "text_contains": "locked out"},
            {"code": "invalid_credentials", "text_contains": "do not match"},
            {"code": "product_not_found", "text_missing": "{{product_name}}"},
            {"code": "checkout_info_missing", "text_contains": "is required"},
        ]),
    ]


def visual_click(name: str, expect: str, box=(100, 100, 80, 30), role: str = "button", confidence: float = 0.9,
                 value: str | None = None) -> VisualDecision:
    x, y, w, h = box
    kind = "visual_type" if value is not None else "visual_click"
    return VisualDecision(kind=kind, value=value, reason="scripted",
                          target=VisualTarget(role=role, name=name, x=x, y=y, width=w, height=h, expect=expect,
                                              reason="scripted", confidence=confidence))


NO_TARGET = VisualDecision(kind="no_target", reason="scripted: nothing to click")


class ScriptedVisionPlanner(ScriptedPlanner):
    """The structured script plus a fixed list of visual decisions, one per fallback call; records every frame."""

    name = "scripted+vision"

    def __init__(self, script: list[ScriptedStep], decisions: list[VisualDecision]):
        super().__init__(script)
        self.decisions = list(decisions)
        self.frames: list[ScreenshotFrame] = []
        self.calls: list[dict] = []

    def decide_visually(self, goal, params, observation, history, frame, remaining_attempts) -> VisualDecision:
        self.frames.append(frame)
        self.calls.append({"params": dict(params), "remaining": remaining_attempts, "width": frame.width,
                           "height": frame.height, "elements": len(observation.elements)})
        if not self.decisions:
            return NO_TARGET
        return self.decisions.pop(0)
