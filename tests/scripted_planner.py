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
    texts: list[str] = field(default_factory=list)   # kind "extract_many": one text prefix per item, in order
    value: str | None = None
    output_name: str | None = None
    pattern: str | None = None
    optional: bool = False
    output_mode: str = "set"      # kind "extract"/"extract_many": set | append
    expect: str | None = None
    outcomes: list[dict] = field(default_factory=list)
    skip_if_absent: bool = False  # skip silently if the control (or candidate) is absent (e.g. a notice)
    stuck_cause: str = "other"    # for "stuck": "missing_control" is the only cause that may lead to vision
    # kind "reuse": pick the offered verified segment from capability `name`; `value` forces an exact id instead


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
        self.chosen: set[str] = set()     # candidate ids already selected: a repeated "reuse" walks an artifact onward

    def decide(self, goal: str, params: dict, observation: Observation, history: list[Action],
               candidates=()) -> Action:
        while self.script:
            step = self.script[0]
            if step.kind == "reuse":
                chosen = step.value or next((c.candidate_id for c in candidates
                                             if c.source_name == step.name and c.candidate_id not in self.chosen), None)
                if chosen is None and step.skip_if_absent:
                    self.script.pop(0)
                    continue
                if chosen is None:
                    return Action(kind="stuck", reason=f"scripted step expected a segment from {step.name!r}",
                                  stuck_cause="planner")
                self.script.pop(0)
                self.chosen.add(chosen)
                return Action(kind="reuse_candidate", candidate_id=chosen, reason="scripted")
            if step.kind in ("done", "stuck", "navigate", "back"):
                self.script.pop(0)
                cause = ""
                if step.kind == "stuck":
                    cause = {"missing_control": "perception", "missing_data": "missing_data"}.get(step.stuck_cause, "planner")
                return Action(kind=step.kind, value=step.value, expect=step.expect, outcomes=step.outcomes,
                              reason="scripted", stuck_cause=cause, output_name=step.output_name)
            if step.kind == "extract_many":
                self.script.pop(0)
                targets = [find_element(observation, ScriptedStep("extract", step.role, text=text), params)
                           for text in step.texts]
                return Action(kind="extract_many", targets=[t for t in targets if t is not None],
                              output_name=step.output_name, pattern=step.pattern, optional=step.optional,
                              output_mode=step.output_mode, expect=step.expect, reason="scripted")
            element = find_element(observation, step, params)
            if element is None and step.skip_if_absent:
                self.script.pop(0)
                continue
            if element is None:
                return Action(kind="stuck", reason=f'scripted step expected {step.role} "{step.name or step.text}"',
                              stuck_cause="planner")
            self.script.pop(0)
            return Action(kind=step.kind, target=element, value=step.value, output_name=step.output_name,
                          pattern=step.pattern, optional=step.optional, output_mode=step.output_mode,
                          expect=step.expect, reason="scripted")
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


def visual_extract(output_name: str, boxes: list[tuple], confidence: float = 0.9, readings=None) -> VisualDecision:
    """A scripted visual extraction: boxes in output order. `readings` mimics a model that also reports what it
    read; the runtime must never use them (they are not part of the decision)."""
    targets = [VisualTarget(role="text", name=str(reading) if reading is not None else "", x=x, y=y, width=w, height=h,
                            expect=None, reason="scripted", confidence=confidence)
               for (x, y, w, h), reading in zip(boxes, readings or [None] * len(boxes))]
    return VisualDecision(kind="visual_extract", output_name=output_name, boxes=targets, reason="scripted")


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
