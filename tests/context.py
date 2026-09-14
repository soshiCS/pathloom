"""Shared test wiring: imports from the package under test plus small builders."""
from __future__ import annotations

from src.cua.artifact import build
from src.cua.escalation import Escalator, SessionControl
from src.cua.evidence import RunLog
from src.cua.models import (Action, Artifact, Element, GraphAction, GraphEdge, GraphNode, Guard,
                            InterventionRequest, InterventionResult, Locator, Observation, Step, TransientError)
from src.cua.policy import Policy

__all__ = ["Action", "Artifact", "Element", "GraphAction", "GraphEdge", "GraphNode", "Guard", "InterventionRequest",
           "InterventionResult", "Locator", "Observation", "Step", "TransientError", "Policy", "Escalator",
           "SessionControl", "RunLog", "build", "checkout_artifact", "ladder", "text_ladder", "RecordingOperator",
           "action_node", "decision_node", "terminal_node", "edge", "HOSTS", "ENTRY", "PARAMS"]

ENTRY = "https://www.saucedemo.com/"
HOSTS = ["www.saucedemo.com"]
PARAMS = {"username": "standard_user", "password": "secret_sauce", "product_name": "Sauce Labs Backpack",
          "first_name": "Test", "last_name": "User", "postal_code": "10001"}


def ladder(role: str, name: str, context: str | None = None) -> Locator:
    rung = {"kind": "role", "role": role, "name": name}
    if context:
        rung["context"] = context
    return Locator(strategies=[rung, {"kind": "css", "selector": f"{role}:{name}:{context or ''}"}])


def text_ladder(prefix: str) -> Locator:
    return Locator(strategies=[{"kind": "text", "text": prefix}])


def type_step(step_id: str, field: str, param: str) -> Step:
    return Step(id=step_id, action="type", target=ladder("textbox", field), value="{{" + param + "}}")


def click_step(step_id: str, name: str, expect: str, context: str | None = None, risk: str = "safe") -> Step:
    return Step(id=step_id, action="click", target=ladder("button", name, context),
                checkpoint={"text_contains": expect}, risk=risk)


def checkout_artifact(extra_step: Step | None = None) -> Artifact:
    """The artifact discovery produces for the checkout-review capability, built by hand for tests."""
    steps = [
        Step(id="s1", action="navigate", target=None, value=ENTRY, checkpoint={"text_contains": "Swag Labs"}),
        type_step("s2", "Username", "username"),
        type_step("s3", "Password", "password"),
        click_step("s4", "Login", "Products"),
        click_step("s5", "Add to cart", "Remove", context="{{product_name}}"),
        Step(id="s6", action="click", target=ladder("link", "cart"), checkpoint={"text_contains": "{{product_name}}"}),
        click_step("s7", "Checkout", "Checkout: Your Information"),
        type_step("s8", "First Name", "first_name"),
        type_step("s9", "Last Name", "last_name"),
        type_step("s10", "Zip/Postal Code", "postal_code"),
        click_step("s11", "Continue", "Checkout: Overview"),
        Step(id="s12", action="extract", target=ladder("link", "View details for {{product_name}}"), value="product_name"),
        Step(id="s13", action="extract", target=text_ladder("Item total:"), value="subtotal"),
        Step(id="s14", action="extract", target=text_ladder("Tax:"), value="tax"),
        Step(id="s15", action="extract", target=text_ladder("Total:"), value="total"),
    ]
    if extra_step is not None:
        steps.append(extra_step)
    money = r"\$\s*([\d.]+)"
    outputs = {
        "product_name": {"type": "string", "required": True, "pattern": r"(.+)"},
        "subtotal": {"type": "number", "required": True, "pattern": money},
        "tax": {"type": "number", "required": True, "pattern": money},
        "total": {"type": "number", "required": True, "pattern": money},
    }
    return build(
        name="checkout_review", goal="add a product and read the checkout overview",
        surface_meta={"kind": "web", "app": "Swag Labs", "entry_url": ENTRY, "allowed_hosts": HOSTS},
        params=PARAMS, sensitive={"password"}, steps=steps, outputs=outputs,
        outcomes=[
            {"code": "dismiss_cookie_notice", "kind": "recoverable", "source": "observed",
             "detect": {"dialog_contains": "We use cookies"},
             "recover": {"action": "click", "target": {"strategies": ladder("button", "Accept").strategies}}},
            {"code": "user_locked_out", "kind": "business", "source": "planner",
             "detect": {"text_contains": "locked out"}},
            {"code": "invalid_credentials", "kind": "business", "source": "planner",
             "detect": {"text_contains": "do not match"}},
            {"code": "product_not_found", "kind": "business", "source": "planner",
             "detect": {"text_missing": "{{product_name}}"}},
            {"code": "checkout_info_missing", "kind": "business", "source": "planner",
             "detect": {"text_contains": "is required"}},
        ],
        success={"text_contains": "Checkout: Overview"}, run_id="test-run",
    )


# ---------- capability graph builders ----------

def action_node(node_id: str, act: GraphAction, effect: str = "reversible",
                retry: str = "verify_before_retry") -> GraphNode:
    return GraphNode(id=node_id, kind="action", action=act, effect=effect, retry_safety=retry)


def decision_node(node_id: str) -> GraphNode:
    return GraphNode(id=node_id, kind="decision", effect="none", retry_safety="safe")


def terminal_node(node_id: str, status: str, outcome_code: str | None = None) -> GraphNode:
    return GraphNode(id=node_id, kind="terminal", effect="none", retry_safety="safe", status=status,
                     outcome_code=outcome_code)


def edge(source: str, target: str, *guards: Guard, priority: int = 0) -> GraphEdge:
    """An edge with the given guards; with none, an unconditional "always" edge."""
    return GraphEdge(source=source, target=target, guards=list(guards) or [Guard(kind="always")], priority=priority)


class RecordingOperator:
    """Scripted human: performs the given surface actions, then answers with a disposition."""

    def __init__(self, disposition: str = "resume", manual: list[tuple] | None = None):
        self.disposition = disposition
        self.manual = manual or []
        self.requests: list[InterventionRequest] = []
        self.surfaces: list = []

    def handle(self, request: InterventionRequest, surface) -> InterventionResult:
        self.requests.append(request)
        self.surfaces.append(surface)
        actions = []
        for kind, name, *rest in self.manual:
            element = next(e for e in surface.observe().elements if e.name == name)
            if kind == "click":
                surface.click(element)
            else:
                surface.type(element, rest[0])
            actions.append({"action": kind, "target": name})
        resolved = self.disposition in ("resume", "restart", "approve")
        return InterventionResult(resolved=resolved, human_actions=actions, note="test", disposition=self.disposition)
