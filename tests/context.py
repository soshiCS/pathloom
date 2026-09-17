"""Shared test wiring: imports from the package under test plus small builders."""
from __future__ import annotations

from src.cua.artifact import build_linear, classify_effect, classify_retry_safety, linear_path
from src.cua.escalation import Escalator, SessionControl
from src.cua.evidence import RunLog
from src.cua.models import (Action, Artifact, Element, GraphAction, GraphEdge, GraphNode, Guard,
                            InterventionRequest, InterventionResult, Locator, Observation, TransientError)
from src.cua.policy import Policy

__all__ = ["Action", "Artifact", "Element", "GraphAction", "GraphEdge", "GraphNode", "Guard", "InterventionRequest",
           "InterventionResult", "Locator", "Observation", "TransientError", "Policy", "Escalator",
           "SessionControl", "RunLog", "build_linear", "linear_path", "checkout_artifact", "checkout_nodes",
           "ladder", "text_ladder", "linear_node", "navigate_node", "type_node", "click_node", "extract_node",
           "finish_node", "RecordingOperator", "action_node", "decision_node", "terminal_node", "edge", "HOSTS",
           "ENTRY", "PARAMS"]

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


# ---------- action nodes as discovery classifies them ----------

def linear_node(node_id: str, act: GraphAction, risky: bool = False) -> GraphNode:
    """An action node with the effect and retry safety discovery would assign."""
    effect = classify_effect(act.action, risky)
    return GraphNode(id=node_id, kind="action", action=act, effect=effect,
                     retry_safety=classify_retry_safety(act.action, effect, act.checkpoint))


def navigate_node(node_id: str, url: str, expect: str | None = None) -> GraphNode:
    checkpoint = {"text_contains": expect} if expect else None
    return linear_node(node_id, GraphAction(action="navigate", target=None, value=url, checkpoint=checkpoint))


def type_node(node_id: str, field: str, param: str) -> GraphNode:
    return linear_node(node_id, GraphAction(action="type", target=ladder("textbox", field), value="{{" + param + "}}"))


def click_node(node_id: str, name: str, expect: str | None, context: str | None = None, risky: bool = False,
               role: str = "button") -> GraphNode:
    checkpoint = {"text_contains": expect} if expect else None
    return linear_node(node_id, GraphAction(action="click", target=ladder(role, name, context), checkpoint=checkpoint),
                       risky=risky)


def extract_node(node_id: str, target: Locator, output: str) -> GraphNode:
    return linear_node(node_id, GraphAction(action="extract", target=target, value=output))


def finish_node(node_id: str = "s16") -> GraphNode:
    """The one click that places the order: the policy calls it risky, so it is irreversible and never retried."""
    return click_node(node_id, "Finish", "Thank you", risky=True)


def checkout_nodes() -> list[GraphNode]:
    return [
        navigate_node("s1", ENTRY, "Swag Labs"),
        type_node("s2", "Username", "username"),
        type_node("s3", "Password", "password"),
        click_node("s4", "Login", "Products"),
        click_node("s5", "Add to cart", "Remove", context="{{product_name}}"),
        click_node("s6", "cart", "{{product_name}}", role="link"),
        click_node("s7", "Checkout", "Checkout: Your Information"),
        type_node("s8", "First Name", "first_name"),
        type_node("s9", "Last Name", "last_name"),
        type_node("s10", "Zip/Postal Code", "postal_code"),
        click_node("s11", "Continue", "Checkout: Overview"),
        extract_node("s12", ladder("link", "View details for {{product_name}}"), "product_name"),
        extract_node("s13", text_ladder("Item total:"), "subtotal"),
        extract_node("s14", text_ladder("Tax:"), "tax"),
        extract_node("s15", text_ladder("Total:"), "total"),
    ]


def checkout_artifact(extra_node: GraphNode | None = None) -> Artifact:
    """The linear graph discovery produces for the checkout-review capability, built by hand for tests."""
    nodes = checkout_nodes()
    if extra_node is not None:
        nodes.append(extra_node)
    money = r"\$\s*([\d.]+)"
    outputs = {
        "product_name": {"type": "string", "required": True, "pattern": r"(.+)"},
        "subtotal": {"type": "number", "required": True, "pattern": money},
        "tax": {"type": "number", "required": True, "pattern": money},
        "total": {"type": "number", "required": True, "pattern": money},
    }
    return build_linear(
        name="checkout_review", goal="add a product and read the checkout overview",
        surface_meta={"kind": "web", "app": "Swag Labs", "entry_url": ENTRY, "allowed_hosts": HOSTS},
        params=PARAMS, sensitive={"password"}, nodes=nodes, outputs=outputs,
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
