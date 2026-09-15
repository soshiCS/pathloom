"""Graph replay: traversal, guards, terminals, effect policy, retry safety, handoff, and version dispatch."""
import json
import subprocess
import sys

import pytest

from src.cua import __main__ as main_module
from src.cua import graph_replay
from src.cua.__main__ import main
from src.cua.artifact import save
from src.cua.escalation import NoOperator
from src.cua.graph import from_linear, nodes_by_id, save_graph, validate_graph
from src.cua.graph_replay import replay_graph
from src.cua.models import ArtifactV2
from src.cua.replay import replay
from tests.context import (ENTRY, HOSTS, PARAMS, Escalator, GraphAction, Guard, Policy, RecordingOperator, RunLog,
                           SessionControl, Step, TransientError, action_node, checkout_artifact, decision_node, edge,
                           ladder, terminal_node)
from tests.fake_surface import FakeSurface


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(graph_replay.time, "sleep", lambda _: None)


# ---------- wiring ----------

def run(surface, graph, params=None, operator=None, policy=None, irreversible="confirm"):
    log = RunLog("replay", secrets=("secret_sauce",))
    escalator = Escalator(operator or NoOperator(), SessionControl(), log)
    result = replay_graph(graph, dict(PARAMS) if params is None else params, surface,
                          policy or Policy(allowed_hosts=HOSTS), escalator, log, irreversible_policy=irreversible)
    return result, log


def events(log, name: str) -> list[dict]:
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def clicks(surface, name: str) -> int:
    return sum(1 for action in surface.actions if action[0] == "click" and action[1] == name)


class RecoveringSurface(FakeSurface):
    """A shop whose reload after a failed login lands on the product list (the login did take effect)."""

    def navigate(self, url: str) -> None:
        if self.logged_in and self.screen == "error":
            self.actions.append(("navigate", url))
            self.screen, self.url = "inventory", ENTRY + "inventory.html"
            return
        super().navigate(url)


# ---------- graphs under test ----------

def login_graph(bad_creds: bool = True, fallback: bool = True, checkpoint: dict | None = None,
                retry: str = "verify_before_retry", extra_edges: list = (), guest_value: str = "guest") -> ArtifactV2:
    """Log in, then a decision node classifies the screen. Every guard kind is exercised."""
    base = checkout_artifact()
    nodes = [
        action_node("s1", GraphAction("navigate", None, ENTRY, {"text_contains": "Swag Labs"}), effect="none",
                    retry="safe"),
        decision_node("d0"),
        action_node("s2", GraphAction("type", ladder("textbox", "Username"), "{{username}}")),
        action_node("s3", GraphAction("type", ladder("textbox", "Password"), "{{password}}")),
        action_node("s4", GraphAction("click", ladder("button", "Login"), checkpoint=checkpoint), retry=retry),
        decision_node("d1"),
        terminal_node("guest", "business_outcome", "guest_not_supported"),
        terminal_node("locked", "business_outcome", "user_locked_out"),
        terminal_node("bad_creds", "business_outcome", "invalid_credentials"),
        terminal_node("verify", "failure", "human_verification_required"),
        terminal_node("ok", "success"),
        terminal_node("unverified", "failure", "login_unverified"),
        terminal_node("wrong", "failure", "should_not_happen"),
    ]
    edges = [
        edge("s1", "d0"),
        edge("d0", "guest", Guard("input_equals", input="username", value=guest_value), priority=0),
        edge("d0", "s2", priority=1),
        edge("s2", "s3"), edge("s3", "s4"), edge("s4", "d1"),
        edge("d1", "locked", Guard("text_visible", value="locked out"), priority=0),
        edge("d1", "verify", Guard("dialog_contains", value="verify you are human"), priority=2),
        edge("d1", "ok", Guard("url_matches", pattern="inventory"),
             Guard("element_present", target=ladder("button", "Add to cart", "{{product_name}}")), priority=3),
        *extra_edges,
    ]
    if bad_creds:
        edges.append(edge("d1", "bad_creds", Guard("text_visible", value="do not match"), priority=1))
    if fallback:
        edges.append(edge("d1", "unverified", priority=9))
    used = {e.target for e in edges} | {"s1"}
    graph = ArtifactV2(
        schema_version="2.0", name="login_check", version=1, status="draft",
        description="log in and classify the screen that follows", surface=base.surface, inputs=base.inputs,
        outputs={}, entry_node="s1", nodes=[n for n in nodes if n.id in used], edges=edges,
        outcomes=base.outcomes + [{"code": "guest_not_supported", "kind": "business", "source": "reviewer",
                                   "detect": {"text_contains": "guest"}}],
        success={"text_contains": "Products"}, provenance={"run_id": "test-run"},
    )
    validate_graph(graph)
    return graph


def finish_step() -> Step:
    return Step(id="s16", action="click", target=ladder("button", "Finish"),
                checkpoint={"text_contains": "Thank you"}, risk="risky")


def finish_graph(effect: str = "irreversible", retry: str = "never_retry") -> ArtifactV2:
    """The converted checkout flow plus a Finish node: the one action that places the order."""
    graph = from_linear(checkout_artifact(extra_step=finish_step()))
    node = nodes_by_id(graph)["s16"]
    node.effect, node.retry_safety = effect, retry
    graph.success = {"text_contains": "Thank you"}
    validate_graph(graph)
    return graph


# ---------- traversal ----------

def test_linear_graph_succeeds_with_typed_outputs_and_stops_at_the_overview():
    surface = FakeSurface()
    result, log = run(surface, from_linear(checkout_artifact()))
    assert result.status == "success"
    assert result.outputs == {"product_name": "Sauce Labs Backpack", "subtotal": 29.99, "tax": 2.4, "total": 32.39}
    assert surface.screen == "overview" and clicks(surface, "Finish") == 0
    assert [e["node_id"] for e in events(log, "node_entered")] == [f"s{i}" for i in range(1, 16)] + ["success"]
    assert events(log, "terminal_reached")[0]["status"] == "success"
    assert all(e["fallback"] is False for e in events(log, "edge_selected"))   # a lone always edge is not a fallback
    assert "secret_sauce" not in log.path.read_text()


SCENARIOS = {
    "success": (dict(PARAMS), {}),
    "locked_out": ({**PARAMS, "username": "locked_out_user"}, {}),
    "bad_password": ({**PARAMS, "password": "wrong"}, {}),
    "missing_product": ({**PARAMS, "product_name": "Sauce Labs Unicorn"}, {}),
    "missing_postal_code": ({**PARAMS, "postal_code": ""}, {}),
    "cookie_notice": (dict(PARAMS), {"show_notice": True}),
    "transient_login": (dict(PARAMS), {"faults": ["transient"]}),
    "other_product": ({**PARAMS, "product_name": "Sauce Labs Bike Light"}, {}),
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_converted_graph_replays_exactly_like_the_linear_artifact(scenario):
    params, surface_options = SCENARIOS[scenario]
    linear_surface, graph_surface = FakeSurface(**surface_options), FakeSurface(**surface_options)
    log = RunLog("replay", secrets=("secret_sauce",))
    linear = replay(checkout_artifact(), params, linear_surface, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    graph, _ = run(graph_surface, from_linear(checkout_artifact()), params)
    assert (graph.status, graph.outcome_code, graph.step_id, graph.outputs) == \
        (linear.status, linear.outcome_code, linear.step_id, linear.outputs)
    assert graph.recoveries == linear.recoveries
    assert graph_surface.actions == linear_surface.actions


def test_replay_is_deterministic():
    first, second = FakeSurface(), FakeSurface()
    run(first, login_graph())
    run(second, login_graph())
    assert first.actions == second.actions


# ---------- guards and edge selection ----------

def test_input_guard_branches_before_touching_the_form():
    surface = FakeSurface()
    result, log = run(surface, login_graph(), params={**PARAMS, "username": "guest"})
    assert result.status == "business_outcome" and result.outcome_code == "guest_not_supported"
    assert result.step_id == "guest" and surface.typed == {}
    assert events(log, "guard_evaluated")[0] == {**events(log, "guard_evaluated")[0], "node_id": "d0",
                                                  "guard": "input_equals", "holds": True}
    assert events(log, "edge_selected")[1]["target"] == "guest"


def test_parameterized_input_guard_compares_the_substituted_value():
    # username must equal the value of another declared input, {{last_name}}.
    graph = login_graph(guest_value="{{last_name}}")
    surface = FakeSurface()
    result, log = run(surface, graph, params={**PARAMS, "username": "User"})          # last_name is "User"
    assert result.status == "business_outcome" and result.outcome_code == "guest_not_supported"
    assert surface.typed == {} and events(log, "guard_evaluated")[0]["holds"] is True

    surface = FakeSurface()
    result, log = run(surface, graph)                                                # "standard_user" != "User"
    assert result.status == "success" and events(log, "guard_evaluated")[0]["holds"] is False
    assert surface.typed["Username"] == "standard_user"


def test_missing_required_inputs_fail_before_traversal():
    surface = FakeSurface()
    result, log = run(surface, login_graph(guest_value="{{last_name}}"), params={"username": "User"})
    assert result.status == "failure" and result.outcome_code == "missing_inputs"
    assert "password" in result.observed and surface.actions == []
    assert events(log, "node_entered") == [] and events(log, "replay_finished")[0]["outcome_code"] == "missing_inputs"


def test_text_guard_selects_the_locked_out_terminal():
    result, log = run(FakeSurface(), login_graph(), params={**PARAMS, "username": "locked_out_user"})
    assert result.status == "business_outcome" and result.outcome_code == "user_locked_out"
    assert result.step_id == "locked" and "locked out" in result.observed
    assert events(log, "terminal_reached")[0] == {**events(log, "terminal_reached")[0], "node_id": "locked",
                                                  "status": "business_outcome", "outcome_code": "user_locked_out"}


def test_url_and_element_guards_select_the_success_terminal():
    result, log = run(FakeSurface(), login_graph())
    assert result.status == "success"
    chosen = [e for e in events(log, "edge_selected") if e["node_id"] == "d1"]
    assert chosen == [{**chosen[0], "target": "ok", "priority": 3, "fallback": False}]
    judged = [(e["target"], e["guard"], e["holds"]) for e in events(log, "guard_evaluated") if e["node_id"] == "d1"]
    assert judged == [("locked", "text_visible", False), ("bad_creds", "text_visible", False),
                      ("verify", "dialog_contains", False), ("ok", "url_matches", True),
                      ("ok", "element_present", True)]


def test_element_guard_respects_the_requested_product():
    # The guard's ladder is parameterized by product; a product that is not listed fails the guard.
    result, _ = run(FakeSurface(), login_graph(), params={**PARAMS, "product_name": "Sauce Labs Unicorn"})
    assert result.status == "failure" and result.outcome_code == "login_unverified"


def test_dialog_guard_routes_an_unknown_challenge_to_its_terminal():
    result, _ = run(FakeSurface(faults=["verification"]), login_graph())
    assert result.status == "failure" and result.outcome_code == "human_verification_required"
    assert result.step_id == "verify" and result.interventions == []


def test_lower_priority_wins_when_several_edges_match():
    rigged = login_graph(extra_edges=[edge("d1", "wrong", Guard("text_visible", value="Products"), priority=-1)])
    result, log = run(FakeSurface(), rigged)
    assert result.status == "failure" and result.outcome_code == "should_not_happen"
    assert [e["target"] for e in events(log, "edge_selected") if e["node_id"] == "d1"] == ["wrong"]


def test_fallback_edge_is_taken_only_after_the_guarded_ones_fail():
    result, log = run(FakeSurface(), login_graph(bad_creds=False), params={**PARAMS, "password": "wrong"})
    assert result.status == "failure" and result.outcome_code == "login_unverified"
    chosen = [e for e in events(log, "edge_selected") if e["node_id"] == "d1"]
    assert chosen == [{**chosen[0], "target": "unverified", "priority": 9, "fallback": True}]


def test_no_matching_edge_without_operator_is_a_failure_naming_the_decision():
    result, _ = run(FakeSurface(), login_graph(bad_creds=False, fallback=False),
                    params={**PARAMS, "password": "wrong"})
    assert result.status == "failure" and result.outcome_code == "no_matching_edge" and result.step_id == "d1"
    assert result.expected.startswith("one of:") and "-> locked when text_visible 'locked out'" in result.expected
    assert "do not match" in result.observed
    assert result.interventions[0]["kind"] == "stuck" and result.interventions[0]["disposition"] == "abort"


def test_human_resolves_a_decision_with_no_matching_edge():
    operator = RecordingOperator(disposition="resume",
                                 manual=[("type", "Password", "secret_sauce"), ("click", "Login")])
    result, _ = run(FakeSurface(), login_graph(bad_creds=False, fallback=False), params={**PARAMS, "password": "wrong"},
                    operator=operator)
    assert result.status == "success" and operator.requests[0].step_id == "d1"


# ---------- terminals ----------

def test_success_terminal_verifies_the_top_level_checkpoint():
    graph = from_linear(checkout_artifact())
    graph.success = {"text_contains": "Order complete"}
    result, _ = run(FakeSurface(), graph)
    assert result.status == "failure" and result.outcome_code == "success_condition_not_met"
    assert result.step_id == "success" and "Checkout: Overview" in result.observed


def test_failure_terminal_returns_its_code_with_the_screen():
    result, _ = run(FakeSurface(), login_graph(bad_creds=False), params={**PARAMS, "password": "wrong"})
    assert result.status == "failure" and result.outcome_code == "login_unverified"
    assert result.step_id == "unverified" and "do not match" in result.observed


# ---------- checkpoints, locators, outcomes ----------

def test_locator_ladder_falls_back_to_the_next_rung():
    drifted = from_linear(checkout_artifact())
    login = nodes_by_id(drifted)["s4"].action
    login.target = ladder("button", "Sign in")
    login.target.strategies[1] = {"kind": "css", "selector": "button:Login:"}
    result, log = run(FakeSurface(), drifted)
    assert result.status == "success"
    resolved = [e for e in events(log, "target_resolved") if e["step_id"] == "s4"]
    assert resolved[0]["rung"] == 1 and resolved[0]["drift_signal"] is True


def test_checkpoint_not_met_reports_node_expected_and_observed():
    broken = from_linear(checkout_artifact())
    nodes_by_id(broken)["s4"].action.checkpoint = {"text_contains": "Welcome back"}
    result, _ = run(FakeSurface(), broken)
    assert result.status == "failure" and result.outcome_code == "checkpoint_not_met"
    assert result.step_id == "s4" and "Welcome back" in result.expected and "Products" in result.observed


def test_action_without_checkpoint_hands_the_screen_to_its_edges():
    result, log = run(FakeSurface(), login_graph(), params={**PARAMS, "password": "wrong"})
    assert result.status == "business_outcome" and result.outcome_code == "invalid_credentials"
    assert result.step_id == "bad_creds"                       # judged by the decision node, not by s4
    assert events(log, "checkpoint_passed")[-1] == {**events(log, "checkpoint_passed")[-1], "step_id": "s4",
                                                    "checkpoint": None}


def test_missing_product_is_an_absence_outcome_not_a_stuck_state():
    result, _ = run(FakeSurface(), from_linear(checkout_artifact()),
                    params={**PARAMS, "product_name": "Sauce Labs Unicorn"})
    assert result.status == "business_outcome" and result.outcome_code == "product_not_found"
    assert result.step_id == "s5" and result.interventions == []


def test_known_notice_is_recovered_while_selecting_an_edge():
    result, log = run(FakeSurface(show_notice=True), login_graph())
    assert result.status == "success" and result.recoveries == ["s1: dismiss_cookie_notice"]


# ---------- retry safety ----------

def test_safe_action_is_simply_repeated_after_a_transient_error():
    surface = FakeSurface(faults=["transient"])
    result, log = run(surface, login_graph(retry="safe"))
    assert result.status == "success" and clicks(surface, "Login") == 2
    assert events(log, "action_repeated")[0]["retry_safety"] == "safe"
    assert any("transient" in note for note in result.recoveries)


def test_verify_before_retry_repeats_only_when_the_checkpoint_is_unmet():
    surface = FakeSurface(faults=["transient"])            # the reload lands on the login page again
    result, log = run(surface, login_graph(checkpoint={"text_contains": "Products"}))
    assert result.status == "success" and clicks(surface, "Login") == 2
    assert events(log, "action_repeated")[0]["retry_safety"] == "verify_before_retry"

    surface = RecoveringSurface(faults=["transient"])      # the reload shows the login took effect
    result, log = run(surface, login_graph(checkpoint={"text_contains": "Products"}))
    assert result.status == "success" and clicks(surface, "Login") == 1
    assert events(log, "step_already_satisfied")[0]["step_id"] == "s4"


def test_verify_before_retry_without_a_checkpoint_asks_instead_of_repeating():
    surface = FakeSurface(faults=["transient"])
    result, _ = run(surface, login_graph())
    assert result.status == "failure" and result.outcome_code == "action_result_uncertain"
    assert result.step_id == "s4" and clicks(surface, "Login") == 1
    assert result.expected == "evidence that the action took effect"
    assert result.interventions[0]["kind"] == "stuck"

    surface = RecoveringSurface(faults=["transient"])
    result, log = run(surface, login_graph(), operator=RecordingOperator(disposition="resume"))
    assert result.status == "success" and clicks(surface, "Login") == 1   # the human vouched: never repeated
    assert events(log, "action_assumed_done")[0]["step_id"] == "s4"


def test_never_retry_continues_on_proof_and_otherwise_asks():
    surface = RecoveringSurface(faults=["transient"])
    result, _ = run(surface, login_graph(checkpoint={"text_contains": "Products"}, retry="never_retry"))
    assert result.status == "success" and clicks(surface, "Login") == 1

    surface = FakeSurface(faults=["transient"])
    result, _ = run(surface, login_graph(checkpoint={"text_contains": "Products"}, retry="never_retry"))
    assert result.status == "failure" and result.outcome_code == "action_result_uncertain"
    assert clicks(surface, "Login") == 1 and result.interventions[0]["disposition"] == "abort"


class FlakyObserveSurface(FakeSurface):
    """The first look at the screen fails before anything was done."""

    def __init__(self):
        super().__init__()
        self.failed_once = False

    def observe(self):
        if not self.failed_once:
            self.failed_once = True
            raise TransientError("server returned HTTP 503", url=ENTRY)
        return super().observe()


def test_transient_error_before_acting_is_never_a_repeat():
    # Observing fails before the node acted: trying again is not repeating an action, even for never_retry.
    graph = login_graph()
    nodes_by_id(graph)["s1"].retry_safety = "never_retry"
    surface = FlakyObserveSurface()
    result, log = run(surface, graph)
    assert result.status == "success" and events(log, "action_repeated") == []
    assert events(log, "recovered")[0] == {**events(log, "recovered")[0], "step_id": "s1", "performed": False}
    assert surface.actions.count(("navigate", ENTRY)) == 2      # the reload, then the node's own navigation


def test_transient_error_during_the_action_counts_as_performed():
    surface = FakeSurface(faults=["slow_entry"])                # the navigation itself times out
    result, log = run(surface, login_graph())
    assert result.status == "success"
    assert events(log, "recovered")[0] == {**events(log, "recovered")[0], "step_id": "s1", "performed": True}


class FlakyTerminalSurface(FakeSurface):
    """Once armed, the page fails to load a given number of times; reloading it keeps the order state."""

    def __init__(self, failures: int, with_url: bool = True):
        super().__init__()
        self.failures, self.with_url, self.armed = failures, with_url, False

    def observe(self):
        if self.armed and self.failures:
            self.failures -= 1
            raise TransientError("server returned HTTP 503", url=self.url if self.with_url else None)
        return super().observe()

    def navigate(self, url: str) -> None:
        if self.armed:
            self.actions.append(("navigate", url))
            return
        super().navigate(url)


@pytest.fixture
def fail_at_the_terminal(monkeypatch):
    """Arm the surface's failures at the moment the engine enters a terminal node."""
    original = graph_replay.terminal_result

    def armed(ctx, node):
        ctx.surface.armed = True
        return original(ctx, node)

    monkeypatch.setattr(graph_replay, "terminal_result", armed)


def test_terminal_observation_rides_out_a_transient_error(fail_at_the_terminal):
    surface = FlakyTerminalSurface(failures=2)
    result, log = run(surface, from_linear(checkout_artifact()))
    assert result.status == "success" and result.outputs["total"] == 32.39
    assert result.recoveries == ["success: transient error (server returned HTTP 503); retry 1",
                                 "success: transient error (server returned HTTP 503); retry 2"]
    assert [e["attempt"] for e in events(log, "recovered") if e["step_id"] == "success"] == [1, 2]
    assert surface.actions.count(("navigate", ENTRY + "checkout-step-two.html")) == 2   # reloaded each time
    assert events(log, "terminal_reached")[0]["node_id"] == "success"

    surface = FlakyTerminalSurface(failures=1, with_url=False)          # nothing to reload, just wait and look again
    result, _ = run(surface, from_linear(checkout_artifact()))
    assert result.status == "success" and ("navigate", ENTRY + "checkout-step-two.html") not in surface.actions


def test_terminal_transient_errors_exhaust_into_a_structured_failure(fail_at_the_terminal):
    result, log = run(FlakyTerminalSurface(failures=5), from_linear(checkout_artifact()))
    assert result.status == "failure" and result.outcome_code == "transient_retries_exhausted"
    assert result.step_id == "success" and "503" in result.observed
    assert len(result.recoveries) == 2 and events(log, "terminal_reached") == []
    assert events(log, "replay_finished")[0]["outcome_code"] == "transient_retries_exhausted"


def test_persistent_errors_become_a_debuggable_hard_failure():
    result, _ = run(FakeSurface(faults=["transient"] * 5), login_graph(retry="safe"))
    assert result.status == "failure" and result.outcome_code == "transient_retries_exhausted"
    assert result.step_id == "s4" and "503" in result.observed


# ---------- effects and the irreversible policy ----------

def test_irreversible_action_is_denied_by_policy_deny():
    surface = FakeSurface()
    result, _ = run(surface, finish_graph(), irreversible="deny")
    assert result.status == "failure" and result.outcome_code == "irreversible_denied" and result.step_id == "s16"
    assert clicks(surface, "Finish") == 0 and result.interventions == [] and surface.screen == "overview"


def test_irreversible_action_is_confirmed_by_a_human_or_not_at_all():
    surface = FakeSurface()
    result, _ = run(surface, finish_graph())                                # default: confirm, nobody there
    assert result.status == "failure" and result.outcome_code == "irreversible_not_confirmed"
    assert clicks(surface, "Finish") == 0 and result.interventions[0]["kind"] == "confirm"

    denied = RecordingOperator(disposition="deny")
    surface = FakeSurface()
    result, _ = run(surface, finish_graph(), operator=denied)
    assert result.outcome_code == "irreversible_not_confirmed" and clicks(surface, "Finish") == 0
    assert denied.requests[0].kind == "confirm" and denied.requests[0].step_id == "s16"

    approved = RecordingOperator(disposition="approve")
    surface = FakeSurface()
    result, log = run(surface, finish_graph(), operator=approved)
    assert result.status == "success" and clicks(surface, "Finish") == 1 and surface.screen == "complete"
    assert events(log, "action_approved")[0]["step_id"] == "s16"


def test_irreversible_action_runs_unattended_under_policy_allow():
    surface = FakeSurface()
    result, log = run(surface, finish_graph(), irreversible="allow")
    assert result.status == "success" and clicks(surface, "Finish") == 1 and result.interventions == []
    assert events(log, "irreversible_allowed")[0]["step_id"] == "s16"


def test_reversible_actions_never_ask_for_effect_confirmation():
    result, log = run(FakeSurface(), from_linear(checkout_artifact()), irreversible="deny")
    assert result.status == "success" and result.interventions == []
    assert events(log, "effect_conflict") == [] and events(log, "irreversible_allowed") == []


def test_unknown_effect_under_deny_is_refused_even_if_a_human_would_approve():
    approving = RecordingOperator(disposition="approve")
    surface = FakeSurface()
    result, log = run(surface, finish_graph(effect="unknown"), operator=approving, irreversible="deny")
    assert result.status == "failure" and result.outcome_code == "unknown_effect_denied" and result.step_id == "s16"
    assert clicks(surface, "Finish") == 0 and surface.screen == "overview"
    assert approving.requests == [] and result.interventions == []            # nobody was even asked
    assert events(log, "policy_checked")[-1]["effect"] == "unknown"


def test_unknown_effect_under_confirm_or_allow_runs_only_after_approval():
    for irreversible in ("confirm", "allow"):
        surface = FakeSurface()
        result, _ = run(surface, finish_graph(effect="unknown"), irreversible=irreversible)
        assert result.status == "failure" and result.outcome_code == "unknown_effect_not_confirmed"
        assert clicks(surface, "Finish") == 0 and result.interventions[0]["kind"] == "confirm"

        surface = FakeSurface()
        result, _ = run(surface, finish_graph(effect="unknown"), operator=RecordingOperator(disposition="deny"),
                        irreversible=irreversible)
        assert result.outcome_code == "unknown_effect_not_confirmed" and clicks(surface, "Finish") == 0

        surface = FakeSurface()
        result, _ = run(surface, finish_graph(effect="unknown"), operator=RecordingOperator(disposition="approve"),
                        irreversible=irreversible)
        assert result.status == "success" and clicks(surface, "Finish") == 1


def test_global_policy_denial_cannot_be_overridden_by_allow():
    tampered = from_linear(checkout_artifact())
    nodes_by_id(tampered)["s1"].action.value = "https://attacker.example.com/"
    surface = FakeSurface()
    result, _ = run(surface, tampered, irreversible="allow")
    assert result.status == "failure" and result.outcome_code == "policy_denied" and surface.actions == []


def test_graph_calling_a_risky_control_reversible_is_treated_as_irreversible():
    # The policy classifies "Finish" as risky; the graph says reversible. The stricter view wins:
    # allow does not run it unattended, and deny refuses it.
    surface = FakeSurface()
    result, log = run(surface, finish_graph(effect="reversible", retry="verify_before_retry"), irreversible="allow")
    assert result.status == "failure" and result.outcome_code == "irreversible_not_confirmed"
    assert clicks(surface, "Finish") == 0 and events(log, "effect_conflict")[0]["graph_effect"] == "reversible"

    surface = FakeSurface()
    result, _ = run(surface, finish_graph(effect="reversible", retry="verify_before_retry"), irreversible="deny")
    assert result.outcome_code == "irreversible_denied" and clicks(surface, "Finish") == 0

    surface = FakeSurface()
    result, _ = run(surface, finish_graph(effect="reversible", retry="verify_before_retry"),
                    operator=RecordingOperator(disposition="approve"), irreversible="allow")
    assert result.status == "success" and clicks(surface, "Finish") == 1


# ---------- handoff ----------

def test_resume_rechecks_the_node_before_redoing_it():
    operator = RecordingOperator(disposition="resume", manual=[("click", "Verify"), ("click", "Login")])
    surface = FakeSurface(faults=["verification"])
    result, log = run(surface, from_linear(checkout_artifact()), operator=operator)
    assert result.status == "success" and operator.surfaces[0] is surface
    assert events(log, "step_already_satisfied")[0]["step_id"] == "s4"
    assert clicks(surface, "Login") == 2                       # once by automation, once by the human
    assert result.interventions[0]["human_actions"] == [{"action": "click", "target": "Verify"},
                                                        {"action": "click", "target": "Login"}]


def test_restart_starts_over_from_the_entry_node():
    operator = RecordingOperator(disposition="restart", manual=[("click", "Verify")])
    surface = FakeSurface(faults=["verification"])
    result, log = run(surface, from_linear(checkout_artifact()), operator=operator)
    assert result.status == "success" and result.outputs["total"] == 32.39
    assert surface.actions.count(("navigate", ENTRY)) == 2 and events(log, "flow_restarted") == [
        {**events(log, "flow_restarted")[0], "restart": 1}]
    assert [e["node_id"] for e in events(log, "node_entered")].count("s1") == 2


def test_abort_returns_the_failure_that_caused_the_handoff():
    result, _ = run(FakeSurface(faults=["verification"]), from_linear(checkout_artifact()),
                    operator=RecordingOperator(disposition="abort"))
    assert result.status == "failure" and result.outcome_code == "unknown_dialog" and result.step_id == "s4"
    assert result.interventions[0]["disposition"] == "abort"


def test_control_goes_to_the_human_and_comes_back_before_automation_continues():
    control = SessionControl()
    log = RunLog("replay")
    escalator = Escalator(RecordingOperator(disposition="approve"), control, log)
    result = replay_graph(finish_graph(), dict(PARAMS), FakeSurface(), Policy(allowed_hosts=HOSTS), escalator, log)
    assert result.status == "success"
    assert [h["to"] for h in control.history] == ["human", "automation"] and control.owner == "automation"


# ---------- version handling and isolation ----------

class ClosableSurface(FakeSurface):
    def close(self) -> None:
        self.closed = True


def test_cli_replays_each_schema_version_with_its_own_engine(monkeypatch, tmp_path, capsys):
    surfaces: list[ClosableSurface] = []
    monkeypatch.setattr(main_module, "PlaywrightSurface",
                        lambda **_: surfaces.append(ClosableSurface()) or surfaces[-1])
    linear_path = save(checkout_artifact(), secrets=("secret_sauce",))
    graph_path = save_graph(from_linear(checkout_artifact()), secrets=("secret_sauce",),
                            path=tmp_path / "checkout_review.graph.json")
    finish_path = save_graph(finish_graph(), secrets=("secret_sauce",), path=tmp_path / "finish.graph.json")
    common = ["--operator", "console", "--quiet", "--sensitive", "password"]   # drafts: supervised only
    common += [f"--param={k}={v}" for k, v in PARAMS.items()]

    assert main(["replay", "--artifact", str(linear_path), *common]) == 0
    assert main(["replay", "--artifact", str(graph_path), *common]) == 0
    assert main(["replay", "--artifact", str(finish_path), "--irreversible-policy", "deny", *common]) == 2
    assert main(["replay", "--artifact", str(finish_path), "--irreversible-policy", "allow", *common]) == 0

    out = capsys.readouterr().out
    assert out.count('"status": "success"') == 3 and '"outcome_code": "irreversible_denied"' in out
    assert surfaces[0].actions == surfaces[1].actions and all(s.closed for s in surfaces)
    assert clicks(surfaces[2], "Finish") == 0 and clicks(surfaces[3], "Finish") == 1
    started = [json.loads(line) for path in tmp_path.glob("evidence/replay-*/run.jsonl")
               for line in path.read_text().splitlines() if '"replay_started"' in line]
    linear = [s for s in started if "steps" in s]
    graphs = sorted((s for s in started if "nodes" in s), key=lambda s: s["nodes"])
    assert [s["steps"] for s in linear] == [15] and "nodes" not in linear[0]
    assert [(s["schema_version"], s["nodes"], s["edges"]) for s in graphs] == [("2.0", 16, 15), ("2.0", 17, 16),
                                                                               ("2.0", 17, 16)]


def test_graph_replay_never_imports_the_planner_or_the_llm_sdk():
    probe = ("import sys, src.cua.graph_replay; "
             "print(sorted(m for m in sys.modules "
             "if m in ('src.cua.planner', 'src.cua.agent', 'anthropic', 'openai')))")
    output = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "[]"
    assert not any(hasattr(graph_replay, name) for name in ("Planner", "ClaudePlanner", "OpenAIPlanner"))


def test_version_1_replay_is_unchanged_by_the_graph_engine():
    surface = FakeSurface()
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(checkout_artifact(extra_step=finish_step()), dict(PARAMS), surface, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "failure" and result.outcome_code == "risky_step_not_confirmed"   # the version-1 code
    assert clicks(surface, "Finish") == 0
    text = log.path.read_text()
    assert '"steps": 15' not in text and '"steps": 16' in text and "node_entered" not in text
