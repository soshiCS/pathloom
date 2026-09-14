"""Deterministic replay: outputs, outcome classification, recovery, and policy at replay time."""
import subprocess
import sys

from src.cua import replay as replay_module
from src.cua.escalation import NoOperator
from src.cua.replay import replay
from tests.context import (HOSTS, PARAMS, Escalator, Policy, RecordingOperator, RunLog, SessionControl, Step,
                           checkout_artifact, ladder)
from tests.fake_surface import ENTRY, FakeSurface


def run(surface, params=None, artifact=None, operator=None, policy=None):
    if params is None:
        params = dict(PARAMS)
    log = RunLog("replay", secrets=("secret_sauce",))
    escalator = Escalator(operator or NoOperator(), SessionControl(), log)
    result = replay(artifact or checkout_artifact(), params, surface, policy or Policy(allowed_hosts=HOSTS),
                    escalator, log)
    return result, log


def finish_step() -> Step:
    return Step(id="s16", action="click", target=ladder("button", "Finish"),
                checkpoint={"text_contains": "Thank you"}, risk="risky")


def test_success_returns_typed_outputs_and_stops_at_the_overview():
    surface = FakeSurface()
    result, log = run(surface)
    assert result.status == "success"
    assert result.outputs == {"product_name": "Sauce Labs Backpack", "subtotal": 29.99, "tax": 2.4, "total": 32.39}
    assert surface.screen == "overview" and "Finish" not in [a[1] for a in surface.actions]
    assert "secret_sauce" not in log.path.read_text()          # the typed password never reaches the log


def test_money_text_becomes_a_number():
    from src.cua.agent import infer_type
    assert infer_type("$ 29.99") == "number" and infer_type("1,299") == "integer" and infer_type("Backpack") == "string"
    assert replay_module.typed_value("$ 29.99", "number") == 29.99
    assert replay_module.typed_value("1,299", "integer") == 1299
    assert replay_module.typed_value("n/a", "number") == "n/a"


def test_replay_never_imports_the_planner_or_the_llm_sdk():
    # Import the replay engine in a clean interpreter: neither the planner module nor the
    # Anthropic/OpenAI SDKs may be loaded as a side effect.
    probe = ("import sys, src.cua.replay; "
             "print(sorted(m for m in sys.modules if m in ('src.cua.planner', 'anthropic', 'openai')))")
    output = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    assert output.strip() == "[]"
    assert not hasattr(replay_module, "Planner") and not hasattr(replay_module, "ClaudePlanner")
    assert not hasattr(replay_module, "OpenAIPlanner")


def test_replay_is_deterministic():
    first, second = FakeSurface(), FakeSurface()
    run(first)
    run(second)
    assert first.actions == second.actions
    assert first.actions[:5] == [("navigate", ENTRY), ("type", "Username", "standard_user"),
                                 ("type", "Password", "secret_sauce"), ("click", "Login", ""),
                                 ("click", "Add to cart", "Sauce Labs Backpack")]


def test_context_qualified_locator_picks_the_requested_product():
    surface = FakeSurface()
    result, _ = run(surface, params={**PARAMS, "product_name": "Sauce Labs Bike Light"})
    assert result.status == "success"
    assert ("click", "Add to cart", "Sauce Labs Bike Light") in surface.actions
    assert result.outputs["product_name"] == "Sauce Labs Bike Light" and result.outputs["subtotal"] == 9.99


def test_finish_in_an_artifact_requires_confirmation_and_is_never_clicked_unattended():
    surface = FakeSurface()
    result, _ = run(surface, artifact=checkout_artifact(extra_step=finish_step()))
    assert result.status == "failure" and result.outcome_code == "risky_step_not_confirmed"
    assert result.step_id == "s16" and "Finish" in result.observed
    assert ("click", "Finish", "") not in surface.actions and surface.screen == "overview"


def test_finish_is_confirmed_by_a_human_or_not_at_all():
    denied = RecordingOperator(disposition="deny")
    surface = FakeSurface()
    result, _ = run(surface, artifact=checkout_artifact(extra_step=finish_step()), operator=denied)
    assert denied.requests[0].kind == "confirm" and result.outcome_code == "risky_step_not_confirmed"
    assert ("click", "Finish", "") not in surface.actions

    approved = RecordingOperator(disposition="approve")
    surface = FakeSurface()
    run(surface, artifact=checkout_artifact(extra_step=finish_step()), operator=approved)
    assert ("click", "Finish", "") in surface.actions                # only after explicit approval


def test_locked_out_user_is_a_business_outcome():
    result, _ = run(FakeSurface(), params={**PARAMS, "username": "locked_out_user"})
    assert result.status == "business_outcome" and result.outcome_code == "user_locked_out"
    assert result.step_id == "s4" and result.outputs == {}


def test_invalid_credentials_is_a_business_outcome():
    result, _ = run(FakeSurface(), params={**PARAMS, "password": "wrong"})
    assert result.status == "business_outcome" and result.outcome_code == "invalid_credentials"


def test_missing_product_is_a_business_outcome_not_a_stuck_state():
    result, _ = run(FakeSurface(), params={**PARAMS, "product_name": "Sauce Labs Unicorn"})
    assert result.status == "business_outcome" and result.outcome_code == "product_not_found"
    assert result.step_id == "s5" and "Sauce Labs Unicorn" in result.observed
    assert result.interventions == []                                  # no human was bothered


def test_structural_fallback_never_substitutes_a_different_product():
    # An artifact whose product locator still carries a structural rung (pointing at the
    # discovery-time product) must not add that product when another one is requested.
    stale = checkout_artifact()
    stale.steps[4].target.strategies.append({"kind": "css", "selector": "button:Add to cart:Sauce Labs Backpack"})
    surface = FakeSurface()
    result, _ = run(surface, params={**PARAMS, "product_name": "Sauce Labs Unicorn"}, artifact=stale)
    assert ("click", "Add to cart", "Sauce Labs Backpack") not in surface.actions
    assert result.status != "success"


def test_absence_outcomes_only_apply_to_the_step_that_looks_for_the_item():
    # A locked-out login with no declared locked-out outcome is a stuck checkpoint, never
    # "product not found" just because the product is not on the login screen.
    undeclared = checkout_artifact()
    undeclared.outcomes = [o for o in undeclared.outcomes if o["code"] != "user_locked_out"]
    result, _ = run(FakeSurface(), params={**PARAMS, "username": "locked_out_user"}, artifact=undeclared)
    assert result.status == "failure" and result.outcome_code == "checkpoint_not_met" and result.step_id == "s4"


def test_missing_checkout_information_is_a_business_outcome():
    result, _ = run(FakeSurface(), params={**PARAMS, "postal_code": ""})
    assert result.status == "business_outcome" and result.outcome_code == "checkout_info_missing"
    assert result.step_id == "s11"


def test_known_notice_is_recovered_not_reported():
    result, _ = run(FakeSurface(show_notice=True))
    assert result.status == "success"
    assert result.recoveries == ["s1: dismiss_cookie_notice"]


def test_transient_error_is_retried_then_succeeds():
    result, _ = run(FakeSurface(faults=["transient"]))
    assert result.status == "success"
    assert any("transient" in note for note in result.recoveries)


def test_persistent_errors_become_a_debuggable_hard_failure(monkeypatch):
    monkeypatch.setattr(replay_module.time, "sleep", lambda _: None)
    result, _ = run(FakeSurface(faults=["transient"] * 5))
    assert result.status == "failure" and result.outcome_code == "transient_retries_exhausted"
    assert result.step_id == "s4" and "503" in result.observed


def test_unknown_dialog_without_operator_is_a_failure_with_context():
    result, _ = run(FakeSurface(faults=["verification"]))
    assert result.status == "failure" and result.outcome_code == "unknown_dialog"
    assert "verify you are human" in result.observed
    assert result.interventions[0]["disposition"] == "abort"


def test_missing_required_input_fails_before_touching_the_surface():
    surface = FakeSurface()
    result, _ = run(surface, params={"username": "standard_user"})
    assert result.status == "failure" and result.outcome_code == "missing_inputs"
    assert "password" in result.observed and surface.actions == []


def test_tampered_artifact_cannot_leave_the_allowlist():
    tampered = checkout_artifact()
    tampered.steps[0].value = "https://attacker.example.com/"
    surface = FakeSurface()
    result, _ = run(surface, artifact=tampered)
    assert result.status == "failure" and result.outcome_code == "policy_denied"
    assert surface.actions == []


def test_checkpoint_not_met_reports_step_expected_and_observed():
    broken = checkout_artifact()
    broken.steps[3].checkpoint = {"text_contains": "Welcome back"}  # text the site never shows
    result, _ = run(FakeSurface(), artifact=broken)
    assert result.status == "failure" and result.outcome_code == "checkpoint_not_met"
    assert result.step_id == "s4" and "Welcome back" in result.expected and "Products" in result.observed


def test_locator_ladder_falls_back_to_the_next_rung():
    drifted = checkout_artifact()
    drifted.steps[3].target = ladder("button", "Sign in")                      # role+name no longer matches ...
    drifted.steps[3].target.strategies[1] = {"kind": "css", "selector": "button:Login:"}   # ... but the path does
    result, log = run(FakeSurface(), artifact=drifted)
    assert result.status == "success"
    events = [line for line in log.path.read_text().splitlines() if '"target_resolved"' in line and '"s4"' in line]
    assert '"rung": 1' in events[0] and '"drift_signal": true' in events[0]
