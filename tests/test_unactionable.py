"""A target the surface proves unactionable (cause `unactionable_target`): with the vision fallback enabled it
becomes a runtime stuck state that the bounded fallback may act on; without it, the planner is told and the
existing per-screen bound applies; a click that may already have been dispatched never reaches vision."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import DiscoveryFailed, discover
from src.cua.artifact import linear_path
from src.cua.escalation import NoOperator
from src.cua.models import ActionError
from tests.context import ENTRY, HOSTS, PARAMS, Escalator, Policy, RecordingOperator, RunLog, SessionControl
from tests.fake_surface import FakeSurface
from tests.scripted_planner import NO_TARGET, ScriptedStep, ScriptedVisionPlanner, visual_click

LOGIN = [ScriptedStep("type", "textbox", "Username", value="{{username}}"),
         ScriptedStep("type", "textbox", "Password", value="{{password}}")]


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


def run(script, decisions, surface, vision=True, operator=None, max_steps=12):
    planner = ScriptedVisionPlanner(script, decisions)
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover(goal="log in", name="login", params=dict(PARAMS), surface=surface, planner=planner,
                        policy=Policy(allowed_hosts=HOSTS), escalator=Escalator(operator or NoOperator(), SessionControl(), log),
                        log=log, entry_url=ENTRY, sensitive={"password"}, max_steps=max_steps,
                        vision=planner if vision else None, max_vision_attempts=2)
    return artifact, planner, log


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def test_the_cause_is_a_structured_field_with_a_closed_vocabulary():
    assert ActionError("x", performed="no", cause="unactionable_target").cause == "unactionable_target"
    assert ActionError("x", performed="no").cause is None
    with pytest.raises(ValueError, match="cause must be one of"):
        ActionError("x", performed="no", cause="covered")


def test_an_unactionable_click_goes_to_the_bounded_vision_fallback_and_is_verified_before_recording():
    surface = FakeSurface(faults=["unactionable_click:Login"])
    artifact, planner, log = run([*LOGIN, ScriptedStep("click", "button", "Login", expect="Products"),
                                  ScriptedStep("done", expect="Products")],
                                 [visual_click("Login", expect="Products")], surface)
    [failed] = events(log, "action_failed")
    assert failed["performed"] == "no" and failed["cause"] == "unactionable_target"
    [stuck] = events(log, "runtime_stuck")
    assert stuck["stuck_cause"] == "actionability" and "no enclosing control" in stuck["reason"]
    assert len(events(log, "vision_fallback_requested")) == 1 and planner.calls[0]["remaining"] == 1
    assert [e["verified"] for e in events(log, "vision_action_verified")] == [True]
    login = linear_path(artifact)[3]
    assert login.action.action == "click" and login.action.checkpoint == {"text_contains": "Products"}
    assert login.action.target.strategies[0]["kind"] == "coords" and login.action.target.strategies[0]["exact"] is True
    assert artifact.provenance["vision_fallback"]["attempts"] == 1 and surface.actions.count(("click", "Login", "")) == 1


def test_without_vision_the_planner_is_told_and_the_existing_bound_hands_off():
    surface = FakeSurface(faults=["unactionable_click:Login", "unactionable_click:Login"])
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run([*LOGIN, ScriptedStep("click", "button", "Login"), ScriptedStep("click", "button", "Login")], [], surface,
            vision=False, operator=operator)
    assert operator.requests[0].reason.startswith("planner is stuck: the same click on button 'Login' failed 2 times")


def test_a_click_that_may_have_been_dispatched_is_neither_sent_to_vision_nor_repeated():
    # a lost "Add to cart": the page's URL does not change and the expected text never appears, so nothing proves it
    surface = FakeSurface(faults=["uncertain_click:Add to cart"])
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run([*LOGIN, ScriptedStep("click", "button", "Login", expect="Products"),
             ScriptedStep("click", "button", "Add to cart", context="{{product_name}}", expect="Checkout: Overview")],
            [visual_click("Add to cart", expect="Checkout: Overview")], surface, operator=operator)
    assert sum(1 for a in surface.actions if a[0] == "click" and a[1] == "Add to cart") == 1   # once, never repeated
    assert operator.requests[0].reason.startswith("planner is stuck: click on button 'Add to cart' ended in an unknown state")


def test_vision_budget_and_screen_rules_still_apply_to_actionability_stuck_states():
    surface = FakeSurface(faults=["unactionable_click:Login", "unactionable_click:Login"])
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run([*LOGIN, ScriptedStep("click", "button", "Login", expect="Products"),
             ScriptedStep("click", "button", "Login", expect="Products")], [NO_TARGET, NO_TARGET], surface,
            operator=operator)
    assert operator.requests[0].reason.startswith("planner is stuck: click on button 'Login' is not actionable")
