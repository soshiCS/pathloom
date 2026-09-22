"""Discovery survives a surface action that fails: not-performed failures return to the planner with context,
repeats are bounded, unknown outcomes are proven or handed to a person, and nothing is recorded twice."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import DiscoveryFailed, discover
from src.cua.artifact import linear_path
from src.cua.escalation import NoOperator
from src.cua.models import InterventionResult
from tests.context import (ENTRY, HOSTS, PARAMS, Element, Escalator, Policy, RecordingOperator, RunLog,
                           SessionControl)
from tests.fake_surface import FakeSurface, el
from tests.scripted_planner import ScriptedPlanner, ScriptedStep, ScriptedVisionPlanner, checkout_script

LOGIN = ScriptedStep("click", "button", "Login", expect="Products")


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


class HistorySpy(ScriptedPlanner):
    def __init__(self, script):
        super().__init__(script)
        self.histories: list[list[str]] = []

    def decide(self, goal, params, observation, history, candidates=()):
        self.histories.append([f"{a.kind}:{a.target.name if a.target else ''}:{a.result}" for a in history])
        return super().decide(goal, params, observation, history, candidates)


class ApproveButAbort:
    """Approves confirmations (so an irreversible click may run) and aborts on a stuck handoff."""

    def __init__(self):
        self.requests = []

    def handle(self, request, surface):
        self.requests.append(request)
        disposition = "approve" if request.kind == "confirm" else "abort"
        return InterventionResult(resolved=disposition == "approve", human_actions=[], note="test",
                                  disposition=disposition)


def run(script, surface=None, operator=None, max_steps=25, planner=None, vision=False):
    planner = planner or (ScriptedVisionPlanner(script, []) if vision else HistorySpy(script))
    surface = surface or FakeSurface()
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover(goal="reach the checkout overview", name="checkout_review", params=dict(PARAMS),
                        surface=surface, planner=planner, policy=Policy(allowed_hosts=HOSTS),
                        escalator=Escalator(operator or NoOperator(), SessionControl(), log), log=log,
                        entry_url=ENTRY, sensitive={"password"}, max_steps=max_steps,
                        vision=planner if vision else None)
    return artifact, planner, surface, log


def events(log, name):
    return [json.loads(l) for l in log.path.read_text().splitlines() if f'"event": "{name}"' in l]


def with_retry(script, step=LOGIN, times=1):
    """The planner tries the same step again after being told it failed."""
    index = next(i for i, s in enumerate(script) if s.kind == step.kind and s.name == step.name)
    return script[:index] + [step] * times + script[index:]


# ---------- not performed ----------

def test_a_pre_action_click_failure_returns_control_to_the_planner():
    surface = FakeSurface(faults=["blocked_click:Login"])
    artifact, planner, surface, log = run(with_retry(checkout_script()), surface=surface)
    [failed] = events(log, "action_failed")
    assert failed == {**failed, "kind": "click", "target": "button 'Login'", "performed": "no"}
    assert "intercepts pointer events" in failed["reason"] and "secret_sauce" not in log.path.read_text()
    assert surface.actions.count(("click", "Login", "")) == 1                    # the retry was the only real click
    logins = [n for n in linear_path(artifact) if n.action.action == "click" and n.action.target.strategies[0]["name"] == "Login"]
    assert len(logins) == 1 and logins[0].action.checkpoint == {"text_contains": "Products"}   # recorded exactly once
    assert artifact.status == "draft" and len(artifact.nodes) == 16


def test_the_planner_receives_the_failure_context_and_can_choose_differently():
    surface = FakeSurface(faults=["blocked_click:Login"])
    script = checkout_script()
    index = next(i for i, s in enumerate(script) if s.name == "Login")
    script = script[:index] + [LOGIN, ScriptedStep("type", "textbox", "Username", value="{{username}}")] + script[index:]
    _, planner, surface, _ = run(script, surface=surface)
    told = planner.histories[index][-1]                      # the decision right after the failed click
    assert told.startswith("click:Login:failed before acting:") and "choose a different control" in told
    assert "has failed 1 time(s) here" in told
    assert surface.actions[index] == ("type", "Username", "standard_user")       # it typed instead of clicking again


def test_repeating_the_same_failed_action_is_bounded_and_goes_to_the_handoff():
    surface = FakeSurface(faults=["blocked_click:Login"] * 5)
    with pytest.raises(DiscoveryFailed, match="aborted"):
        run(with_retry(checkout_script(), times=5), surface=surface)
    assert surface.actions.count(("click", "Login", "")) == 0

    operator = RecordingOperator(disposition="abort")
    surface = FakeSurface(faults=["blocked_click:Login"] * 5)
    with pytest.raises(DiscoveryFailed):
        run(with_retry(checkout_script(), times=5), surface=surface, operator=operator)
    assert operator.requests[0].kind == "stuck" and "failed 2 times on this screen" in operator.requests[0].reason

    resumed = RecordingOperator(disposition="resume")                            # the person unblocks the screen
    surface = FakeSurface(faults=["blocked_click:Login"] * 2)
    artifact, _, surface, log = run(with_retry(checkout_script(), times=2), surface=surface, operator=resumed)
    assert len(resumed.requests) == 1 and len(events(log, "action_failed")) == 2
    assert surface.actions.count(("click", "Login", "")) == 1 and len(artifact.nodes) == 16


def test_max_steps_still_bounds_the_whole_discovery():
    surface = FakeSurface(faults=["blocked_click:Login"] * 20)
    with pytest.raises(DiscoveryFailed, match="steps"):
        run(with_retry(checkout_script(), times=20), surface=surface, operator=RecordingOperator("resume"),
            max_steps=6)


def test_an_ordinary_execution_failure_never_captures_a_vision_screenshot():
    surface = FakeSurface(faults=["blocked_click:Login"])
    artifact, planner, surface, log = run(with_retry(checkout_script()), surface=surface, vision=True)
    assert surface.viewport_shots == [] and planner.frames == []
    assert not any(e["event"].startswith("vision_") for e in map(json.loads, log.path.read_text().splitlines()))
    assert len(artifact.nodes) == 16


# ---------- performed or unknown ----------

def test_a_proven_postcondition_after_an_unknown_outcome_records_the_action_once():
    surface = FakeSurface(faults=["uncertain_click:Login"])                     # the click landed, the surface lost it
    artifact, planner, surface, log = run(checkout_script(), surface=surface)
    assert surface.actions.count(("click", "Login", "")) == 1
    [proven] = events(log, "action_proven_after_failure")
    assert proven == {**proven, "kind": "click", "expect": "Products"}
    logins = [n for n in linear_path(artifact) if n.action.action == "click" and n.action.target.strategies[0]["name"] == "Login"]
    assert len(logins) == 1 and logins[0].action.checkpoint == {"text_contains": "Products"}
    assert planner.histories[4][-1].endswith(":ok")


def test_an_uncertain_irreversible_action_is_never_repeated_automatically():
    finish = ScriptedStep("click", "button", "Finish", expect="Thank you")
    operator = ApproveButAbort()
    surface = FakeSurface(faults=["lost_click:Finish"])                          # not dispatched, but nobody can tell
    script = checkout_script()
    script.insert(-1, finish)
    script.insert(-1, finish)                                                    # a planner that would try again
    with pytest.raises(DiscoveryFailed, match="aborted"):
        run(script, surface=surface, operator=operator)
    assert surface.actions.count(("click", "Finish", "")) == 0 and surface.screen == "overview"
    assert [r.kind for r in operator.requests] == ["confirm", "stuck"]
    assert "must not be repeated blindly" in operator.requests[1].reason


def test_an_unproven_reversible_action_goes_to_the_handoff_and_resume_continues():
    surface = FakeSurface(faults=["lost_click:Login"])
    resumed = RecordingOperator(disposition="resume", manual=[("click", "Login")])   # the person does it
    artifact, _, surface, log = run(checkout_script(), surface=surface, operator=resumed)
    assert surface.actions.count(("click", "Login", "")) == 1 and resumed.requests[0].kind == "stuck"
    assert events(log, "action_proven_after_failure") == []
    logins = [n for n in linear_path(artifact) if n.action.action == "click" and n.action.target.strategies[0]["name"] == "Login"]
    assert [n.id for n in logins] == artifact.provenance["human_steps"] == ["s4"]   # the person's click, recorded once
    assert logins[0].action.checkpoint == {"url_contains": "/inventory.html"} and artifact.status == "draft"


# ---------- wording never decides ----------

def test_an_unknown_outcome_is_never_repeated_whatever_the_error_text_says():
    # The fake's unknown-outcome message quotes every actionability phrase; only the state matters.
    surface = FakeSurface(faults=["lost_click:Login"])
    with pytest.raises(DiscoveryFailed, match="aborted"):
        run(with_retry(checkout_script()), surface=surface)
    assert surface.actions.count(("click", "Login", "")) == 0          # never retried on its own

    from src.cua.artifact import nodes_by_id
    from src.cua.replay import replay as engine
    from tests.context import checkout_artifact
    guarded = checkout_artifact()
    nodes_by_id(guarded)["s4"].retry_safety = "never_retry"
    surface = FakeSurface(faults=["lost_click:Login"])
    log = RunLog("replay", secrets=("secret_sauce",))
    result = engine(guarded, dict(PARAMS), surface, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.outcome_code == "action_result_uncertain" and surface.actions.count(("click", "Login", "")) == 0
    assert "intercepts pointer events" in log.path.read_text()            # the wording was logged, and ignored


# ---------- a lost action is proven only by what changed after it ----------

LOGIN_STEPS = [ScriptedStep("type", "textbox", "Username", value="{{username}}"),
               ScriptedStep("type", "textbox", "Password", value="{{password}}")]


def test_text_already_on_screen_before_a_lost_click_proves_nothing():
    # "Products" is on the inventory page before and after "Add to cart": the lost click is not recorded
    surface = FakeSurface(faults=["uncertain_click:Add to cart"])
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run(LOGIN_STEPS + [ScriptedStep("click", "button", "Login", expect="Products"),
                           ScriptedStep("click", "button", "Add to cart", context="{{product_name}}", expect="Products")],
            surface, operator=operator)
    assert operator.requests[0].reason.startswith("planner is stuck: click on button 'Add to cart' ended in an unknown state")


def test_text_that_newly_appears_after_a_lost_click_is_proof():
    surface = FakeSurface(faults=["uncertain_click:Login"])
    artifact, _, _, log = run(LOGIN_STEPS + [ScriptedStep("click", "button", "Login", expect="Products"),
                                             ScriptedStep("done", expect="Products")], surface)
    [proven] = events(log, "action_proven_after_failure")
    assert proven["proof"] == "text" and linear_path(artifact)[3].action.checkpoint == {"text_contains": "Products"}


def test_an_unchanged_url_with_pre_existing_text_is_not_proof_and_the_click_is_never_repeated():
    surface = FakeSurface(faults=["uncertain_click:Add to cart"])
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run(LOGIN_STEPS + [ScriptedStep("click", "button", "Login", expect="Products"),
                           ScriptedStep("click", "button", "Add to cart", context="{{product_name}}", expect="Swag")],
            surface, operator=operator)
    assert sum(1 for a in surface.actions if a[0] == "click" and a[1] == "Add to cart") == 1


def test_a_genuine_url_change_after_a_lost_click_is_proof_recorded_as_a_url_checkpoint():
    # the product name is on the inventory page and on the cart page: only the URL change proves the cart opened
    surface = FakeSurface(faults=["uncertain_click:cart"])
    artifact, _, _, log = run(LOGIN_STEPS + [ScriptedStep("click", "button", "Login", expect="Products"),
                                             ScriptedStep("click", "button", "Add to cart", context="{{product_name}}"),
                                             ScriptedStep("click", "link", "cart", expect="{{product_name}}"),
                                             ScriptedStep("done", expect="Your Cart")], surface)
    [proven] = events(log, "action_proven_after_failure")
    assert proven["proof"] == "url"
    cart = linear_path(artifact)[5]
    assert cart.action.action == "click" and cart.action.checkpoint == {"url_contains": "/cart.html"}
    assert events(log, "checkpoint_not_evidence")[0]["expected"] == "Sauce Labs Backpack"
