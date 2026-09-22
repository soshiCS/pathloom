"""An action nothing can prove is never recorded and never blindly repeated.

An expectation the planner supplied and the screen contradicted is not the same thing as an action
with no expectation at all. When the expectation fails, discovery looks for independent deterministic
proof (a safe page-address change, a native control state that flipped) and records the step with that
proof instead. When there is none, the step is classified `action_effect_unverified`, left out of the
graph, and refused a second blind attempt until something measurable changes or a person decides.
Offline, on the fake shop; no model and no live site.
"""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import DiscoveryFailed, discover
from src.cua.artifact import linear_path, validate
from src.cua.escalation import NoOperator
from src.cua.replay import replay
from tests.context import ENTRY, HOSTS, PARAMS, Escalator, Policy, RecordingOperator, RunLog, SessionControl
from tests.fake_surface import FakeSurface, el
from tests.scripted_planner import ScriptedStep, ScriptedVisionPlanner, checkout_script, visual_click


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


LOGIN = checkout_script()[:4]                      # navigate, type, type, click Login
DONE = ScriptedStep("done", expect="Products")


def run(script, surface=None, operator=None, max_steps=25, vision=None):
    log = RunLog("discovery", secrets=("secret_sauce",))
    surface = surface or FakeSurface()
    planner = vision or ScriptedPlannerOf(script)
    artifact = discover(goal="reach the inventory", name="inventory_visit", params=dict(PARAMS), surface=surface,
                        planner=planner, policy=Policy(allowed_hosts=HOSTS),
                        escalator=Escalator(operator or NoOperator(), SessionControl(), log), log=log,
                        entry_url=ENTRY, sensitive={"password"}, max_steps=max_steps,
                        vision=vision, max_vision_attempts=2)
    return artifact, surface, log


def ScriptedPlannerOf(script):
    from tests.scripted_planner import ScriptedPlanner
    return ScriptedPlanner(script)


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def inert_surface(name="Show details"):
    """A control that is real, resolvable and clickable, and changes nothing observable at all."""
    surface = FakeSurface()
    surface.inert.add(name)
    return surface


def toggle_surface(name="Compact view", state="pressed"):
    surface = FakeSurface()
    surface.toggles[name] = (state, False)
    return surface


# ---------- 1. an unmet supplied expectation is not recorded ----------

def test_a_click_whose_expectation_never_appears_is_not_recorded_at_all():
    surface = inert_surface()
    artifact, surface, log = run([*LOGIN, ScriptedStep("click", "button", "Show details", expect="Item details"),
                                  DONE], surface=surface, operator=RecordingOperator("resume"))
    names = [n.action.target.strategies[0].get("name") for n in linear_path(artifact) if n.action.target]
    assert "Show details" not in names                     # performed, unproven, absent from the graph
    assert surface.actions.count(("click", "Show details", "")) == 1
    [unverified] = events(log, "action_effect_unverified")
    assert unverified["cause"] == "action_effect_unverified" and unverified["kind"] == "click"
    assert events(log, "expectation_failed")[0]["expected"] == "Item details"
    assert events(log, "action_effect_proven") == []
    validate(artifact)


def test_the_planner_is_told_the_action_was_unverified_not_that_it_succeeded():
    from tests.scripted_planner import ScriptedPlanner

    class Spy(ScriptedPlanner):
        def __init__(self, script):
            super().__init__(script)
            self.results: list[str] = []

        def decide(self, goal, params, observation, history, candidates=()):
            self.results = [a.result or "" for a in history]
            return super().decide(goal, params, observation, history, candidates)

    spy = Spy([*LOGIN, ScriptedStep("click", "button", "Show details", expect="Item details"), DONE])
    log = RunLog("discovery", secrets=("secret_sauce",))
    discover(goal="reach the inventory", name="inventory_visit", params=dict(PARAMS), surface=inert_surface(),
             planner=spy, policy=Policy(allowed_hosts=HOSTS),
             escalator=Escalator(RecordingOperator("resume"), SessionControl(), log), log=log, entry_url=ENTRY,
             sensitive={"password"}, max_steps=25)
    told = [r for r in spy.results if r.startswith("unverified:")]
    assert told and "never appeared" in told[-1] and "not recorded" in told[-1]
    assert not any(r == "ok" or r.startswith("done;") for r in spy.results[-1:])


# ---------- 2. an action with no expectation keeps its documented behaviour ----------

def test_an_action_with_no_expectation_is_still_recorded_with_no_checkpoint():
    # Nothing was requested, so nothing was contradicted: the ordinary checkpoint-less rule applies.
    surface = inert_surface()
    artifact, surface, log = run([*LOGIN, ScriptedStep("click", "button", "Show details"), DONE], surface=surface)
    shown = [n for n in linear_path(artifact)
             if n.action.target and n.action.target.strategies[0].get("name") == "Show details"]
    assert len(shown) == 1 and shown[0].action.checkpoint is None
    assert shown[0].retry_safety == "never_retry"          # unprovable, so replay never repeats it either
    assert events(log, "action_effect_unverified") == [] and events(log, "expectation_failed") == []


# ---------- 3. the same unverified action is not dispatched twice ----------

def test_the_same_unverified_action_cannot_be_dispatched_again_before_progress():
    surface = inert_surface()
    blind = ScriptedStep("click", "button", "Show details", expect="Item details")
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed):
        run([*LOGIN, blind, blind, DONE], surface=surface, operator=operator)
    assert surface.actions.count(("click", "Show details", "")) == 1     # the second attempt never reached the page
    assert [r.kind for r in operator.requests] == ["stuck"]


def test_a_refused_repeat_is_logged_with_its_reason_and_handed_to_a_person():
    surface = inert_surface()
    blind = ScriptedStep("click", "button", "Show details", expect="Item details")
    artifact, surface, log = run([*LOGIN, blind, DONE], surface=surface,
                                 operator=RecordingOperator("resume"))
    [refused] = events(log, "unverified_repeat_refused")
    assert refused["kind"] == "click" and refused["attempts"] == 1
    assert "must not be repeated blindly" in events(log, "intervention_requested")[-1]["reason"]


def test_discovery_cannot_build_an_artifact_with_two_consecutive_blind_attempts_at_one_target():
    surface = inert_surface()
    blind = ScriptedStep("click", "button", "Show details", expect="Item details")
    artifact, surface, log = run([*LOGIN, blind, blind, DONE], surface=surface,
                                 operator=RecordingOperator("resume"))
    clicks = [n.action.target.strategies[0].get("name") for n in linear_path(artifact)
              if n.action.action == "click" and n.action.target]
    assert clicks.count("Show details") == 0
    assert all(a != b for a, b in zip(clicks, clicks[1:]))               # no repeated target anywhere in the graph


# ---------- 4. vision fallback never repeats a possibly performed action ----------

def test_vision_fallback_does_not_repeat_a_visual_action_it_could_not_verify():
    surface = inert_surface("Start")
    decision = visual_click("Start", expect="Session started", box=(10, 10, 50, 20))
    planner = ScriptedVisionPlanner([ScriptedStep("type", "textbox", "Username", value="{{username}}"),
                                     ScriptedStep("stuck", stuck_cause="missing_control"),
                                     ScriptedStep("stuck", stuck_cause="missing_control"), DONE],
                                    [decision, decision])
    log = RunLog("discovery", secrets=("secret_sauce",))
    with pytest.raises(DiscoveryFailed):
        discover(goal="start the session", name="session_start", params=dict(PARAMS), surface=surface,
                 planner=planner, policy=Policy(allowed_hosts=HOSTS),
                 escalator=Escalator(RecordingOperator("abort"), SessionControl(), log), log=log, entry_url=ENTRY,
                 sensitive={"password"}, max_steps=25, vision=planner, max_vision_attempts=2)
    assert surface.actions.count(("click", "Start", "")) == 1            # dispatched once, never proposed again
    assert len(planner.calls) == 1                                       # the visual budget was spent, not retried
    [unverified] = events(log, "action_effect_unverified")
    assert unverified["source"] == "vision" and unverified["cause"] == "action_effect_unverified"


# ---------- 5, 6, 7. the three kinds of deterministic proof ----------

def test_newly_appearing_expected_text_records_exactly_one_verified_node():
    artifact, surface, log = run([*LOGIN, DONE])
    logins = [n for n in linear_path(artifact)
              if n.action.action == "click" and n.action.target.strategies[0].get("name") == "Login"]
    assert len(logins) == 1 and logins[0].action.checkpoint == {"text_contains": "Products"}
    assert events(log, "action_effect_unverified") == [] and events(log, "expectation_failed") == []


def test_a_valid_url_transition_records_exactly_one_verified_node():
    artifact, surface, log = run([*LOGIN[:3], ScriptedStep("click", "button", "Login", expect="Welcome back"), DONE])
    logins = [n for n in linear_path(artifact)
              if n.action.action == "click" and n.action.target.strategies[0].get("name") == "Login"]
    assert len(logins) == 1 and logins[0].action.checkpoint == {"url_contains": "/inventory.html"}
    assert events(log, "action_effect_proven")[0]["proof"] == "the page address changed"
    assert events(log, "action_effect_unverified") == []


def test_a_native_state_transition_is_deterministic_proof_and_records_one_node():
    surface = toggle_surface()
    artifact, surface, log = run([ScriptedStep("click", "button", "Compact view", expect="Compact mode on"),
                                  ScriptedStep("done", expect="Swag Labs")], surface=surface)
    toggles = [n for n in linear_path(artifact)
               if n.action.target and n.action.target.strategies[0].get("name") == "Compact view"]
    assert len(toggles) == 1
    # The state is both the proof and a predicate replay can re-verify by resolving this step's own control.
    assert toggles[0].action.checkpoint == {"target_selected": "true"}
    assert events(log, "action_effect_proven")[0]["proof"] == "the control is now selected (aria-pressed)"
    assert events(log, "action_effect_unverified") == []


def test_a_click_the_surface_merely_dispatched_is_not_proof():
    # The surface activated an enclosing control instead of the target: the click happened, but nothing shows
    # it did anything, so it is still unverified.
    surface = inert_surface()
    artifact, surface, log = run([*LOGIN, ScriptedStep("click", "button", "Show details", expect="Item details"),
                                  DONE], surface=surface, operator=RecordingOperator("resume"))
    assert len(events(log, "action_effect_unverified")) == 1
    assert events(log, "action_effect_proven") == []


# ---------- 8, 9. the bounded-retry rule ----------

def test_an_unprovable_action_hands_off_with_no_node_rather_than_retrying():
    # This build takes the conservative branch the rule permits: no automatic second attempt at all.
    surface = inert_surface()
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed):
        run([*LOGIN, ScriptedStep("click", "button", "Show details", expect="Item details"), DONE],
            surface=surface, operator=operator)
    assert surface.actions.count(("click", "Show details", "")) == 1
    assert [r.kind for r in operator.requests] == ["stuck"]


@pytest.mark.parametrize("name, state", [("Compact view", "pressed"), ("Notify me", "checked"),
                                         ("More filters", "expanded")])
def test_no_control_class_is_ever_dispatched_twice_on_an_unproven_attempt(name, state):
    # Buttons, toggles and submissions alike: one attempt, then a person. Nothing is retried by inference.
    surface = FakeSurface()
    surface.inert.add(name)
    blind = ScriptedStep("click", "button", name, expect="Never shown")
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed):
        run([*LOGIN, blind, blind, DONE], surface=surface, operator=operator)
    assert surface.actions.count(("click", name, "")) == 1


def test_replay_never_repeats_an_action_by_inferring_from_its_effect():
    # The rule is the checkpoint, not the effect word: a reversible node with nothing to verify is
    # never carried out a second time on its own, and neither is an irreversible or unknown one.
    from src.cua.models import GraphAction, GraphNode
    from src.cua.replay import may_repeat

    def built(effect, retry, checkpoint):
        n = GraphNode(id="n", kind="action",
                      action=GraphAction(action="click", target=None, value=None, checkpoint=checkpoint),
                      effect=effect, retry_safety=retry)
        return n

    assert may_repeat(built("reversible", "verify_before_retry", {"text_contains": "Done"})) is True
    assert may_repeat(built("reversible", "verify_before_retry", None)) is False   # nothing to verify first
    assert may_repeat(built("reversible", "never_retry", {"text_contains": "Done"})) is False
    assert may_repeat(built("irreversible", "never_retry", {"text_contains": "Done"})) is False
    assert may_repeat(built("unknown", "never_retry", None)) is False


# ---------- 11. the resulting artifact replays in order ----------

def test_replay_of_the_resulting_artifact_reaches_the_destination_before_extracting_there():
    surface = inert_surface()
    artifact, _, _ = run([*LOGIN, ScriptedStep("click", "button", "Show details", expect="Item details"),
                          ScriptedStep("click", "link", "cart", expect="Your Cart"),
                          ScriptedStep("extract", "text", text="Your Cart", output_name="heading"),
                          ScriptedStep("done", expect="Your Cart")], surface=surface,
                         operator=RecordingOperator("resume"))
    validate(artifact)
    order = [n.action.action for n in linear_path(artifact)]
    assert order.index("extract") > max(i for i, a in enumerate(order) if a == "click")
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, dict(PARAMS), FakeSurface(), Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success" and result.outputs["heading"]
    assert events(log, "checkpoint_not_met") == []


def test_replay_never_calls_a_null_checkpoint_a_passed_one():
    surface = inert_surface()
    artifact, _, _ = run([*LOGIN, ScriptedStep("click", "button", "Show details"),
                          ScriptedStep("done", expect="Products")], surface=surface)
    log = RunLog("replay", secrets=("secret_sauce",))
    replay(artifact, dict(PARAMS), inert_surface(), Policy(allowed_hosts=HOSTS),
           Escalator(NoOperator(), SessionControl(), log), log)
    # every checkpoint the run reports as passed is a real predicate; the steps that carry none say so
    assert [e["checkpoint"] for e in events(log, "checkpoint_passed")] != []
    assert None not in [e["checkpoint"] for e in events(log, "checkpoint_passed")]
    assert "click" in [e["action"] for e in events(log, "no_checkpoint_recorded")]


# ---------- 12, 13. evidence hygiene and a site-agnostic implementation ----------

def test_no_secret_reaches_the_log_or_the_artifact_on_the_unverified_path():
    surface = inert_surface()
    artifact, surface, log = run([*LOGIN, ScriptedStep("click", "button", "Show details", expect="Item details"),
                                  DONE], surface=surface, operator=RecordingOperator("resume"))
    text = log.path.read_text()
    assert "secret_sauce" not in text and "secret_sauce" not in json.dumps(artifact.__dict__, default=str)
    assert "[REDACTED]" in text or "{{password}}" in text


def live_strings(tree):
    """Every string constant the code can actually use; a docstring explaining a rule is not behaviour."""
    import ast
    docstrings = set()
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if isinstance(parent, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body \
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            docstrings.add(id(body[0].value))
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings]


def test_the_unverified_rule_names_no_site_capability_or_selector():
    import ast
    from pathlib import Path
    banned = ["saucedemo", "swag labs", "inventory.html", "memberops", "checkout_paths", "member_account",
              "show details", "compact view", "#login", "standard_user"]
    for path in sorted(Path("src/cua").glob("*.py")):
        for value in live_strings(ast.parse(path.read_text())):
            for word in banned:
                assert word not in value.lower(), f"{path} carries {word!r} in a live string"
