"""A proven interface dismissal whose own control is gone has demonstrably happened.

Closing a modal, a panel or a sidebar often changes no text the planner can predict, changes no address,
and leaves nothing selected: the one thing it reliably does is remove the thing it closed, including the
control that closed it. That is proof, but only for a control the safety rules already prove is a UI
dismissal. For an ordinary button, link or submission, and for anything destructive or ambiguous,
disappearance proves nothing. Offline, on the fake shop; no model and no live site.
"""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import discover
from src.cua.artifact import linear_path, validate
from src.cua.escalation import NoOperator
from src.cua.models import Action, Element, Observation
from src.cua.replay import replay
from tests.context import ENTRY, HOSTS, PARAMS, Escalator, Policy, RecordingOperator, RunLog, SessionControl
from tests.fake_surface import FakeSurface
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

PANEL = "Filters"
CLOSE = "Close"


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


def panel_surface(name=PANEL, control=CLOSE, **spec):
    surface = FakeSurface()
    surface.panels[name] = {"control": control, "body": f"{name} body", **spec}
    return surface


def script(expect="Award Amount", control=CLOSE, panel=PANEL):
    return [ScriptedStep("click", "button", control, context=panel, expect=expect),
            ScriptedStep("done", expect="Swag Labs")]


def run(surface, steps=None, operator=None):
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover(goal="close the filters", name="filter_close", params=dict(PARAMS), surface=surface,
                        planner=ScriptedPlanner(steps or script()), policy=Policy(allowed_hosts=HOSTS),
                        escalator=Escalator(operator or RecordingOperator("resume"), SessionControl(), log), log=log,
                        entry_url=ENTRY, sensitive={"password"}, max_steps=20)
    return artifact, log


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def closes(artifact, control=CLOSE):
    return [n for n in linear_path(artifact)
            if n.action.action == "click" and n.action.target.strategies[0].get("name") == control]


# ---------- the dismissal is proof ----------

@pytest.mark.parametrize("landmark, name", [("dialog", "Details"), ("region", "Filters"), ("menu", "Options")])
def test_closing_a_modal_panel_or_sidebar_is_proven_by_its_control_disappearing(landmark, name):
    surface = panel_surface(name=name, landmark=landmark)
    artifact, log = run(surface, steps=script(panel=name))
    [node] = closes(artifact)
    assert node.action.checkpoint == {"target_absent": "true"}
    assert events(log, "action_effect_proven")[0]["proof"] == "the dismissal's own control is gone"
    assert events(log, "action_effect_unverified") == []
    validate(artifact)


def test_the_panel_hiding_while_a_different_same_named_close_remains_is_still_proven():
    # A control elsewhere shares the name; it is not the control that was clicked, and the clicked one is
    # matched by its own structural reference rather than by its label.
    surface = panel_surface(twin=True)
    artifact, log = run(surface)
    [node] = closes(artifact)
    assert node.action.checkpoint == {"target_absent": "true"}
    assert any(e.name == CLOSE for e in surface.observe().elements)   # the twin is still there
    assert events(log, "dismissal_not_proven") == []


def test_a_dismissal_followed_by_delayed_loading_is_still_proven_at_once():
    # The results area is still loading, so no expected text has appeared. The dismissal is proven anyway.
    surface = panel_surface(reveals="Loading")
    artifact, log = run(surface, steps=script(expect="Award Amount"))
    [node] = closes(artifact)
    assert node.action.checkpoint == {"target_absent": "true"}
    assert events(log, "expectation_failed")[0]["expected"] == "Award Amount"


def test_a_dismissal_with_no_expectation_at_all_is_still_proven():
    artifact, log = run(panel_surface(), steps=[ScriptedStep("click", "button", CLOSE, context=PANEL),
                                                ScriptedStep("done", expect="Swag Labs")])
    [node] = closes(artifact)
    assert node.action.checkpoint == {"target_absent": "true"}


def test_a_dismissal_whose_expected_text_was_already_on_screen_is_proven_by_the_disappearance():
    artifact, log = run(panel_surface(), steps=script(expect="Swag Labs"))
    [node] = closes(artifact)
    assert node.action.checkpoint == {"target_absent": "true"}
    assert events(log, "checkpoint_not_evidence")[0]["reason"] == "the text was already on screen before acting"


# ---------- what disappearance never proves ----------

def test_an_ordinary_button_disappearing_after_a_rerender_is_not_proof():
    surface = FakeSurface()
    vanishing = Element(role="button", name="Apply", text="Apply", ref="button:Apply", context="")
    surface.extra_elements = [vanishing]

    original = surface.dispatch_click

    def rerender(target):
        if target.name == "Apply":
            surface.extra_elements = []          # the page re-rendered it away; it dismissed nothing
            return
        original(target)

    surface.dispatch_click = rerender
    artifact, log = run(surface, steps=[ScriptedStep("click", "button", "Apply", expect="Never shown"),
                                        ScriptedStep("done", expect="Swag Labs")],
                        operator=RecordingOperator("resume"))
    assert closes(artifact, "Apply") == []
    assert events(log, "action_effect_unverified")[0]["cause"] == "action_effect_unverified"


@pytest.mark.parametrize("control, panel", [("Close account", "Settings"), ("Close", "Delete workspace")])
def test_a_destructive_close_is_never_proven_by_vanishing(control, panel):
    # The safety rules stop these before the click even happens, which is the stronger guarantee; the
    # dismissal predicate refuses them too, so vanishing could not prove one even if it were dispatched.
    surface = panel_surface(name=panel, control=control)
    artifact, log = run(surface, steps=script(expect="Never shown", control=control, panel=panel),
                        operator=RecordingOperator("resume"))
    assert closes(artifact, control) == []
    assert events(log, "policy_checked")[0]["decision"] in ("confirm", "deny")
    assert not dismissal_verdict(surface, control, panel)


def test_an_unknown_close_with_no_dismissal_evidence_is_never_proven_by_vanishing():
    # No platform dismissal relationship, no dismissible container, no interface word in the wording.
    surface = panel_surface(name="Record 4471", dismisses=False, landmark="")
    artifact, log = run(surface, steps=script(expect="Never shown", panel="Record 4471"),
                        operator=RecordingOperator("resume"))
    assert closes(artifact) == []
    assert not dismissal_verdict(surface, CLOSE, "Record 4471")


def test_a_submission_control_is_never_a_dismissal_however_it_is_named():
    surface = panel_surface(control="Close", native="button:submit", dismisses=False)
    assert not dismissal_verdict(surface, CLOSE, PANEL)


def dismissal_verdict(surface, control, panel) -> bool:
    """What the policy says about the control as it is actually perceived on the fake screen."""
    element = next(e for e in surface.observe().elements
                   if e.name == control and e.context == panel)
    return Policy(allowed_hosts=HOSTS).dismisses_interface(Action(kind="click", target=element))


def test_a_control_still_on_screen_after_the_click_is_not_proven():
    surface = panel_surface()
    surface.dispatch_click = lambda target: None      # the click reaches the page and nothing moves
    artifact, log = run(surface, steps=script(expect="Never shown"), operator=RecordingOperator("resume"))
    assert closes(artifact) == []
    assert events(log, "dismissal_not_proven")[-1]["reason"] == "the control is still on screen"


def test_a_control_absent_before_the_click_can_never_have_disappeared():
    from src.cua.agent import dismissal_transition

    log = RunLog("discovery")
    target = Element(role="button", name=CLOSE, ref="panel:X:Close", landmark="dialog", dismisses=True)
    action = Action(kind="click", target=target)
    empty = Observation(url="about:blank", elements=[])
    assert dismissal_transition(action, empty, empty, Policy(allowed_hosts=HOSTS), log) is None
    # Nothing is even reported: there was no control to have disappeared, so no refusal arises.
    assert not log.path.exists() or events(log, "dismissal_not_proven") == []


def test_a_navigation_replacing_the_document_is_not_read_as_a_dismissal():
    # Everything vanishes when a page is replaced. The URL rung runs first, so that is what proves it.
    surface = panel_surface()
    original = surface.dispatch_click

    def navigate_away(target):
        if target.name == CLOSE:
            surface.dismissed.add(PANEL)
            surface.screen, surface.url = "inventory", ENTRY + "inventory.html"
            return
        original(target)

    surface.dispatch_click = navigate_away
    artifact, log = run(surface, steps=[ScriptedStep("click", "button", CLOSE, context=PANEL,
                                                     expect="Never shown"),
                                        ScriptedStep("done", expect="Products")])
    [node] = closes(artifact)
    assert node.action.checkpoint == {"url_contains": "/inventory.html"}
    assert events(log, "action_effect_proven")[0]["proof"] == "the page address changed"


# ---------- discovery to replay ----------

def test_replay_verifies_the_dismissal_without_a_model():
    artifact, _ = run(panel_surface())
    fresh = panel_surface()
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, dict(PARAMS), fresh, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success" and PANEL in fresh.dismissed
    assert {"target_absent": "true"} in [e["checkpoint"] for e in events(log, "checkpoint_passed")]


def test_replay_fails_safely_when_the_control_survives_the_click():
    artifact, _ = run(panel_surface())
    stuck = panel_surface()
    stuck.dispatch_click = lambda target: None
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, dict(PARAMS), stuck, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "failure" and result.outcome_code == "checkpoint_not_met"
    assert events(log, "target_absent_unmet")[-1]["reason"] == "the control is still on screen"


def test_replay_clicks_the_dismissal_exactly_once():
    artifact, _ = run(panel_surface())
    fresh = panel_surface()
    log = RunLog("replay", secrets=("secret_sauce",))
    replay(artifact, dict(PARAMS), fresh, Policy(allowed_hosts=HOSTS),
           Escalator(NoOperator(), SessionControl(), log), log)
    assert [a for a in fresh.actions if a[0] == "click" and a[1] == CLOSE] == [("click", CLOSE, PANEL)]


def test_an_unproven_dismissal_click_is_never_dispatched_twice():
    surface = panel_surface()
    surface.dispatch_click = lambda target: None
    blind = ScriptedStep("click", "button", CLOSE, context=PANEL, expect="Never shown")
    artifact, log = run(surface, steps=[blind, blind, ScriptedStep("done", expect="Swag Labs")],
                        operator=RecordingOperator("resume"))
    assert [a for a in surface.actions if a[0] == "click"].count(("click", CLOSE, PANEL)) == 1
    assert events(log, "unverified_repeat_refused") != []


# ---------- the artifact stays abstract ----------

def test_the_checkpoint_carries_no_selector_container_name_or_dismissal_metadata():
    artifact, _ = run(panel_surface())
    [node] = closes(artifact)
    assert node.action.checkpoint == {"target_absent": "true"}
    written = json.dumps([node.action.checkpoint, [dict(s) for s in node.action.target.strategies]]).lower()
    for token in ("dismiss", "landmark", "region", "dialog", "panel:"):
        assert token not in written


def test_the_dismissal_rule_names_no_site_capability_or_selector():
    import ast
    from pathlib import Path
    banned = ["usaspending", "saucedemo", "filter sidebar", "award amount", "advanced search",
              "search results", "prime awards"]
    for path in sorted(Path("src/cua").glob("*.py")):
        tree = ast.parse(path.read_text())
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                      and getattr(node, "body", None) and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                for word in banned:
                    assert word not in node.value.lower(), f"{path} carries {word!r} in a live string"
