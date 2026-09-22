"""A control that became selected proves its own click, by browser state and never by styling.

Many controls neither navigate, nor commit a field, nor change any text: clicking one simply selects it.
That transition is deterministic, so it is proof, but only from state the browser itself reports: native
checked or selected, the ARIA state a control publishes, a boolean state attribute a custom control sets,
or a control structurally associated with it. Only false-to-true counts. Offline; no model, no live site.
"""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import discover
from src.cua.artifact import linear_path, validate
from src.cua.escalation import NoOperator
from src.cua.replay import replay
from tests.context import ENTRY, HOSTS, PARAMS, Escalator, Policy, RecordingOperator, RunLog, SessionControl
from tests.fake_surface import FakeSurface
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

CONTROL = "FY 2025"
GROUP = "Time Period"


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


def surface_with(**spec):
    surface = FakeSurface()
    surface.selectable[CONTROL] = {"group": GROUP, "source": "aria-pressed", "selected": False, **spec}
    return surface


def script(expect="Show New Awards Only"):
    return [ScriptedStep("click", "button", CONTROL, context=GROUP, expect=expect),
            ScriptedStep("done", expect="Swag Labs")]


def run(surface, steps=None, operator=None):
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover(goal="choose a period", name="period_choice", params=dict(PARAMS), surface=surface,
                        planner=ScriptedPlanner(steps or script()), policy=Policy(allowed_hosts=HOSTS),
                        escalator=Escalator(operator or RecordingOperator("resume"), SessionControl(), log), log=log,
                        entry_url=ENTRY, sensitive={"password"}, max_steps=20)
    return artifact, log


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def chosen(artifact):
    return [n for n in linear_path(artifact)
            if n.action.action == "click" and n.action.target.strategies[0].get("name") == CONTROL]


# ---------- the transition is proof, across every evidence source ----------

@pytest.mark.parametrize("source", ["checked", "selected", "aria-pressed", "aria-checked", "aria-selected",
                                    "aria-current", "data-selected", "partner-checked", "partner-stored"])
def test_a_control_that_became_selected_proves_its_click_whatever_reports_it(source):
    artifact, log = run(surface_with(source=source))
    [node] = chosen(artifact)
    assert node.action.checkpoint == {"target_selected": "true"}
    assert events(log, "action_effect_proven")[0]["proof"] == f"the control is now selected ({source})"
    assert events(log, "action_effect_unverified") == []
    validate(artifact)


def test_the_recorded_checkpoint_resolves_the_original_locator_and_stores_no_markup_detail():
    artifact, _ = run(surface_with())
    [node] = chosen(artifact)
    assert node.action.checkpoint == {"target_selected": "true"}    # no class, no framework id, no attribute name
    rung = node.action.target.strategies[0]
    assert rung["kind"] == "role" and rung["role"] == "button" and rung["name"] == CONTROL
    assert not any("data-" in str(value) or "aria-" in str(value)
                   for strategy in node.action.target.strategies for value in strategy.values())


# ---------- the planner's expectation never decides whether a selection is proof ----------

def test_a_pre_existing_expected_text_does_not_hide_a_successful_selection():
    # The live shape: the planner expected text that was already on screen, so the text proves nothing.
    # The selection does, and it is what gets recorded instead of a null checkpoint.
    artifact, log = run(surface_with(), steps=[ScriptedStep("click", "button", CONTROL, context=GROUP,
                                                            expect="Swag Labs"),
                                               ScriptedStep("done", expect="Swag Labs")])
    [node] = chosen(artifact)
    assert node.action.checkpoint == {"target_selected": "true"}
    assert events(log, "checkpoint_not_evidence")[0]["reason"] == "the text was already on screen before acting"
    assert events(log, "action_effect_proven")[0]["proof"] == "the control is now selected (aria-pressed)"
    assert node.retry_safety == "verify_before_retry"          # it carries proof, so replay may verify first
    validate(artifact)


def test_the_planner_is_told_the_selection_stood_in_for_its_expectation():
    from tests.scripted_planner import ScriptedPlanner as Base

    class Spy(Base):
        def __init__(self, script):
            super().__init__(script)
            self.results: list[str] = []

        def decide(self, goal, params, observation, history, candidates=()):
            self.results = [a.result or "" for a in history]
            return super().decide(goal, params, observation, history, candidates)

    spy = Spy([ScriptedStep("click", "button", CONTROL, context=GROUP, expect="Swag Labs"),
               ScriptedStep("done", expect="Swag Labs")])
    log = RunLog("discovery", secrets=("secret_sauce",))
    discover(goal="choose a period", name="period_choice", params=dict(PARAMS), surface=surface_with(),
             planner=spy, policy=Policy(allowed_hosts=HOSTS),
             escalator=Escalator(RecordingOperator("resume"), SessionControl(), log), log=log, entry_url=ENTRY,
             sensitive={"password"}, max_steps=20)
    told = [r for r in spy.results if "already on screen" in r]
    assert told and "recorded instead" in told[-1] and "is now selected" in told[-1]


def test_a_selection_with_no_expectation_at_all_is_still_recorded_as_proof():
    artifact, log = run(surface_with(), steps=[ScriptedStep("click", "button", CONTROL, context=GROUP),
                                               ScriptedStep("done", expect="Swag Labs")])
    [node] = chosen(artifact)
    assert node.action.checkpoint == {"target_selected": "true"}
    assert events(log, "checkpoint_not_evidence") == []        # nothing was expected, so nothing was rejected


def test_genuinely_new_expected_text_still_wins_over_the_selection():
    # New text describes the screen the next step acts on, so it stays the checkpoint when it is real proof.
    # The control both selects itself and reveals a banner; the banner was not on screen before the click.
    surface = surface_with(reveals="Period chosen")
    artifact, log = run(surface, steps=[ScriptedStep("click", "button", CONTROL, context=GROUP,
                                                     expect="Period chosen"),
                                        ScriptedStep("done", expect="Swag Labs")])
    [node] = chosen(artifact)
    assert node.action.checkpoint == {"text_contains": "Period chosen"}
    assert events(log, "checkpoint_not_evidence") == []
    assert events(log, "action_effect_proven") == []       # the text was proof; the selection was never asked


def test_a_failed_expectation_plus_a_successful_selection_still_records_the_selection():
    artifact, log = run(surface_with(), steps=[ScriptedStep("click", "button", CONTROL, context=GROUP,
                                                            expect="Never shown anywhere"),
                                               ScriptedStep("done", expect="Swag Labs")])
    [node] = chosen(artifact)
    assert node.action.checkpoint == {"target_selected": "true"}
    assert events(log, "expectation_failed")[0]["expected"] == "Never shown anywhere"
    assert events(log, "action_effect_proven")[0]["proof"] == "the control is now selected (aria-pressed)"


def test_the_selection_is_evaluated_and_logged_only_once_per_action():
    artifact, log = run(surface_with(), steps=[ScriptedStep("click", "button", CONTROL, context=GROUP,
                                                            expect="Swag Labs"),
                                               ScriptedStep("done", expect="Swag Labs")])
    assert len(events(log, "action_effect_proven")) == 1
    assert len(chosen(artifact)) == 1


def test_a_refused_selection_is_reported_once_even_though_both_paths_ask():
    artifact, log = run(surface_with(toggles=False), steps=[ScriptedStep("click", "button", CONTROL, context=GROUP,
                                                                         expect="Swag Labs"),
                                                            ScriptedStep("done", expect="Swag Labs")],
                        operator=RecordingOperator("resume"))
    assert len(events(log, "selection_not_proven")) == 1
    assert chosen(artifact)[0].action.checkpoint is None       # recorded, but with nothing to verify


# ---------- what is not proof ----------

def test_a_control_that_was_already_selected_proves_nothing():
    artifact, log = run(surface_with(selected=True, toggles=False), operator=RecordingOperator("resume"))
    assert chosen(artifact) == []
    assert events(log, "selection_not_proven")[-1]["reason"] == "the control was already selected"
    assert events(log, "action_effect_unverified") != []


def test_a_control_whose_state_did_not_change_proves_nothing():
    artifact, log = run(surface_with(toggles=False), operator=RecordingOperator("resume"))
    assert chosen(artifact) == []
    assert events(log, "selection_not_proven")[-1]["reason"] == "the control is not selected"


def test_a_change_of_styling_alone_is_never_proof():
    artifact, log = run(surface_with(styling_only=True), operator=RecordingOperator("resume"))
    assert chosen(artifact) == []
    assert events(log, "action_effect_unverified") != []


def test_a_control_that_reports_no_selected_state_yields_no_proof_rather_than_a_false_negative():
    artifact, log = run(surface_with(unknown=True), operator=RecordingOperator("resume"))
    assert chosen(artifact) == []
    assert events(log, "selection_not_proven") == []        # nothing to compare: silently no proof
    assert events(log, "action_effect_unverified") != []


def test_a_reading_from_different_evidence_is_not_treated_as_a_transition():
    from src.cua.agent import selection_transition
    from src.cua.models import Action, Element

    log = RunLog("discovery")
    surface = surface_with(source="aria-selected", selected=True)
    action = Action(kind="click", target=Element(role="button", name=CONTROL, ref="r"))
    action.selected_before = {"known": True, "selected": False, "source": "data-selected", "ref": "r"}
    assert selection_transition(action, surface, log) is None
    assert events(log, "selection_not_proven")[-1]["reason"].startswith("the control's selected state is reported")


# ---------- discovery to replay ----------

def test_replay_verifies_the_selected_state_without_a_model():
    artifact, _ = run(surface_with())
    fresh = surface_with()
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, dict(PARAMS), fresh, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success" and fresh.selectable[CONTROL]["selected"] is True
    assert {"target_selected": "true"} in [e["checkpoint"] for e in events(log, "checkpoint_passed")]


def test_replay_fails_safely_when_the_control_never_becomes_selected():
    artifact, _ = run(surface_with())
    stuck = surface_with(toggles=False)
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, dict(PARAMS), stuck, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "failure" and result.outcome_code == "checkpoint_not_met"


def test_replay_fails_safely_when_the_control_is_absent():
    artifact, _ = run(surface_with())
    empty = FakeSurface()
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, dict(PARAMS), empty, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "failure" and result.outcome_code in ("target_not_found", "checkpoint_not_met")


def test_the_checkpoint_is_unmet_when_the_control_cannot_be_resolved_uniquely():
    # A ladder whose every rung is ambiguous resolves to nothing, so the selected state cannot be read and
    # the checkpoint is simply not met. Nothing is clicked and no element is guessed at.
    from tests.fake_surface import el
    from src.cua.models import GraphAction, GraphNode, Locator

    artifact, _ = run(surface_with())
    twins = surface_with()
    twin = el("button", CONTROL, context="Another panel")
    twin.ref = "button:twin"
    twins.extra_elements = [twin]
    log = RunLog("replay", secrets=("secret_sauce",))
    ctx = replay_context(artifact, twins, log)
    node = GraphNode(id="n", kind="action",
                     action=GraphAction(action="click",
                                        target=Locator(strategies=[{"kind": "role", "role": "button",
                                                                    "name": CONTROL}]),
                                        value=None, checkpoint={"target_selected": "true"}),
                     effect="reversible", retry_safety="verify_before_retry")
    from src.cua.replay import checkpoint_met
    assert checkpoint_met({"target_selected": "true"}, twins.observe(), dict(PARAMS), ctx, node) is False
    assert events(log, "target_selected_unmet")[-1]["reason"] == "the control is ambiguous"


def replay_context(artifact, surface, log):
    from src.cua.replay import ReplayContext
    return ReplayContext(artifact=artifact, params=dict(PARAMS), surface=surface,
                         policy=Policy(allowed_hosts=HOSTS),
                         escalator=Escalator(NoOperator(), SessionControl(), log), log=log)


def test_a_surface_that_cannot_report_selectedness_fails_the_checkpoint_rather_than_passing_it():
    from src.cua.replay import target_selected
    assert target_selected("true", None, None, None) is False


# ---------- the earlier guarantees still hold ----------

def test_a_wrong_expectation_still_cannot_prove_an_action_on_its_own():
    # Nothing about this control is selectable, so a contradicted expectation proves nothing at all.
    surface = FakeSurface()
    surface.inert.add(CONTROL)
    artifact, log = run(surface, steps=[ScriptedStep("click", "button", CONTROL, expect="Never shown"),
                                        ScriptedStep("done", expect="Swag Labs")],
                        operator=RecordingOperator("resume"))
    assert chosen(artifact) == []
    assert events(log, "action_effect_unverified")[0]["cause"] == "action_effect_unverified"


def test_an_unproven_selection_click_is_never_dispatched_twice():
    surface = surface_with(toggles=False)
    blind = ScriptedStep("click", "button", CONTROL, context=GROUP, expect="Never shown")
    artifact, log = run(surface, steps=[blind, blind, ScriptedStep("done", expect="Swag Labs")],
                        operator=RecordingOperator("resume"))
    assert surface.actions.count(("click", CONTROL, GROUP)) == 1
    assert events(log, "unverified_repeat_refused") != []


def test_the_state_token_vocabulary_is_small_closed_and_unambiguous():
    """The words that may stand for selectedness, and the ones deliberately kept out.

    Whole tokens only: the page script compares against this list with an exact match, so a token that
    merely contains one of these words is a different token. `active` is excluded on purpose, because it
    equally means focused, running, enabled or hovered.
    """
    from src.cua.surface import STATE_TOKENS

    assert set(STATE_TOKENS) == {"selected", "is-selected", "checked", "is-checked",
                                 "chosen", "is-chosen", "current", "is-current"}
    for ambiguous in ("active", "is-active", "on", "open", "focus", "hover", "highlight", "primary"):
        assert ambiguous not in STATE_TOKENS
    for trap in ("unselected", "selected-item", "button-selected-style", "deselected", "preselected"):
        assert trap not in STATE_TOKENS               # exact matching is what keeps these out


def test_the_page_script_matches_state_tokens_exactly_and_never_as_substrings():
    from src.cua.surface import SELECTED_JS

    # Only the selection logic itself, not the shared helpers it is prepended with (those read computed
    # style for visibility, which is a different question from whether something is selected).
    body = SELECTED_JS.split("const STATE_TOKENS", 1)[1]
    # The comparison is an exact list membership test, not a substring or prefix search.
    assert "STATE_TOKENS.includes(token)" in body
    assert "indexOf" not in body and "startsWith" not in body and "includes(" not in body.replace(
        "STATE_TOKENS.includes(token)", "")
    assert "getComputedStyle" not in body             # styling is never consulted
    assert "className" not in body                    # the class string is never read as a string
    assert "classList" in body                        # tokens are read as tokens


def test_the_selection_rule_names_no_site_capability_or_selector():
    import ast
    from pathlib import Path
    banned = ["usaspending", "saucedemo", "fy 2025", "time period", "advanced search", "prime awards",
              "show new awards only", "fiscal year"]
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
