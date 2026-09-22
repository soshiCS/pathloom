"""The select action, offline: choosing an entry in a chooser is not typing; a typed value nobody accepted is not
a selection; the artifact keeps placeholders; replay selects with no model."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import DiscoveryFailed, discover
from src.cua.artifact import ArtifactError, classify_effect, classify_retry_safety, linear_path, to_dict, validate
from src.cua.escalation import NoOperator
from src.cua.models import GraphAction, GraphNode, Locator
from src.cua.policy import Policy
from src.cua.replay import replay
from src.cua.surface import match_option, matching_suggestion, pick_option
from tests.context import ENTRY, HOSTS, PARAMS, Action, Element, Escalator, RecordingOperator, RunLog, SessionControl
from tests.fake_surface import FakeSurface, el
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

LOCATION = el("combobox", "Location")
CHOICES = ["Boston, Massachusetts", "Boston, Lincolnshire, United Kingdom", "Bostonia, California"]
PARAMS_WITH_LOCATION = {**PARAMS, "location": "Boston, Massachusetts"}


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


def chooser(options=CHOICES, faults=None, extra=()) -> FakeSurface:
    surface = FakeSurface(extra_elements=[LOCATION, *extra], faults=faults)
    surface.options = {"Location": list(options)}
    return surface


def run(script, surface, params=None, operator=None):
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover(goal="pick the location", name="pick_location", params=dict(params or PARAMS_WITH_LOCATION),
                        surface=surface, planner=ScriptedPlanner(script), policy=Policy(allowed_hosts=HOSTS),
                        escalator=Escalator(operator or NoOperator(), SessionControl(), log), log=log, entry_url=ENTRY,
                        sensitive={"password"}, max_steps=10)
    return artifact, log


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


SELECT = ScriptedStep("select", "combobox", "Location", value="{{location}}")
DONE = ScriptedStep("done", expect="Swag Labs")


def test_a_matching_suggestion_is_selected_and_recorded_with_its_placeholder():
    surface = chooser()
    artifact, log = run([SELECT, DONE], surface)
    validate(artifact)
    assert surface.selected == {"Location": "Boston, Massachusetts"} and surface.actions[-1][0] == "select"
    node = linear_path(artifact)[1]
    assert node.action.action == "select" and node.action.value == "{{location}}"
    assert node.action.target.strategies[0] == {"kind": "role", "role": "combobox", "name": "Location"}
    assert (node.effect, node.retry_safety) == ("reversible", "never_retry")
    assert "Boston" not in json.dumps(to_dict(artifact))
    assert [e["value"] for e in events(log, "acted")] == ["[typed]"]           # the value never reaches the log


def test_replay_selects_another_value_with_no_model():
    artifact, _ = run([SELECT, DONE], chooser())
    fresh = chooser()
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, {**PARAMS, "location": "Bostonia"}, fresh, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success" and fresh.selected == {"Location": "Bostonia, California"}
    assert [e["value"] for e in events(log, "acted") if e["action"] == "select"] == ["[typed]"]


def test_a_value_no_suggestion_matches_is_typed_but_never_a_selection():
    surface = chooser()
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run([SELECT], surface, params={**PARAMS, "location": "Paris"}, operator=operator)
    assert surface.typed["Location"] == "Paris" and surface.selected == {}
    assert operator.requests[0].reason.startswith("planner is stuck: select on combobox 'Location' ended in an unknown state")


def test_an_uncertain_post_action_failure_is_handed_off_and_never_repeated():
    surface = chooser(faults=["select_unknown:Location"])
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run([SELECT, SELECT], surface, operator=operator)
    assert sum(1 for a in surface.actions if a[0] == "select") == 1


def test_a_control_that_is_not_a_chooser_is_refused_before_acting():
    surface = FakeSurface(extra_elements=[LOCATION])                          # no options configured: not a chooser
    artifact, log = run([SELECT, ScriptedStep("type", "textbox", "Username", value="{{username}}"), DONE], surface)
    [failed] = events(log, "action_failed")
    assert failed["performed"] == "no" and "not an enabled chooser" in failed["reason"]
    assert [n.action.action for n in linear_path(artifact)] == ["navigate", "type"]


def test_option_matching_needs_exactly_one_match_at_a_level_and_never_picks_by_order():
    options = [{"label": "Boston, Lincolnshire"}, {"label": "boston, massachusetts"}, {"label": "Bostonia"},
               {"label": "Boston, Massachusetts", "disabled": True}]
    assert match_option(options, "Boston, Massachusetts") == ("match", options[1], "exact")   # unique exact, case aside
    assert match_option(options, "Boston") == ("ambiguous", None, "prefix")                    # two prefix matches
    assert match_option(options, "Lincoln") == ("match", options[0], "containing")
    assert match_option(options, "Paris") == ("none", None, "")
    duplicates = [{"label": "Boston"}, {"label": "Boston"}]
    assert match_option(duplicates, "Boston") == ("ambiguous", None, "exact") and pick_option(duplicates, "Boston") is None
    similar = [{"label": "Springfield, Illinois"}, {"label": "Springfield, Illinois West"}, {"label": "Springfield, Illinois East"}]
    assert match_option(similar, "Springfield, Illinois") == ("match", similar[0], "exact")   # exact beats the prefixes


def test_an_ambiguous_selection_is_told_to_the_planner_and_nothing_is_recorded():
    surface = chooser(["Springfield, Illinois", "Springfield, Illinois"])
    artifact, log = run([ScriptedStep("select", "combobox", "Location", value="Springfield, Illinois"),
                         ScriptedStep("type", "textbox", "Username", value="{{username}}"), DONE], surface)
    [failed] = events(log, "action_failed")
    assert failed["performed"] == "yes" and failed["cause"] == "ambiguous_selection"
    assert surface.selected == {} and [n.action.action for n in linear_path(artifact)] == ["navigate", "type"]
    assert surface.typed["Location"] == "Springfield, Illinois"                 # typed, never chosen


def test_the_same_ambiguity_twice_on_a_screen_hands_off():
    surface = chooser(["Boston, Lincolnshire", "Boston, Massachusetts"])
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run([ScriptedStep("select", "combobox", "Location", value="Boston")] * 2, surface, operator=operator)
    assert operator.requests[0].reason.startswith("planner is stuck: the same select on combobox 'Location' was ambiguous 2 times")


def test_multiple_unlinked_popups_are_ambiguous_too():
    surface = chooser()
    surface.popups["Location"] = 2
    artifact, log = run([SELECT, ScriptedStep("type", "textbox", "Username", value="{{username}}"), DONE], surface)
    [failed] = events(log, "action_failed")
    assert failed["cause"] == "ambiguous_selection" and "2 unrelated popups" in failed["reason"] and surface.selected == {}


def test_replay_fails_structurally_on_ambiguity_instead_of_choosing_or_retrying():
    artifact, _ = run([SELECT, DONE], chooser())
    ambiguous = chooser(["Boston, Massachusetts", "Boston, Massachusetts"])
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, dict(PARAMS_WITH_LOCATION), ambiguous, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "failure" and result.outcome_code == "ambiguous_selection" and result.step_id == "s2"
    assert result.expected == "exactly one option matching the value" and "exact match" in result.observed
    assert ambiguous.selected == {} and sum(1 for a in ambiguous.actions if a[0] == "select") == 1
    assert events(log, "ambiguous_selection")[0]["performed"] == "yes" and result.performed_attempts == 2   # navigate + the typing


def test_the_artifact_contract_and_policy_know_select():
    action = GraphAction(action="select", target=Locator(strategies=[{"kind": "role", "role": "combobox", "name": "Location"}]),
                         value="{{location}}")
    assert classify_effect("select", "safe") == "reversible" and classify_retry_safety("select", "reversible", None) == "never_retry"
    assert classify_retry_safety("select", "reversible", {"text_contains": "x"}) == "verify_before_retry"
    artifact, _ = run([SELECT, DONE], chooser())
    node = next(n for n in artifact.nodes if n.action and n.action.action == "select")
    node.action.value = None
    with pytest.raises(ArtifactError, match="select needs a value"):
        validate(artifact)
    policy = Policy(allowed_hosts=HOSTS)
    verdict = policy.check(Action(kind="select", target=Element(role="combobox", name="Location"), value="Paris"), ENTRY)
    assert verdict.decision == "allow" and policy.risk_of(Action(kind="select", target=LOCATION, value="x")) == "safe"


# ---------- a displayed gloss must not defeat the caller's plain name ----------

AGENCIES = ["Environmental Protection Agency (EPA)", "Agriculture Department (USDA)", "Department of Energy (DOE)"]


def test_a_trailing_acronym_does_not_stop_the_plain_name_matching():
    status, option, level = match_option([{"label": text} for text in AGENCIES], "Environmental Protection Agency")
    assert (status, option["label"], level) == ("match", AGENCIES[0], "exact without a trailing bracket")
    for shown, wanted in [("Boston, MA (Suffolk)", "Boston, MA"), ("Inbox [12]", "Inbox"), ("Team {archived}", "Team"),
                          ("Downloads (3 files)", "Downloads")]:
        assert match_option([{"label": shown}], wanted)[0] == "match", shown
    # only a trailing bracket is forgiven: a gloss in the middle of the text is not stripped
    assert match_option([{"label": "Energy (DOE) office"}], "Energy")[2] == "prefix"
    # the acronym itself still matches only through the ordinary containing level, never as the name
    assert match_option([{"label": "Environmental Protection Agency (EPA)"}], "EPA")[2] == "containing"


def test_the_gloss_rule_never_picks_by_document_order():
    both = [{"label": "Energy Department (DOE)"}, {"label": "Energy Department (2 offices)"}]
    assert match_option(both, "Energy Department") == ("ambiguous", None, "exact without a trailing bracket")
    assert pick_option(both, "Energy Department") is None
    # an exact display still beats a glossed one, so a real duplicate is the only ambiguity
    assert match_option([{"label": "Energy Department"}, {"label": "Energy Department (DOE)"}],
                        "Energy Department")[2] == "exact"


def test_a_glossed_suggestion_is_selected_end_to_end_and_replays():
    surface = chooser(AGENCIES)
    artifact, log = run([ScriptedStep("select", "combobox", "Location", value="{{location}}"), DONE], surface,
                        params={**PARAMS, "location": "Environmental Protection Agency"})
    assert surface.selected == {"Location": AGENCIES[0]}
    node = linear_path(artifact)[1]
    assert node.action.action == "select" and node.action.value == "{{location}}"
    fresh = chooser(AGENCIES)
    replay_log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, {**PARAMS, "location": "Department of Energy"}, fresh, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success" and fresh.selected == {"Location": AGENCIES[2]}


# ---------- a failed selection is never proven by the screen ----------

def test_typed_text_and_a_visible_suggestion_never_count_as_an_accepted_selection():
    # the chooser offers nothing matching, so the value is only typed; the text is on screen either way
    surface = chooser(["Some other agency"], extra=[el("text", "", "Environmental Protection Agency")])
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run([ScriptedStep("select", "combobox", "Location", value="{{location}}", expect="{{location}}")], surface,
            params={**PARAMS, "location": "Environmental Protection Agency"}, operator=operator)
    assert surface.selected == {} and surface.typed["Location"] == "Environmental Protection Agency"
    assert operator.requests[0].reason.startswith("planner is stuck: select on combobox 'Location' ended in an "
                                                  "unknown state")


def test_no_select_node_is_recorded_after_a_failed_acceptance():
    surface = chooser(["Some other agency"])
    artifact, log = run([ScriptedStep("select", "combobox", "Location", value="{{location}}", expect="{{location}}"),
                         ScriptedStep("type", "textbox", "Username", value="{{username}}"), DONE], surface,
                        params={**PARAMS, "location": "Environmental Protection Agency"},
                        operator=RecordingOperator("resume"))     # a person looked and let automation carry on
    assert [n.action.action for n in linear_path(artifact)] == ["navigate", "type"]
    assert events(log, "action_proven_after_failure") == []
    [refused] = events(log, "checkpoint_not_evidence")
    assert refused["kind"] == "select" and "cannot be proven from the screen" in refused["reason"]


# ---------- several plausible popups: the one holding the match decides ----------

def shown(groups, linked=False):
    rows = [[{"text": text} for text in group] for group in groups]
    return {"linked": linked, "popups": len(groups), "options": [o for g in rows for o in g], "groups": rows}


def test_only_the_container_holding_the_requested_option_is_used():
    three = shown([["Some other agency"], ["Environmental Protection Agency (EPA)"], ["Yet another thing"]])
    status, option, level = matching_suggestion(three, "Environmental Protection Agency")
    assert (status, option["text"]) == ("match", "Environmental Protection Agency (EPA)")
    assert level == "exact without a trailing bracket"


def test_matches_in_several_containers_are_ambiguous():
    both = shown([["Environmental Protection Agency (EPA)"], ["Environmental Protection Agency (region 2)"], ["Other"]])
    assert matching_suggestion(both, "Environmental Protection Agency")[:2] == ("ambiguous_popups", None)


def test_an_ambiguity_inside_the_matching_container_is_still_ambiguous():
    inside = shown([["Other"], ["Energy Department (DOE)", "Energy Department (2 offices)"]])
    status, option, level = matching_suggestion(inside, "Energy Department")
    assert (status, option) == ("ambiguous", None) and level == "exact without a trailing bracket"


def test_no_container_holding_the_option_is_simply_no_match():
    assert matching_suggestion(shown([["Other"], ["Different"]]), "Environmental Protection Agency")[0] == "none"
    assert matching_suggestion(shown([]), "Anything")[0] == "none"


def test_a_linked_control_still_trusts_the_popup_it_names():
    named = shown([["Environmental Protection Agency (EPA)"], ["Environmental Protection Agency (other)"]], linked=True)
    # a named popup's options are matched as one list, so two matches there are an ordinary ambiguity
    assert matching_suggestion(named, "Environmental Protection Agency")[0] == "ambiguous"


# ---------- distinct same-named elements stay distinct through recording ----------

def test_three_same_named_links_record_three_distinct_targets_and_replay_one_to_one():
    from src.cua.artifact import validate
    from src.cua.surface import is_ambiguous, locator_for

    def agency_rows():
        rows = [el("link", "Environmental Protection Agency") for _ in range(3)]
        for index, row in enumerate(rows, start=1):
            row.ref = f"li:nth-of-type({index}) > a:nth-of-type(1)"
            row.context = "Environmental Protection Agency"
        return rows

    rows = agency_rows()
    surface = FakeSurface(extra_elements=rows)
    contract = {"agencies": {"type": "list", "required": True, "min_items": 3, "max_items": 3,
                             "items": {"type": "string", "pattern": "(.+)"}}}
    class RowPlanner(ScriptedPlanner):
        """Picks the three same-named links by their own references, as a real planner picking rows would."""

        def decide(self, goal, params, observation, history, candidates=()):
            if not history:
                picked = [e for e in observation.elements if e.ref.startswith("li:nth-of-type")]
                return Action(kind="extract_many", targets=picked, output_name="agencies", pattern="(.+)",
                              reason="scripted")
            return super().decide(goal, params, observation, history, candidates)

    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover(goal="read the agencies", name="agency_list", params={"agency": "Environmental Protection Agency"},
                        surface=surface, planner=RowPlanner([ScriptedStep("done", expect="Swag Labs")]),
                        policy=Policy(allowed_hosts=HOSTS), escalator=Escalator(NoOperator(), SessionControl(), log),
                        log=log, entry_url=ENTRY, output_contract=contract)
    validate(artifact)                                   # the duplicate-target rule still applies and passes
    node = linear_path(artifact)[-1]
    ladders = node.action.targets
    assert len(ladders) == 3 and len({tuple(map(str, l.strategies)) for l in ladders}) == 3
    assert [l.strategies[0] for l in ladders] == [{"kind": "role", "role": "link", "name": "{{agency}}",
                                                   "context": "{{agency}}"}] * 3
    assert [l.strategies[-1]["selector"] for l in ladders] == [r.ref for r in rows]
    assert artifact.outputs["agencies"]["example"] == ["{{agency}}"] * 3       # the same value three times is fine

    fresh = FakeSurface(extra_elements=agency_rows())
    replay_log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, {"agency": "Environmental Protection Agency"}, fresh, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success"
    assert result.outputs == {"agencies": ["Environmental Protection Agency"] * 3}
    resolved = [json.loads(l) for l in replay_log.path.read_text().splitlines() if '"target_resolved"' in l]
    assert [e["target_index"] for e in resolved if "target_index" in e] == [0, 1, 2]


def test_a_single_target_keeps_css_only_for_guarded_disambiguation_at_replay():
    from src.cua.artifact import parameterize_locator
    from src.cua.models import Locator

    ladder = Locator(strategies=[{"kind": "role", "role": "link", "name": "Environmental Protection Agency"},
                                 {"kind": "css", "selector": "a:nth-of-type(1)"},
                                 {"kind": "coords", "x": 5, "y": 5}])
    params = {"agency": "Environmental Protection Agency"}
    assert [s["kind"] for s in parameterize_locator(ladder, params).strategies] == ["role", "css"]
    assert [s["kind"] for s in parameterize_locator(ladder, params, keep_structure=True).strategies] == ["role", "css"]


# ---------- keyboard commit: the proof rule, offline ----------

def test_a_committed_field_is_proven_only_by_state_that_is_not_the_typed_query():
    from src.cua.surface import commit_proof

    before = {"value": "", "active_text": "", "stored": [], "chips": []}
    typed = "10 Example Street"
    resolved = {"value": "10 Example Street, Springfield", "active_text": "", "stored": [], "chips": []}
    assert commit_proof(resolved, before, typed) == "the field resolved into a different value"
    echoed = {"value": typed, "active_text": "", "stored": [], "chips": []}
    assert commit_proof(echoed, before, typed) is None                       # the raw query proves nothing
    assert commit_proof({"value": "", "active_text": "", "stored": [], "chips": []}, before, typed) is None
    stored = {"value": "", "active_text": "", "stored": ["id-4821"], "chips": []}
    assert commit_proof(stored, before, typed) == "a stored value"
    assert commit_proof({"value": "", "active_text": "", "stored": [typed], "chips": []}, before, typed) is None
    chipped = {"value": "", "active_text": "", "stored": [], "chips": ["10 Example Street ×"]}
    assert commit_proof(chipped, before, typed) == "a selected chip"
    active = {"value": "", "active_text": "10 Example Street, Springfield", "stored": [], "chips": []}
    assert commit_proof(active, before, typed) == "the active descendant"
    # state that was already there is never fresh proof
    had_chip = {"value": "", "active_text": "", "stored": [], "chips": ["10 Example Street ×"]}
    assert commit_proof(had_chip, {**before, "chips": ["10 Example Street ×"]}, typed) is None


def test_the_fake_chooser_still_refuses_a_value_no_option_matches():
    surface = chooser(["Some other agency"])
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run([SELECT], surface, params={**PARAMS, "location": "Nothing like it"}, operator=operator)
    assert surface.selected == {}
