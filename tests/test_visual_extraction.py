"""Grounded visual extraction, offline: the vision planner only says where to look; the page element under each
box supplies the value, the declared contract parses it, and the recorded node replays with no model."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import DiscoveryFailed, discover
from src.cua.artifact import linear_path, validate
from src.cua.escalation import NoOperator
from src.cua.models import VisualDecision
from src.cua.planner import MAX_VISUAL_BOXES, action_from_tool_input, visual_decision_from_tool_input
from src.cua.replay import replay
from tests.context import ENTRY, HOSTS, PARAMS, Escalator, Observation, Policy, RecordingOperator, RunLog, SessionControl
from tests.fake_surface import VIEWPORT, FakeSurface
from tests.scripted_planner import MONEY, ScriptedStep, ScriptedVisionPlanner, checkout_script, visual_click, visual_extract

TOTALS = {"totals": {"type": "list", "required": True, "min_items": 3, "max_items": 3,
                     "items": {"type": "number", "pattern": MONEY}}}
BOXES = [(10, 10, 50, 20), (70, 10, 50, 20), (130, 10, 50, 20)]
GROUNDING = {BOXES[0]: "Item total:", BOXES[1]: "Tax:", BOXES[2]: "Total:"}
OVERVIEW = checkout_script()[:11]                       # login .. Continue: the overview screen is reached
MISSING = ScriptedStep("stuck", stuck_cause="missing_data", output_name="totals")
DONE = ScriptedStep("done", expect="Checkout: Overview")


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


def run(script, decisions, surface=None, contract=TOTALS, operator=None, max_attempts=2):
    planner = ScriptedVisionPlanner(script, decisions)
    surface = surface or grounded_surface()
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover(goal="read the totals", name="checkout_totals", params=dict(PARAMS), surface=surface,
                        planner=planner, policy=Policy(allowed_hosts=HOSTS),
                        escalator=Escalator(operator or NoOperator(), SessionControl(), log), log=log, entry_url=ENTRY,
                        sensitive={"password"}, max_steps=25, vision=planner, max_vision_attempts=max_attempts,
                        output_contract=contract)
    return artifact, planner, surface, log


def grounded_surface(grounding=None):
    surface = FakeSurface()
    surface.grounding = dict(GROUNDING if grounding is None else grounding)
    return surface


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


# ---------- the trust boundary: boxes select page elements, the page supplies the values ----------

def test_the_page_supplies_the_values_and_the_model_reading_is_ignored():
    artifact, planner, surface, log = run(OVERVIEW + [MISSING, DONE],
                                          [visual_extract("totals", BOXES, readings=["999", "998", "997"])])
    validate(artifact)
    assert artifact.outputs["totals"]["example"] == ["29.99", "2.40", "32.39"]     # read from the elements
    text = log.path.read_text()
    assert "999" not in text and "998" not in text and "997" not in text
    node = linear_path(artifact)[-1]
    assert node.action.action == "extract_many" and node.action.value == "totals" and len(node.action.targets) == 3
    first = node.action.targets[0].strategies
    assert first[0] == {"kind": "text", "text": "Item total:"}                    # a semantic rung first
    assert first[1]["kind"] == "css" and first[-1]["kind"] == "coords"
    assert first[-1] == {"kind": "coords", "x": 35, "y": 20, "exact": True, "read": True,
                         "viewport": {"width": VIEWPORT[0], "height": VIEWPORT[1]}, "scroll": {"x": 0, "y": 0}}
    provenance = artifact.provenance["vision_fallback"]
    assert provenance["attempts"] == 1 and provenance["steps"] == [node.id]
    assert provenance["grounded_extractions"] == [{"step_id": node.id, "output": "totals", "boxes": 3,
                                                   "ladders": [["text", "css", "coords"]] * 3}]
    assert "png" not in json.dumps(provenance).lower() and "secret_sauce" not in json.dumps(artifact.provenance)
    assert events(log, "vision_extraction_grounded") == [{**events(log, "vision_extraction_grounded")[0],
                                                          "output_name": "totals", "boxes": 3, "grounded": [True] * 3}]


def test_three_grounded_values_become_one_ordered_list_that_replays_without_a_model():
    artifact, _, _, _ = run(OVERVIEW + [MISSING, DONE], [visual_extract("totals", BOXES)])
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, dict(PARAMS), FakeSurface(), Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success" and result.outputs == {"totals": [29.99, 2.40, 32.39]}
    assert '"vision' not in log.path.read_text()


def test_a_box_without_a_page_backed_value_rejects_the_whole_proposal_and_hands_off():
    surface = grounded_surface({**GROUNDING, BOXES[1]: None})              # the middle box covers nothing textual
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run(OVERVIEW + [MISSING], [visual_extract("totals", BOXES)], surface=surface, operator=operator)
    assert operator.requests[0].reason.startswith("planner is stuck")
    assert surface.actions[-1] == ("click", "Continue", "")                   # nothing else was touched


def test_a_grounded_value_that_does_not_parse_under_the_contract_records_nothing():
    strict = {"totals": {**TOTALS["totals"], "items": {"type": "number", "pattern": r"€\s*([\d.]+)"}}}
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run(OVERVIEW + [MISSING], [visual_extract("totals", BOXES)], contract=strict, operator=operator)
    assert "could not be parsed" in operator.requests[0].reason


# ---------- triggering ----------

def test_missing_data_needs_a_declared_unrecorded_output_and_a_matching_proposal():
    undeclared = ScriptedStep("stuck", stuck_cause="missing_data", output_name="fees")
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed):
        run(OVERVIEW + [undeclared], [visual_extract("fees", BOXES[:1])], operator=operator)
    assert operator.requests[0].reason.startswith("planner is stuck")

    recorded = ScriptedStep("extract_many", "text", texts=["Item total:", "Tax:", "Total:"], output_name="totals",
                            pattern=MONEY)
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed):
        _, planner, _, log = None, None, None, None
        run(OVERVIEW + [recorded, MISSING], [visual_extract("totals", BOXES)], operator=operator)

    for cause_step, decision, why in [
        (ScriptedStep("stuck", stuck_cause="missing_control"), visual_extract("totals", BOXES),
         "visual_extract does not answer a perception stuck state"),
        (MISSING, visual_click("Total", expect="Nothing"), "visual_click does not answer a missing_data stuck state"),
        (MISSING, visual_extract("fees", BOXES), "visual_extract names 'fees', not the missing output 'totals'"),
    ]:
        operator = RecordingOperator("abort")
        with pytest.raises(DiscoveryFailed):
            run(OVERVIEW + [cause_step], [decision], operator=operator)


def test_gate_reasons_are_logged_before_any_screenshot_is_taken():
    operator = RecordingOperator("abort")
    surface = grounded_surface()
    with pytest.raises(DiscoveryFailed):
        run(OVERVIEW + [ScriptedStep("stuck", stuck_cause="missing_data", output_name="fees")],
            [visual_extract("fees", BOXES[:1])], surface=surface, operator=operator)
    assert surface.viewport_shots == []


def test_a_scalar_output_takes_exactly_one_box_and_budgets_still_count():
    scalar = {"total": {"type": "number", "required": True, "pattern": MONEY}}
    one = ScriptedStep("stuck", stuck_cause="missing_data", output_name="total")
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed):
        run(OVERVIEW + [one], [visual_extract("total", BOXES[:2])], contract=scalar, operator=operator)
    artifact, _, _, log = run(OVERVIEW + [one, DONE], [visual_extract("total", BOXES[2:])], contract=scalar)
    assert artifact.outputs["total"]["example"] == "32.39" and linear_path(artifact)[-1].action.action == "extract"
    assert artifact.provenance["vision_fallback"]["attempts"] == 1
    assert len(events(log, "vision_fallback_requested")) == 1


def test_viewport_or_scroll_drift_since_the_capture_rejects_the_proposal(monkeypatch):
    surface = grounded_surface()
    monkeypatch.setattr(FakeSurface, "scroll_position", lambda self: (0, 120))
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed):
        run(OVERVIEW + [MISSING], [visual_extract("totals", BOXES)], surface=surface, operator=operator)
    assert operator.requests[0].reason.startswith("planner is stuck")


def test_replay_never_reads_from_a_coordinate_when_the_structure_is_gone_and_the_rung_cannot_be_honoured():
    artifact, _, _, _ = run(OVERVIEW + [MISSING, DONE], [visual_extract("totals", BOXES)])
    node = linear_path(artifact)[-1]
    for ladder in node.action.targets:
        ladder.strategies = [rung for rung in ladder.strategies if rung["kind"] == "coords"]    # only the exact point
    operator = RecordingOperator("abort")
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, dict(PARAMS), FakeSurface(), Policy(allowed_hosts=HOSTS),
                    Escalator(operator, SessionControl(), log), log)
    assert result.status == "failure" and result.outcome_code == "target_not_found"   # the fake cannot honour the point


# ---------- the planner boundary ----------

def test_visual_extract_tool_input_is_validated_and_readings_are_dropped():
    good = {"kind": "visual_extract", "output_name": "totals", "reason": "three totals", "role": "", "name": "",
            "x": 0, "y": 0, "width": 0, "height": 0, "value": None, "expect": None, "confidence": 0,
            "boxes": [{"x": 10, "y": 10, "width": 50, "height": 20, "reason": "item total", "confidence": 0.9,
                       "reading": "999"},
                      {"x": 70, "y": 10, "width": 50, "height": 20, "reason": "tax", "confidence": 0.8}]}
    decision = visual_decision_from_tool_input(good, 800, 600)
    assert decision.kind == "visual_extract" and decision.output_name == "totals" and len(decision.boxes) == 2
    assert [(b.x, b.y, b.width, b.height, b.expect) for b in decision.boxes] == [(10, 10, 50, 20, None), (70, 10, 50, 20, None)]
    assert "999" not in json.dumps(decision.__dict__, default=str)
    bad = [({**good, "output_name": None}, "needs an output_name"),
           ({**good, "boxes": []}, "needs at least one box"),
           ({**good, "boxes": [{**good["boxes"][0], "x": 790}]}, "not inside the 800 x 600 viewport"),
           ({**good, "boxes": [{**good["boxes"][0], "confidence": 0.2}]}, "below"),
           ({**good, "boxes": [{**good["boxes"][0], "width": "wide"}]}, "not a finite number"),
           ({**good, "boxes": [good["boxes"][0]] * (MAX_VISUAL_BOXES + 1)}, "exceed the limit")]
    for data, message in bad:
        rejected = visual_decision_from_tool_input(data, 800, 600)
        assert rejected.kind == "no_target" and message in rejected.rejected


def test_the_planner_reports_missing_data_as_its_own_stuck_cause():
    data = {"kind": "stuck", "stuck_cause": "missing_data", "output_name": "depths", "reason": "depths are drawn but not listed"}
    action = action_from_tool_input(data, Observation(url=ENTRY, elements=[]))
    assert action.kind == "stuck" and action.stuck_cause == "missing_data" and action.output_name == "depths"
    assert action_from_tool_input({**data, "stuck_cause": "other"}, Observation(url=ENTRY, elements=[])).stuck_cause == "planner"
