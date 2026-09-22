"""Ordered multi-element extraction: the planner selects several elements for one list output, discovery
records one node with one ladder per element, and replay reads the same list with no model."""
import json
from dataclasses import replace

import pytest

from src.cua import agent as agent_module
from src.cua import artifact as artifact_module
from src.cua.agent import ActionError, discover, extraction_problem, perform
from src.cua.artifact import (ArtifactError, from_dict, linear_path, load_artifact, nodes_by_id, save_artifact, to_dict,
                              validate)
from src.cua.campaign import CampaignError, outputs_from_dict
from src.cua.escalation import NoOperator
from src.cua.library import build_library, execute_candidate, find_candidates, importable_nodes, verified_entries
from src.cua.merge import MergeError, merge_traces
from src.cua.models import Action, Element, GraphAction, Locator
from src.cua.planner import CHOOSE_ACTION_TOOL, OpenAIPlanner, action_from_tool_input
from src.cua.replay import replay
from tests.context import (ENTRY, HOSTS, PARAMS, Escalator, Observation, Policy, RecordingOperator, RunLog,
                           SessionControl, build_linear, checkout_artifact, checkout_nodes, linear_node, text_ladder)
from tests.fake_surface import FakeSurface, el
from tests.scripted_planner import ScriptedPlanner, ScriptedStep, ScriptedVisionPlanner, checkout_script
from tests.test_merge import link_trace, url_trace

MONEY = r"\$\s*([\d.]+)"
TOTALS = {"totals": {"type": "list", "required": True, "items": {"type": "number", "pattern": MONEY}}}


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)


def many_step(texts=("Item total:", "Tax:", "Total:"), output="totals", pattern=MONEY, optional=False):
    return ScriptedStep("extract_many", "text", texts=list(texts), output_name=output, pattern=pattern, optional=optional)


def totals_script(step=None):
    script = checkout_script()
    return script[:11] + [step or many_step(), ScriptedStep("done", expect="Checkout: Overview")]


def run(script, surface=None, planner=None, output_contract=None, vision=False, operator=None):
    planner = planner or (ScriptedVisionPlanner(script, []) if vision else ScriptedPlanner(script))
    surface = surface or FakeSurface()
    log = RunLog("discovery", secrets=("secret_sauce",))
    artifact = discover(goal="read the checkout totals", name="checkout_totals", params=dict(PARAMS), surface=surface,
                        planner=planner, policy=Policy(allowed_hosts=HOSTS),
                        escalator=Escalator(operator or NoOperator(), SessionControl(), log), log=log,
                        entry_url=ENTRY, sensitive={"password"}, max_steps=25, output_contract=output_contract,
                        vision=planner if vision else None)
    return artifact, planner, surface, log


def events(log, name):
    return [json.loads(l) for l in log.path.read_text().splitlines() if f'"event": "{name}"' in l]


def replays(artifact, surface=None, params=None, operator=None):
    surface = surface or FakeSurface()
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(artifact, dict(params or PARAMS), surface, Policy(allowed_hosts=HOSTS),
                    Escalator(operator or NoOperator(), SessionControl(), log), log)
    return result, log


def totals_node(node_id="s12", texts=("Item total:", "Tax:", "Total:"), output="totals"):
    return linear_node(node_id, GraphAction(action="extract_many", target=None, value=output,
                                            targets=[text_ladder(t) for t in texts]))


def totals_artifact(spec=None, node=None):
    """A valid artifact, then the requested node or output spec swapped in unvalidated (for rejection tests)."""
    artifact = build_linear(name="checkout_totals", goal="read the checkout totals",
                            surface_meta=dict(checkout_artifact().surface), params=PARAMS, sensitive={"password"},
                            nodes=checkout_nodes()[:11] + [totals_node()], outputs=TOTALS,
                            outcomes=[dict(o) for o in checkout_artifact().outcomes],
                            success={"text_contains": "Checkout: Overview"}, run_id="test-run")
    if node is not None:
        artifact.nodes[11] = node
    if spec is not None:
        artifact.outputs = spec
    return artifact


# ---------- discovery ----------

def test_three_structured_elements_become_one_ordered_list_in_one_node():
    artifact, _, surface, log = run(totals_script())
    node = linear_path(artifact)[-1]
    assert node.action.action == "extract_many" and node.action.value == "totals" and node.action.target is None
    assert [ladder.strategies[0] for ladder in node.action.targets] == [{"kind": "text", "text": "Item total:"},
                                                                         {"kind": "text", "text": "Tax:"},
                                                                         {"kind": "text", "text": "Total:"}]
    assert all(len(ladder.strategies) == 3 for ladder in node.action.targets)      # text, structural path, coordinates
    assert (node.effect, node.retry_safety) == ("none", "safe") and node.action.checkpoint is None
    assert artifact.outputs["totals"] == {"type": "list", "required": True,
                                          "items": {"type": "number", "pattern": MONEY},
                                          "description": "Ordered values read from 3 controls",
                                          "example": ["29.99", "2.40", "32.39"]}
    assert len([n for n in artifact.nodes if n.action and n.action.action == "extract_many"]) == 1
    [recorded] = events(log, "recorded_output")
    assert recorded["value"] == ["29.99", "2.40", "32.39"] and recorded["items"] == 3
    assert "secret_sauce" not in log.path.read_text()


def test_replay_reads_the_same_ordered_list_without_a_planner():
    artifact, _, _, _ = run(totals_script())
    result, log = replays(artifact)
    assert result.status == "success" and result.outputs == {"totals": [29.99, 2.4, 32.39]}
    resolved = [e for e in events(log, "target_resolved") if e["step_id"] == "s12"]
    assert [e["target_index"] for e in resolved] == [0, 1, 2] and all(e["rung"] == 0 for e in resolved)
    assert not any(e["event"].startswith("vision_") for e in map(json.loads, log.path.read_text().splitlines()))


class ReorderedOverview(FakeSurface):
    """The overview lists the totals in another order; semantic ladders still give the requested order."""

    def overview_screen(self):
        screen = super().overview_screen()
        return list(reversed(screen))


def test_a_reordered_screen_keeps_the_requested_logical_order():
    artifact, _, _, _ = run(totals_script())
    result, _ = replays(artifact, surface=ReorderedOverview())
    assert result.outputs == {"totals": [29.99, 2.4, 32.39]}


def test_a_required_item_that_does_not_parse_fails_the_whole_extraction_and_the_planner_is_told():
    script = totals_script(many_step(texts=("Item total:", "Checkout: Overview", "Total:")))   # no money in item 1
    script.insert(-1, many_step())                                                             # then the right selection
    artifact, planner, _, log = run(script)
    [failed] = events(log, "extraction_failed")
    assert failed == {**failed, "output_name": "totals", "target_index": 1}
    assert events(log, "recorded_output")[0]["value"] == ["29.99", "2.40", "32.39"]          # only the second attempt
    assert len([n for n in artifact.nodes if n.action and n.action.action == "extract_many"]) == 1
    assert artifact.outputs["totals"]["example"] == ["29.99", "2.40", "32.39"]


def with_targets(artifact, texts, required=True):
    """The totals node re-targeted, and the output's requiredness set, without rebuilding the artifact."""
    node = replace(artifact.nodes[11], action=replace(artifact.nodes[11].action, targets=[text_ladder(t) for t in texts]))
    outputs = {"totals": {**artifact.outputs["totals"], "required": required}}
    changed = replace(artifact, nodes=artifact.nodes[:11] + [node] + artifact.nodes[12:], outputs=outputs)
    validate(changed)
    return changed


def test_a_required_list_with_a_missing_middle_target_fails_without_a_partial_value():
    artifact, _, _, _ = run(totals_script())
    broken = with_targets(artifact, ("Item total:", "Shipping:", "Total:"))
    result, log = replays(broken, operator=RecordingOperator("abort"))
    assert result.status == "failure" and result.outcome_code == "target_not_found" and result.step_id == "s12"
    assert "totals" not in result.outputs and events(log, "output_extracted") == []
    assert events(log, "target_unresolved")[0]["target_index"] == 1


def test_an_optional_list_with_a_missing_middle_target_is_absent_as_a_whole():
    artifact, _, _, _ = run(totals_script())
    optional = with_targets(artifact, ("Item total:", "Shipping:", "Total:"), required=False)
    result, log = replays(optional)
    assert result.status == "success" and "totals" not in result.outputs          # never ["29.99", "32.39"]
    [absent] = events(log, "optional_output_absent")
    assert absent == {**absent, "step_id": "s12", "output": "totals", "target_index": 1, "reason": "control not on screen"}
    assert events(log, "output_extracted") == []


def test_an_optional_list_with_a_pattern_failure_is_absent_and_the_log_never_quotes_the_text():
    secret = FakeSurface(extra_elements=[el("text", "", "Voucher: secret_sauce")])
    artifact, _, _, _ = run(totals_script())
    optional = with_targets(artifact, ("Item total:", "Voucher:", "Total:"), required=False)
    result, log = replays(optional, surface=secret)
    assert result.status == "success" and "totals" not in result.outputs
    [absent] = events(log, "optional_output_absent")
    assert absent["target_index"] == 1 and absent["reason"] == "pattern did not match the item's text"
    assert "secret_sauce" not in log.path.read_text() and "text" not in absent


def test_a_complete_optional_list_is_emitted_in_order_and_never_shifts():
    artifact, _, _, _ = run(totals_script())
    optional = with_targets(artifact, ("Item total:", "Tax:", "Total:"), required=False)
    assert replays(optional)[0].outputs == {"totals": [29.99, 2.4, 32.39]}
    assert replays(optional, surface=ReorderedOverview())[0].outputs == {"totals": [29.99, 2.4, 32.39]}
    reversed_targets = with_targets(artifact, ("Total:", "Tax:", "Item total:"), required=False)
    assert replays(reversed_targets)[0].outputs == {"totals": [32.39, 2.4, 29.99]}


def test_discovery_never_records_a_partial_list_for_a_required_or_optional_selection():
    for optional in (False, True):
        script = totals_script(many_step(texts=("Item total:", "Checkout: Overview", "Total:"), optional=optional))
        script.insert(-1, many_step(optional=optional))
        artifact, planner, _, log = run(script)
        [failed] = events(log, "extraction_failed")
        assert failed["target_index"] == 1 and failed["optional"] is optional
        assert events(log, "recorded_output")[0]["value"] == ["29.99", "2.40", "32.39"]     # the second selection only
        assert len([n for n in artifact.nodes if n.action and n.action.action == "extract_many"]) == 1
        assert artifact.outputs["totals"]["example"] == ["29.99", "2.40", "32.39"]
        assert artifact.outputs["totals"]["required"] is (not optional)


def test_scalar_optional_extraction_is_unchanged():
    artifact, _, _, _ = run(checkout_script())
    node = nodes_by_id(artifact)["s15"]                                            # the total, read from "Total:"
    node.action.target = text_ladder("Shipping:")
    optional = replace(artifact, outputs={**artifact.outputs, "total": {**artifact.outputs["total"], "required": False}})
    validate(optional)
    result, log = replays(optional)
    assert result.status == "success" and result.outputs["total"] is None            # present, as null
    assert events(log, "output_missing")[0]["output"] == "total"


def test_scalar_extraction_is_unchanged():
    artifact, _, _, _ = run(checkout_script())
    result, _ = replays(artifact)
    assert result.outputs == {"product_name": "Sauce Labs Backpack", "subtotal": 29.99, "tax": 2.4, "total": 32.39}
    assert all(n.action.targets is None for n in artifact.nodes if n.action)


def test_vision_is_not_involved_when_the_items_are_structured():
    artifact, planner, surface, log = run(totals_script(), vision=True)
    assert planner.frames == [] and surface.viewport_shots == []
    assert artifact.outputs["totals"]["example"] == ["29.99", "2.40", "32.39"]


def test_the_declared_list_contract_wins_and_sensitive_values_stay_redacted():
    contract = {"totals": {"type": "list", "required": True, "items": {"type": "string", "pattern": r"(\$\s*[\d.]+)"}}}
    artifact, _, _, log = run(totals_script(many_step(pattern=r"([\d.]+)")), output_contract=contract)
    assert artifact.outputs["totals"]["items"] == {"type": "string", "pattern": r"(\$\s*[\d.]+)"}
    assert replays(artifact)[0].outputs == {"totals": ["$ 29.99", "$ 2.40", "$ 32.39"]}
    secret = FakeSurface(extra_elements=[el("text", "", "Password hint: secret_sauce"), el("text", "", "Code: 0042")])
    artifact, _, _, log = run(totals_script(many_step(texts=("Password hint:", "Code:"), output="notes",
                                                      pattern=r":\s*(.+)$")), surface=secret)
    text = log.path.read_text()
    assert "secret_sauce" not in text and "[REDACTED]" in text and "0042" in json.dumps(artifact.outputs)


# ---------- the planner boundary and direct execution ----------

def observation():
    return Observation(url=ENTRY, elements=[Element("text", "", "Item total: $ 29.99"), Element("text", "", "Tax: $ 2.40"),
                                            Element("text", "", "Total: $ 32.39")])


def test_tool_input_validation_for_extract_and_extract_many():
    base = {"reason": "", "outcomes": [], "output_name": "totals", "pattern": MONEY}
    good = action_from_tool_input({**base, "kind": "extract_many", "element_indexes": [2, 0]}, observation())
    assert good.kind == "extract_many" and [t.text for t in good.targets] == ["Total: $ 32.39", "Item total: $ 29.99"]
    for data, reason in [
        ({**base, "kind": "extract_many", "element_indexes": []}, "gave no element_indexes"),
        ({**base, "kind": "extract_many", "element_indexes": None}, "gave no element_indexes"),
        ({**base, "kind": "extract_many", "element_indexes": [0, 0]}, "more than once"),
        ({**base, "kind": "extract_many", "element_indexes": [0, 7]}, "does not exist"),
        ({**base, "kind": "extract_many", "element_indexes": list(range(21))}, "at most 20"),
        ({**base, "kind": "extract_many", "element_indexes": [0], "output_name": ""}, "without an output_name"),
        ({**base, "kind": "extract", "element_index": None}, "without exactly one target"),
        ({**base, "kind": "extract", "element_index": 0, "output_name": None}, "without an output_name"),
    ]:
        stuck = action_from_tool_input(data, observation())
        assert stuck.kind == "stuck" and stuck.stuck_cause == "provider" and reason in stuck.reason, data
    assert "extract_many" in CHOOSE_ACTION_TOOL["input_schema"]["properties"]["kind"]["enum"]
    assert "element_indexes" in CHOOSE_ACTION_TOOL["input_schema"]["required"]


def test_openai_and_anthropic_adapters_reject_malformed_extractions_before_execution():
    from types import SimpleNamespace
    from src.cua.planner import ClaudePlanner

    class Responses:
        def create(self, **kwargs):
            call = SimpleNamespace(type="function_call", name="choose_action", arguments=json.dumps(
                {"kind": "extract", "element_index": None, "output_name": "total", "pattern": MONEY, "value": None,
                 "optional": False, "expect": None, "reason": "", "outcomes": [], "stuck_cause": None,
                 "candidate_id": None, "element_indexes": None}))
            return SimpleNamespace(output=[call])

    planner = OpenAIPlanner(model="m", client=SimpleNamespace(responses=Responses()))
    stuck = planner.decide("g", {}, observation(), [])
    assert stuck.kind == "stuck" and "without exactly one target" in stuck.reason

    class Messages:
        def create(self, **kwargs):
            block = SimpleNamespace(type="tool_use", input={"kind": "extract_many", "element_indexes": [1, 1],
                                                            "output_name": "totals", "reason": "", "outcomes": []})
            return SimpleNamespace(content=[block], stop_reason="tool_use")

    claude = ClaudePlanner.__new__(ClaudePlanner)
    claude.client, claude.model, claude.vision_model, claude.name = SimpleNamespace(messages=Messages()), "m", "m", "c"
    stuck = claude.decide("g", {}, observation(), [])
    assert stuck.kind == "stuck" and stuck.stuck_cause == "provider" and "more than once" in stuck.reason


def test_direct_execution_refuses_malformed_extractions_without_logging_them_as_acted():
    surface = FakeSurface()
    log = RunLog("discovery")
    for action, reason in [
        (Action(kind="extract", target=None, output_name="x"), "no target control"),
        (Action(kind="extract", target=Element("text", "", "Total: 1"), output_name=""), "no output_name"),
        (Action(kind="extract_many", targets=[], output_name="x"), "no target controls"),
        (Action(kind="extract_many", targets=[Element("text", "", "a", ref="p")] * 2, output_name="x"), "more than once"),
        (Action(kind="extract_many", targets=[Element("text", "", str(i), ref=f"p{i}") for i in range(21)],
                output_name="x"), "exceed the limit"),
    ]:
        assert reason in (extraction_problem(action) or ""), action.kind
        with pytest.raises(ActionError, match=reason) as refused:
            perform(action, {}, surface, log)
        assert refused.value.performed == "no"
    assert not log.path.exists()                                                  # nothing was logged at all

    # A scripted (internal) malformed extract goes through the same recovery: told, never recorded.
    script = checkout_script()
    script.insert(11, ScriptedStep("extract", "text", text="Total:", output_name=None, pattern=MONEY))
    artifact, _, _, log = run(script)
    assert events(log, "action_failed")[0]["reason"].endswith("no output_name was given")
    assert sorted(artifact.outputs) == ["product_name", "subtotal", "tax", "total"]
    assert len([e for e in events(log, "acted") if e["kind"] == "extract"]) == 4      # the four well-formed ones only


# ---------- the artifact ----------

def test_extract_many_nodes_are_validated_serialized_and_round_tripped():
    artifact = totals_artifact()
    validate(artifact)
    data = json.loads(json.dumps(to_dict(artifact)))
    node = next(n for n in data["nodes"] if n["id"] == "s12")
    assert node["action"]["target"] is None and [t["strategies"][0]["text"] for t in node["action"]["targets"]] == [
        "Item total:", "Tax:", "Total:"]
    assert from_dict(data) == artifact
    path = save_artifact(artifact, secrets=("secret_sauce",))
    assert load_artifact(path) == artifact and path.read_text() == artifact_module.dumps(artifact, secrets=("secret_sauce",))

    def rejects(bad, message):
        with pytest.raises(ArtifactError, match=message):
            validate(bad)

    rejects(totals_artifact(node=totals_node(texts=())), "non-empty list of targets")
    rejects(totals_artifact(node=totals_node(texts=("Tax:", "Tax:"))), "duplicates an earlier target")
    rejects(totals_artifact(node=totals_node(texts=[f"t{i}:" for i in range(21)])), "at most 20 targets")
    rejects(totals_artifact(node=totals_node(output="shipping")), "writes undeclared output")
    rejects(totals_artifact(spec={"totals": {"type": "string", "required": True}}), "which is not a list")
    rejects(totals_artifact(spec={"totals": {"type": "list", "required": True}}), "needs items with a type")
    rejects(totals_artifact(spec={"totals": {"type": "list", "required": True, "items": {"type": "number", "pattern": "("}}}),
            "not a valid regex")
    single = replace(totals_node(), action=GraphAction(action="extract", target=text_ladder("Total:"), value="totals"))
    rejects(totals_artifact(node=single), "is a list \\(use extract_many, or mode append")
    with_target = replace(totals_node(), action=replace(totals_node().action, target=text_ladder("Total:")))
    rejects(totals_artifact(node=with_target), "uses targets, not a single target")
    with pytest.raises(ArtifactError, match="targets must be a JSON list"):
        from_dict({**to_dict(artifact), "nodes": [{**n, "action": {**n["action"], "targets": {}}} if n["id"] == "s12" else n
                                                   for n in to_dict(artifact)["nodes"]]})


def test_campaign_list_contracts_are_parsed_and_merged_only_when_identical():
    parsed = outputs_from_dict({"totals": {"type": "list", "items": {"type": "number", "pattern": MONEY}}})
    assert parsed == {"totals": {"type": "list", "required": True, "items": {"type": "number", "pattern": MONEY}}}
    for raw, message in [({"totals": {"type": "list"}}, "needs an items object"),
                         ({"totals": {"type": "list", "items": {"type": "list", "pattern": "x"}}}, "items type must be"),
                         ({"totals": {"type": "list", "items": {"type": "number"}}}, "a regex pattern is required"),
                         ({"totals": {"type": "list", "items": {"type": "number", "pattern": "x"}, "pattern": "x"}},
                          "takes its pattern inside items"),
                         ({"total": {"type": "number", "pattern": "x", "items": {"type": "number"}}}, "only a list output")]:
        with pytest.raises(CampaignError, match=message):
            outputs_from_dict(raw)

    same = {"totals": {"type": "list", "required": True, "items": {"type": "number", "pattern": MONEY}}}
    merged = merge_traces([link_trace(outputs=same), url_trace(outputs=same)], ["cart_route"], "c")
    assert merged.outputs == same
    other_items = {"totals": {**same["totals"], "items": {"type": "string", "pattern": MONEY}}}
    with pytest.raises(MergeError, match="output 'totals' items"):
        merge_traces([link_trace(outputs=same), url_trace(outputs=other_items)], ["cart_route"], "c")
    with pytest.raises(MergeError, match="output 'totals' type 'list' vs 'string'"):
        merge_traces([link_trace(outputs=same), url_trace(outputs={"totals": {"type": "string", "required": True,
                                                                             "pattern": MONEY}})], ["cart_route"], "c")


def test_reuse_preserves_every_target_ladder_and_the_list_contract():
    approved = replace(totals_artifact(), status="approved")
    save_artifact(approved, secrets=("secret_sauce",))
    log = RunLog("discovery", secrets=("secret_sauce",))
    library = build_library(None, ENTRY, HOSTS, dict(PARAMS), {"password"}, log)
    overview = FakeSurface()
    replay(checkout_artifact(), dict(PARAMS), overview, Policy(allowed_hosts=HOSTS),
           Escalator(NoOperator(), SessionControl(), RunLog("replay")), RunLog("replay"))
    offered = find_candidates(library, overview.observe(), dict(PARAMS), overview, log, 1)
    candidate = next(c for c in offered if c.candidate_id == "checkout_totals.v1@s12")
    assert candidate.node_ids == ["s12"] and candidate.outputs == ["totals"] and candidate.action_count == 1
    assert candidate.outline == ["read list output 'totals' from 3 controls"]
    execution = execute_candidate(library, candidate, dict(PARAMS), overview, Policy(allowed_hosts=HOSTS), log, 1)
    assert execution.result.status == "success" and execution.result.outputs == {"totals": [29.99, 2.4, 32.39]}
    [entry] = execution.result.executed_path
    assert [t["strategies"][0]["text"] for t in entry["action"]["targets"]] == ["Item total:", "Tax:", "Total:"]
    nodes, outputs = importable_nodes(verified_entries(execution.result, complete=True), execution.segment, TOTALS)
    assert nodes[0].action == totals_node().action and outputs == {"totals": approved.outputs["totals"]}
    assert importable_nodes(verified_entries(execution.result, complete=True), execution.segment,
                            {"totals": {**TOTALS["totals"], "items": {"type": "string", "pattern": MONEY}}}) == ([], {})
