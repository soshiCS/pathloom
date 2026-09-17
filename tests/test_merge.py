"""Prefix-tree merge of linear graphs: sharing rules, node metadata, conflicts, conditional inputs, gated replay."""
import json
from dataclasses import replace

import pytest

from src.cua import artifact as artifact_module
from src.cua.artifact import copy_node, linear_path, missing_inputs, nodes_by_id, outgoing, to_dict, validate
from src.cua.escalation import NoOperator
from src.cua.merge import MergeError, merge_traces, node_key
from src.cua.models import Scenario, ScenarioTrace
from src.cua.replay import replay
from tests.context import (ENTRY, HOSTS, Escalator, GraphAction, GraphNode, Policy, RunLog, SessionControl,
                           build_linear, click_node, ladder, linear_node, navigate_node, type_node)
from tests.fake_surface import FakeSurface

LOGIN = {"username": "standard_user", "password": "secret_sauce"}
SURFACE = {"kind": "web", "app": "Swag Labs", "entry_url": ENTRY, "allowed_hosts": HOSTS}


def login_nodes() -> list[GraphNode]:
    return [navigate_node("s1", ENTRY, "Swag Labs"), type_node("s2", "Username", "username"),
            type_node("s3", "Password", "password"), click_node("s4", "Login", "Products")]


def add_node(node_id: str = "s5") -> GraphNode:
    return click_node(node_id, "Add to cart", "Remove", context="{{product_name}}")


def cart_link_node(node_id: str) -> GraphNode:
    return click_node(node_id, "cart", "Your Cart", role="link")


def cart_url_node(node_id: str) -> GraphNode:
    return navigate_node(node_id, "{{cart_url}}", "Your Cart")


def trace(scenario: str, nodes: list[GraphNode], params: dict, sensitive=(), **overrides) -> ScenarioTrace:
    """A verified linear graph as one scenario's discovery would record it (node ids renumbered per run)."""
    nodes = [copy_node(node, f"s{index}") for index, node in enumerate(nodes, start=1)]
    fields = {"name": "checkout_paths", "goal": "add {{product_name}} and reach the cart", "surface_meta": SURFACE,
              "outputs": {}, "outcomes": [], "success": {"text_contains": "Your Cart"},
              "run_id": f"discovery-{scenario}"}
    fields.update(overrides)
    artifact = build_linear(params=params, nodes=nodes, sensitive=set(sensitive), **fields)
    return ScenarioTrace(scenario=Scenario(name=scenario, params=params, sensitive=list(sensitive)),
                         artifact=artifact, run_id=fields["run_id"], planner="scripted")


def link_trace(**overrides) -> ScenarioTrace:
    params = {**LOGIN, "product_name": "Sauce Labs Backpack", "cart_route": "cart_link"}
    return trace("via_link", login_nodes() + [add_node(), cart_link_node("s6")], params, ["password"], **overrides)


def url_trace(**overrides) -> ScenarioTrace:
    params = {**LOGIN, "product_name": "Sauce Labs Backpack", "cart_route": "cart_url", "cart_url": ENTRY + "cart.html"}
    return trace("via_url", login_nodes() + [add_node(), cart_url_node("s6")], params, ["password"], **overrides)


def edges_out(graph, node_id: str) -> list[tuple]:
    return [(e.target, e.priority, [(g.kind, g.input, g.value) for g in e.guards]) for e in outgoing(graph, node_id)]


def actions_named(graph, name: str) -> list[str]:
    return [n.id for n in graph.nodes if n.kind == "action" and n.action.target
            and n.action.target.strategies[0].get("name") == name]


# ---------- sharing rules ----------

def test_two_scenarios_share_the_prefix_and_branch_at_the_first_difference():
    graph = merge_traces([link_trace(), url_trace()], ["cart_route"], campaign_id="campaign-test")
    assert [n.id for n in graph.nodes] == ["d1", "s1", "s2", "s3", "s4", "s5", "d2", "s6", "s7", "success"]
    assert [n.kind for n in graph.nodes] == ["decision"] + ["action"] * 5 + ["decision", "action", "action",
                                                                             "terminal"]
    # The entry gate admits only the declared selector assignments, both onto the shared first action.
    assert graph.entry_node == "d1"
    assert edges_out(graph, "d1") == [("s1", 0, [("input_equals", "cart_route", "cart_link")]),
                                      ("s1", 1, [("input_equals", "cart_route", "cart_url")])]
    assert edges_out(graph, "s5") == [("d2", 0, [("always", None, None)])]
    assert edges_out(graph, "d2") == [("s6", 0, [("input_equals", "cart_route", "cart_link")]),
                                      ("s7", 1, [("input_equals", "cart_route", "cart_url")])]
    assert nodes_by_id(graph)["s6"].action.action == "click" and nodes_by_id(graph)["s7"].action.value == "{{cart_url}}"
    assert edges_out(graph, "s6") == edges_out(graph, "s7") == [("success", 0, [("always", None, None)])]
    assert nodes_by_id(graph)["s4"].effect == "reversible" and nodes_by_id(graph)["s7"].effect == "none"


def test_merged_nodes_keep_their_effect_and_retry_safety_exactly():
    # A trace may carry classifications that differ from the defaults (a reused prefix, a hand-reviewed node);
    # the merge copies them verbatim and shares only nodes whose classification agrees.
    strict = link_trace()
    tweak = {"s2": ("reversible", "never_retry"), "s4": ("reversible", "safe"), "s1": ("none", "safe")}
    for node in strict.artifact.nodes:
        if node.id in tweak:
            node.effect, node.retry_safety = tweak[node.id]
    other = url_trace()
    for node in other.artifact.nodes:
        if node.id in tweak:
            node.effect, node.retry_safety = tweak[node.id]
    graph = merge_traces([strict, other], ["cart_route"], campaign_id="c")
    merged = nodes_by_id(graph)
    assert (merged["s2"].effect, merged["s2"].retry_safety) == ("reversible", "never_retry")
    assert (merged["s4"].effect, merged["s4"].retry_safety) == ("reversible", "safe")
    assert (merged["s1"].effect, merged["s1"].retry_safety) == ("none", "safe")
    assert [n.id for n in graph.nodes] == ["d1", "s1", "s2", "s3", "s4", "s5", "d2", "s6", "s7", "success"]
    assert merged["s2"].action == strict.artifact.nodes[1].action and merged["s2"].action is not strict.artifact.nodes[1].action

    # A node that differs only in retry safety is a different node: the traces diverge there.
    disagreeing = url_trace()
    disagreeing.artifact.nodes[1].retry_safety = "safe"
    graph = merge_traces([link_trace(), disagreeing], ["cart_route"], campaign_id="c")
    assert [n.id for n in graph.nodes][:4] == ["d1", "s1", "d2", "s2"]
    assert {n.retry_safety for n in graph.nodes if n.kind == "action" and n.action.value == "{{username}}"} == \
        {"never_retry", "safe"}


def test_three_scenarios_diverge_twice_and_never_share_a_suffix():
    checkout = [click_node("x", "Checkout", "Checkout: Your Information")]
    postal = [type_node("x", "Zip/Postal Code", "postal_code"), click_node("x", "Continue", "Checkout: Overview")]
    phone = [type_node("x", "Zip/Postal Code", "phone"), click_node("x", "Continue", "Checkout: Overview")]
    common = {**LOGIN, "product_name": "Sauce Labs Backpack"}
    success = {"text_contains": "Checkout: Overview"}
    traces = [
        trace("link_postal", login_nodes() + [add_node(), cart_link_node("x")] + checkout + postal,
              {**common, "cart_route": "cart_link", "zip_source": "postal_code", "postal_code": "10001"},
              ["password"], success=success),
        trace("url_postal", login_nodes() + [add_node(), cart_url_node("x")] + checkout + postal,
              {**common, "cart_route": "cart_url", "zip_source": "postal_code", "postal_code": "10001",
               "cart_url": ENTRY + "cart.html"}, ["password"], success=success),
        trace("url_phone", login_nodes() + [add_node(), cart_url_node("x")] + checkout + phone,
              {**common, "cart_route": "cart_url", "zip_source": "phone", "phone": "+15551234567",
               "cart_url": ENTRY + "cart.html"}, ["password", "phone"], success=success),
    ]
    graph = merge_traces(traces, ["cart_route", "zip_source"], campaign_id="campaign-test")
    decisions = [n.id for n in graph.nodes if n.kind == "decision"]
    assert decisions == ["d1", "d2", "d3"]
    assert [t for t, _, _ in edges_out(graph, "d1")] == ["s1", "s1", "s1"]
    assert edges_out(graph, "d2") == [
        ("s6", 0, [("input_equals", "cart_route", "cart_link"), ("input_equals", "zip_source", "postal_code")]),
        ("s10", 1, [("input_equals", "cart_route", "cart_url"), ("input_equals", "zip_source", "postal_code")]),
        ("s10", 2, [("input_equals", "cart_route", "cart_url"), ("input_equals", "zip_source", "phone")])]
    assert [t for t, _, _ in edges_out(graph, "d3")] == ["s12", "s14"]
    assert actions_named(graph, "Checkout") == ["s7", "s11"]
    assert actions_named(graph, "Continue") == ["s9", "s13", "s15"]
    paths = {s["name"]: s["node_path"] for s in graph.provenance["scenarios"]}
    assert paths["link_postal"] == ["d1", "s1", "s2", "s3", "s4", "s5", "d2", "s6", "s7", "s8", "s9", "success"]
    assert paths["url_postal"] == ["d1", "s1", "s2", "s3", "s4", "s5", "d2", "s10", "s11", "d3", "s12", "s13",
                                   "success"]
    assert paths["url_phone"] == ["d1", "s1", "s2", "s3", "s4", "s5", "d2", "s10", "s11", "d3", "s14", "s15",
                                  "success"]
    assert graph.inputs["phone"] == {"type": "string", "required": False, "sensitive": True,
                                     "description": "Value typed where 'phone' was used during discovery",
                                     "required_when": [{"cart_route": "cart_url", "zip_source": "phone"}]}
    assert graph.inputs["postal_code"]["required_when"] == [{"cart_route": "cart_link", "zip_source": "postal_code"},
                                                            {"cart_route": "cart_url", "zip_source": "postal_code"}]
    assert graph.inputs["cart_url"]["required_when"] == [{"cart_route": "cart_url", "zip_source": "postal_code"},
                                                         {"cart_route": "cart_url", "zip_source": "phone"}]
    assert graph.inputs["cart_route"] == {**graph.inputs["cart_route"], "required": True, "selector": True}
    assert graph.inputs["username"]["required"] is True and "required_when" not in graph.inputs["username"]


def test_identical_traces_share_one_path_and_provenance_names_both():
    first = link_trace()
    second = trace("via_link_again", login_nodes() + [add_node(), cart_link_node("s6")],
                   {**first.scenario.params, "cart_route": "cart_link_again"}, ["password"])
    graph = merge_traces([first, second], ["cart_route"], campaign_id="campaign-test")
    assert [n.id for n in graph.nodes] == ["d1", "s1", "s2", "s3", "s4", "s5", "s6", "success"]
    assert [t for t, _, _ in edges_out(graph, "d1")] == ["s1", "s1"]    # only the gate; no branch inside
    paths = [s["node_path"] for s in graph.provenance["scenarios"]]
    assert paths[0] == paths[1] == ["d1", "s1", "s2", "s3", "s4", "s5", "s6", "success"]
    assert [s["name"] for s in graph.provenance["scenarios"]] == ["via_link", "via_link_again"]


def test_a_trace_that_is_a_prefix_of_another_ends_at_a_guarded_terminal():
    short = trace("login_only", login_nodes(), {**LOGIN, "mode": "login_only"}, ["password"],
                  success={"text_contains": "Products"})
    longer = trace("add_item", login_nodes() + [add_node()],
                   {**LOGIN, "mode": "add_item", "product_name": "Sauce Labs Backpack"}, ["password"],
                   success={"text_contains": "Products"})
    graph = merge_traces([short, longer], ["mode"], campaign_id="campaign-test")
    assert [n.id for n in graph.nodes] == ["d1", "s1", "s2", "s3", "s4", "d2", "s5", "success"]
    assert edges_out(graph, "d2") == [("success", 0, [("input_equals", "mode", "login_only")]),
                                      ("s5", 1, [("input_equals", "mode", "add_item")])]
    assert edges_out(graph, "s5") == [("success", 0, [("always", None, None)])]
    assert graph.inputs["product_name"]["required_when"] == [{"mode": "add_item"}]


def test_nodes_are_compared_by_meaning_not_by_id():
    renumbered = url_trace()
    for node in renumbered.artifact.nodes:
        if node.kind == "action":
            node.id = "t" + node.id[1:]
    for e in renumbered.artifact.edges:
        e.source, e.target = ("t" + e.source[1:] if e.source.startswith("s") else e.source,
                              "t" + e.target[1:] if e.target.startswith("s") and e.target != "success" else e.target)
    renumbered.artifact.entry_node = "t1"
    validate(renumbered.artifact)
    graph = merge_traces([link_trace(), renumbered], ["cart_route"], campaign_id="campaign-test")
    assert [n.id for n in graph.nodes] == ["d1", "s1", "s2", "s3", "s4", "s5", "d2", "s6", "s7", "success"]
    assert node_key(login_nodes()[0]) == node_key(copy_node(login_nodes()[0], "zzz"))
    risky = click_node("s4", "Login", "Products", risky=True)
    assert node_key(login_nodes()[3]) != node_key(risky)                         # effect is part of the meaning
    assert node_key(login_nodes()[3]) != node_key(replace(login_nodes()[3], retry_safety="safe"))


def test_only_linear_traces_can_be_merged():
    branching = merge_traces([link_trace(), url_trace()], ["cart_route"], campaign_id="c")
    bad = ScenarioTrace(scenario=Scenario(name="branchy", params=link_trace().scenario.params, sensitive=[]),
                        artifact=branching, run_id="r", planner="scripted")
    with pytest.raises(MergeError, match="scenario 'branchy' did not record a linear graph"):
        merge_traces([bad], [], campaign_id="c")


def test_merge_is_deterministic():
    one = merge_traces([link_trace(), url_trace()], ["cart_route"], campaign_id="campaign-test")
    two = merge_traces([link_trace(), url_trace()], ["cart_route"], campaign_id="campaign-test")
    strip = lambda graph: {**to_dict(graph), "provenance": {**graph.provenance, "recorded_at": None}}
    assert json.dumps(strip(one), sort_keys=False) == json.dumps(strip(two), sort_keys=False)
    assert [(e.source, e.target, e.priority) for e in one.edges] == [
        ("d1", "s1", 0), ("d1", "s1", 1), ("s1", "s2", 0), ("s2", "s3", 0), ("s3", "s4", 0), ("s4", "s5", 0),
        ("s5", "d2", 0), ("d2", "s6", 0), ("d2", "s7", 1), ("s6", "success", 0), ("s7", "success", 0)]


def test_merged_provenance_records_the_campaign_and_every_scenario():
    graph = merge_traces([link_trace(), url_trace()], ["cart_route"], campaign_id="campaign-20260914T000000Z-abcd")
    prov = graph.provenance
    assert prov["campaign_id"] == "campaign-20260914T000000Z-abcd" and prov["merge_strategy"] == "prefix_tree"
    assert (prov["scenario_count"], prov["successful_scenario_count"]) == (2, 2)
    assert prov["selectors"] == ["cart_route"] and prov["planners"] == ["scripted"]
    assert prov["scenarios"][1] == {**prov["scenarios"][1], "name": "via_url", "selectors": {"cart_route": "cart_url"},
                                    "run_id": "discovery-via_url", "planner": "scripted", "action_count": 6,
                                    "node_path": ["d1", "s1", "s2", "s3", "s4", "s5", "d2", "s7", "success"],
                                    "verified": True}
    assert graph.status == "draft" and graph.version == 1 and graph.schema_version == "2.0"
    assert "secret_sauce" not in json.dumps(to_dict(graph)) and '"risk"' not in json.dumps(to_dict(graph))


# ---------- conflicts ----------

def test_incompatible_traces_are_rejected_with_the_field_named():
    money = {"type": "number", "required": True, "pattern": r"\$\s*([\d.]+)"}
    cases = [
        (url_trace(goal="a different goal"), "disagree on description"),
        (url_trace(surface_meta={**SURFACE, "entry_url": "https://other.example/"}), "disagree on surface.entry_url"),
        (url_trace(surface_meta={**SURFACE, "kind": "desktop"}), "disagree on surface.kind"),
        (url_trace(surface_meta={**SURFACE, "allowed_hosts": ["other.example"]}), "disagree on surface.allowed_hosts"),
        (url_trace(success={"text_contains": "Products"}), "disagree on success"),
        (url_trace(outputs={"total": money}), "disagree on outputs: \\[\\] vs \\['total'\\]"),
        (url_trace(name="other_name"), "disagree on name"),
    ]
    for other, message in cases:
        with pytest.raises(MergeError, match=message):
            merge_traces([link_trace(), other], ["cart_route"], campaign_id="c")


def test_outputs_must_agree_on_type_requiredness_and_pattern():
    money = {"type": "number", "required": True, "pattern": r"\$\s*([\d.]+)"}
    first = link_trace(outputs={"total": money})
    with pytest.raises(MergeError, match="disagree on outputs: output 'total' type 'number' vs 'string'"):
        merge_traces([first, url_trace(outputs={"total": {**money, "type": "string"}})], ["cart_route"], "c")
    with pytest.raises(MergeError, match="disagree on outputs: output 'total' required True vs False"):
        merge_traces([first, url_trace(outputs={"total": {**money, "required": False}})], ["cart_route"], "c")
    with pytest.raises(MergeError) as rejected:
        merge_traces([first, url_trace(outputs={"total": {**money, "pattern": "Total: (.+)"}})], ["cart_route"], "c")
    assert str(rejected.value) == ("scenarios 'via_link' and 'via_url' disagree on outputs: output 'total' pattern "
                                   "'\\\\$\\\\s*([\\\\d.]+)' vs 'Total: (.+)'")
    with pytest.raises(MergeError, match="output 'total' pattern"):
        merge_traces([first, url_trace(outputs={"total": {"type": "number", "required": True}})], ["cart_route"], "c")
    merged = merge_traces([first, url_trace(outputs={"total": {**money, "example": "1.00", "description": "x"}})],
                          ["cart_route"], "c")
    assert merged.outputs == {"total": money}                     # descriptive metadata is free to differ


def test_outcomes_are_unioned_by_code_and_conflicts_are_rejected():
    locked = {"code": "user_locked_out", "kind": "business", "source": "planner",
              "detect": {"text_contains": "locked out"}}
    denied = {"code": "invalid_credentials", "kind": "business", "source": "reviewer",
              "detect": {"text_contains": "do not match"}}
    merged = merge_traces([link_trace(outcomes=[locked]),
                           url_trace(outcomes=[denied, {**locked, "source": "reviewer"}])], ["cart_route"], "c")
    assert merged.outcomes == [locked, denied]                    # first definition and its source win
    with pytest.raises(MergeError, match="outcome 'user_locked_out' is defined differently by scenarios 'via_link' "
                                         "and 'via_url'"):
        merge_traces([link_trace(outcomes=[locked]),
                      url_trace(outcomes=[{**locked, "detect": {"text_contains": "locked"}}])], ["cart_route"], "c")


def test_conflicting_input_types_are_rejected():
    other = url_trace()
    other.artifact.inputs["username"]["type"] = "integer"
    with pytest.raises(MergeError, match="input 'username' has conflicting types"):
        merge_traces([link_trace(), other], ["cart_route"], "c")


def test_nothing_to_merge_is_an_error():
    with pytest.raises(MergeError, match="no successful traces"):
        merge_traces([], ["cart_route"], "c")


# ---------- conditional inputs ----------

def test_conditional_input_rules_are_validated():
    graph = merge_traces([link_trace(), url_trace()], ["cart_route"], "c")

    def rejects(mutate, message):
        broken = merge_traces([link_trace(), url_trace()], ["cart_route"], "c")
        mutate(broken.inputs)
        with pytest.raises(artifact_module.ArtifactError, match=message):
            validate(broken)

    validate(graph)
    rejects(lambda i: i["cart_url"].update(required_when=[{"username": "x"}]),
            "required_when refers to 'username', which is not a selector input")
    rejects(lambda i: i["cart_url"].update(required_when=[]), "must be a non-empty list")
    rejects(lambda i: i["cart_url"].update(required_when=[{}]), "must be a non-empty object")
    rejects(lambda i: i["cart_url"].update(required_when={"cart_route": "cart_url"}), "must be a non-empty list")
    rejects(lambda i: i["cart_url"].update(required_when=[{"cart_route": 1}]), "must be a string")
    rejects(lambda i: i["cart_url"].update(required=True), "meaningless on an input that is always required")
    rejects(lambda i: i["cart_route"].update(required=False), "a selector input must be required")


def test_missing_inputs_follow_the_selected_path():
    graph = merge_traces([link_trace(), url_trace()], ["cart_route"], "c")
    base = {**LOGIN, "product_name": "Sauce Labs Backpack"}
    assert missing_inputs(graph.inputs, {**base, "cart_route": "cart_link"}) == []
    assert missing_inputs(graph.inputs, {**base, "cart_route": "cart_url"}) == ["cart_url"]
    assert missing_inputs(graph.inputs, {**base, "cart_route": "cart_url", "cart_url": "x"}) == []
    assert missing_inputs(graph.inputs, {"cart_route": "cart_link"}) == ["username", "password", "product_name"]
    assert missing_inputs(graph.inputs, {}) == ["username", "password", "product_name", "cart_route"]


# ---------- replaying the merged graph ----------

def run(graph, params, surface=None):
    surface = surface or FakeSurface()
    log = RunLog("replay", secrets=("secret_sauce",))
    result = replay(graph, params, surface, Policy(allowed_hosts=HOSTS),
                    Escalator(NoOperator(), SessionControl(), log), log)
    return result, surface, log


def test_selector_guards_replay_each_declared_path():
    graph = merge_traces([link_trace(), url_trace()], ["cart_route"], "c")
    base = {**LOGIN, "product_name": "Sauce Labs Backpack"}
    result, surface, log = run(graph, {**base, "cart_route": "cart_link"})
    assert result.status == "success" and ("click", "cart", "") in surface.actions
    assert [e for e in surface.actions if e[0] == "navigate"] == [("navigate", ENTRY)]
    assert '"target": "s6"' in log.path.read_text()
    assert [e["id"] for e in result.executed_path] == ["s1", "s2", "s3", "s4", "s5", "s6"]

    result, surface, _ = run(graph, {**base, "cart_route": "cart_url", "cart_url": ENTRY + "cart.html"})
    assert result.status == "success" and ("navigate", ENTRY + "cart.html") in surface.actions
    assert ("click", "cart", "") not in surface.actions
    assert [e["id"] for e in result.executed_path] == ["s1", "s2", "s3", "s4", "s5", "s7"]


def test_undeclared_selector_combination_stops_at_the_gate_before_any_action():
    graph = merge_traces([link_trace(), url_trace()], ["cart_route"], "c")
    result, surface, _ = run(graph, {**LOGIN, "product_name": "Sauce Labs Backpack", "cart_route": "teleport"})
    assert result.status == "failure" and result.outcome_code == "no_matching_edge" and result.step_id == "d1"
    assert result.interventions[0]["disposition"] == "abort"
    assert surface.actions == [] and surface.screen == "blank"                            # nothing ran
    assert result.executed_path == [] and result.performed_attempts == 0


def test_a_single_scenario_with_selectors_is_still_gated():
    graph = merge_traces([link_trace()], ["cart_route"], "c")
    assert graph.entry_node == "d1" and [n.id for n in graph.nodes][:2] == ["d1", "s1"]
    assert edges_out(graph, "d1") == [("s1", 0, [("input_equals", "cart_route", "cart_link")])]
    assert graph.provenance["scenarios"][0]["node_path"][:2] == ["d1", "s1"]
    result, surface, _ = run(graph, {**LOGIN, "product_name": "Sauce Labs Backpack", "cart_route": "cart_link"})
    assert result.status == "success"
    result, surface, _ = run(graph, {**LOGIN, "product_name": "Sauce Labs Backpack", "cart_route": "cart_url"})
    assert result.outcome_code == "no_matching_edge" and surface.actions == []


def test_a_campaign_without_selectors_is_a_linear_graph():
    only = trace("only", login_nodes() + [add_node(), cart_link_node("s6")],
                 {**LOGIN, "product_name": "Sauce Labs Backpack"}, ["password"])
    graph = merge_traces([only], [], "c")
    assert graph.entry_node == "s1" and all(n.kind != "decision" for n in graph.nodes)
    assert [n.id for n in linear_path(graph)] == ["s1", "s2", "s3", "s4", "s5", "s6"]
    assert graph.provenance["scenarios"][0]["node_path"] == ["s1", "s2", "s3", "s4", "s5", "s6", "success"]
    assert all(not spec.get("selector") and "required_when" not in spec for spec in graph.inputs.values())
    result, _, _ = run(graph, {**LOGIN, "product_name": "Sauce Labs Backpack"})
    assert result.status == "success"


def test_missing_conditional_input_fails_before_any_surface_action():
    graph = merge_traces([link_trace(), url_trace()], ["cart_route"], "c")
    result, surface, _ = run(graph, {**LOGIN, "product_name": "Sauce Labs Backpack", "cart_route": "cart_url"})
    assert result.status == "failure" and result.outcome_code == "missing_inputs"
    assert "cart_url" in result.observed and surface.actions == []


def test_prefix_scenario_replays_to_its_own_terminal():
    short = trace("login_only", login_nodes(), {**LOGIN, "mode": "login_only"}, ["password"],
                  success={"text_contains": "Products"})
    longer = trace("add_item", login_nodes() + [add_node()],
                   {**LOGIN, "mode": "add_item", "product_name": "Sauce Labs Backpack"}, ["password"],
                   success={"text_contains": "Products"})
    graph = merge_traces([short, longer], ["mode"], "c")
    result, surface, _ = run(graph, {**LOGIN, "mode": "login_only"})
    assert result.status == "success" and surface.cart == []
    result, surface, _ = run(graph, {**LOGIN, "mode": "add_item", "product_name": "Sauce Labs Backpack"})
    assert result.status == "success" and surface.cart == ["Sauce Labs Backpack"]
