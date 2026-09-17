"""MemberOps campaign: the spec, the merged graph's conditional inputs, the gate, the policy, redaction."""
import json
from dataclasses import replace

import pytest

from src.cua.artifact import build_linear, load_artifact, missing_inputs, outgoing, save_artifact, to_dict
from src.cua.campaign import CampaignError, load_spec, reconcile_outcomes, spec_from_dict
from src.cua.escalation import NoOperator
from src.cua.merge import merge_traces
from src.cua.models import Action, Element, GraphAction, GraphNode, Scenario, ScenarioTrace
from src.cua.policy import Policy
from src.cua.replay import replay
from tests.context import Escalator, Locator, RunLog, SessionControl, linear_node
from tests.fake_surface import FakeSurface

SPEC = "scenarios/member_account_prepare.json"
BASE = "http://127.0.0.1:8765/"
SURFACE = {"kind": "web", "app": "MemberOps Sandbox", "entry_url": BASE, "allowed_hosts": ["127.0.0.1"]}
OUTPUTS = {"member_name": {"type": "string", "required": True, "pattern": r"Member name:\s*(.+)"},
           "account_type": {"type": "string", "required": True, "pattern": r"Account type:\s*(\w+)"},
           "opening_deposit": {"type": "number", "required": True, "pattern": r"Opening deposit:\s*\$\s*([\d.]+)"}}
SUCCESS = {"text_contains": "Review new sub-account"}


def role(role_name: str, name: str) -> Locator:
    return Locator(strategies=[{"kind": "role", "role": role_name, "name": name}])


def text(prefix: str) -> Locator:
    return Locator(strategies=[{"kind": "text", "text": prefix}])


def act(action: str, target: Locator | None, value: str | None = None, checkpoint: dict | None = None,
        risky: bool = False) -> GraphNode:
    return linear_node("x", GraphAction(action=action, target=target, value=value, checkpoint=checkpoint), risky=risky)


def member_ops_nodes(lookup_method: str, account_type: str) -> list[GraphNode]:
    """What a discovery run records for one scenario, as the app's controls are perceived."""
    lookup = ([act("click", role("link", "By phone"), checkpoint={"text_contains": "Phone number"}),
               act("type", role("textbox", "Phone number"), "{{phone}}")]
              if lookup_method == "phone" else
              [act("type", role("textbox", "Member ID"), "{{member_id}}")])
    label = "Savings" if account_type == "savings" else "Checking"
    nodes = [
        act("navigate", None, BASE, checkpoint={"text_contains": "Sign in"}),
        act("type", role("textbox", "Username"), "{{username}}"),
        act("type", role("textbox", "Password"), "{{password}}"),
        act("click", role("button", "Sign in"), checkpoint={"text_contains": "Member lookup"}),
        *lookup,
        act("click", role("button", "Search"), checkpoint={"text_contains": "Member profile"}),
        act("click", role("link", "New sub-account"), checkpoint={"text_contains": "Choose the account type"}),
        act("click", role("link", label), checkpoint={"text_contains": "Opening deposit"}),
        act("type", role("textbox", "Opening deposit (USD)"), "{{opening_deposit}}"),
        act("click", role("button", "Continue to review"), checkpoint=SUCCESS),
        act("extract", text("Member name:"), "member_name"),
        act("extract", text("Account type:"), "account_type"),
        act("extract", text("Opening deposit:"), "opening_deposit"),
    ]
    return [replace(node, id=f"s{index}") for index, node in enumerate(nodes, start=1)]


def trace(scenario: Scenario, outcomes=()) -> ScenarioTrace:
    artifact = build_linear(name="member_account_prepare", goal="prepare a sub-account", surface_meta=SURFACE,
                            params=scenario.params, nodes=member_ops_nodes(scenario.params["lookup_method"],
                                                                            scenario.params["account_type"]),
                            outputs=OUTPUTS, outcomes=[dict(o) for o in outcomes], success=SUCCESS,
                            run_id=f"discovery-{scenario.name}", sensitive=set(scenario.sensitive))
    return ScenarioTrace(scenario=scenario, artifact=artifact, run_id=f"discovery-{scenario.name}", planner="scripted")


@pytest.fixture
def merged():
    spec = load_spec(SPEC)
    return merge_traces([trace(s, spec.outcomes) for s in spec.scenarios], spec.selectors, campaign_id="campaign-test")


OPEN_SPEC = "scenarios/member_account_open.json"


def open_nodes() -> list[GraphNode]:
    """What a supervised discovery records for the side-effecting capability: the human approved the final click."""
    prepare = member_ops_nodes("member_id", "savings")[:-3]         # up to Continue to review, no extracts
    final = [act("click", role("button", "Confirm and Open Account"),
                 checkpoint={"text_contains": "Sub-account opened"}, risky=True),
             act("extract", text("Account "), "account_number"),
             act("extract", text("Account "), "opening_deposit")]
    return [replace(node, id=f"s{index}") for index, node in enumerate(prepare + final, start=1)]


# ---------- the specification ----------

def test_spec_declares_three_paths_with_only_the_inputs_they_need():
    spec = load_spec(SPEC)
    assert spec.name == "member_account_prepare" and spec.selectors == ["lookup_method", "account_type"]
    assert [s.name for s in spec.scenarios] == ["member_id_savings", "phone_savings", "member_id_checking"]
    combos = [(s.params["lookup_method"], s.params["account_type"]) for s in spec.scenarios]
    assert combos == [("member_id", "savings"), ("phone", "savings"), ("member_id", "checking")]
    assert ("phone", "checking") not in combos                       # deliberately undeclared
    assert "phone" not in spec.scenarios[0].params and "member_id" not in spec.scenarios[1].params
    assert all(s.sensitive == ["password"] for s in spec.scenarios)
    assert sorted(spec.outputs) == ["account_type", "member_name", "opening_deposit"]
    assert spec.outputs["opening_deposit"]["type"] == "number"
    for phrase in ("lookup_method", "account_type", "Never click 'Confirm and Open Account'", "opening deposit"):
        assert phrase in spec.goal
    data = json.load(open(SPEC))
    data["scenarios"].append({"name": "dup", "params": {**data["scenarios"][0]["params"]}, "sensitive": []})
    with pytest.raises(CampaignError, match="same selector values"):
        spec_from_dict(data)


# ---------- the merged graph ----------

def test_merged_graph_has_conditional_inputs_per_path(merged):
    inputs = merged.inputs
    assert inputs["lookup_method"]["selector"] is True and inputs["account_type"]["selector"] is True
    for always in ("username", "password", "opening_deposit"):
        assert inputs[always]["required"] is True and "required_when" not in inputs[always]
    assert inputs["password"]["sensitive"] is True
    assert inputs["member_id"] == {**inputs["member_id"], "required": False, "required_when": [
        {"lookup_method": "member_id", "account_type": "savings"},
        {"lookup_method": "member_id", "account_type": "checking"}]}
    assert inputs["phone"] == {**inputs["phone"], "required": False,
                               "required_when": [{"lookup_method": "phone", "account_type": "savings"}]}
    base = {"username": "operator", "password": "training_only", "opening_deposit": "500"}
    assert missing_inputs(inputs, {**base, "lookup_method": "phone", "account_type": "savings"}) == ["phone"]
    assert missing_inputs(inputs, {**base, "lookup_method": "member_id", "account_type": "checking"}) == ["member_id"]
    complete = {**base, "lookup_method": "phone", "account_type": "savings", "phone": "5550101"}
    assert missing_inputs(inputs, complete) == []


def test_merged_graph_shares_login_and_branches_on_lookup_then_account_type(merged):
    gate = [(e.priority, [(g.input, g.value) for g in e.guards]) for e in outgoing(merged, merged.entry_node)]
    assert gate == [(0, [("lookup_method", "member_id"), ("account_type", "savings")]),
                    (1, [("lookup_method", "phone"), ("account_type", "savings")]),
                    (2, [("lookup_method", "member_id"), ("account_type", "checking")])]
    paths = {s["name"]: s["node_path"] for s in merged.provenance["scenarios"]}
    shared = ["d1", "s1", "s2", "s3", "s4"]                              # the login prefix
    assert all(path[:5] == shared for path in paths.values())
    assert paths["member_id_savings"][5] == paths["member_id_checking"][5] == "d2"   # lookup split
    assert paths["phone_savings"][6] != paths["member_id_savings"][6]
    assert merged.outputs == {name: {**spec, "pattern": spec["pattern"]} for name, spec in OUTPUTS.items()}


def test_undeclared_phone_checking_is_rejected_before_any_surface_action(merged):
    surface = FakeSurface()
    log = RunLog("replay", secrets=("training_only",))
    params = {"username": "operator", "password": "training_only", "opening_deposit": "250",
              "lookup_method": "phone", "phone": "5550101", "account_type": "checking"}
    result = replay(merged, params, surface, Policy(allowed_hosts=["127.0.0.1"]),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "failure" and result.outcome_code == "no_matching_edge" and result.step_id == "d1"
    assert surface.actions == [] and result.interventions[0]["disposition"] == "abort"


# ---------- policy and redaction ----------

def test_policy_classifies_the_final_button_as_irreversible_but_not_blocked():
    policy = Policy(allowed_hosts=["127.0.0.1"])
    click = lambda name: Action(kind="click", target=Element(role="button", name=name))
    assert policy.check(click("Confirm and Open Account"), BASE).decision == "confirm"
    assert policy.risk_of(click("Confirm and Open Account")) == "risky"
    for safe in ("Sign in", "Search", "New sub-account", "Continue to review", "Dismiss", "Acknowledge", "Back"):
        assert policy.check(click(safe), BASE).decision == "allow", safe
    # Self-registration stays blocked outright; the sandbox's final action is deliberately not named that way.
    assert policy.check(click("Confirm and Create Account"), BASE).decision == "deny"
    assert policy.check(click("Sign in"), "http://evil.example/").decision == "deny"


def test_password_never_reaches_the_artifact_or_the_log(merged, tmp_path):
    path = save_artifact(merged, secrets=("training_only",), path=tmp_path / "member_account_prepare.v1.json")
    saved = path.read_text()
    assert "training_only" not in saved and "{{password}}" in saved
    assert json.loads(saved)["inputs"]["password"]["sensitive"] is True
    log = RunLog("replay", secrets=("training_only",))
    log.event("acted", kind="type", value="training_only", note={"password": "training_only"})
    assert "training_only" not in log.path.read_text() and "[REDACTED]" in log.path.read_text()


# ---------- the reviewer-declared outcome contract ----------

def test_spec_declares_reviewer_outcomes_with_the_apps_exact_text():
    spec = load_spec(SPEC)
    by_code = {o["code"]: o for o in spec.outcomes}
    assert set(by_code) == {"invalid_credentials", "member_not_found", "member_access_denied",
                            "invalid_opening_deposit", "dismiss_maintenance_notice"}
    assert all(o["source"] == "reviewer" for o in spec.outcomes)
    assert by_code["member_access_denied"]["detect"] == {"text_contains": "This member record is restricted"}
    notice = by_code["dismiss_maintenance_notice"]
    assert notice["kind"] == "recoverable" and notice["detect"] == {"dialog_contains": "Scheduled maintenance"}
    assert notice["recover"]["target"]["strategies"] == [{"kind": "role", "role": "button", "name": "Dismiss"}]
    from examples.member_ops.app import NOTICE_TEXT
    assert "Scheduled maintenance" in NOTICE_TEXT


def test_spec_outcomes_are_validated_by_the_artifact_rules():
    def rejects(outcomes, message):
        data = json.load(open(SPEC))
        data["outcomes"] = outcomes
        with pytest.raises(CampaignError, match=message):
            spec_from_dict(data)

    business = {"code": "x", "kind": "business", "detect": {"text_contains": "y"}}
    rejects({"code": "x"}, "outcomes must be a list")
    rejects([{**business, "kind": "fatal"}], "needs a code and a kind")
    rejects([{**business, "detect": {}}], "detect needs text_contains, text_missing, or dialog_contains")
    rejects([{**business, "detect": {"regex": "y"}}], "detect has unknown keys \\['regex'\\]")
    rejects([{**business, "detect": {"text_contains": ""}}], "every detect value must be a non-empty string")
    rejects([{**business, "recover": {"action": "click"}}], "only recoverable outcomes carry a recover action")
    rejects([{**business, "kind": "recoverable"}], "recoverable outcomes need a recover action")
    rejects([{**business, "kind": "recoverable", "recover": {"action": "press"}}], "recover action in")
    rejects([{**business, "kind": "recoverable", "recover": {"action": "click", "target": {"strategies": []}}}],
            "non-empty list of locator strategies")
    rejects([business, business], "outcome code 'x' is declared twice")
    rejects([{**business, "source": "planner"}], "source must be 'reviewer'")
    rejects([{**business, "priority": 1}], "outcome 0 has unknown keys: \\['priority'\\]")
    data = json.load(open(SPEC))
    data.pop("outcomes")
    assert spec_from_dict(data).outcomes == []                       # optional: specs without it are unchanged


def test_reviewer_outcomes_are_authoritative_for_their_codes():
    from tests.context import RunLog
    declared = [{"code": "a", "kind": "business", "source": "reviewer", "detect": {"text_contains": "A"}}]
    planner_same = {"code": "a", "kind": "business", "source": "planner", "detect": {"text_contains": "A"}}
    planner_other = {"code": "b", "kind": "business", "source": "planner", "detect": {"text_contains": "B"}}
    planner_diff = {**planner_same, "detect": {"text_contains": "A!"}}

    log = RunLog("discovery")
    log.event("probe")
    assert reconcile_outcomes([planner_other, planner_same], declared, log) == [planner_other, declared[0]]
    assert "reviewer_outcome_overrode_planner" not in log.path.read_text()      # identical: silent dedupe

    log = RunLog("discovery")
    assert reconcile_outcomes([planner_diff, planner_other], declared, log) == [planner_other, declared[0]]
    [event] = [json.loads(l) for l in log.path.read_text().splitlines() if "reviewer_outcome_overrode_planner" in l]
    assert event["code"] == "a" and event["planner"]["detect"] == {"text_contains": "A!"}
    assert event["reviewer"]["detect"] == {"text_contains": "A"}

    log = RunLog("discovery")
    log.event("probe")
    many = [planner_diff, planner_same, {**planner_diff, "detect": {"text_contains": "A?"}}]
    assert reconcile_outcomes(many, declared, log) == declared                   # exactly one reviewer version
    assert log.path.read_text().count("reviewer_outcome_overrode_planner") == 2
    assert reconcile_outcomes([], declared) == declared and reconcile_outcomes([planner_other], []) == [planner_other]


def test_declared_outcomes_survive_trace_merge_and_round_trip(merged, tmp_path):
    spec = load_spec(SPEC)
    assert [o["code"] for o in merged.outcomes] == [o["code"] for o in spec.outcomes]
    assert all(o["source"] == "reviewer" for o in merged.outcomes)
    path = save_artifact(merged, secrets=("training_only",), path=tmp_path / "prepare.json")
    loaded = load_artifact(path)
    assert loaded.outcomes == merged.outcomes == spec.outcomes


# ---------- the separate side-effecting capability ----------

def test_open_spec_is_a_single_supervised_scenario_that_really_opens_an_account():
    spec = load_spec(OPEN_SPEC)
    assert spec.name == "member_account_open" and spec.selectors == [] and len(spec.scenarios) == 1
    assert "really opens the sub-account" in spec.goal and "Confirm and Open Account" in spec.goal
    assert sorted(spec.outputs) == ["account_number", "opening_deposit"]
    assert spec.scenarios[0].sensitive == ["password"] and "member_id" in spec.scenarios[0].params
    prepare = load_spec(SPEC)
    assert "Never click 'Confirm and Open Account'" in prepare.goal   # the safe capability stays safe


def test_open_spec_shares_the_prepare_outcomes_it_needs():
    prepare = {o["code"]: o for o in load_spec(SPEC).outcomes}
    opener = {o["code"]: o for o in load_spec(OPEN_SPEC).outcomes}
    assert set(opener) == {"invalid_credentials", "member_not_found", "member_access_denied",
                           "invalid_opening_deposit", "dismiss_maintenance_notice"}
    for code, outcome in opener.items():
        assert outcome == prepare[code], code                    # the exact same definitions


def test_open_capability_records_the_final_click_as_irreversible_and_never_retry():
    spec = load_spec(OPEN_SPEC)
    scenario = spec.scenarios[0]
    artifact = build_linear(name=spec.name, goal=spec.goal, surface_meta=SURFACE, params=scenario.params,
                            nodes=open_nodes(), outputs=spec.outputs, outcomes=spec.outcomes,
                            success={"text_contains": "Sub-account opened"}, run_id="discovery-open",
                            sensitive={"password"})
    graph = merge_traces([ScenarioTrace(scenario, artifact, "discovery-open", "scripted")], [], "campaign-open")
    final = next(n for n in graph.nodes if n.kind == "action" and n.action.target
                 and n.action.target.strategies[0].get("name") == "Confirm and Open Account")
    assert (final.effect, final.retry_safety) == ("irreversible", "never_retry")
    assert graph.entry_node == "s1" and all(n.kind != "decision" for n in graph.nodes)
    assert "training_only" not in json.dumps(to_dict(graph)) and '"risk"' not in json.dumps(to_dict(graph))
